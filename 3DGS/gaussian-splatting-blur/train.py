#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Note by lathika 
"""
python train.py -s /home/lathika/Workspace/Data_Sets/My_Data_sets/Chair -m /home/lathika/Workspace/test/Dump/Dump/wavelets_7 --eval --iterations 7000 --ortho_gauss
"""

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, OptimizeBlurParams   # Code by lathika - added "OptimizeBlurParams"
from utils.blur_utils import apply_gaussian_blur, detect_H_freq_with_DoG, save_torch_image_3D, save_torch_image_2D, apply_mag_kernel, get_3d_points_and_gaussian_index, get_world_scales # Code by lathika 
import numpy as np # Code by lathika
import cv2  # Code by lathika
import math # Code by lathika

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# Code by lathika - nothing new
def densify_func_adc(dataset, opt, iteration, gaussians, visibility_filter, viewspace_point_tensor, scene, radii):
    if dataset.ortho_gauss:
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration > opt.densify_from_iter and iteration % (opt.densification_interval) == 0:  # (opt.densification_interval//2)
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent , size_threshold, radii) # *2
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()
    else:
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

def training(dataset, opt, bopt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):   # Code by lathika - added "bopt"

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, bopt, gaussians)   # Code by lathika - added "bopt"
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    iterations_list =[] # Code by lathika - test

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        
        gt_image = viewpoint_cam.original_image.cuda()

        # Code by lathika
        gt_original_img = gt_image
        
        if dataset.ortho_gauss:

            sigma_base = bopt.sigma_base
            num_of_stages = bopt.num_of_stages
            bsdp = bopt.blur_stage_divider_pow # 1

            # For testing
            if (iteration==1):
                print(f"Margins = {[min( math.floor(((i/num_of_stages)**bsdp)*bopt.blur_until_iter) + 1, math.floor(((i/num_of_stages)**bsdp)*opt.iterations) + 1) for i in range(num_of_stages)]}")

            for i in range(1,num_of_stages):    # If there are 5 stages, we blur 4 times and last one is the original
                low = min( math.floor((((i-1)/num_of_stages)**bsdp)*bopt.blur_until_iter) + 1, math.floor((((i-1)/num_of_stages)**bsdp)*opt.iterations) + 1)    #min((i-1)*(opt.densify_until_iter - opt.densify_from_iter)//num_of_stages + opt.densify_from_iter, (i-1)*opt.iterations//num_of_stages + 1)
                high = min( math.floor(((i/num_of_stages)**bsdp)*bopt.blur_until_iter) + 1, math.floor(((i/num_of_stages)**bsdp)*opt.iterations) + 1)   #min(i*(opt.densify_until_iter - opt.densify_from_iter)//num_of_stages + opt.densify_from_iter, i*opt.iterations//num_of_stages + 1)
                if low <= iteration < high:
                    stage = i
                    sigma = sigma_base**(num_of_stages-1-i)    # According to the SIFT paper, it is better to increase the sigma by root 2 times when incrementing
                    gt_image, kernel_size = apply_gaussian_blur(gt_image, sigma)
                    # Test
                    if high - 1 <= iteration < high: #low <= iteration < low+1:
                        iterations_list.append(high)

                    if iteration == high -1:
                        print(f"Iteration_margines = {iteration}, points = {gaussians._xyz.shape[0]}")
                    break
                else:
                    stage = num_of_stages   # No blurring
                    low = min( math.floor((((num_of_stages-1)/num_of_stages)**bsdp)*bopt.blur_until_iter) + 1, math.floor((((num_of_stages-1)/num_of_stages)**bsdp)*opt.iterations) + 1)    
                    high = min( bopt.blur_until_iter + 1, opt.iterations + 1)

        ##
        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Code by lathika - test
            """
            print(f"Viewmat = {viewpoint_cam.world_view_transform}\n")
            print(f"Full proj = {viewpoint_cam.full_proj_transform}\n")
            print(f"H = {gt_original_img.shape[1]}, W = {gt_original_img.shape[2]}")
            if iteration > 1:
                break
            it_1 = 7000
            if iteration==it_1 and dataset.ortho_gauss:
                print(f"iterations_list={iterations_list}")
                depth_extract = render_pkg["depth_extract"]
                #print(f"depth_extract shape = {depth_extract.squeeze(0).shape}")
                np.savetxt("/home/lathika/Workspace/test/Dump/Dump_blur/depths.txt", depth_extract.squeeze(0).cpu().numpy().round(2) , delimiter=', ', fmt='%.2f')
            """

            """
            # Test
            if len(iterations_list)> num_of_stages -2: # Need to change this condition #len(iterations_list)>0:
                print(f"iterations_list = {iterations_list}")
                if  iteration >= iterations_list[-1]:
                    # Test
                    sigma_2 = 2**(0.5*(num_of_stages-stage))    # For testing, when stage ==1 , need to think more
                    gt_image_2, kernel_size_2 = apply_gaussian_blur(gt_original_img, sigma_2)

                    dog_image, results, out_image, kernel_size_vis = detect_H_freq_with_DoG(gt_image, gt_image_2, kernel_size) 
                    #dog_image, results, out_image, kernel_size_vis = detect_H_freq_with_DoG(gt_image, image, kernel_size) # Here the Ground truth is also adjusted (Blured or not) according to the stage
                    out_image_magnified = apply_mag_kernel(out_image, kernel_size_vis)
                    print(f"iteration = {iteration}")
                    print(f"dog_image_shape = {dog_image.shape}")
                    save_torch_image_2D(dog_image, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/dog_{iteration}.jpg")
                    print(f"results length = {len(results)}")
                    print(f"out_image_magnified shape = {out_image_magnified.shape}")
                    save_torch_image_2D(out_image_magnified, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/mag_{iteration}.jpg")
                    save_torch_image_3D(image, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/output{iteration}.jpg")
                    save_torch_image_3D(gt_image, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/output_gt{iteration}.jpg")

                    # Depth check around the residual points
                    depth_extract = render_pkg["depth_extract"]
                    point = results[2000]
                    print(f"selected point ={point}")
                    depth_portion = depth_extract.squeeze(0).cpu().numpy().round(2)[max(0,point[0]-20):min(dog_image.shape[0],point[0]+20), max(0,point[1]-20):min(dog_image.shape[1],point[1]+20)]
                    depth_portion[depth_portion.shape[0]//2,depth_portion.shape[1]//2] = 100000.0   # To recognize the mid point
                    np.savetxt("/home/lathika/Workspace/test/Dump/Dump_blur/depths_portion.txt", depth_portion , delimiter=', ', fmt='%.2f')
                    
                    prom_gauss_idx = render_pkg["prom_gauss_idx"]
                    prom_gauss_portion = prom_gauss_idx.squeeze(0).cpu().numpy().round(2)[max(0,point[0]-20):min(dog_image.shape[0],point[0]+20), max(0,point[1]-20):min(dog_image.shape[1],point[1]+20)]
                    prom_gauss_portion[prom_gauss_portion.shape[0]//2,prom_gauss_portion.shape[1]//2] = 99999   # To recognize the mid point
                    np.savetxt("/home/lathika/Workspace/test/Dump/Dump_blur/prom_gauss_portion.txt", prom_gauss_portion , delimiter=', ')

                    min_d = torch.min(depth_extract.squeeze(0).flatten())
                    max_d = torch.max(depth_extract.squeeze(0).flatten())
                    d_img =  (depth_extract.squeeze(0)-min_d)/(max_d-min_d+0.0001)*255
                    print(d_img[100:110,100:110])
                    save_torch_image_2D(d_img, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/depth_{iteration}.jpg")

                    break
            """

            # Code by lathika - Test - Saving the images
            if iteration == 7000:
                save_torch_image_3D(image, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/output{iteration}.jpg")
                save_torch_image_3D(gt_image, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/output_gt{iteration}.jpg")

            # Densification
            
            # Code by lathika
            if dataset.ortho_gauss and bopt.activate_blur_densify:
                blur_densify_interval = bopt.blur_densify_interval  
                if (bopt.blur_densify_method ==1):
                    densify_low = low + math.floor((high-low)*bopt.blur_in_stage_densify_start_portion)
                    densify_high = low + math.floor((high-low)*bopt.blur_in_stage_densify_end_portion)
                else:
                    densify_low = low + bopt.blur_in_stage_densify_start_after_iter
                    densify_high = low + bopt.blur_in_stage_densify_end_after_iter

                # Test scales
                test =0
                if (densify_low  <= iteration < densify_low + blur_densify_interval):
                    test = 1

            
                if (bopt.blur_densify_until_stage>stage>1) and (densify_low <= iteration < densify_high) and (iteration%blur_densify_interval == 0):
                    if (bopt.use_rendered_image):
                        image_compare = image
                    else:
                        sigma_2 = sigma_base**(num_of_stages-stage)    # For testing, when stage ==1 sigma_2 will be 4*root(2), need to think more
                                                                    # This method will intriduce points at the edges and will reduce the number of residual points 
                                                                    # when compared with using the original rendered image with the blurred gt_image
                        gt_image_2, kernel_size_2 = apply_gaussian_blur(gt_original_img, sigma_2)
                        image_compare = gt_image_2

                    dog_image, results, out_image, kernel_size_vis = detect_H_freq_with_DoG(gt_image, image_compare, kernel_size) # image, gt_image_2 # Here gt_image_2 is a blurred gt_truth with sigma is root(2) [base sigma] times bigger w.r.t gt_image
                    print(f"results shape = {results.shape} at iteration = {iteration} stage = {stage}")
                    depth_extract = render_pkg["depth_extract"].squeeze(0)
                    prom_gauss_idx = render_pkg["prom_gauss_idx"].squeeze(0)
                    orig_coordinates, gaussian_indexes = get_3d_points_and_gaussian_index(depth_extract, prom_gauss_idx, results, viewpoint_cam.full_proj_transform, portion = bopt.residual_portion) #0.1
                    tanfovx, tanfovy = math.tan(viewpoint_cam.FoVx * 0.5), math.tan(viewpoint_cam.FoVy * 0.5)
                    H, W = depth_extract.shape
                    new_scales = get_world_scales(sigma, viewpoint_cam.world_view_transform, tanfovx, tanfovy, orig_coordinates,H, W)            
                    gaussians.blur_densify(orig_coordinates, gaussian_indexes, new_scales, radii, test)   # Added "test" for testing  # We dont need radii  
                    test  =0 # test
                    radii = gaussians.tmp_radii

                    # Test
                    save_torch_image_2D(dog_image, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/dog_{iteration}.jpg")
                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                    im_new = render_pkg["render"]
                    save_torch_image_3D(im_new, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/output_old_angle_{iteration}.jpg")
                    for i in range(5):
                        if not viewpoint_stack:
                            viewpoint_stack = scene.getTrainCameras().copy()
                            viewpoint_indices = list(range(len(viewpoint_stack)))
                        rand_idx = randint(0, len(viewpoint_indices) - 1)
                        viewpoint_cam = viewpoint_stack.pop(rand_idx)
                        vind = viewpoint_indices.pop(rand_idx)
                        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                        im_new = render_pkg["render"]
                        save_torch_image_3D(im_new, f"/home/lathika/Workspace/test/Dump/Dump_blur/images/output_new_angle__{iteration+i}.jpg")
                    
                    #break

            # Code by lathika
            densify_func_adc(dataset, opt, iteration, gaussians, visibility_filter, viewspace_point_tensor, scene, radii)
            if iteration == opt.iterations:
                print(f"Iteration = {iteration}, points = {gaussians._xyz.shape[0]}")
            
            """
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            """

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    bopt = OptimizeBlurParams(parser)   # Code by lathika
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), bopt.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)  # Code by lathika - added "bopt"

    # All done
    print("\nTraining complete.")
