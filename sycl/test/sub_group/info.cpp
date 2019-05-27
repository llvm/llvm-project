// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------- info.cpp - SYCL sub_group parameters test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
class kernel_sg;
using namespace cl::sycl;

int main() {

  queue Queue;
  device Device = Queue.get_device();
  /* Check info::device parameters. */
  if (Queue.is_host()) {
    try {
      Device.get_info<info::device::max_num_sub_groups>();
      std::cout << "Expected exception has not been generated\n";
      return 1;
    } catch (runtime_error e) {
      /* Expected exception. */
    }
    try {
      Device.get_info<info::device::sub_group_independent_forward_progress>();
      std::cout << "Expected exception has not been generated\n";
      return 1;
    } catch (runtime_error e) {
      /* Expected exception. */
    }
    try {
      Device.get_info<info::device::sub_group_sizes>();
      std::cout << "Expected exception has not been generated\n";
      return 1;
    } catch (runtime_error e) {
       /* Expected exception. */
    }
  } else {
    Device.get_info<info::device::sub_group_independent_forward_progress>();
    Device.get_info<info::device::max_num_sub_groups>();
    // There is no support on CPU and accelerator devices for now.
    if ( Device.get_info<info::device::device_type>() == info::device_type::cpu ||
         Device.get_info<info::device::device_type>() == 
             info::device_type::accelerator) {
      try {
        Device.get_info<info::device::sub_group_sizes>();
        std::cout << "Expected exception has not been generated\n";
        return 1;
      } catch (runtime_error e) {
        /* Expected exception. */
      }
    } else {
      Device.get_info<info::device::sub_group_sizes>();
    }

    /* Basic sub-group functionality is supported as part of cl_khr_subgrou
     * extension or as core OpenCL 2.1 feature. */
    if (!core_sg_supported(Device)) {
      std::cout << "Skipping test\n";
      return 0;
    }
    try {
      size_t max_sg_num = get_sg_size(Device);
      size_t max_wg_size = Device.get_info<info::device::max_work_group_size>();
      program Prog(Queue.get_context());
      /* TODO: replace with pure SYCL code when fixed problem with consumption 
       * kernels defined using program objects on GPU device
      Prog.build_with_kernel_type<kernel_sg>();
      kernel Kernel = Prog.get_kernel<kernel_sg>();

      Queue.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<kernel_sg>(
            nd_range<2>(range<2>(50, 40), range<2>(10, 20)), Kernel,
            [=](nd_item<2> index) {});
      });*/
      Prog.build_with_source("kernel void "
                             "kernel_sg(global double* a, global double* b, "
                             "global double* c) {*a=*b+*c; }\n");
      kernel Kernel = Prog.get_kernel("kernel_sg");
      size_t Res = 0;
      for (auto r : {range<3>(3, 4, 5), range<3>(1, 1, 1), range<3>(4, 2, 1),
                     range<3>(32, 3, 4), range<3>(7, 9, 11)}) {
        Res = Kernel.get_sub_group_info<
            info::kernel_sub_group::max_sub_group_size_for_ndrange>(Device, r);
        exit_if_not_equal(Res, min(r.size(), max_sg_num),
                          "max_sub_group_size_for_ndrange");
        Res = Kernel.get_sub_group_info<
            info::kernel_sub_group::sub_group_count_for_ndrange>(Device, r);
        exit_if_not_equal<size_t>(
            Res, r.size() / max_sg_num + (r.size() % max_sg_num ? 1 : 0),
            "sub_group_count_for_ndrange");
      }

      Res = Kernel.get_sub_group_info<
          info::kernel_sub_group::compile_num_sub_groups>(Device);

      /* Sub-group size is not specified in kernel or IL*/
      exit_if_not_equal<size_t>(Res, 0, "compile_num_sub_groups");

      /* Check work-group sizea which can accommodate the requested number of
       * sub-groups*/
      for (auto s : {(size_t)200, (size_t)1, (size_t)3, (size_t)5, (size_t)7,
                     (size_t)13, max_sg_num, max_sg_num + 1}) {
        range<3> ResRange = Kernel.get_sub_group_info<
            info::kernel_sub_group::local_size_for_sub_group_count>(Device, s);
        if (s * max_sg_num <= max_wg_size) {
          exit_if_not_equal<size_t>(ResRange[0], s * max_sg_num,
                                    "local_size_for_sub_group_count[0]");
          exit_if_not_equal<size_t>(ResRange[1], 1,
                                    "local_size_for_sub_group_count[1]");
          exit_if_not_equal<size_t>(ResRange[2], 1,
                                    "local_size_for_sub_group_count[2]");

        } else {
          exit_if_not_equal<size_t>(ResRange[0], 0,
                                    "local_size_for_sub_group_count[0]");
          exit_if_not_equal<size_t>(ResRange[1], 0,
                                    "local_size_for_sub_group_count[1]");
          exit_if_not_equal<size_t>(ResRange[2], 0,
                                    "local_size_for_sub_group_count[2]");
        }
      }
    } catch (exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }
  std::cout << "Test passed.\n";
  return 0;
}

