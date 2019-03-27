// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %S/sg.cl -triple spir64-unknown-unknown -emit-llvm-bc -o %T/kernel_ocl.bc -include opencl-c.h
// RUN: llvm-spirv %T/kernel_ocl.bc -o %T/kernel_ocl.spv
// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %T/kernel_ocl.spv
// RUN: %GPU_RUN_PLACEHOLDER %t.out %T/kernel_ocl.spv
// RUN: %ACC_RUN_PLACEHOLDER %t.out %T/kernel_ocl.spv
//==--- common_ocl.cpp - basic SG methods in SYCL vs OpenCL  ---*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
using namespace cl::sycl;
struct Data {
  unsigned int local_id;
  unsigned int local_range;
  unsigned int max_local_range;
  unsigned int group_id;
  unsigned int group_range;
  unsigned int uniform_group_range;
};

void check(queue &Queue, const int G, const int L, const char *SpvFile) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<struct Data, 1> oclbuf(G);
    buffer<struct Data, 1> syclbuf(G);

    std::ifstream File(SpvFile, std::ios::binary);
    if (!File.is_open()) {
      std::cerr << std::strerror(errno);
      throw compile_program_error("Cannot open SPIRV file\n");
    }
    File.seekg(0, std::ios::end);
    vector_class<char> Spv(File.tellg());
    File.seekg(0);
    File.read(Spv.data(), Spv.size());
    File.close();
    int Err;
    cl_program ClProgram = clCreateProgramWithIL(Queue.get_context().get(),
                                                 Spv.data(), Spv.size(), &Err);
    CHECK_OCL_CODE(Err);
    CHECK_OCL_CODE(
        clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr));
    program Prog(Queue.get_context(), ClProgram);
    Queue.submit([&](handler &cgh) {
      auto oclacc = oclbuf.get_access<access::mode::read_write>(cgh);
      cgh.set_args(oclacc);
      cgh.parallel_for(NdRange, Prog.get_kernel("ocl_subgr"));
    });
    size_t NumSG = Prog.get_kernel("ocl_subgr")
                       .get_sub_group_info<
                           info::kernel_sub_group::sub_group_count_for_ndrange>(
                           Queue.get_device(), range<3>(G, 1, 1));
    auto oclacc = oclbuf.get_access<access::mode::read_write>();

    Queue.submit([&](handler &cgh) {
      auto syclacc = syclbuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class sycl_subgr>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        syclacc[NdItem.get_global_id()].local_id = SG.get_local_id().get(0);
        syclacc[NdItem.get_global_id()].local_range =
            SG.get_local_range().get(0);
        syclacc[NdItem.get_global_id()].max_local_range =
            SG.get_max_local_range().get(0);
        syclacc[NdItem.get_global_id()].group_id = SG.get_group_id().get(0);
        syclacc[NdItem.get_global_id()].group_range = SG.get_group_range();
        syclacc[NdItem.get_global_id()].uniform_group_range =
            SG.get_uniform_group_range();
      });
    });
    auto syclacc = syclbuf.get_access<access::mode::read_write>();
    for (int j = 0; j < G; j++) {
      exit_if_not_equal(syclacc[j].local_id, oclacc[j].local_id, "local_id");
      exit_if_not_equal(syclacc[j].local_range, oclacc[j].local_range,
                        "local_range");
      exit_if_not_equal(syclacc[j].max_local_range, oclacc[j].max_local_range,
                        "max_local_range");
      exit_if_not_equal(syclacc[j].group_id, oclacc[j].group_id, "group_id");
      exit_if_not_equal(syclacc[j].group_range, oclacc[j].group_range,
                        "group_range");
      exit_if_not_equal(syclacc[j].uniform_group_range,
                        oclacc[j].uniform_group_range, "uniform_group_range");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
int main(int argc, char **argv) {
  queue Queue;
  if (!core_sg_supported(Queue.get_device()) || argc != 2) {
    std::cout << "Skipping test\n";
    return 0;
  }

  check(Queue, 240, 80, argv[1]);
  check(Queue, 8, 4, argv[1]);
  check(Queue, 24, 12, argv[1]);
  check(Queue, 1024, 256, argv[1]);
  std::cout << "Test passed." << std::endl;
  return 0;
}
