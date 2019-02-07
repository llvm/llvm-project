// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==-------------- common.cpp - SYCL sub_group common test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
using namespace cl::sycl;
struct Data {
  unsigned int local_id;
  unsigned int local_range;
  unsigned int max_local_range;
  unsigned int group_id;
  unsigned int group_range;
  unsigned int uniform_group_range;
};

void check(queue &Queue, unsigned int G, unsigned int L) {

  try {
    nd_range<1> NdRange(G, L);
    buffer<struct Data, 1> syclbuf(G);

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
    unsigned int max_sg = get_sg_size(Queue.get_device());
    unsigned int num_sg = L / max_sg + (L % max_sg ? 1 : 0);
    for (int j = 0; j < G; j++) {
      unsigned int group_id = j % L / max_sg;
      unsigned int local_range =
          (group_id + 1 == num_sg) ? (L - group_id * max_sg) : max_sg;
      exit_if_not_equal(syclacc[j].local_id, j % L % max_sg, "local_id");
      exit_if_not_equal(syclacc[j].local_range, local_range, "local_range");
      // TODO: Currently workgroup size affects this paramater on CPU and does
      // not on GPU. Remove if when it is aligned.
      if (Queue.get_device().get_info<info::device::device_type>() ==
          info::device_type::cpu) {
        exit_if_not_equal(syclacc[j].max_local_range, std::min(max_sg, L),
                          "max_local_range");
      } else {
        exit_if_not_equal(syclacc[j].max_local_range, max_sg,
                          "max_local_range");
      }
      exit_if_not_equal(syclacc[j].group_id, group_id, "group_id");
      exit_if_not_equal(syclacc[j].group_range, num_sg, "group_range");
      exit_if_not_equal(syclacc[j].uniform_group_range, num_sg,
                        "uniform_group_range");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  check(Queue, 240, 80);
  check(Queue, 8, 4);
  check(Queue, 24, 12);
  check(Queue, 1024, 256);
  std::cout << "Test passed." << std::endl;
  return 0;
}
