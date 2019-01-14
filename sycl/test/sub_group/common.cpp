// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable when use SPIRV operations instead direct built-ins calls.
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==-------------- common.cpp - SYCL sub_group common test -----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "helper.hpp"
#include <CL/sycl.hpp>

using namespace cl::sycl;
struct Data {
  size_t local_id;
  size_t local_range;
  size_t max_local_range;
  size_t group_id;
  size_t group_range;
  size_t uniform_group_range;
};

void check(queue &Queue, const int G, const int L) {

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
    size_t max_sg =
        Queue.get_device().get_info<info::device::max_num_sub_groups>();
    size_t num_sg = L / max_sg + (L % max_sg ? 1 : 0);
    for (int j = 0; j < G; j++) {
      size_t group_id = j % L / max_sg;
      size_t local_range =
          (group_id + 1 == num_sg) ? (L - group_id * max_sg) : max_sg;
      exit_if_not_equal(syclacc[j].local_id, j % L % max_sg, "local_id");
      exit_if_not_equal(syclacc[j].local_range, local_range, "local_range");
      exit_if_not_equal(syclacc[j].max_local_range, max_sg, "max_local_range");
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
