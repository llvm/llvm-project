// RUN: %clang -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------- barrier.cpp - SYCL sub_group barrier test -------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
#include <limits>
#include <numeric>
template <typename T> class sycl_subgr;
using namespace cl::sycl;
template <typename T> void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    std::vector<T> data(G);
    std::iota(data.begin(), data.end(), sizeof(T));
    buffer<T> addbuf(data.data(), range<1>(G));
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto addacc = addbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<sycl_subgr<T>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        size_t lid = SG.get_local_id().get(0);
        size_t gid = NdItem.get_global_id(0);
        size_t SGoff = gid - lid;

        T res = 0;
        for (size_t i = 0; i <= lid; i++) {
          res += addacc[SGoff + i];
        }
        SG.barrier(access::fence_space::global_space);
        addacc[gid] = res;
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
      });
    });
    auto addacc = addbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();

    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    T add = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        add = 0;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      add += j + sizeof(T);
      exit_if_not_equal<T>(addacc[j], add, "barrier");
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
  check<int>(Queue);
  check<unsigned int>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  check<float>(Queue);
  if (Queue.get_device().has_extension("cl_khr_fp64")) {
    check<double>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
