// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------------- reduce.cpp - SYCL sub_group reduce test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
template <typename T> class sycl_subgr;
using namespace cl::sycl;
template <typename T> void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> minbuf(G);
    buffer<T> maxbuf(G);
    buffer<T> addbuf(G);
    Queue.submit([&](handler &cgh) {
      auto minacc = minbuf.template get_access<access::mode::read_write>(cgh);
      auto maxacc = maxbuf.template get_access<access::mode::read_write>(cgh);
      auto addacc = addbuf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group sg = NdItem.get_sub_group();
        minacc[NdItem.get_global_id()] =
            sg.reduce<T, intel::minimum>(NdItem.get_global_id(0));
        maxacc[NdItem.get_global_id()] =
            sg.reduce<T, intel::maximum>(NdItem.get_global_id(0));
        addacc[NdItem.get_global_id()] =
            sg.reduce<T, intel::plus>(NdItem.get_global_id(0));
      });
    });
    auto minacc = minbuf.template get_access<access::mode::read_write>();
    auto maxacc = maxbuf.template get_access<access::mode::read_write>();
    auto addacc = addbuf.template get_access<access::mode::read_write>();
    size_t sg_size = get_sg_size(Queue.get_device());
    int WGid = -1, SGid = 0;
    T max = 0, add = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        max = 0;
        add = 0;
        for (int i = j; (i % L && i % L % sg_size) || (i == j); i++) {
          add += i;
          max = i;
        }
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      exit_if_not_equal<T>(minacc[j], L * WGid + SGid * sg_size, "reduce_min");
      exit_if_not_equal<T>(maxacc[j], max, "reduce_max");
      exit_if_not_equal<T>(addacc[j], add, "reduce_add");
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
