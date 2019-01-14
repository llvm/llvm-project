// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable when use SPIRV operations instead direct built-ins calls.
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------------- scan.cpp - SYCL sub_group scan test --------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
#include <limits>
template <typename T> class sycl_subgr;
using namespace cl::sycl;
template <typename T> void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> minexbuf(G);
    buffer<T> maxexbuf(G);
    buffer<T> addexbuf(G);
    buffer<T> mininbuf(G);
    buffer<T> maxinbuf(G);
    buffer<T> addinbuf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto minexacc =
          minexbuf.template get_access<access::mode::read_write>(cgh);
      auto maxexacc =
          maxexbuf.template get_access<access::mode::read_write>(cgh);
      auto addexacc =
          addexbuf.template get_access<access::mode::read_write>(cgh);
      auto mininacc =
          mininbuf.template get_access<access::mode::read_write>(cgh);
      auto maxinacc =
          maxinbuf.template get_access<access::mode::read_write>(cgh);
      auto addinacc =
          addinbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        minexacc[NdItem.get_global_id()] =
            SG.exclusive_scan<T, intel::minimum>(NdItem.get_global_id(0));
        maxexacc[NdItem.get_global_id()] =
            SG.exclusive_scan<T, intel::maximum>(NdItem.get_global_id(0));
        addexacc[NdItem.get_global_id()] =
            SG.exclusive_scan<T, intel::plus>(NdItem.get_global_id(0));
        mininacc[NdItem.get_global_id()] =
            SG.inclusive_scan<T, intel::minimum>(NdItem.get_global_id(0));
        maxinacc[NdItem.get_global_id()] =
            SG.inclusive_scan<T, intel::maximum>(NdItem.get_global_id(0));
        addinacc[NdItem.get_global_id()] =
            SG.inclusive_scan<T, intel::plus>(NdItem.get_global_id(0));
      });
    });
    auto minexacc = minexbuf.template get_access<access::mode::read_write>();
    auto maxexacc = maxexbuf.template get_access<access::mode::read_write>();
    auto addexacc = addexbuf.template get_access<access::mode::read_write>();
    auto mininacc = mininbuf.template get_access<access::mode::read_write>();
    auto maxinacc = maxinbuf.template get_access<access::mode::read_write>();
    auto addinacc = addinbuf.template get_access<access::mode::read_write>();

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
      /*skip check for empty array*/
      if (j % L % sg_size != 0) {
        exit_if_not_equal<T>(minexacc[j], L * WGid + SGid * sg_size,
                             "scan_exc_min");
        exit_if_not_equal<T>(maxexacc[j], j - 1, "scan_exc_max");
      }
      exit_if_not_equal<T>(addexacc[j], add, "scan_exc_add");
      add += j;
      exit_if_not_equal<T>(mininacc[j], L * WGid + SGid * sg_size,
                           "scan_inc_min");
      exit_if_not_equal<T>(maxinacc[j], j, "scan_inc_max");
      exit_if_not_equal<T>(addinacc[j], add, "scan_inc_add");
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
  /* Limit work-group size to avoid type overflow. */
  check<char>(Queue, 120, 30);
  check<short>(Queue);
  check<int>(Queue);
  check<uint>(Queue);
  check<long>(Queue);
  check<ulong>(Queue);
  if (!Queue.get_device().has_extension("cl_khr_fp16")) {
    check<half>(Queue);
  }
  check<float>(Queue);
  if (!Queue.get_device().has_extension("cl_khr_fp64")) {
    check<double>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
