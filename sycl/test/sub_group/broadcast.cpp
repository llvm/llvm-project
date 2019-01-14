// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable when use SPIRV operations instead direct built-ins calls.
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------- broadcast.cpp - SYCL sub_group broadcast test ----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
template <typename T> class sycl_subgr;
using namespace cl::sycl;
template <typename T> void check(queue &Queue) {
  const int G = 240, L = 60;
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto syclacc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        /*Broadcast GID of element with SGLID == SGID */
        syclacc[NdItem.get_global_id()] =
            SG.broadcast<T>(NdItem.get_global_id(0), SG.get_group_id());
      });
    });
    auto syclacc = syclbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      exit_if_not_equal<T>(syclacc[j], L * WGid + SGid + SGid * sg_size,
                           "broadcasted value");
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
  check<char>(Queue);
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
