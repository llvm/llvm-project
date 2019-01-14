// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable when use SPIRV operations instead direct built-ins calls.
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ shuffle.cpp - SYCL sub_group shuffle test ----------------==//
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
template <typename T> void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf2(G);
    buffer<T> buf2_up(G);
    buffer<T> buf2_down(G);
    buffer<T> buf(G);
    buffer<T> buf_up(G);
    buffer<T> buf_down(G);
    buffer<T> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto acc2 = buf2.template get_access<access::mode::read_write>(cgh);
      auto acc2_up = buf2_up.template get_access<access::mode::read_write>(cgh);
      auto acc2_down =
          buf2_down.template get_access<access::mode::read_write>(cgh);

      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      auto acc_up = buf_up.template get_access<access::mode::read_write>(cgh);
      auto acc_down =
          buf_down.template get_access<access::mode::read_write>(cgh);
      auto acc_xor = buf_xor.template get_access<access::mode::read_write>(cgh);

      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        /* 1 for odd subgroups and 2 for even*/
        acc2[NdItem.get_global_id()] = SG.shuffle<T>(
            1, 2,
            (SG.get_group_id().get(0) % 2) ? 1 : SG.get_max_local_range()[0]);
        /* GID-SGID */
        acc2_up[NdItem.get_global_id()] =
            SG.shuffle_up<T>(NdItem.get_global_id(0), NdItem.get_global_id(0),
                             SG.get_group_id().get(0));
        /* GID-SGID or SGLID if GID+SGID > SGsize*/
        acc2_down[NdItem.get_global_id()] = SG.shuffle_down<T>(
            NdItem.get_global_id(0), SG.get_local_id().get(0),
            SG.get_group_id().get(0));

        /*GID of middle element in every subgroup*/
        acc[NdItem.get_global_id()] = SG.shuffle<T>(
            NdItem.get_global_id(0), SG.get_max_local_range()[0] / 2);
        /* Save GID-SGID */
        acc_up[NdItem.get_global_id()] =
            SG.shuffle_up<T>(NdItem.get_global_id(0), SG.get_group_id().get(0));
        /* Save GID+SGID */
        acc_down[NdItem.get_global_id()] = SG.shuffle_down<T>(
            NdItem.get_global_id(0), SG.get_group_id().get(0));
        /* Save GID XOR SGID */
        acc_xor[NdItem.get_global_id()] =
            SG.shuffle_xor<T>(NdItem.get_global_id(0), SG.get_group_id());
      });
    });
    auto acc = buf.template get_access<access::mode::read_write>();
    auto acc_up = buf_up.template get_access<access::mode::read_write>();
    auto acc_down = buf_down.template get_access<access::mode::read_write>();
    auto acc2 = buf2.template get_access<access::mode::read_write>();
    auto acc2_up = buf2_up.template get_access<access::mode::read_write>();
    auto acc2_down = buf2_down.template get_access<access::mode::read_write>();
    auto acc_xor = buf_xor.template get_access<access::mode::read_write>();

    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    size_t sg_size = sgsizeacc[0];
    int SGid = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
      }
      if (j % L == 0) {
        SGid = 0;
      }
      /*GID of middle element in every subgroup*/
      exit_if_not_equal<T>(acc[j], j / L * L + SGid * sg_size + sg_size / 2,
                           "shuffle");
      /* 1 for odd subgroups and 2 for even*/
      exit_if_not_equal<T>(acc2[j], (SGid % 2) ? 1 : 2, "shuffle2");
      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal<T>(acc_down[j], j + SGid, "shuffle_down");
        exit_if_not_equal<T>(acc2_down[j], j + SGid, "shuffle2_down");
      } else {                /* SGLID for GID+SGid */
        if (j % L + SGid < L) /* Do not go out  LG*/
          exit_if_not_equal<T>(acc2_down[j], (j + SGid) % L % sg_size,
                               "shuffle2_down");
      }
      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal<T>(acc_up[j], j - SGid, "shuffle_up");
        exit_if_not_equal<T>(acc2_up[j], j - SGid, "shuffle2_up");
      } else {                          /* SGLID for GID-SGid */
        if (j % L - SGid + sg_size < L) /* Do not go out  LG*/
          exit_if_not_equal<T>(acc2_up[j], j - SGid + sg_size, "shuffle2_up");
      }
      /* GID XOR SGID */
      exit_if_not_equal<T>(acc_xor[j], j ^ SGid, "shuffle_xor");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
int main() {
  queue Queue;
  if (!Queue.get_device().has_extension("cl_intel_subgroups")) {
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
