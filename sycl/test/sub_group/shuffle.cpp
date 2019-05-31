// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ shuffle.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
template <typename T, int N> class sycl_subgr;

using namespace cl::sycl;

// TODO remove this workaround when clang will support correct generation of
// half typename in integration header
struct wa_half;

template <typename T, int N>
void check(queue &Queue, size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<vec<T, N>> buf2(G);
    buffer<vec<T, N>> buf2_up(G);
    buffer<vec<T, N>> buf2_down(G);
    buffer<vec<T, N>> buf(G);
    buffer<vec<T, N>> buf_up(G);
    buffer<vec<T, N>> buf_down(G);
    buffer<vec<T, N>> buf_xor(G);
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

      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        vec<T, N> vwggid(wggid), vsgid(sgid);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        /* 1 for odd subgroups and 2 for even*/
        acc2[NdItem.get_global_id()] =
            SG.shuffle(vec<T, N>(1), vec<T, N>(2),
                       (sgid % 2) ? 1 : SG.get_max_local_range()[0]);
        /* GID-SGID */
        acc2_up[NdItem.get_global_id()] = SG.shuffle_up(vwggid, vwggid, sgid);
        /* GID-SGID or SGLID if GID+SGID > SGsize*/
        acc2_down[NdItem.get_global_id()] =
            SG.shuffle_down(vwggid, vec<T, N>(SG.get_local_id().get(0)), sgid);

        /*GID of middle element in every subgroup*/
        acc[NdItem.get_global_id()] =
            SG.shuffle(vwggid, SG.get_max_local_range()[0] / 2);
        /* Save GID-SGID */
        acc_up[NdItem.get_global_id()] = SG.shuffle_up(vwggid, sgid);
        /* Save GID+SGID */
        acc_down[NdItem.get_global_id()] = SG.shuffle_down(vwggid, sgid);
        /* Save GID XOR SGID */
        acc_xor[NdItem.get_global_id()] = SG.shuffle_xor(vwggid, sgid);
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
      exit_if_not_equal_vec<T, N>(
          acc[j], vec<T, N>(j / L * L + SGid * sg_size + sg_size / 2),
          "shuffle");
      /* 1 for odd subgroups and 2 for even*/
      exit_if_not_equal_vec<T, N>(acc2[j], vec<T, N>((SGid % 2) ? 1 : 2),
                                  "shuffle2");
      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal_vec(acc_down[j], vec<T, N>(j + SGid), "shuffle_down");
        exit_if_not_equal_vec(acc2_down[j], vec<T, N>(j + SGid),
                              "shuffle2_down");
      } else {                /* SGLID for GID+SGid */
        if (j % L + SGid < L) /* Do not go out  LG*/
          exit_if_not_equal_vec<T, N>(acc2_down[j],
                                      vec<T, N>((j + SGid) % L % sg_size),
                                      "shuffle2_down");
      }
      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal_vec(acc_up[j], vec<T, N>(j - SGid), "shuffle_up");
        exit_if_not_equal_vec(acc2_up[j], vec<T, N>(j - SGid), "shuffle2_up");
      } else {                          /* SGLID for GID-SGid */
        if (j % L - SGid + sg_size < L) /* Do not go out  LG*/
          exit_if_not_equal_vec(acc2_up[j], vec<T, N>(j - SGid + sg_size),
                                "shuffle2_up");
      }
      /* GID XOR SGID */
      exit_if_not_equal_vec(acc_xor[j], vec<T, N>(j ^ SGid), "shuffle_xor");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

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

      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        /* 1 for odd subgroups and 2 for even*/
        acc2[NdItem.get_global_id()] =
            SG.shuffle<T>(1, 2, (sgid % 2) ? 1 : SG.get_max_local_range()[0]);
        /* GID-SGID */
        acc2_up[NdItem.get_global_id()] = SG.shuffle_up<T>(wggid, wggid, sgid);
        /* GID-SGID or SGLID if GID+SGID > SGsize*/
        acc2_down[NdItem.get_global_id()] =
            SG.shuffle_down<T>(wggid, SG.get_local_id().get(0), sgid);

        /*GID of middle element in every subgroup*/
        acc[NdItem.get_global_id()] =
            SG.shuffle<T>(wggid, SG.get_max_local_range()[0] / 2);
        /* Save GID-SGID */
        acc_up[NdItem.get_global_id()] = SG.shuffle_up<T>(wggid, sgid);
        /* Save GID+SGID */
        acc_down[NdItem.get_global_id()] = SG.shuffle_down<T>(wggid, sgid);
        /* Save GID XOR SGID */
        acc_xor[NdItem.get_global_id()] = SG.shuffle_xor<T>(wggid, sgid);
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

  if (Queue.get_device().has_extension("cl_intel_subgroups_short")) {
    check<short>(Queue);
    check<unsigned short>(Queue);
  }
  check<int>(Queue);
  check<int, 2>(Queue);
  check<int, 4>(Queue);
  check<int, 8>(Queue);
  check<int, 16>(Queue);
  check<unsigned int>(Queue);
  check<unsigned int, 2>(Queue);
  check<unsigned int, 4>(Queue);
  check<unsigned int, 8>(Queue);
  check<unsigned int, 16>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  if (Queue.get_device().has_extension("cl_khr_fp16")) {
    check<half>(Queue);
  }
  check<float>(Queue);
  if (Queue.get_device().has_extension("cl_khr_fp64")) {
    check<double>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
