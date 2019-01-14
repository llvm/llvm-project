// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable when use SPIRV operations instead direct built-ins calls.
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==----------- load_store.cpp - SYCL sub_group load/store test ------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
template <typename T, int N> class sycl_subgr;

using namespace cl::sycl;
template <typename T1, int N> struct utils {
  static T1 add_vec(const vec<T1, N> &v);
};
template <typename T2> struct utils<T2, 1> {
  static T2 add_vec(const vec<T2, 1> &v) { return v.s0(); }
};
template <typename T2> struct utils<T2, 2> {
  static T2 add_vec(const vec<T2, 2> &v) { return v.s0() + v.s1(); }
};
template <typename T2> struct utils<T2, 4> {
  static T2 add_vec(const vec<T2, 4> &v) {
    return v.s0() + v.s1() + v.s2() + v.s3();
  }
};
template <typename T2> struct utils<T2, 8> {
  static T2 add_vec(const vec<T2, 8> &v) {
    return v.s0() + v.s1() + v.s2() + v.s3() + v.s4() + v.s5() + v.s6() +
           v.s7();
  }
};

template <typename T, int N> void check(queue &Queue) {
  const int G = 128, L = 128;
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G);
    buffer<size_t> sgsizebuf(1);
    {
      auto acc = syclbuf.template get_access<access::mode::read_write>();
      for (int i = 0; i < G; i++) {
        acc[i] = i;
        acc[i] += 0.1; // Check that floating point types are not casted to int
      }
    }
    Queue.submit([&](handler &cgh) {
      auto acc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        if (SG.get_group_id().get(0) % N == 0) {
          if (NdItem.get_global_id(0) == 0)
            sgsizeacc[0] = SG.get_max_local_range()[0];

          size_t WGSGoffset =
              NdItem.get_group(0) * L +
              SG.get_group_id().get(0) * SG.get_max_local_range().get(0);
          multi_ptr<T, access::address_space::global_space> mp(
              &acc[WGSGoffset]);
          // Add all values in read block
          vec<T, N> v(utils<T, N>::add_vec(SG.load<N, T>(mp)));
          SG.store<N, T>(mp, v);
        }
      });
    });
    auto acc = syclbuf.template get_access<access::mode::read_write>();
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
      T ref = 0;
      if (SGid % N) {
        ref = acc[j - (SGid % N) * sg_size];
      } else {
        for (int i = 0; i < N; i++) {
          ref += (T)(j + i * sg_size) + 0.1;
        }
      }
      std::string s("Vector<");
      s += std::string(typeid(ref).name()) + std::string(",") +
           std::to_string(N) + std::string(">[") + std::to_string(j) +
           std::string("]");
      exit_if_not_equal<T>(acc[j], ref, s.c_str());
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
template <typename T> void check(queue &Queue) {
  const int G = 128, L = 64;
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G);
    buffer<size_t> sgsizebuf(1);
    {
      auto acc = syclbuf.template get_access<access::mode::read_write>();
      for (int i = 0; i < G; i++) {
        acc[i] = i;
        acc[i] += 0.1; // Check that floating point types are not casted to int
      }
    }

    Queue.submit([&](handler &cgh) {
      auto acc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        intel::sub_group SG = NdItem.get_sub_group();
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
        size_t WGSGoffset =
            NdItem.get_group(0) * L +
            SG.get_group_id().get(0) * SG.get_max_local_range().get(0);
        multi_ptr<T, access::address_space::global_space> mp(&acc[WGSGoffset]);
        T s = SG.load<T>(mp) + (T)SG.get_local_id().get(0);
        SG.store<T>(mp, s);
      });
    });
    auto acc = syclbuf.template get_access<access::mode::read_write>();
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
      std::string s("Scalar<");
      s += std::string(typeid(acc[j]).name()) + std::string(">[") +
           std::to_string(j) + std::string("]");

      exit_if_not_equal<T>(acc[j], (T)(j + j % L % sg_size) + 0.1, s.c_str());
    }

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

int main() {
  queue Queue;
  if (!Queue.get_device().has_extension("cl_intel_subgroups") &&
      !Queue.get_device().has_extension("cl_intel_subgroups_short")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  if (Queue.get_device().has_extension("cl_intel_subgroups")) {
    typedef int aligned_int __attribute__((aligned(16)));
    check<aligned_int>(Queue);
    check<aligned_int, 1>(Queue);
    check<aligned_int, 2>(Queue);
    check<aligned_int, 4>(Queue);
    check<aligned_int, 8>(Queue);
    typedef uint aligned_uint __attribute__((aligned(16)));
    check<aligned_uint>(Queue);
    check<aligned_uint, 1>(Queue);
    check<aligned_uint, 2>(Queue);
    check<aligned_uint, 4>(Queue);
    check<aligned_uint, 8>(Queue);
    typedef float aligned_float __attribute__((aligned(16)));
    check<aligned_float>(Queue);
    check<aligned_float, 1>(Queue);
    check<aligned_float, 2>(Queue);
    check<aligned_float, 4>(Queue);
    check<aligned_float, 8>(Queue);
  }
  if (Queue.get_device().has_extension("cl_khr_fp16") &&
      Queue.get_device().has_extension("cl_intel_subgroups_short")) {
    typedef short aligned_short __attribute__((aligned(16)));
    check<aligned_short>(Queue);
    check<aligned_short, 1>(Queue);
    check<aligned_short, 2>(Queue);
    check<aligned_short, 4>(Queue);
    check<aligned_short, 8>(Queue);
    typedef half aligned_half __attribute__((aligned(16)));
    check<aligned_half>(Queue);
    check<aligned_half, 1>(Queue);
    check<aligned_half, 2>(Queue);
    check<aligned_half, 4>(Queue);
    check<aligned_half, 8>(Queue);
  }
  std::cout << "Test passed." << std::endl;
  return 0;
}
