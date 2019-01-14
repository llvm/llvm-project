// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==----------------accessor.cpp - SYCL accessor basic test ----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>

namespace sycl {
using namespace cl::sycl;
}

struct IdxID1 {
  int x;

  IdxID1(int x) : x(x) {}
  operator sycl::id<1>() { return x; }
};

struct IdxID3 {
  int x;
  int y;
  int z;

  IdxID3(int x, int y, int z) : x(x), y(y), z(z) {}
  operator sycl::id<3>() { return sycl::id<3>(x, y, z); }
};

struct IdxSzT {
  int x;

  IdxSzT(int x) : x(x) {}
  operator size_t() { return x; }
};

int main() {
  // Host accessor.
  {
    int src[2] = {3, 7};
    int dst[2];

    sycl::buffer<int, 1> buf_src(src, sycl::range<1>(2),
                                 {cl::sycl::property::buffer::use_host_ptr()});
    sycl::buffer<int, 1> buf_dst(dst, sycl::range<1>(2),
                                 {cl::sycl::property::buffer::use_host_ptr()});

    sycl::id<1> id1(1);
    auto acc_src = buf_src.get_access<sycl::access::mode::read>();
    auto acc_dst = buf_dst.get_access<sycl::access::mode::read_write>();

    assert(!acc_src.is_placeholder());
    assert(acc_src.get_size() == sizeof(src));
    assert(acc_src.get_count() == 2);
    assert(acc_src.get_range() == sycl::range<1>(2));
    assert(acc_src.get_pointer() == src);

    // Make sure that operator[] is defined for both size_t and id<1>.
    // Implicit conversion from IdxSzT to size_t guarantees that no
    // implicit conversion from size_t to id<1> will happen.
    assert(acc_src[IdxSzT(0)] + acc_src[IdxID1(1)] == 10);

    acc_dst[0] = acc_src[0] + acc_src[IdxID1(0)];
    acc_dst[id1] = acc_src[1] + acc_src[IdxSzT(1)];
    assert(dst[0] == 6 && dst[1] == 14);
  }

  // Three-dimensional host accessor.
  {
    int data[24];
    for (int i = 0; i < 24; ++i)
      data[i] = i;
    {
      sycl::buffer<int, 3> buf(data, sycl::range<3>(2, 3, 4));
      auto acc = buf.get_access<sycl::access::mode::read_write>();

      assert(!acc.is_placeholder());
      assert(acc.get_size() == sizeof(data));
      assert(acc.get_count() == 24);
      assert(acc.get_range() == sycl::range<3>(2, 3, 4));
      assert(acc.get_pointer() != data);

      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 4; ++k)
            acc[IdxID3(i, j, k)] += acc[sycl::id<3>(i, j, k)];
    }
    for (int i = 0; i < 24; ++i) {
      assert(data[i] == 2 * i);
    }
  }
  int data = 5;
  // Device accessor.
  {
    sycl::queue Queue;

    sycl::buffer<int, 1> buf(&data, sycl::range<1>(1),
                             {cl::sycl::property::buffer::use_host_ptr()});

    Queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      assert(!acc.is_placeholder());
      assert(acc.get_size() == sizeof(int));
      assert(acc.get_count() == 1);
      assert(acc.get_range() == sycl::range<1>(1));
      cgh.single_task<class kernel>(
          [=]() { acc[IdxSzT(0)] += acc[IdxID1(0)]; });
    });
    Queue.wait();
  }
  assert(data == 10);

  // Device accessor with 2-dimensional subscript operators.
  {
    sycl::queue Queue;
    if (!Queue.is_host()) {
      int array[2][3] = {0};
      {
        sycl::range<2> Range(2, 3);
        sycl::buffer<int, 2> buf((int *)array, Range,
                                 {cl::sycl::property::buffer::use_host_ptr()});

        Queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<class dim2_subscr>(Range, [=](sycl::item<2> itemID) {
            acc[itemID.get_id(0)][itemID.get_id(1)] += itemID.get_linear_id();
          });
        });
        Queue.wait();
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          std::cout << "array[" << i << "][" << j << "]=" << array[i][j]
                    << std::endl;
          assert(array[i][j] == i * 3 + j);
        }
      }
    }
  }

  // Device accessor with 3-dimensional subscript operators.
  {
    sycl::queue Queue;
    if (!Queue.is_host()) {
      int array[2][3][4] = {0};
      {
        sycl::range<3> Range(2, 3, 4);
        sycl::buffer<int, 3> buf((int *)array, Range,
                                 {cl::sycl::property::buffer::use_host_ptr()});

        Queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<class dim3_subscr>(Range, [=](sycl::item<3> itemID) {
            acc[itemID.get_id(0)][itemID.get_id(1)][itemID.get_id(2)] +=
                itemID.get_linear_id();
          });
        });
        Queue.wait();
      }
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 4; k++) {
            std::cout << "array[" << i << "][" << j << "][" << k
                      << "]=" << array[i][j][k] << std::endl;
            assert(array[i][j][k] == k + 4 * (j + 3 * i));
          }
        }
      }
    }
  }
}
