// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==----------------accessor.cpp - SYCL accessor basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

template <typename Acc> struct AccWrapper { Acc accessor; };

template <typename Acc1, typename Acc2> struct AccsWrapper {
  int a;
  Acc1 accessor1;
  int b;
  Acc2 accessor2;
};

struct Wrapper1 {
  int a;
  int b;
};

template <typename Acc> struct Wrapper2 {
  Wrapper1 w1;
  AccWrapper<Acc> wrapped;
};

template <typename Acc> struct Wrapper3 { Wrapper2<Acc> w2; };

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

  // Discard write accessor.
  {
    try {
      sycl::queue Queue;
      sycl::buffer<int, 1> buf(sycl::range<1>(3));

      Queue.submit([&](sycl::handler& cgh) {
        auto dev_acc = buf.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class test_discard_write>(
            sycl::range<1>{3},
            [=](sycl::id<1> index) { dev_acc[index] = 42; });
      });

      auto host_acc = buf.get_access<sycl::access::mode::read>();
      for (int i = 0; i != 3; ++i)
        assert(host_acc[i] == 42);

    } catch (cl::sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Discard read-write accessor.
  {
    try {
      sycl::queue Queue;
      sycl::buffer<int, 1> buf(sycl::range<1>(3));

      Queue.submit([&](sycl::handler& cgh) {
        auto dev_acc = buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class test_discard_read_write>(
            sycl::range<1>{3},
            [=](sycl::id<1> index) { dev_acc[index] = 42; });
      });

      auto host_acc =
        buf.get_access<sycl::access::mode::discard_read_write>();
    } catch (cl::sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }

  // Check that accessor is initialized when accessor is wrapped to some class.
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      int array[10] = {0};
      {
        sycl::buffer<int, 1> buf((int *)array, sycl::range<1>(10),
                                 {cl::sycl::property::buffer::use_host_ptr()});
        queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
          cgh.parallel_for<class wrapped_access1>(
              sycl::range<1>(buf.get_count()), [=](sycl::item<1> it) {
                auto idx = it.get_linear_id();
                acc_wrapped.accessor[idx] = 333;
              });
        });
        queue.wait();
      }
      for (int i = 0; i < 10; i++) {
        std::cout << "array[" << i << "]=" << array[i] << std::endl;
        assert(array[i] == 333);
      }
    }
  }

  // Case when several accessors are wrapped to some class. Check that they are
  // initialized in proper way and value is assigned.
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      int array1[10] = {0};
      int array2[10] = {0};
      {
        sycl::buffer<int, 1> buf1((int *)array1, sycl::range<1>(10),
                                  {cl::sycl::property::buffer::use_host_ptr()});
        sycl::buffer<int, 1> buf2((int *)array2, sycl::range<1>(10),
                                  {cl::sycl::property::buffer::use_host_ptr()});
        queue.submit([&](sycl::handler &cgh) {
          auto acc1 = buf1.get_access<sycl::access::mode::read_write>(cgh);
          auto acc2 = buf2.get_access<sycl::access::mode::read_write>(cgh);
          auto acc_wrapped =
              AccsWrapper<decltype(acc1), decltype(acc2)>{10, acc1, 5, acc2};
          cgh.parallel_for<class wrapped_access2>(
              sycl::range<1>(10), [=](sycl::item<1> it) {
                auto idx = it.get_linear_id();
                acc_wrapped.accessor1[idx] = 333;
                acc_wrapped.accessor2[idx] = 666;
              });
        });
        queue.wait();
      }
      for (int i = 0; i < 10; i++) {
        std::cout << "array1[" << i << "]=" << array1[i] << std::endl;
        std::cout << "array2[" << i << "]=" << array2[i] << std::endl;
        assert(array1[i] == 333);
        assert(array2[i] == 666);
      }
    }
  }

  // Several levels of wrappers for accessor.
  {
    sycl::queue queue;
    if (!queue.is_host()) {
      int array[10] = {0};
      {
        sycl::buffer<int, 1> buf((int *)array, sycl::range<1>(10),
                                 {cl::sycl::property::buffer::use_host_ptr()});
        queue.submit([&](sycl::handler &cgh) {
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
          Wrapper1 wr1;
          auto wr2 = Wrapper2<decltype(acc)>{wr1, acc_wrapped};
          auto wr3 = Wrapper3<decltype(acc)>{wr2};
          cgh.parallel_for<class wrapped_access3>(
              sycl::range<1>(buf.get_count()), [=](sycl::item<1> it) {
                auto idx = it.get_linear_id();
                wr3.w2.wrapped.accessor[idx] = 333;
              });
        });
        queue.wait();
      }
      for (int i = 0; i < 10; i++) {
        std::cout << "array[" << i << "]=" << array[i] << std::endl;
        assert(array[i] == 333);
      }
    }
  }

  // Two accessors to the same buffer.
  {
    try {
      sycl::queue queue;
      int array[3] = {1, 1, 1};
      sycl::buffer<int, 1> buf(array, sycl::range<1>(3));

      queue.submit([&](sycl::handler& cgh) {
        auto acc1 = buf.get_access<sycl::access::mode::read>(cgh);
        auto acc2 = buf.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class two_accessors_to_buf>(
            sycl::range<1>{3},
            [=](sycl::id<1> index) { acc2[index] = 41 + acc1[index]; });
      });

      auto host_acc = buf.get_access<sycl::access::mode::read>();
      for (int i = 0; i != 3; ++i)
        assert(host_acc[i] == 42);

    } catch (cl::sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }
}
