// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------- reinterpret.cpp --- SYCL buffer reinterpret basic test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

// This tests verifies basic cases of using cl::sycl::buffer::reinterpret
// functionality - changing buffer type and range. This test checks that
// original buffer updates when we write to reinterpreted buffer and also checks
// that we can't create reinterpreted buffer when total size in bytes will be
// not same as total size in bytes of original buffer.

int main() {

  bool failed = false;
  cl::sycl::queue q;

  cl::sycl::range<1> r1(1);
  cl::sycl::range<1> r2(sizeof(unsigned int) / sizeof(unsigned char));
  cl::sycl::buffer<unsigned int, 1> buf_i(r1);
  auto buf_char = buf_i.reinterpret<unsigned char>(r2);
  q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf_char.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class chars>(
        r2, [=](cl::sycl::id<1> i) { acc[i] = UCHAR_MAX; });
  });

  {
    auto acc = buf_i.get_access<cl::sycl::access::mode::read>();
    if (acc[0] != UINT_MAX) {
      std::cout << acc[0] << std::endl;
      std::cout << "line: " << __LINE__ << " array[" << 0 << "] is " << acc[0]
                << " expected " << UINT_MAX << std::endl;
      failed = true;
    }
  }

  cl::sycl::range<1> r1d(9);
  cl::sycl::range<2> r2d(3, 3);
  cl::sycl::buffer<unsigned int, 1> buf_1d(r1d);
  auto buf_2d = buf_1d.reinterpret<unsigned int>(r2d);
  q.submit([&](cl::sycl::handler &cgh) {
    auto acc2d = buf_2d.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class ones>(r2d, [=](cl::sycl::item<2> itemID) {
      size_t i = itemID.get_id(0);
      size_t j = itemID.get_id(1);
      if (i == j)
        acc2d[i][j] = 1;
      else
        acc2d[i][j] = 0;
    });
  });

  {
    auto acc = buf_1d.get_access<cl::sycl::access::mode::read>();
    for (auto i = 0u; i < r1d.size(); i++) {
      size_t expected = (i % 4) ? 0 : 1;
      if (acc[i] != expected) {
        std::cout << "line: " << __LINE__ << " array[" << i << "] is " << acc[i]
                  << " expected " << expected << std::endl;
        failed = true;
      }
    }
  }

  try {
    cl::sycl::buffer<float, 1> buf_fl(r1d);
    auto buf_d = buf_1d.reinterpret<double>(r2d);
  } catch (cl::sycl::invalid_object_error e) {
    std::cout << "Expected exception has been caught: " << e.what()
              << std::endl;
  }

  return failed;
}
