// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
//==--------------- types.cpp - SYCL types test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <cassert>
#include <iostream>
#include <type_traits>

using namespace std;

using cl_schar = cl_char;
using cl_schar4 = cl_char4;

#define CHECK_TYPE(type)                                                       \
  static_assert(sizeof(cl_##type) == sizeof(cl::sycl::cl_##type), "Wrong "     \
                                                                  "size")

#define CHECK_SIZE(T, S) static_assert(sizeof(T) == S, "Wrong size of type");

#define CHECK_SIZE_TYPE_I(T, S)                                                \
  CHECK_SIZE(T, S)                                                             \
  static_assert(std::is_signed<T>::value, "Expected signed type");

#define CHECK_SIZE_TYPE_UI(T, S)                                               \
  CHECK_SIZE(T, S)                                                             \
  static_assert(std::is_unsigned<T>::value, "Expected unsigned type");

#define CHECK_SIZE_TYPE_F(T, S)                                                \
  CHECK_SIZE(T, S)                                                             \
  static_assert(std::numeric_limits<T>::is_iec559,                             \
                "Expected type conformed to the IEEE 754");

int main() {
  CHECK_TYPE(bool);
  CHECK_TYPE(char);
  CHECK_TYPE(schar);
  CHECK_TYPE(uchar);
  CHECK_TYPE(short);
  CHECK_TYPE(ushort);
  CHECK_TYPE(half);
  CHECK_TYPE(int);
  CHECK_TYPE(uint);
  CHECK_TYPE(long);
  CHECK_TYPE(ulong);
  CHECK_TYPE(float);
  CHECK_TYPE(double);
  CHECK_TYPE(bool);
  CHECK_TYPE(char2);
  CHECK_TYPE(uchar3);
  CHECK_TYPE(short4);
  CHECK_TYPE(ushort8);
  CHECK_TYPE(half16);
  CHECK_TYPE(int2);
  CHECK_TYPE(uint3);
  CHECK_TYPE(long4);
  CHECK_TYPE(schar4);
  CHECK_TYPE(ulong8);
  CHECK_TYPE(float16);
  CHECK_TYPE(double2);

  // Table 4.93: Scalar data type aliases supported by SYCL
  CHECK_SIZE_TYPE_UI(cl::sycl::byte, 1);

  CHECK_SIZE_TYPE_I(cl::sycl::cl_char, 1);
  CHECK_SIZE_TYPE_I(cl::sycl::cl_short, 2);
  CHECK_SIZE_TYPE_I(cl::sycl::cl_int, 4);
  CHECK_SIZE_TYPE_I(cl::sycl::cl_long, 8);

  CHECK_SIZE_TYPE_UI(cl::sycl::cl_uchar, 1);
  CHECK_SIZE_TYPE_UI(cl::sycl::cl_ushort, 2);
  CHECK_SIZE_TYPE_UI(cl::sycl::cl_uint, 4);
  CHECK_SIZE_TYPE_UI(cl::sycl::cl_ulong, 8);

  CHECK_SIZE_TYPE_F(cl::sycl::cl_float, 4);
  CHECK_SIZE_TYPE_F(cl::sycl::cl_double, 8);
  // CHECK_SIZE_TYPE_F(cl::sycl::cl_half, 2);

  using value_type = decltype(std::declval<cl::sycl::item<1>>()[0]);
  static_assert(!std::is_reference<value_type>::value,
                "Expected a non-reference type");
}
