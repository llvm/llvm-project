// RUN: %clang --sycl %s -c -o %T/kernel.spv
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %t.out
//==--------------- vectors.cpp - SYCL vectors test ------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
using namespace cl::sycl;

void check_vectors(int4 a, int4 b, int4 c, int4 gold) {
  int4 result = a * (int4)b.y() + c;
  assert((int)result.x() == (int)gold.x());
  assert((int)result.y() == (int)gold.y());
  assert((int)result.w() == (int)gold.w());
  assert((int)result.z() == (int)gold.z());
}

int main() {
  int4 a = {1, 2, 3, 4};
  const int4 b = {10, 20, 30, 40};
  const int4 gold = {21, 42, 90, 120};
  const int2 a_xy = a.xy();
  check_vectors(a, b, {1, 2, 30, 40}, gold);
  check_vectors(a, b, {a.x(), a.y(), b.z(), b.w()}, gold);
  check_vectors(a, b, {a.x(), 2, b.z(), 40}, gold);
  check_vectors(a, b, {a.x(), 2, b.zw()}, gold);
  check_vectors(a, b, {a_xy, b.z(), 40}, gold);
  check_vectors(a, b, {a.xy(), b.zw()}, gold);

  // Constructing vector from a scalar
  cl::sycl::vec<int, 1> vec_from_one_elem(1);

  // implicit conversion
  cl::sycl::vec<unsigned char, 2> vec_2(1, 2);
  cl::sycl::vec<unsigned char, 4> vec_4(0, vec_2, 3);

  assert(vec_4.get_count() == 4);
  assert(static_cast<unsigned char>(vec_4.x()) == static_cast<unsigned char>(0));
  assert(static_cast<unsigned char>(vec_4.y()) == static_cast<unsigned char>(1));
  assert(static_cast<unsigned char>(vec_4.z()) == static_cast<unsigned char>(2));
  assert(static_cast<unsigned char>(vec_4.w()) == static_cast<unsigned char>(3));

  // explicit conversion
  int64_t(vec_2.x());
  cl::sycl::int4(vec_2.x());

  return 0;
}
