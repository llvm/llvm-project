// RUN: %clang --sycl %s -c -o %T/kernel.spv
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %t.out

//==--------------- id.cpp - SYCL id test ----------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using cl::sycl::detail::Builder;

using namespace std;
int main() {
  /* id()
   * Construct a SYCL id with the value 0 for each dimension. */
  cl::sycl::id<1> one_dim_zero_id;
  assert(one_dim_zero_id.get(0) == 0);
  cl::sycl::id<2> two_dim_zero_id;
  assert(two_dim_zero_id.get(0) == 0 && two_dim_zero_id.get(1) == 0);
  cl::sycl::id<3> three_dim_zero_id;
  assert(three_dim_zero_id.get(0) == 0 && three_dim_zero_id.get(1) == 0 &&
         three_dim_zero_id.get(2) == 0);

  /* id(size_t dim0)
   * Construct a 1D id with value dim0. Only valid when the template parameter
   * dimensions is equal to 1 */
  cl::sycl::id<1> one_dim_id(64);
  assert(one_dim_id.get(0) == 64);

  /* id(size_t dim0, size_t dim1)
   * Construct a 2D id with values dim0, dim1. Only valid when the template
   * parameter dimensions is equal to 2. */
  cl::sycl::id<2> two_dim_id(128, 256);
  assert(two_dim_id.get(0) == 128 && two_dim_id.get(1) == 256);

  /* id(size_t dim0, size_t dim1, size_t dim2)
   * Construct a 3D id with values dim0, dim1, dim2. Only valid when the
   * template parameter dimensions is equal to 3. */
  cl::sycl::id<3> three_dim_id(64, 1, 2);
  assert(three_dim_id.get(0) == 64 && three_dim_id.get(1) == 1 &&
         three_dim_id.get(2) == 2);

  /* id(const range<dimensions> &range)
   * Construct an id from the dimensions of r. */
  cl::sycl::range<1> one_dim_range(2);
  cl::sycl::id<1> one_dim_id_range(one_dim_range);
  assert(one_dim_id_range.get(0) == 2);
  cl::sycl::range<2> two_dim_range(4, 8);
  cl::sycl::id<2> two_dim_id_range(two_dim_range);
  assert(two_dim_id_range.get(0) == 4 && two_dim_id_range.get(1) == 8);
  cl::sycl::range<3> three_dim_range(16, 32, 64);
  cl::sycl::id<3> three_dim_id_range(three_dim_range);
  assert(three_dim_id_range.get(0) == 16 && three_dim_id_range.get(1) == 32 &&
         three_dim_id_range.get(2) == 64);

  /* id(const item<dimensions> &item)
   * Construct an id from item.get_id().*/
  cl::sycl::item<1, true> one_dim_item_with_offset =
      Builder::createItem<1, true>({4}, {2}, {1});
  cl::sycl::id<1> one_dim_id_item(one_dim_item_with_offset);
  assert(one_dim_id_item.get(0) == 2);
  cl::sycl::item<2, true> two_dim_item_with_offset =
      Builder::createItem<2, true>({8, 16}, {4, 8}, {1, 1});
  cl::sycl::id<2> two_dim_id_item(two_dim_item_with_offset);
  assert(two_dim_id_item.get(0) == 4 && two_dim_id_item.get(1) == 8);
  cl::sycl::item<3, true> three_dim_item_with_offset =
      Builder::createItem<3, true>({32, 64, 128}, {16, 32, 64}, {1, 1, 1});
  cl::sycl::id<3> three_dim_id_item(three_dim_item_with_offset);
  assert(three_dim_id_item.get(0) == 16 && three_dim_id_item.get(1) == 32 &&
         three_dim_id_item.get(2) == 64);
  /* size_t get(int dimension)const
   * Return the value of the id for dimension dimension. */

  /* size_t &operator[](int dimension)const
   * Return a reference to the requested dimension of the id object. */

  /* size_t &operator[](int dimension)const
   * Return a reference to the requested dimension of the id object. */

  /* id<dimensions> operatorOP(const id<dimensions> &rhs) const
   * Where OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=.
   * Constructs and returns a new instance of the SYCL id class template with
   * the same dimensionality as this SYCL id, where each element of the new SYCL
   * id instance is the result of an element-wise OP operator between each
   * element of this SYCL id and each element of the rhs id. If the operator
   * returns a bool the result is the cast to size_t */

  /* id<dimensions> operatorOP(const id<dimensions> &rhs) const
   * Where OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=.
   * Constructs and returns a new instance of the SYCL id class template with
   * the same dimensionality as this SYCL id, where each element of the new SYCL
   * id instance is the result of an element-wise OP operator between each
   * element of this SYCL id and each element of the rhs id. If the operator
   * returns a bool the result is the cast to size_t */
}
