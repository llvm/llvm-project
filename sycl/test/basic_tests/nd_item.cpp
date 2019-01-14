// RUN: %clang --sycl %s -c -o %T/kernel.spv
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %t.out
//==--------------- nd_item.cpp - SYCL nd_item test ------------------------==//
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

using namespace std;
using cl::sycl::detail::Builder;

int main() {
  // one dimension nd_item
  cl::sycl::nd_item<1> one_dim =
      Builder::createNDItem<1>(Builder::createItem<1, true>({4}, {2}, {0}),
                               Builder::createItem<1, false>({2}, {0}),
                               Builder::createGroup<1>({4}, {2}, {1}));
  assert((one_dim.get_global_id() == cl::sycl::id<1>{2}));
  assert(one_dim.get_global_id(0) == 2);
  assert(one_dim.get_global_linear_id() == 2);
  assert((one_dim.get_local_id() == cl::sycl::id<1>{0}));
  assert(one_dim.get_local_id(0) == 0);
  assert(one_dim.get_local_linear_id() == 0);
  assert((one_dim.get_group() == Builder::createGroup<1>({4}, {2}, {1})));
  assert(one_dim.get_group(0) == 1);
  assert(one_dim.get_group_linear_id() == 1);
  assert((one_dim.get_group_range() == cl::sycl::id<1>{2}));
  assert(one_dim.get_group_range(0) == 2);
  assert((one_dim.get_global_range() == cl::sycl::range<1>{4}));
  assert((one_dim.get_global_range(0) == 4));
  assert((one_dim.get_local_range() == cl::sycl::range<1>{2}));
  assert((one_dim.get_local_range(0) == 2));
  assert((one_dim.get_offset() == cl::sycl::id<1>{0}));
  assert((one_dim.get_nd_range() == cl::sycl::nd_range<1>({4}, {2}, {0})));

  // two dimension nd_item
  cl::sycl::nd_item<2> two_dim = Builder::createNDItem<2>(
      Builder::createItem<2, true>({4, 4}, {2, 2}, {0, 0}),
      Builder::createItem<2, false>({2, 2}, {0, 0}),
      Builder::createGroup<2>({4, 4}, {2, 2}, {1, 1}));
  assert((two_dim.get_global_id() == cl::sycl::id<2>{2, 2}));
  assert(two_dim.get_global_id(0) == 2);
  assert(two_dim.get_global_id(1) == 2);
  assert(two_dim.get_global_linear_id() == 10);
  assert((two_dim.get_local_id() == cl::sycl::id<2>{0, 0}));
  assert(two_dim.get_local_id(0) == 0);
  assert(two_dim.get_local_id(1) == 0);
  assert(two_dim.get_local_linear_id() == 0);
  assert(
      (two_dim.get_group() == Builder::createGroup<2>({4, 4}, {2, 2}, {1, 1})));
  assert(two_dim.get_group(0) == 1);
  assert(two_dim.get_group(1) == 1);
  assert(two_dim.get_group_linear_id() == 3);
  assert((two_dim.get_group_range() == cl::sycl::id<2>{2, 2}));
  assert(two_dim.get_group_range(0) == 2);
  assert(two_dim.get_group_range(1) == 2);
  assert((two_dim.get_global_range() == cl::sycl::range<2>{4, 4}));
  assert((two_dim.get_global_range(0) == 4 &&
          two_dim.get_global_range(1) == 4));
  assert((two_dim.get_local_range() == cl::sycl::range<2>{2, 2}));
  assert((two_dim.get_local_range(0) == 2 &&
          two_dim.get_local_range(1) == 2));
  assert((two_dim.get_offset() == cl::sycl::id<2>{0, 0}));
  assert((two_dim.get_nd_range() ==
          cl::sycl::nd_range<2>({4, 4}, {2, 2}, {0, 0})));

  // three dimension nd_item
  cl::sycl::nd_item<3> three_dim = Builder::createNDItem<3>(
      Builder::createItem<3, true>({4, 4, 4}, {2, 2, 2}, {0, 0, 0}),
      Builder::createItem<3, false>({2, 2, 2}, {0, 0, 0}),
      Builder::createGroup<3>({4, 4, 4}, {2, 2, 2}, {1, 1, 1}));
  assert((three_dim.get_global_id() == cl::sycl::id<3>{2, 2, 2}));
  assert(three_dim.get_global_id(0) == 2);
  assert(three_dim.get_global_id(1) == 2);
  assert(three_dim.get_global_id(2) == 2);
  assert(three_dim.get_global_linear_id() == 42);
  assert((three_dim.get_local_id() == cl::sycl::id<3>{0, 0, 0}));
  assert(three_dim.get_local_id(0) == 0);
  assert(three_dim.get_local_id(1) == 0);
  assert(three_dim.get_local_id(2) == 0);
  assert(three_dim.get_local_linear_id() == 0);
  assert((three_dim.get_group() ==
          Builder::createGroup<3>({4, 4, 4}, {2, 2, 2}, {1, 1, 1})));
  assert(three_dim.get_group(0) == 1);
  assert(three_dim.get_group(1) == 1);
  assert(three_dim.get_group(2) == 1);
  assert(three_dim.get_group_linear_id() == 7);
  assert((three_dim.get_group_range() == cl::sycl::id<3>{2, 2, 2}));
  assert(three_dim.get_group_range(0) == 2);
  assert(three_dim.get_group_range(1) == 2);
  assert(three_dim.get_group_range(2) == 2);
  assert((three_dim.get_global_range() == cl::sycl::range<3>{4, 4, 4}));
  assert((three_dim.get_global_range(0) == 4 &&
          three_dim.get_global_range(1) == 4 &&
          three_dim.get_global_range(2) == 4));
  assert((three_dim.get_local_range() == cl::sycl::range<3>{2, 2, 2}));
  assert((three_dim.get_local_range(0) == 2 &&
          three_dim.get_local_range(1) == 2 &&
          three_dim.get_local_range(2) == 2));
  assert((three_dim.get_offset() == cl::sycl::id<3>{0, 0, 0}));
  assert((three_dim.get_nd_range() ==
          cl::sycl::nd_range<3>({4, 4, 4}, {2, 2, 2}, {0, 0, 0})));
}
