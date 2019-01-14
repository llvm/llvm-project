// RUN: %clang --sycl %s -c -o %T/kernel.spv
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %t.out

//==--------------- group.cpp - SYCL group test ----------------------------==//
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
  cl::sycl::group<1> one = Builder::createGroup<1>({8}, {4}, {1});
  // one dimension group
  cl::sycl::group<1> one_dim = Builder::createGroup<1>({8}, {4}, {1});
  assert(one_dim.get_id() == cl::sycl::id<1>{1});
  assert(one_dim.get_id(0) == 1);
  assert((one_dim.get_global_range() == cl::sycl::range<1>{8}));
  assert(one_dim.get_global_range(0) == 8);
  assert((one_dim.get_local_range() == cl::sycl::range<1>{4}));
  assert(one_dim.get_local_range(0) == 4);
  assert((one_dim.get_group_range() == cl::sycl::range<1>{4}));
  assert(one_dim.get_group_range(0) == 4);
  assert(one_dim[0] == 1);
  assert(one_dim.get_linear() == 1);

  // two dimension group
  cl::sycl::group<2> two_dim = Builder::createGroup<2>({8, 4}, {4, 2}, {1, 1});
  assert((two_dim.get_id() == cl::sycl::id<2>{1, 1}));
  assert(two_dim.get_id(0) == 1);
  assert(two_dim.get_id(1) == 1);
  assert((two_dim.get_global_range() == cl::sycl::range<2>{8, 4}));
  assert(two_dim.get_global_range(0) == 8);
  assert(two_dim.get_global_range(1) == 4);
  assert((two_dim.get_local_range() == cl::sycl::range<2>{4, 2}));
  assert(two_dim.get_local_range(0) == 4);
  assert(two_dim.get_local_range(1) == 2);
  assert((two_dim.get_group_range() == cl::sycl::range<2>{4, 2}));
  assert(two_dim.get_group_range(0) == 4);
  assert(two_dim.get_group_range(1) == 2);
  assert(two_dim[0] == 1);
  assert(two_dim[1] == 1);
  assert(two_dim.get_linear() == 3);

  // three dimension group
  cl::sycl::group<3> three_dim =
      Builder::createGroup<3>({16, 8, 4}, {8, 4, 2}, {1, 1, 1});
  assert((three_dim.get_id() == cl::sycl::id<3>{1, 1, 1}));
  assert(three_dim.get_id(0) == 1);
  assert(three_dim.get_id(1) == 1);
  assert(three_dim.get_id(2) == 1);
  assert((three_dim.get_global_range() == cl::sycl::range<3>{16, 8, 4}));
  assert(three_dim.get_global_range(0) == 16);
  assert(three_dim.get_global_range(1) == 8);
  assert(three_dim.get_global_range(2) == 4);
  assert((three_dim.get_local_range() == cl::sycl::range<3>{8, 4, 2}));
  assert(three_dim.get_local_range(0) == 8);
  assert(three_dim.get_local_range(1) == 4);
  assert(three_dim.get_local_range(2) == 2);
  assert((three_dim.get_group_range() == cl::sycl::range<3>{8, 4, 2}));
  assert(three_dim.get_group_range(0) == 8);
  assert(three_dim.get_group_range(1) == 4);
  assert(three_dim.get_group_range(2) == 2);
  assert(three_dim[0] == 1);
  assert(three_dim[1] == 1);
  assert(three_dim[2] == 1);
  assert(three_dim.get_linear() == 7);
}
