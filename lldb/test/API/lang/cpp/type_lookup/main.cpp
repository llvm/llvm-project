//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In this test, we define struct that exist might exist at the different
// levels in the code and test that we can properly locate these types with
// a varienty of different expressions.

namespace nsp_a {
  struct namespace_only {
    int a;
  };
  struct namespace_and_file {
    int aa;
  };
  struct contains_type {
    struct in_contains_type {
      int aaa;
    };
  };
};
namespace nsp_b {
  struct namespace_only {
    int b;
  };
  struct namespace_and_file {
    int bb;
  };
  struct contains_type {
    struct in_contains_type {
      int bbb;
    };
  };
};

struct namespace_and_file {
  int ff;
};

struct contains_type {
  struct in_contains_type {
    int fff;
  };
};


int main (int argc, char const *argv[]) {
  nsp_a::namespace_only a_namespace_only = { 1 };
  nsp_a::namespace_and_file a_namespace_and_file = { 2 };
  nsp_a::contains_type::in_contains_type a_in_contains_type = { 3 };
  nsp_b::namespace_only b_namespace_only = { 11 };
  nsp_b::namespace_and_file b_namespace_and_file = { 22 };
  nsp_b::contains_type::in_contains_type b_in_contains_type = { 33 };
  namespace_and_file file_namespace_and_file = { 44 };
  contains_type::in_contains_type file_in_contains_type = { 55 };
  int i = 123; // Provide an integer that can be used for casting
  // Take address of "i" to ensure it is in memory
  if (&i == &argc) {
    i = -1;
  }
  return i == -1; // Set a breakpoint here
}
