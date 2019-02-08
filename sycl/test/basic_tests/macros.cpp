// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
//==------------------- macros.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  std::cout << "SYCL language version: " << CL_SYCL_LANGUAGE_VERSION
            << std::endl;
  std::cout << "SYCL compiler version: " << __SYCL_COMPILER_VERSION
            << std::endl;
  return 0;
}
