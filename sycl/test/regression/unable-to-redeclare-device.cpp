// RUN: %clang -I %sycl_include -std=c++11 -fsyntax-only -Xclang -verify -DCL_TARGET_OPENCL_VERSION=220 %s
// expected-no-diagnostics
//
//==-- unable-to-redeclare-device.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks that the following symbols (Device, GroupOperation) are not
// defined in global namespace by sycl.hpp and available to user
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

enum GroupOperation {
  ADD, SUB
};

int Device = 10, spv = 20;

int main() { return 0; }
