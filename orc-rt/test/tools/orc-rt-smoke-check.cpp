//===- orc-rt-smoke-check.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A minimal regression-test tool. It exists only to smoke-check that the ORC
// runtime regression test-tool infrastructure works end to end: that a tool
// under test/tools is built, placed where lit can find it, and that its output
// can be matched by a regression test.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

int main() {
  std::puts("smoke check");
  return 0;
}
