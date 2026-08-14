//===- CompilerTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the utilities in orc-rt/Compiler.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Compiler.h"

#include "gtest/gtest.h"

namespace {

int reachablePathReturns(int X) {
  switch (X) {
  case 0:
    return 0;
  default:
    ORC_RT_UNREACHABLE("only 0 is expected");
  }
}

} // anonymous namespace

TEST(CompilerTest, UnreachableCompilesInReturningFunction) {
  EXPECT_EQ(reachablePathReturns(0), 0);
}

// ORC_RT_UNREACHABLE only aborts in +Asserts builds; under NDEBUG it lowers to
// a bare optimizer hint whose execution is undefined behavior, so there is
// nothing well-defined to assert on in a release build. Test for !NDEBUG builds
// only.
#ifndef NDEBUG
TEST(CompilerDeathTest, UnreachableAborts) {
  EXPECT_DEATH(ORC_RT_UNREACHABLE("unreachable reached"),
               "unreachable reached");
}
#endif
