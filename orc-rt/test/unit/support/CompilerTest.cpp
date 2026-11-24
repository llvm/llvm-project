//===- CompilerTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the utilities in orc-rt/support/Compiler.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/Compiler.h"

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

} // namespace

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

// Defined in CAPICompileTest.c, which is compiled as C. The value it returns
// is decided by the preprocessor when that file is built, so comparing it
// against the answer here checks that ORC_RT_HAS_BUILTIN works in C and agrees
// across the two languages.
extern "C" int orc_rt_test_hasBuiltinExpect(void);

TEST(CompilerTest, HasBuiltinAgreesBetweenCAndCxx) {
  EXPECT_EQ(orc_rt_test_hasBuiltinExpect(),
            ORC_RT_HAS_BUILTIN(__builtin_expect) ? 1 : 0);
}

// Also defined in CAPICompileTest.c: these check that the macros moved into
// orc-rt-c/support/Compiler.h behave correctly when used from C, not merely
// that they parse there.
extern "C" {
int orc_rt_test_likely(int X);
int orc_rt_test_unlikely(int X);
int orc_rt_test_unreachable(int X);
}

TEST(CompilerTest, LikelyMacrosPreserveTruthinessInC) {
  EXPECT_EQ(orc_rt_test_likely(0), 0);
  EXPECT_EQ(orc_rt_test_likely(1), 1);
  // A value whose low bits are zero must still be truthy: !! normalizes, a
  // truncating conversion would not.
  EXPECT_EQ(orc_rt_test_likely(256), 1);
  EXPECT_EQ(orc_rt_test_unlikely(0), 0);
  EXPECT_EQ(orc_rt_test_unlikely(1), 1);
  EXPECT_EQ(orc_rt_test_unlikely(256), 1);
}

TEST(CompilerTest, UnreachableCompilesInReturningCFunction) {
  EXPECT_EQ(orc_rt_test_unreachable(0), 0);
}

#ifndef NDEBUG
TEST(CompilerDeathTest, UnreachableAbortsFromC) {
  EXPECT_DEATH(orc_rt_test_unreachable(1), "only 0 is expected");
}
#endif
