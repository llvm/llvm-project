//===- unittest/Support/ProgramStackTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ProgramStack.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"

using namespace llvm;

static uintptr_t func(int &A) {
  A = 7;
  return getStackPointer();
}

static void func2(int &A) {
  A = 5;
}

TEST(ProgramStackTest, runOnNewStack) {
  int A = 0;
  uintptr_t Stack = runOnNewStack(0, function_ref<uintptr_t(int &)>(func), A);
  EXPECT_EQ(A, 7);
  intptr_t StackDiff = (intptr_t)llvm::getStackPointer() - (intptr_t)Stack;
  size_t StackDistance = (size_t)std::abs(StackDiff);
  // Page size is used as it's large enough to guarantee were not on the same
  // stack but not too large to cause spurious failures.
  EXPECT_GT(StackDistance, llvm::sys::Process::getPageSizeEstimate());
  runOnNewStack(0, function_ref<void(int &)>(func2), A);
  EXPECT_EQ(A, 5);
}
