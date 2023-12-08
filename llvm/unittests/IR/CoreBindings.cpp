//===- llvm/unittest/IR/CoreBindings.cpp - Tests for C-API bindings -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/Config/llvm-config.h"
#include "gtest/gtest.h"

namespace {

TEST(CoreBindings, VersionTest) {
  // Test ability to ignore output parameters
  LLVMGetVersion(nullptr, nullptr, nullptr);

  unsigned Major, Minor, Patch;
  LLVMGetVersion(&Major, &Minor, &Patch);
  EXPECT_EQ(Major, (unsigned)LLVM_VERSION_MAJOR);
  EXPECT_EQ(Minor, (unsigned)LLVM_VERSION_MINOR);
  EXPECT_EQ(Patch, (unsigned)LLVM_VERSION_PATCH);
}

} // end anonymous namespace
