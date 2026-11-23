//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Use the umbrella header for -Wdocumentation.
#include "lldb/API/LLDB.h"

#include "gtest/gtest.h"

using namespace lldb;

TEST(SBInitializationOptionsTest, Initialize) {
  SBInitializationOptions options;
  EXPECT_TRUE(SBDebugger::Initialize(options).Success());
  SBDebugger::Terminate();
}
