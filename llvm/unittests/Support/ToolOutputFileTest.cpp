//===- ToolOutputFileTest.cpp - ToolOutputFile tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ToolOutputFileTest, DashOpensOuts) {
  std::error_code EC;
  EXPECT_EQ(&ToolOutputFile("-", EC, sys::fs::OF_None).os(), &outs());
}

} // namespace
