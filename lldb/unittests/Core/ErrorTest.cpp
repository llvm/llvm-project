//===-- ErrorTest.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(ErrorTest, Formatv) {
  EXPECT_EQ("", llvm::formatv("{0}", Error()).str());
  EXPECT_EQ("Hello Error", llvm::formatv("{0}", Error("Hello Error")).str());
  EXPECT_EQ("Hello", llvm::formatv("{0:5}", Error("Hello Error")).str());
}
