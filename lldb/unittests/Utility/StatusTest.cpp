//===-- StatusTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Status.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(StatusTest, Formatv) {
  EXPECT_EQ("", llvm::formatv("{0}", Status()).str());
  EXPECT_EQ("Hello Status", llvm::formatv("{0}", Status("Hello Status")).str());
  EXPECT_EQ("Hello", llvm::formatv("{0:5}", Status("Hello Error")).str());
}
