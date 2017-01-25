//===-- StateTest.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/State.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(StateTest, Formatv) {
  EXPECT_EQ("exited", llvm::formatv("{0}", eStateExited).str());
  EXPECT_EQ("stopped", llvm::formatv("{0}", eStateStopped).str());
  EXPECT_EQ("unknown", llvm::formatv("{0}", StateType(-1)).str());
}
