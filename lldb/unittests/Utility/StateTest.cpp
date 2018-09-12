//===-- StateTest.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/State.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(StateTest, Formatv) {
  EXPECT_EQ("invalid", llvm::formatv("{0}", eStateInvalid).str());
  EXPECT_EQ("unloaded", llvm::formatv("{0}", eStateUnloaded).str());
  EXPECT_EQ("connected", llvm::formatv("{0}", eStateConnected).str());
  EXPECT_EQ("attaching", llvm::formatv("{0}", eStateAttaching).str());
  EXPECT_EQ("launching", llvm::formatv("{0}", eStateLaunching).str());
  EXPECT_EQ("stopped", llvm::formatv("{0}", eStateStopped).str());
  EXPECT_EQ("running", llvm::formatv("{0}", eStateRunning).str());
  EXPECT_EQ("stepping", llvm::formatv("{0}", eStateStepping).str());
  EXPECT_EQ("crashed", llvm::formatv("{0}", eStateCrashed).str());
  EXPECT_EQ("detached", llvm::formatv("{0}", eStateDetached).str());
  EXPECT_EQ("exited", llvm::formatv("{0}", eStateExited).str());
  EXPECT_EQ("suspended", llvm::formatv("{0}", eStateSuspended).str());
  
}
