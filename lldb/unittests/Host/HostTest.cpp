//===-- HostTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

TEST(Host, WaitStatusFormat) {
  EXPECT_EQ("W01", formatv("{0:g}", WaitStatus{WaitStatus::Exit, 1}).str());
  EXPECT_EQ("X02", formatv("{0:g}", WaitStatus{WaitStatus::Signal, 2}).str());
  EXPECT_EQ("S03", formatv("{0:g}", WaitStatus{WaitStatus::Stop, 3}).str());
  EXPECT_EQ("Exited with status 4",
            formatv("{0}", WaitStatus{WaitStatus::Exit, 4}).str());
}
