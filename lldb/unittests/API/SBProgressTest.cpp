//===-- SBProgressTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "gtest/gtest.h"

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/API/LLDB.h"

using namespace lldb_private;
using namespace lldb;

class SBProgressTest : public testing::Test {
protected:
  void SetUp() override {
    debugger = SBDebugger::Create(/*source_init_files=*/false);
  }

  void TearDown() override { SBDebugger::Destroy(debugger); }

  SubsystemRAII<SBDebugger> subsystems;
  SBDebugger debugger;
};

TEST_F(SBProgressTest, Constructor) {
  // Calling the constructor with null values does not crash.
  SBProgress progress(nullptr, nullptr, debugger);
  SBProgress progress2(nullptr, nullptr, /*total_units=*/10, debugger);

  progress.Increment(2, nullptr);
  progress.Increment(2, "Other");
  progress.Finalize();
}
