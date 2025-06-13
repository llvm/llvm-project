//===-- PipeTestUtilities.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_PIPETESTUTILITIES_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_PIPETESTUTILITIES_H

#include "lldb/Host/Pipe.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

/// A base class for tests that need a pair of pipes for communication.
class PipeTest : public testing::Test {
protected:
  lldb_private::Pipe input;
  lldb_private::Pipe output;

  void SetUp() override {
    ASSERT_THAT_ERROR(input.CreateNew(false).ToError(), llvm::Succeeded());
    ASSERT_THAT_ERROR(output.CreateNew(false).ToError(), llvm::Succeeded());
  }
};

#endif
