//===-- TestCommandReturnObject.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandReturnObject.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(CommandReturnObjectTest, DefaultStatusIsInvalid) {
  CommandReturnObject result(/*colors=*/false);
  EXPECT_EQ(result.GetStatus(), eReturnStatusInvalid);
}

TEST(CommandReturnObjectTest, SetStatusUpdatesStatus) {
  CommandReturnObject result(false);
  result.SetStatus(eReturnStatusSuccessFinishResult);
  EXPECT_EQ(result.GetStatus(), eReturnStatusSuccessFinishResult);
}

TEST(CommandReturnObjectTest, AppendErrorSetsFailed) {
  CommandReturnObject result(false);
  result.AppendError("boom");
  EXPECT_EQ(result.GetStatus(), eReturnStatusFailed);
}

TEST(CommandReturnObjectTest, ClearResetsToInvalid) {
  CommandReturnObject result(false);
  result.SetStatus(eReturnStatusSuccessFinishResult);
  ASSERT_EQ(result.GetStatus(), eReturnStatusSuccessFinishResult);
  result.Clear();
  EXPECT_EQ(result.GetStatus(), eReturnStatusInvalid);
}
