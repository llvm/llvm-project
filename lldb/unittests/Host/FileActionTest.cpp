//===-- FileActionTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileAction.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(FileActionTest, Open) {
  FileAction Action;
  Action.Open(47, FileSpec("/tmp"), /*read*/ true, /*write*/ false);
  EXPECT_EQ(Action.GetAction(), FileAction::eFileActionOpen);
  EXPECT_EQ(Action.GetFileSpec(), FileSpec("/tmp"));
}
