//===-- FileActionTest.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
