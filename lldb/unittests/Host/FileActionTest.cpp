//===-- FileActionTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>

#include "lldb/Host/FileAction.h"
#include "gtest/gtest.h"
#if defined(_WIN32)
#include "lldb/Host/windows/PosixApi.h"
#endif

using namespace lldb_private;

TEST(FileActionTest, Open) {
  FileAction Action;
  Action.Open(47, FileSpec("/tmp"), /*read*/ true, /*write*/ false);
  EXPECT_EQ(Action.GetAction(), FileAction::eFileActionOpen);
  EXPECT_EQ(Action.GetFileSpec(), FileSpec("/tmp"));
}

TEST(FileActionTest, OpenReadWrite) {
  FileAction Action;
  Action.Open(48, FileSpec("/tmp_0"), /*read*/ true, /*write*/ true);
  EXPECT_TRUE(Action.GetActionArgument() & (O_NOCTTY | O_CREAT | O_RDWR));
  EXPECT_FALSE(Action.GetActionArgument() & O_RDONLY);
  EXPECT_FALSE(Action.GetActionArgument() & O_WRONLY);
}

TEST(FileActionTest, OpenReadOnly) {
  FileAction Action;
  Action.Open(49, FileSpec("/tmp_1"), /*read*/ true, /*write*/ false);
#ifndef _WIN32
  EXPECT_TRUE(Action.GetActionArgument() & (O_NOCTTY | O_RDONLY));
#endif
  EXPECT_FALSE(Action.GetActionArgument() & O_WRONLY);
}

TEST(FileActionTest, OpenWriteOnly) {
  FileAction Action;
  Action.Open(50, FileSpec("/tmp_2"), /*read*/ false, /*write*/ true);
  EXPECT_TRUE(Action.GetActionArgument() &
              (O_NOCTTY | O_CREAT | O_WRONLY | O_TRUNC));
  EXPECT_FALSE(Action.GetActionArgument() & O_RDONLY);
}
