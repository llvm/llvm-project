//===-- SupportTest.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/linux/Support.h"
#include "llvm/Support/Threading.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(Support, getProcFile_Pid) {
  auto BufferOrError = getProcFile(getpid(), "maps");
  ASSERT_TRUE(BufferOrError);
  ASSERT_TRUE(*BufferOrError);
}

TEST(Support, getProcFile_Tid) {
  auto BufferOrError = getProcFile(getpid(), llvm::get_threadid(), "comm");
  ASSERT_TRUE(BufferOrError);
  ASSERT_TRUE(*BufferOrError);
}
