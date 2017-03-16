//===-- SupportTest.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
