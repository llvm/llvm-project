//===-- SupportTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/Support.h"
#include "lldb/Host/aix/Support.h"
#include "llvm/Support/Threading.h"
#include "gtest/gtest.h"

using namespace lldb_private;

#ifndef __APPLE__
TEST(Support, getProcFile_Pid) {
  auto BufferOrError = getProcFile(getpid(), "status");
  ASSERT_TRUE(BufferOrError);
  ASSERT_TRUE(*BufferOrError);
}
#endif // #ifndef __APPLE__

#if defined(_AIX) && defined(LLVM_ENABLE_THREADING)
TEST(Support, getProcFile_Tid) {
  auto BufferOrError = getProcFile(getpid(), llvm::get_threadid(), "lwpstatus");
  ASSERT_TRUE(BufferOrError);
  ASSERT_TRUE(*BufferOrError);
}
#endif // #ifdef _AIX && LLVM_ENABLE_THREADING
