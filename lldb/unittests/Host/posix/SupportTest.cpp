//===-- SupportTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/Support.h"
#include "llvm/Support/Threading.h"
#include "gtest/gtest.h"

#if defined(_AIX)
#include "lldb/Host/aix/Support.h"
#elif defined(__linux__)
#include "lldb/Host/linux/Support.h"
#endif

using namespace lldb_private;

#ifndef __APPLE__
TEST(Support, getProcFile_Pid) {
  auto BufferOrError = getProcFile(getpid(), "status");
  ASSERT_TRUE(BufferOrError);
  ASSERT_TRUE(*BufferOrError);
}
#endif // #ifndef __APPLE__

#if defined(_AIX) || defined(__linux__)
TEST(Support, getProcFile_Tid) {
  auto BufferOrError = getProcFile(getpid(), llvm::get_threadid(),
#ifdef _AIX
                                   "lwpstatus"
#else
                                   "status"
#endif
  );
  ASSERT_TRUE(BufferOrError);
  ASSERT_TRUE(*BufferOrError);
}
#endif // #if defined(_AIX) || defined(__linux__)
