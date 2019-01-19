//===-- GDBRemoteTestUtils.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteTestUtils.h"

#if defined(_MSC_VER)
#include "lldb/Host/windows/windows.h"
#include <WinSock2.h>
#endif

namespace lldb_private {
namespace process_gdb_remote {

void GDBRemoteTest::SetUpTestCase() {
#if defined(_MSC_VER)
  WSADATA data;
  ::WSAStartup(MAKEWORD(2, 2), &data);
#endif
}

void GDBRemoteTest::TearDownTestCase() {
#if defined(_MSC_VER)
  ::WSACleanup();
#endif
}

} // namespace process_gdb_remote
} // namespace lldb_private
