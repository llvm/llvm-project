//===-- GDBRemoteTestUtils.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
