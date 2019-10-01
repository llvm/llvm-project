//===-- GDBRemoteTestUtils.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteTestUtils.h"
#include "lldb/Host/Socket.h"
#include "llvm/Testing/Support/Error.h"

namespace lldb_private {
namespace process_gdb_remote {

void GDBRemoteTest::SetUpTestCase() {
  ASSERT_THAT_ERROR(Socket::Initialize(), llvm::Succeeded());
}

void GDBRemoteTest::TearDownTestCase() { Socket::Terminate(); }

} // namespace process_gdb_remote
} // namespace lldb_private
