//===-- Transport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "DAPLog.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

namespace lldb_dap {

Transport::Transport(lldb_dap::Log &log, lldb_private::MainLoop &loop,
                     lldb::IOObjectSP input, lldb::IOObjectSP output)
    : HTTPDelimitedJSONTransport(loop, input, output), m_log(log) {}

void Transport::Log(llvm::StringRef message) {
  // Emit the message directly, since this log was forwarded.
  m_log.Emit(message);
}

} // namespace lldb_dap
