//===-- LLDBServerNativeProcess.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERNATIVEPROCESS_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERNATIVEPROCESS_H

#include "lldb/Utility/Status.h"
#include "lldb/lldb-types.h"
#include <memory>

namespace lldb_private {

class NativeProcessProtocol;

namespace lldb_server {

/// A class that interfaces back to the lldb-server native process for
/// LLDBServerNativeProcess objects.
class LLDBServerNativeProcess {
  NativeProcessProtocol *m_native_process;

public:
  LLDBServerNativeProcess(NativeProcessProtocol *native_process);
  ~LLDBServerNativeProcess();

  /// Set a breakpoint in the native process.
  ///
  /// When the breakpoints gets hit, lldb-server will call
  /// LLDBServerPlugin::BreakpointWasHit with this address. This will allow
  /// LLDBServerPlugin plugins to synchronously handle a breakpoint hit in
  /// the native process.
  lldb::user_id_t SetBreakpoint(lldb::addr_t address);
  Status RegisterSignalCatcher(int signo);
  Status HaltProcess();
  Status ContinueProcess();
};

} // namespace lldb_server
} // namespace lldb_private

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERNATIVEPROCESS_H
