//===-- LLDBServerNativeProcess.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H

#include "lldb/Utility/Status.h"
#include "lldb/lldb-types.h"
#include <memory>

namespace lldb_private {
namespace lldb_server {

/// A class that interfaces back to the lldb-server native process for
/// LLDBServerNativeProcess objects.
class LLDBServerNativeProcess {
public:
  static LLDBServerNativeProcess *GetNativeProcess();
  // lldb-server will call this function to set the native process object prior
  // to any plug-ins being loaded.
  static void SetNativeProcess(LLDBServerNativeProcess *process);

  virtual ~LLDBServerNativeProcess();
  /// Set a breakpoint in the native process.
  ///
  /// When the breakpoints gets hit, lldb-server will call
  /// LLDBServerPlugin::BreakpointWasHit with this address. This will allow
  /// LLDBServerPlugin plugins to synchronously handle a breakpoint hit in the
  /// native process.
  virtual lldb::user_id_t SetBreakpoint(lldb::addr_t address) = 0;

  virtual Status RegisterSignalCatcher(int signo) = 0;

  virtual Status HaltProcess() = 0;
  virtual Status ContinueProcess() = 0;
};

} // namespace lldb_server
} // namespace lldb_private

#endif
