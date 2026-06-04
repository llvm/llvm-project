//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_LLDBSERVERACCELERATORPLUGIN_H
#define LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_LLDBSERVERACCELERATORPLUGIN_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Utility/AcceleratorGDBRemotePackets.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace lldb_private {

namespace process_gdb_remote {
class GDBRemoteCommunicationServerLLGS;
} // namespace process_gdb_remote

namespace lldb_server {

class LLDBServerAcceleratorPlugin {
public:
  using GDBServer = process_gdb_remote::GDBRemoteCommunicationServerLLGS;

  LLDBServerAcceleratorPlugin(GDBServer &gdb_server, MainLoop &main_loop);
  virtual ~LLDBServerAcceleratorPlugin();

  virtual llvm::StringRef GetPluginName() = 0;

  virtual std::optional<AcceleratorActions> GetInitializeActions() = 0;

  virtual llvm::Expected<AcceleratorBreakpointHitResponse>
  BreakpointWasHit(AcceleratorBreakpointHitArgs &args) = 0;

protected:
  GDBServer &m_gdb_server;
  MainLoop &m_main_loop;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_LLDBSERVERACCELERATORPLUGIN_H
