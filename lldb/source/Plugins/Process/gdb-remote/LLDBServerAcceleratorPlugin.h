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
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/AcceleratorGDBRemotePackets.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>

namespace lldb_private {

namespace process_gdb_remote {
class GDBRemoteCommunicationServerLLGS;
} // namespace process_gdb_remote

namespace lldb_server {

class LLDBServerAcceleratorPlugin {
public:
  using GDBServer = process_gdb_remote::GDBRemoteCommunicationServerLLGS;
  using Manager = NativeProcessProtocol::Manager;

  LLDBServerAcceleratorPlugin(GDBServer &native_gdb_server,
                              MainLoop &native_main_loop);
  virtual ~LLDBServerAcceleratorPlugin();

  virtual llvm::StringRef GetPluginName() = 0;

  virtual std::optional<AcceleratorActions> GetInitializeActions() = 0;

  virtual llvm::Expected<AcceleratorBreakpointHitResponse>
  BreakpointWasHit(AcceleratorBreakpointHitArgs &args) = 0;

  /// Create an AcceleratorActions with an identifier unique within this plugin,
  /// so identifiers from different actions don't collide.
  AcceleratorActions GetNewAcceleratorAction() {
    return AcceleratorActions(GetPluginName(),
                              ++m_accelerator_action_identifier);
  }

protected:
  GDBServer &m_native_gdb_server;
  MainLoop &m_native_main_loop;
  std::unique_ptr<Manager> m_process_manager_up;
  std::unique_ptr<GDBServer> m_accelerator_gdb_server;
  int64_t m_accelerator_action_identifier = 0;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_LLDBSERVERACCELERATORPLUGIN_H
