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

  /// Stop the native process if it is running.
  ///
  /// An accelerator plugin can use this when its accelerator stops. This can
  /// be called from any thread and returns without waiting: the native
  /// process's GDB server interrupts it on its own main loop. The client's stop
  /// reply reports lldb::eStopReasonTargetSync and names this plugin so the
  /// client can identify the corresponding accelerator target.
  ///
  /// \param[in] pid
  ///     The native process to stop. Capture this while handling the event
  ///     associated with that process so a later multiprocess selection change
  ///     cannot redirect the interrupt.
  void HaltNativeProcessIfNeeded(lldb::pid_t pid);

  virtual std::optional<AcceleratorActions> GetInitializeActions() = 0;

  virtual llvm::Expected<AcceleratorBreakpointHitResponse>
  BreakpointWasHit(AcceleratorBreakpointHitArgs &args) = 0;

  /// Create an AcceleratorActions with an identifier unique within this plugin,
  /// so identifiers from different actions don't collide.
  AcceleratorActions GetNewAcceleratorAction() {
    return AcceleratorActions(GetPluginName(),
                              ++m_accelerator_action_identifier);
  }

  virtual std::optional<AcceleratorDynamicLoaderResponse>
  GetDynamicLoaderLibraryInfos(const AcceleratorDynamicLoaderArgs &args) {
    return std::nullopt;
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
