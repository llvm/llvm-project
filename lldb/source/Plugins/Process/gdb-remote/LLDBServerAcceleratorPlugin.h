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
#include <atomic>
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

  /// Return accelerator actions to include when native process \p pid stops.
  /// Returning std::nullopt adds no actions to the stop reply. Servers can
  /// debug multiple processes, so plugins should ignore unrelated PIDs. LLGS
  /// calls this only for all-stop replies. The call can occur on a process
  /// state-notification thread, so implementations must be thread-safe. If an
  /// action is returned more than once for the same event, it must reuse its
  /// identifier.
  virtual std::optional<AcceleratorActions>
  NativeProcessIsStopping(lldb::pid_t pid) {
    return std::nullopt;
  }

  virtual llvm::Expected<AcceleratorBreakpointHitResponse>
  BreakpointWasHit(AcceleratorBreakpointHitArgs &args) = 0;

  /// Create an AcceleratorActions with an identifier unique within this plugin.
  /// Other action sources and connections for the plugin must coordinate with
  /// this sequence so no identifier is reused during a debugging session.
  AcceleratorActions GetNewAcceleratorAction() {
    return AcceleratorActions(GetPluginName(),
                              m_next_accelerator_action_identifier.fetch_add(
                                  1, std::memory_order_relaxed));
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
  // Atomic signed fetch_add has defined two's-complement wrapping. The full
  // int64_t identifier space is used before an identifier can be reused.
  std::atomic<int64_t> m_next_accelerator_action_identifier{1};
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_LLDBSERVERACCELERATORPLUGIN_H
