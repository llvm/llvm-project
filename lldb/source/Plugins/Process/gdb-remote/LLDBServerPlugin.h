//===-- LLDBServerPlugin.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"

#include <functional>
#include <optional>
#include <stdint.h>
#include <string>
#include <mutex>

namespace lldb_private {

namespace process_gdb_remote {
  class GDBRemoteCommunicationServerLLGS;
}

namespace lldb_server {

class LLDBServerPlugin {
protected:
  // Add a version field to allow the APIs to change over time.
  using GDBServer = process_gdb_remote::GDBRemoteCommunicationServerLLGS;
  using Manager = NativeProcessProtocol::Manager;
  GDBServer &m_native_process;
  MainLoop m_main_loop;
  std::unique_ptr<Manager> m_process_manager_up;
  std::unique_ptr<GDBServer> m_gdb_server;
  bool m_is_listening = false;
  bool m_is_connected = false;

public:
  LLDBServerPlugin(GDBServer &native_process);
  virtual ~LLDBServerPlugin();

  /// Check if we are already connected.
  bool IsConnected() const { return m_is_connected; }

  virtual llvm::StringRef GetPluginName() = 0;

  /// Stop the native process if it is running.
  ///
  /// Some plug-ins might want to stop the native process if it is running so
  /// that the plug-in can return some GPUActions from the call to the
  /// NativeProcessIsStopping(). This function will trigger the native process
  /// to stop only if it is running.
  ///
  /// \param[out] was_halted The \a was_halted parameter will be set to
  ///   true if the process was running and was halted. It will be false if the
  ///   process was already stopped.
  ///
  /// \param[out] timeout_sec The timeout in seconds to wait for the process to
  ///   enter the stopped state.
  ///
  /// \return The actual state of the process in case the process was not able
  /// to be stopped within the specified timeout.
  lldb::StateType HaltNativeProcessIfNeeded(bool &was_halted,
                                            uint32_t timeout_sec = 5);

  /// Get notified when the process is stopping.
  ///
  /// This function will get called each time native process stops as the stop
  /// reply packet is being created. If the plug-in is ready to be activated,
  /// return a GPUPluginConnectionInfo with a value connection URL to use with
  /// "process connect" which can connect to this plug-in. Plug-ins should wait
  /// for a connection to be made before trying to do any blocking code. The
  /// plug-in should assume the users do not want to use any features unless a
  /// connection is made.
  virtual std::optional<GPUActions> NativeProcessIsStopping() {
    return std::nullopt;
  };

  /// Get the GPU plug-in initialization actions.
  ///
  /// Each GPU plugin can return a structure that describes the GPU plug-in and
  /// any actions that should be performed right away .Actions include setting
  /// any breakpoints it requires in the native process. GPU plug-ins might want
  /// to set breakpoints in the native process to know when the GPU has been
  /// initialized, or when the GPU has shared libraries that get loaded.
  /// They can do this by populating returning any actions needed when this
  /// function is called.
  ///
  /// The contents of this structure will be converted to JSON and sent to the
  /// LLDB client. The structure allows plug-ins to set breakpoints by name
  /// and to also request symbol values that should be sent when the breakpoint
  /// gets hit. When the breakpoint is hit, the BreakpointWasHit(...) method
  /// will get called with a structure that identifies the plugin,
  /// breakpoint and it will supply any requested symbol values.
  virtual GPUActions GetInitializeActions() = 0;

  /// Get a file descriptor to listen for in the ptrace epoll loop.
  ///
  /// When polling for process ptrace events, plug-ins can supply extra file
  /// descriptors that should be listened to. When a file descriptor has
  /// events, the LLDBServerPlugin::HandleFileDescriptorEvent(...) function
  /// will get called synchronously from the event loop listening for events.
  /// This allows synchronization with the ptrace event loop.
  ///
  /// \param idx The index of the file descriptor to add.
  ///
  /// \return A valid file descriptor if \a idx is a valid index, or -1.
  virtual int GetEventFileDescriptorAtIndex(size_t idx) { return -1; }

  /// Handle a file descriptor event that was registered with the lldb-server
  /// from previous calls to LLDBServerPlugin::GetEventFileDescriptorAtIndex()
  ///
  /// \param fd The file descriptor event to handle.
  virtual bool HandleEventFileDescriptorEvent(int fd) { return false; }

  /// Called when a breakpoint is hit in the native process.
  ///
  /// LLDBServerPlugin objects can set breakpoints in the native process by
  /// calling m_process.SetBreakpoint(...) to help implement funcionality,
  /// such as dynamic library loading in GPUs or to synchronize in any other
  /// way with the native process.
  virtual GPUPluginBreakpointHitResponse
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) = 0;

  protected:
    std::mutex m_connect_mutex;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H
