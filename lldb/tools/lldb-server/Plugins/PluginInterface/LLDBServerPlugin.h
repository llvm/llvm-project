//===-- LLDBServerPlugin.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/lldb-types.h"

#include <functional>
#include <optional>
#include <stdint.h>
#include <string>
namespace lldb_private {

namespace process_gdb_remote {
  class GDBRemoteCommunicationServerLLGS;
}
  
namespace lldb_server {


class LLDBServerNativeProcess;

class LLDBServerPlugin {
protected:
  // Add a version field to allow the APIs to change over time.
  using GDBServer = process_gdb_remote::GDBRemoteCommunicationServerLLGS;
  using Manager = NativeProcessProtocol::Manager;
  GDBServer &m_native_process;
  MainLoop m_main_loop;
  std::unique_ptr<Manager> m_process_manager_up;
  std::unique_ptr<GDBServer> m_gdb_server;
  bool m_is_connected = false;

public:
  using CreateCallback = llvm::function_ref<LLDBServerPlugin *()>;
  static void RegisterCreatePlugin(CreateCallback callback);
  static size_t GetNumCreateCallbacks();
  static CreateCallback GetCreateCallbackAtIndex(size_t i);
  LLDBServerPlugin(GDBServer &native_process) : 
      m_native_process(native_process) {}

  virtual ~LLDBServerPlugin();

  /// Check if we are already connected.
  bool IsConnected() const { return m_is_connected; }
  /// Get an connection URL to connect to this plug-in.
  ///
  /// This function will get called each time native process stops if this
  /// object is not connected already. If the plug-in is ready to be activated,
  /// return a valid URL to use with "process connect" that can connect to this
  /// plug-in. Execution should wait for a connection to be made before trying
  /// to do any blocking code. The plug-in should assume the users do not want
  /// to use any features unless a connection is made.
  virtual std::optional<std::string> GetConnectionURL() {
    return std::nullopt;
  };

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
  virtual void BreakpointWasHit(lldb::addr_t address) {}
};

} // namespace lldb_server
} // namespace lldb_private

#endif
