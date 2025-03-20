//===-- LLDBServerPlugin.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGIN_H

#include "lldb/lldb-types.h"
#include <functional>
#include <stdint.h>
#include <string>
namespace lldb_private {
namespace lldb_server {

class LLDBServerNativeProcess;

typedef std::function<std::string(const char *)>
    CStrArgWithReturnStringCallback;
typedef std::function<std::string()> NoArgsReturnStringCallback;

// lldb-server will decode all packets and any packets and for any packets that
// have handlers in this structure, the functions will be called. This removes
// the need for plug-ins to have to parse the packets and args.
struct GDBRemotePacketCallbacks {
  // Handle any "general query packets" here.
  // Handle the "qSupported" query.

  CStrArgWithReturnStringCallback qSupported = nullptr;
  NoArgsReturnStringCallback qHostInfo = nullptr;
  NoArgsReturnStringCallback qProcessInfo = nullptr;
  CStrArgWithReturnStringCallback qXfer = nullptr;

  // Handle "vCont?" packet
  NoArgsReturnStringCallback vContQuery = nullptr;
};

class LLDBServerPlugin {
  // Add a version field to allow the APIs to change over time.
  const uint32_t m_version = 1;
  LLDBServerNativeProcess &m_process;

public:
  LLDBServerPlugin(LLDBServerNativeProcess &process) : m_process(process) {}

  virtual LLDBServerPlugin() = default;

  virtual void GetCallbacks(GDBRemotePacketCallbacks &callbacks);

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

  /// Handle a received a GDB remote packet that doesn't have a callback
  /// specified in the GDBRemotePacketCallbacks structure after a call to
  /// LLDBServerPlugin::GetCallbacks(...).
  ///
  /// \return
  ///   The resonse packet to send. If the empty string is returned, this will
  ///   cause an unimplemented packet ($#00) to be sent signaling this packet
  ///   is not supported.
  virtual std::string HandlePacket(const uint8_t *payload, size_t payload_size);

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
