//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_LLDBSERVERMOCKACCELERATORPLUGIN_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_LLDBSERVERMOCKACCELERATORPLUGIN_H

#include "Plugins/Process/gdb-remote/LLDBServerAcceleratorPlugin.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"

#include <memory>
#include <vector>

namespace lldb_private {

class TCPSocket;

namespace lldb_server {

class LLDBServerMockAcceleratorPlugin : public LLDBServerAcceleratorPlugin {
public:
  LLDBServerMockAcceleratorPlugin(GDBServer &native_gdb_server,
                                  MainLoop &native_main_loop);
  ~LLDBServerMockAcceleratorPlugin() override;

  llvm::StringRef GetPluginName() override;
  std::optional<AcceleratorActions> GetInitializeActions() override;
  llvm::Expected<AcceleratorBreakpointHitResponse>
  BreakpointWasHit(AcceleratorBreakpointHitArgs &args) override;

private:
  // Lazily bring up the mock accelerator GDB server and return its connection
  // info. Called on the connection breakpoint hit, so inferiors that never
  // connect create nothing.
  std::optional<AcceleratorConnectionInfo> CreateConnection();

  // Breakpoint set during initialization, by function name with no shared
  // library. Requests the "compute" symbol value when hit.
  static constexpr int64_t kBreakpointIDInitialize = 1;
  // Breakpoint set by address, using the "compute" symbol value delivered when
  // the initialize breakpoint was hit.
  static constexpr int64_t kBreakpointIDByAddress = 2;
  // Breakpoint set by function name scoped to a shared library.
  static constexpr int64_t kBreakpointIDByNameShlib = 3;
  // Breakpoint on the dedicated "mock_gpu_accelerator_connect" hook; only the
  // connection test defines that function. See BreakpointWasHit for the hit.
  static constexpr int64_t kBreakpointIDConnect = 4;

  // The mock accelerator server runs on its own main loop and thread so its
  // packets don't contend with the native process. Populated by
  // CreateConnection(), torn down in the destructor.
  MainLoop m_mock_main_loop;
  HostThread m_mock_main_loop_thread;
  std::unique_ptr<TCPSocket> m_listen_socket;
  std::vector<MainLoopBase::ReadHandleUP> m_read_handles;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_LLDBSERVERMOCKACCELERATORPLUGIN_H
