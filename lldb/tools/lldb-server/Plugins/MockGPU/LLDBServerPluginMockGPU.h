//===-- LLDBServerPluginMockGPU.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINMOCKGPU_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINMOCKGPU_H

#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {
  
  class TCPSocket;

namespace lldb_server {

class LLDBServerPluginMockGPU : public lldb_private::lldb_server::LLDBServerPlugin {
public:
  LLDBServerPluginMockGPU(lldb_private::lldb_server::LLDBServerPlugin::GDBServer &native_process);
  ~LLDBServerPluginMockGPU() override;
  int GetEventFileDescriptorAtIndex(size_t idx) override;
  bool HandleEventFileDescriptorEvent(int fd) override;
  std::optional<std::string> GetConnectionURL() override;

private:
  void CloseFDs();
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  // Used with a socketpair to get events on the native ptrace event queue.
  int m_fds[2] = {-1, -1};
  Status m_main_loop_status;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINMOCKGPU_H
