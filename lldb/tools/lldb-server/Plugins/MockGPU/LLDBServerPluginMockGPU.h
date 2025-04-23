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

// This is a mock GPU plugin that is used for testing the LLDBServerPlugin. It
// should be run with the following code as the main binary:
/*

$ cat main.cpp
#include <stdio.h>

struct ShlibInfo {
  const char *path = nullptr;
  ShlibInfo *next = nullptr;
};

ShlibInfo g_shlib_list = { "/tmp/a.out", nullptr};

int gpu_initialize() {
  return puts(__FUNCTION__);
}
int gpu_shlib_load() {
  return puts(__FUNCTION__);
}
int main(int argc, const char **argv) {
  gpu_initialize();
  gpu_shlib_load();
  return 0; // Break here
}

$ clang++ -g -O0 -o a.out main.cpp
$ ninja lldb lldb-server
$ ./bin/lldb a.out -o 'b /Break here/ -o run

*/
// If the above code is run, you will be stopped at the breakpoint and the Mock
// GPU target will be selected. Try doing a "reg read --all" to see the state
// of the GPU registers. Then you can select the native process target with 
// "target select 0" and issue commands to the native process, and then select
// the GPU target with "target select 1" and issue commands to the GPU target.

namespace lldb_private {
  
  class TCPSocket;

namespace lldb_server {

class LLDBServerPluginMockGPU : public LLDBServerPlugin {
public:
  LLDBServerPluginMockGPU(LLDBServerPlugin::GDBServer &native_process);
  ~LLDBServerPluginMockGPU() override;
  llvm::StringRef GetPluginName() override;
  int GetEventFileDescriptorAtIndex(size_t idx) override;
  bool HandleEventFileDescriptorEvent(int fd) override;
  GPUActions GetInitializeActions() override;
  std::optional<struct GPUActions> NativeProcessIsStopping() override;  
  GPUPluginBreakpointHitResponse 
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

private:
  std::optional<GPUPluginConnectionInfo> CreateConnection();
  void CloseFDs();
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  // Used with a socketpair to get events on the native ptrace event queue.
  int m_fds[2] = {-1, -1};
  Status m_main_loop_status;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINMOCKGPU_H
