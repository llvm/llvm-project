//===-- LLDBServerPluginAMDGPU.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINAMDGPU_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINAMDGPU_H

#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "ProcessAMDGPU.h"
#include <amd-dbgapi/amd-dbgapi.h>

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

class GPUIOObject : public IOObject {
public:
  GPUIOObject(int notifier_fd)
      : lldb_private::IOObject(eFDTypeSocket), m_notifier_fd(notifier_fd) {}

  Status Read(void *buf, size_t &num_bytes) override {
    Status error;
    return error;
  }
  Status Write(const void *buf, size_t &num_bytes) override {
    Status error;
    return error;
  }
  virtual bool IsValid() const override { return true; }
  virtual Status Close() override {
    Status error;
    return error;
  }

  virtual WaitableHandle GetWaitableHandle() override { return m_notifier_fd; }

private:
  int m_notifier_fd = -1;
};

class LLDBServerPluginAMDGPU : public LLDBServerPlugin {
public:
  LLDBServerPluginAMDGPU(LLDBServerPlugin::GDBServer &native_process,
                         MainLoop &main_loop);
  ~LLDBServerPluginAMDGPU() override;
  llvm::StringRef GetPluginName() override;
  int GetEventFileDescriptorAtIndex(size_t idx) override;
  bool HandleEventFileDescriptorEvent(int fd) override;
  GPUActions GetInitializeActions() override;
  std::optional<struct GPUActions> NativeProcessIsStopping() override;
  GPUPluginBreakpointHitResponse
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  NativeProcessProtocol *GetNativeProcess() {
    return m_native_process.GetCurrentProcess();
  }
  ProcessAMDGPU *GetGPUProcess() {
    return (ProcessAMDGPU *)m_gdb_server->GetCurrentProcess();
  }

  bool CreateGPUBreakpoint(uint64_t addr);

  // TODO: make this private
  struct GPUInternalBreakpoinInfo {
    uint64_t addr;
    amd_dbgapi_breakpoint_id_t breakpoind_id;
  };
  std::optional<GPUInternalBreakpoinInfo> m_gpu_internal_bp;
  bool m_wait_for_gpu_internal_bp_stop = false;
  amd_dbgapi_architecture_id_t m_architecture_id = AMD_DBGAPI_ARCHITECTURE_NONE;

private:
  std::optional<GPUPluginConnectionInfo> CreateConnection();
  void CloseFDs();
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  bool initRocm();
  bool HandleGPUInternalBreakpointHit(const GPUInternalBreakpoinInfo &bp,
                                      bool &has_new_libraries);
  amd_dbgapi_event_id_t
  process_event_queue(amd_dbgapi_event_kind_t until_event_kind);
  bool processGPUEvent();
  bool SetGPUBreakpoint(uint64_t addr, const uint8_t *bp_instruction,
                        size_t size);

  // Used with a socketpair to get events on the native ptrace event queue.
  int m_fds[2] = {-1, -1};
  Status m_main_loop_status;
  MainLoopBase::ReadHandleUP m_gpu_event_read_up;
  std::vector<MainLoopBase::ReadHandleUP> m_read_handles;
  std::unique_ptr<TCPSocket> m_listen_socket; // Keep socket alive for main_loop
  std::shared_ptr<GPUIOObject> m_gpu_event_io_obj_sp;

  amd_dbgapi_process_id_t m_gpu_pid = AMD_DBGAPI_PROCESS_NONE;
  int m_notifier_fd = -1;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINAMDGPU_H
