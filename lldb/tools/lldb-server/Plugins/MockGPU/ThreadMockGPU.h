//===-- ThreadMockGPU.h --------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_THREADMOCKGPU_H
#define LLDB_TOOLS_LLDB_SERVER_THREADMOCKGPU_H

#include "RegisterContextMockGPU.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/lldb-private-forward.h"
#include <string>

namespace lldb_private {
namespace lldb_server {
class ProcessMockGPU;

class NativeProcessLinux;

class ThreadMockGPU : public NativeThreadProtocol {
  friend class ProcessMockGPU;

public:
  ThreadMockGPU(ProcessMockGPU &process, lldb::tid_t tid);

  // NativeThreadProtocol Interface
  std::string GetName() override;

  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  RegisterContextMockGPU &GetRegisterContext() override {
    return m_reg_context;
  }

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override;

  Status RemoveWatchpoint(lldb::addr_t addr) override;

  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override;

  ProcessMockGPU &GetProcess();

  const ProcessMockGPU &GetProcess() const;

private:
  // Member Variables
  lldb::StateType m_state;
  ThreadStopInfo m_stop_info;
  RegisterContextMockGPU m_reg_context;
  std::string m_stop_description;
};
} // namespace lldb_server
} // namespace lldb_private

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_THREADMOCKGPU_H
