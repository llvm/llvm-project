//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_THREADMOCKACCELERATOR_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_THREADMOCKACCELERATOR_H

#include "RegisterContextMockAccelerator.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include <string>

namespace lldb_private {
namespace lldb_server {

class ProcessMockAccelerator;

/// A single, always-stopped thread for the mock accelerator process.
class ThreadMockAccelerator : public NativeThreadProtocol {
public:
  ThreadMockAccelerator(ProcessMockAccelerator &process, lldb::tid_t tid);

  std::string GetName() override;
  lldb::StateType GetState() override;
  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;
  RegisterContextMockAccelerator &GetRegisterContext() override {
    return m_reg_context;
  }

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override;
  Status RemoveWatchpoint(lldb::addr_t addr) override;
  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;
  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override;

private:
  RegisterContextMockAccelerator m_reg_context;
  ThreadStopInfo m_stop_info;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_THREADMOCKACCELERATOR_H
