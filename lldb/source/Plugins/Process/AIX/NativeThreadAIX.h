//===-- NativeThreadAIX.h ----------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVETHREADAIX_H_
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVETHREADAIX_H_

#include "lldb/Host/common/NativeThreadProtocol.h"

namespace lldb_private::process_aix {

class NativeProcessAIX;

class NativeThreadAIX : public NativeThreadProtocol {
  friend class NativeProcessAIX;

public:
  NativeThreadAIX(NativeProcessAIX &process, lldb::tid_t tid);

  // NativeThreadProtocol Interface
  std::string GetName() override;

  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override;

  Status RemoveWatchpoint(lldb::addr_t addr) override;

  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override;

  NativeProcessAIX &GetProcess();

  const NativeProcessAIX &GetProcess() const;

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  GetSiginfo() const override;

private:
  lldb::StateType m_state;
};
} // namespace lldb_private::process_aix

#endif // #ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_NATIVETHREADAIX_H_
