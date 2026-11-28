//===-- NativeThreadOpenBSD.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVETHREADOPENBSD_H
#define LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVETHREADOPENBSD_H

#include "lldb/Host/common/NativeThreadProtocol.h"

#include <csignal>
#include <string>

namespace lldb_private {
namespace process_openbsd {

class NativeProcessOpenBSD;

class NativeThreadOpenBSD : public NativeThreadProtocol {
  friend class NativeProcessOpenBSD;

public:
  NativeThreadOpenBSD(NativeProcessOpenBSD &process, lldb::tid_t tid);

  // NativeThreadProtocol Interface
  std::string GetName() override;

  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  NativeRegisterContext &GetRegisterContext() override;

  // OpenBSD does not expose hardware debug registers to userland
  // so these functions will just return a Status error.
  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;
  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override;

  // Similarly, until software watchpoints are implemented in lldb,
  // these functions will just return a Status error.
  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override;
  Status RemoveWatchpoint(lldb::addr_t addr) override;

private:
  // Interface for friend classes

  void SetStoppedBySignal(uint32_t signo);
  void SetStoppedByBreakpoint();
  void SetStoppedByTrace();
  void SetStoppedByExec();
  void SetStoppedByFork(lldb::pid_t child_pid, lldb::tid_t child_tid);
  void SetStoppedWithNoReason();
  void SetStopped();
  void SetRunning();
  void SetStepping();
  void SetName(std::string name) { m_name = std::move(name); }

  // Member Variables
  lldb::StateType m_state;
  std::unique_ptr<NativeRegisterContext> m_reg_context_up;
  std::string m_name;
};

} // namespace process_openbsd
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVETHREADOPENBSD_H
