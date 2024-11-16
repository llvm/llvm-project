//===-- NativeThreadQNX.h ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeThreadQNX_H_
#define liblldb_NativeThreadQNX_H_

#include "lldb/Host/common/NativeThreadProtocol.h"

#include "Plugins/Process/QNX/NativeRegisterContextQNX.h"

namespace lldb_private {
namespace process_qnx {

class NativeProcessQNX;

class NativeThreadQNX : public NativeThreadProtocol {
  friend class NativeProcessQNX;

public:
  NativeThreadQNX(NativeProcessQNX &process, lldb::tid_t tid);

  // NativeThreadProtocol Interface
  std::string GetName() override;

  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  NativeRegisterContextQNX &GetRegisterContext() override;

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override;

  Status RemoveWatchpoint(lldb::addr_t addr) override;

  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override;

private:
  // Interface for friend classes

  Status Resume();

  Status SingleStep();

  void SetStoppedBySignal(uint32_t signo, const siginfo_t *info = nullptr);

  void SetStoppedByBreakpoint();

  void SetStoppedByTrace();

  void SetStoppedWithNoReason();

  void SetStopped();

  // Member Variables
  lldb::StateType m_state;
  ThreadStopInfo m_stop_info;
  std::unique_ptr<NativeRegisterContextQNX> m_reg_context_up;
  std::string m_stop_description;
};

} // namespace process_qnx
} // namespace lldb_private

#endif // #ifndef liblldb_NativeThreadQNX_H_
