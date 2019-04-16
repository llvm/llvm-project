//===-- ThreadDpu.h ----------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadDpu_H_
#define liblldb_ThreadDpu_H_

#include "Plugins/Process/Dpu/RegisterContextDpu.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/lldb-private-forward.h"

namespace lldb_private {
namespace process_dpu {

class ProcessDpu;

class ThreadDpu : public NativeThreadProtocol {
  friend class ProcessDpu;

public:
  ThreadDpu(ProcessDpu &process, lldb::tid_t tid, int index);

  // ---------------------------------------------------------------------
  // NativeThreadProtocol Interface
  // ---------------------------------------------------------------------
  std::string GetName() override;

  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  RegisterContextDpu &GetRegisterContext() override {
    return *m_reg_context_up;
  }

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override;

  Status RemoveWatchpoint(lldb::addr_t addr) override;

  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override;

  int GetIndex() { return m_thread_index; }

private:
  // ---------------------------------------------------------------------
  // Interface for friend classes
  // ---------------------------------------------------------------------

  // ---------------------------------------------------------------------
  // Private interface
  // ---------------------------------------------------------------------
  ProcessDpu &GetProcess();

  // ---------------------------------------------------------------------
  // Member Variables
  // ---------------------------------------------------------------------
  int m_thread_index;
  std::unique_ptr<RegisterContextDpu> m_reg_context_up;
  lldb::StateType m_state;
};
} // namespace process_dpu
} // namespace lldb_private

#endif // #ifndef liblldb_ThreadDpu_H_
