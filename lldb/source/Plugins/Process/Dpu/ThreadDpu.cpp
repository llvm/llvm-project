//===-- ThreadDpu.cpp --------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThreadDpu.h"

#include <signal.h>
#include <sstream>

// For llvm::to_string
#include "llvm/Support/ScopedPrinter.h"

#include "ProcessDpu.h"
#include "RegisterContextDpu.h"

#include "lldb/Core/State.h"
#include "lldb/Host/HostNativeThread.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_dpu;

ThreadDpu::ThreadDpu(ProcessDpu &process, lldb::tid_t tid, int index)
    : NativeThreadProtocol(process, tid), m_thread_index(index),
      m_state(eStateStopped),
      m_reg_context_up(llvm::make_unique<RegisterContextDpu>(*this, process)) {}

std::string ThreadDpu::GetName() {
  return "DPUthread" + llvm::to_string(m_thread_index);
}

void ThreadDpu::SetThreadStepping() {
  m_state = lldb::StateType::eStateStepping;
}

lldb::StateType ThreadDpu::GetState() { return m_state; }

bool ThreadDpu::GetStopReason(ThreadStopInfo &stop_info,
                              std::string &description) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));

  description.clear();
  stop_info.details.signal.signo = 0;
  m_state = GetProcess().GetThreadState(m_thread_index, description,
                                        stop_info.reason);

  return true;
}

Status ThreadDpu::SetWatchpoint(lldb::addr_t addr, size_t size,
                                uint32_t watch_flags, bool hardware) {
  return Status("watchpoint not implemented");
}

Status ThreadDpu::RemoveWatchpoint(lldb::addr_t addr) {
  return Status("watchpoint not implemented");
}

Status ThreadDpu::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status("no hardware breakpoint");
}

Status ThreadDpu::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status("no hardware breakpoint");
}

ProcessDpu &ThreadDpu::GetProcess() {
  return static_cast<ProcessDpu &>(m_process);
}
