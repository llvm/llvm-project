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
      m_state(StateType::eStateStopped),
      m_reg_context_up(llvm::make_unique<RegisterContextDpu>(*this, process)) {}

std::string ThreadDpu::GetName() {
  // ProcessDpu &process = GetProcess();
  return "TODO DPUthreadNN";
}

lldb::StateType ThreadDpu::GetState() { return m_state; }

bool ThreadDpu::GetStopReason(ThreadStopInfo &stop_info,
                              std::string &description) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));

  description.clear();

  switch (m_state) {
  case eStateStopped:
  case eStateCrashed:
  case eStateExited:
  case eStateSuspended:
  case eStateUnloaded:
    // TODO get info from Process (polling status + rank context)
    // cached version will be in m_state ... ???
    stop_info.reason = eStopReasonNone;
    // eStopReasonBreakpoint, eStopReasonException, eStopReasonThreadExiting

    return true;

  case eStateInvalid:
  case eStateConnected:
  case eStateAttaching:
  case eStateLaunching:
  case eStateRunning:
  case eStateStepping:
  case eStateDetached:
    if (log) {
      log->Printf("ThreadDpu::%s tid %" PRIu64
                  " in state %s cannot answer stop reason",
                  __FUNCTION__, GetID(), StateAsCString(m_state));
    }
    return false;
  }
  llvm_unreachable("unhandled StateType!");
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
