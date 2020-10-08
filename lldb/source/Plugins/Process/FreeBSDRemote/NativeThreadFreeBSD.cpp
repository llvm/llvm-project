//===-- NativeThreadFreeBSD.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeThreadFreeBSD.h"
#include "NativeRegisterContextFreeBSD.h"

#include "NativeProcessFreeBSD.h"

#include "Plugins/Process/POSIX/CrashReason.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/State.h"
#include "llvm/Support/Errno.h"

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/sysctl.h>
// clang-format on

#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_freebsd;

NativeThreadFreeBSD::NativeThreadFreeBSD(NativeProcessFreeBSD &process,
                                         lldb::tid_t tid)
    : NativeThreadProtocol(process, tid), m_state(StateType::eStateInvalid),
      m_stop_info(),
      m_reg_context_up(
          NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
              process.GetArchitecture(), *this)),
      m_stop_description() {}

Status NativeThreadFreeBSD::Resume() {
  Status ret = NativeProcessFreeBSD::PtraceWrapper(PT_RESUME, m_process.GetID(),
                                                   nullptr, GetID());
  if (!ret.Success())
    return ret;
  ret = NativeProcessFreeBSD::PtraceWrapper(PT_CLEARSTEP, m_process.GetID(),
                                            nullptr, GetID());
  if (ret.Success())
    SetRunning();
  return ret;
}

Status NativeThreadFreeBSD::SingleStep() {
  Status ret = NativeProcessFreeBSD::PtraceWrapper(PT_RESUME, m_process.GetID(),
                                                   nullptr, GetID());
  if (!ret.Success())
    return ret;
  ret = NativeProcessFreeBSD::PtraceWrapper(PT_SETSTEP, m_process.GetID(),
                                            nullptr, GetID());
  if (ret.Success())
    SetStepping();
  return ret;
}

Status NativeThreadFreeBSD::Suspend() {
  Status ret = NativeProcessFreeBSD::PtraceWrapper(
      PT_SUSPEND, m_process.GetID(), nullptr, GetID());
  if (ret.Success())
    SetStopped();
  return ret;
}

void NativeThreadFreeBSD::SetStoppedBySignal(uint32_t signo,
                                             const siginfo_t *info) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD));
  LLDB_LOG(log, "tid = {0} in called with signal {1}", GetID(), signo);

  SetStopped();

  m_stop_info.reason = StopReason::eStopReasonSignal;
  m_stop_info.details.signal.signo = signo;

  m_stop_description.clear();
  if (info) {
    switch (signo) {
    case SIGSEGV:
    case SIGBUS:
    case SIGFPE:
    case SIGILL:
      const auto reason = GetCrashReason(*info);
      m_stop_description = GetCrashReasonString(reason, *info);
      break;
    }
  }
}

void NativeThreadFreeBSD::SetStoppedByBreakpoint() {
  SetStopped();
  m_stop_info.reason = StopReason::eStopReasonBreakpoint;
  m_stop_info.details.signal.signo = SIGTRAP;
}

void NativeThreadFreeBSD::SetStoppedByTrace() {
  SetStopped();
  m_stop_info.reason = StopReason::eStopReasonTrace;
  m_stop_info.details.signal.signo = SIGTRAP;
}

void NativeThreadFreeBSD::SetStoppedByExec() {
  SetStopped();
  m_stop_info.reason = StopReason::eStopReasonExec;
  m_stop_info.details.signal.signo = SIGTRAP;
}

void NativeThreadFreeBSD::SetStoppedByWatchpoint(uint32_t wp_index) {
  SetStopped();
}

void NativeThreadFreeBSD::SetStoppedWithNoReason() {
  SetStopped();

  m_stop_info.reason = StopReason::eStopReasonNone;
  m_stop_info.details.signal.signo = 0;
}

void NativeThreadFreeBSD::SetStopped() {
  const StateType new_state = StateType::eStateStopped;
  m_state = new_state;
  m_stop_description.clear();
}

void NativeThreadFreeBSD::SetRunning() {
  m_state = StateType::eStateRunning;
  m_stop_info.reason = StopReason::eStopReasonNone;
}

void NativeThreadFreeBSD::SetStepping() {
  m_state = StateType::eStateStepping;
  m_stop_info.reason = StopReason::eStopReasonNone;
}

std::string NativeThreadFreeBSD::GetName() { return ""; }

lldb::StateType NativeThreadFreeBSD::GetState() { return m_state; }

bool NativeThreadFreeBSD::GetStopReason(ThreadStopInfo &stop_info,
                                        std::string &description) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD));
  description.clear();

  switch (m_state) {
  case eStateStopped:
  case eStateCrashed:
  case eStateExited:
  case eStateSuspended:
  case eStateUnloaded:
    stop_info = m_stop_info;
    description = m_stop_description;

    return true;

  case eStateInvalid:
  case eStateConnected:
  case eStateAttaching:
  case eStateLaunching:
  case eStateRunning:
  case eStateStepping:
  case eStateDetached:
    LLDB_LOG(log, "tid = {0} in state {1} cannot answer stop reason", GetID(),
             StateAsCString(m_state));
    return false;
  }
  llvm_unreachable("unhandled StateType!");
}

NativeRegisterContextFreeBSD &NativeThreadFreeBSD::GetRegisterContext() {
  assert(m_reg_context_up);
  return *m_reg_context_up;
}

Status NativeThreadFreeBSD::SetWatchpoint(lldb::addr_t addr, size_t size,
                                          uint32_t watch_flags, bool hardware) {
  return Status("not implemented");
}

Status NativeThreadFreeBSD::RemoveWatchpoint(lldb::addr_t addr) {
  auto wp = m_watchpoint_index_map.find(addr);
  if (wp == m_watchpoint_index_map.end())
    return Status();
  return Status("not implemented");
}

Status NativeThreadFreeBSD::SetHardwareBreakpoint(lldb::addr_t addr,
                                                  size_t size) {
  if (m_state == eStateLaunching)
    return Status();

  Status error = RemoveHardwareBreakpoint(addr);
  if (error.Fail())
    return error;

  return Status("not implemented");
}

Status NativeThreadFreeBSD::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  auto bp = m_hw_break_index_map.find(addr);
  if (bp == m_hw_break_index_map.end())
    return Status();

  return Status("not implemented");
}

Status NativeThreadFreeBSD::CopyWatchpointsFrom(NativeThreadFreeBSD &source) {
  return Status("not implemented");
}
