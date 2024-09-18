//===-- NativeThreadQNX.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeThreadQNX.h"
#include "NativeRegisterContextQNX.h"

#include "NativeProcessQNX.h"

#include "Plugins/Process/POSIX/CrashReason.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Utility/State.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_qnx;

NativeThreadQNX::NativeThreadQNX(NativeProcessQNX &process, lldb::tid_t tid)
    : NativeThreadProtocol(process, tid), m_state(StateType::eStateInvalid),
      m_stop_info(),
      m_reg_context_up(
          NativeRegisterContextQNX::CreateHostNativeRegisterContextQNX(
              process.GetArchitecture(), *this)),
      m_stop_description() {}

Status NativeThreadQNX::Resume() {
  m_state = StateType::eStateRunning;
  m_stop_info.reason = StopReason::eStopReasonNone;
  return Status();
}

Status NativeThreadQNX::SingleStep() {
  m_state = StateType::eStateStepping;
  m_stop_info.reason = StopReason::eStopReasonNone;
  return Status();
}

void NativeThreadQNX::SetStoppedBySignal(uint32_t signo,
                                         const siginfo_t *info) {
  Log *log = GetLog(POSIXLog::Thread);
  LLDB_LOG(log, "tid = {0} in called with signal {1}", GetID(), signo);

  SetStopped();

  m_stop_info.reason = StopReason::eStopReasonSignal;
  m_stop_info.signo = signo;

  m_stop_description.clear();
  if (info) {
    switch (signo) {
    case SIGSEGV:
    case SIGBUS:
    case SIGFPE:
    case SIGILL:
      m_stop_description = GetCrashReasonString(*info);
      break;
    }
  }
}

void NativeThreadQNX::SetStoppedByBreakpoint() {
  SetStopped();
  m_stop_info.reason = StopReason::eStopReasonBreakpoint;
  m_stop_info.signo = SIGTRAP;
}

void NativeThreadQNX::SetStoppedByTrace() {
  SetStopped();
  m_stop_info.reason = StopReason::eStopReasonTrace;
  m_stop_info.signo = SIGTRAP;
}

void NativeThreadQNX::SetStoppedWithNoReason() {
  SetStopped();
  m_stop_info.reason = StopReason::eStopReasonNone;
  m_stop_info.signo = 0;
}

void NativeThreadQNX::SetStopped() {
  const StateType new_state = StateType::eStateStopped;
  m_state = new_state;
  m_stop_description.clear();
}

std::string NativeThreadQNX::GetName() { return ""; }

lldb::StateType NativeThreadQNX::GetState() { return m_state; }

bool NativeThreadQNX::GetStopReason(ThreadStopInfo &stop_info,
                                    std::string &description) {
  Log *log = GetLog(POSIXLog::Thread);
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

NativeRegisterContextQNX &NativeThreadQNX::GetRegisterContext() {
  return *m_reg_context_up;
}

Status NativeThreadQNX::SetWatchpoint(lldb::addr_t addr, size_t size,
                                      uint32_t watch_flags, bool hardware) {
  return Status("not implemented");
}

Status NativeThreadQNX::RemoveWatchpoint(lldb::addr_t addr) {
  return Status("not implemented");
}

Status NativeThreadQNX::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status("not implemented");
}

Status NativeThreadQNX::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status("not implemented");
}
