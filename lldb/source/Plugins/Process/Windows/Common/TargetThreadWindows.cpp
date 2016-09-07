//===-- TargetThreadWindows.cpp----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Log.h"
#include "lldb/Core/Logging.h"
#include "lldb/Core/State.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Target/RegisterContext.h"

#include "ProcessWindows.h"
#include "ProcessWindowsLog.h"
#include "TargetThreadWindows.h"
#include "UnwindLLDB.h"

using namespace lldb;
using namespace lldb_private;

TargetThreadWindows::TargetThreadWindows(ProcessWindows &process,
                                         const HostThread &thread)
    : Thread(process, thread.GetNativeThread().GetThreadId()),
      m_host_thread(thread) {}

TargetThreadWindows::~TargetThreadWindows() { DestroyThread(); }

void TargetThreadWindows::RefreshStateAfterStop() {
  ::SuspendThread(m_host_thread.GetNativeThread().GetSystemHandle());
  SetState(eStateStopped);
  GetRegisterContext()->InvalidateIfNeeded(false);
}

void TargetThreadWindows::WillResume(lldb::StateType resume_state) {}

void TargetThreadWindows::DidStop() {}

bool TargetThreadWindows::CalculateStopInfo() {
  SetStopInfo(m_stop_info_sp);
  return true;
}

Unwind *TargetThreadWindows::GetUnwinder() {
  // FIXME: Implement an unwinder based on the Windows unwinder exposed through
  // DIA SDK.
  if (m_unwinder_ap.get() == NULL)
    m_unwinder_ap.reset(new UnwindLLDB(*this));
  return m_unwinder_ap.get();
}

bool TargetThreadWindows::DoResume() {
  StateType resume_state = GetTemporaryResumeState();
  StateType current_state = GetState();
  if (resume_state == current_state)
    return true;

  if (resume_state == eStateStepping) {
    uint32_t flags_index =
        GetRegisterContext()->ConvertRegisterKindToRegisterNumber(
            eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);
    uint64_t flags_value =
        GetRegisterContext()->ReadRegisterAsUnsigned(flags_index, 0);
    flags_value |= 0x100; // Set the trap flag on the CPU
    GetRegisterContext()->WriteRegisterFromUnsigned(flags_index, flags_value);
  }

  if (resume_state == eStateStepping || resume_state == eStateRunning) {
    DWORD previous_suspend_count = 0;
    HANDLE thread_handle = m_host_thread.GetNativeThread().GetSystemHandle();
    do {
      previous_suspend_count = ::ResumeThread(thread_handle);
    } while (previous_suspend_count > 0);
  }
  return true;
}
