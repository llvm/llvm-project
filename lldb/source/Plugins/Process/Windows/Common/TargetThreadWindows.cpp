//===-- TargetThreadWindows.cpp----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"
#include "lldb/Utility/State.h"

#include "Plugins/Process/Utility/UnwindLLDB.h"
#include "ProcessWindows.h"
#include "ProcessWindowsLog.h"
#include "TargetThreadWindows.h"

#if defined(__x86_64__) || defined(_M_AMD64)
#include "x64/RegisterContextWindows_x64.h"
#elif defined(__i386__) || defined(_M_IX86)
#include "x86/RegisterContextWindows_x86.h"
#elif defined(__aarch64__) || defined(_M_ARM64)
#include "arm64/RegisterContextWindows_arm64.h"
#elif defined(__arm__) || defined(_M_ARM)
#include "arm/RegisterContextWindows_arm.h"
#endif

using namespace lldb;
using namespace lldb_private;

TargetThreadWindows::TargetThreadWindows(ProcessWindows &process,
                                         const HostThread &thread)
    : Thread(process, thread.GetNativeThread().GetThreadId()),
      m_thread_reg_ctx_sp(), m_host_thread(thread) {}

TargetThreadWindows::~TargetThreadWindows() { DestroyThread(); }

void TargetThreadWindows::RefreshStateAfterStop() {
  ::SuspendThread(m_host_thread.GetNativeThread().GetSystemHandle());
  SetState(eStateStopped);
  GetRegisterContext()->InvalidateIfNeeded(false);
}

void TargetThreadWindows::WillResume(lldb::StateType resume_state) {}

void TargetThreadWindows::DidStop() {}

RegisterContextSP TargetThreadWindows::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);

  return m_reg_context_sp;
}

RegisterContextSP
TargetThreadWindows::CreateRegisterContextForFrame(StackFrame *frame) {
  RegisterContextSP reg_ctx_sp;
  uint32_t concrete_frame_idx = 0;
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));

  if (frame)
    concrete_frame_idx = frame->GetConcreteFrameIndex();

  if (concrete_frame_idx == 0) {
    if (!m_thread_reg_ctx_sp) {
      ArchSpec arch = HostInfo::GetArchitecture();
      switch (arch.GetMachine()) {
      case llvm::Triple::arm:
      case llvm::Triple::thumb:
#if defined(__arm__) || defined(_M_ARM)
        m_thread_reg_ctx_sp.reset(
            new RegisterContextWindows_arm(*this, concrete_frame_idx));
#else
        LLDB_LOG(log, "debugging foreign targets is currently unsupported");
#endif
        break;

      case llvm::Triple::aarch64:
#if defined(__aarch64__) || defined(_M_ARM64)
        m_thread_reg_ctx_sp.reset(
            new RegisterContextWindows_arm64(*this, concrete_frame_idx));
#else
        LLDB_LOG(log, "debugging foreign targets is currently unsupported");
#endif
        break;

      case llvm::Triple::x86:
#if defined(__i386__) || defined(_M_IX86)
        m_thread_reg_ctx_sp.reset(
            new RegisterContextWindows_x86(*this, concrete_frame_idx));
#else
        LLDB_LOG(log, "debugging foreign targets is currently unsupported");
#endif
        break;

      case llvm::Triple::x86_64:
#if defined(__x86_64__) || defined(_M_AMD64)
        m_thread_reg_ctx_sp.reset(
            new RegisterContextWindows_x64(*this, concrete_frame_idx));
#else
        LLDB_LOG(log, "debugging foreign targets is currently unsupported");
#endif
        break;

      default:
        break;
      }
    }
    reg_ctx_sp = m_thread_reg_ctx_sp;
  } else {
    Unwind *unwinder = GetUnwinder();
    if (unwinder != nullptr)
      reg_ctx_sp = unwinder->CreateRegisterContextForFrame(frame);
  }

  return reg_ctx_sp;
}

bool TargetThreadWindows::CalculateStopInfo() {
  SetStopInfo(m_stop_info_sp);
  return true;
}

Unwind *TargetThreadWindows::GetUnwinder() {
  // FIXME: Implement an unwinder based on the Windows unwinder exposed through
  // DIA SDK.
  if (!m_unwinder_up)
    m_unwinder_up.reset(new UnwindLLDB(*this));
  return m_unwinder_up.get();
}

Status TargetThreadWindows::DoResume() {
  StateType resume_state = GetTemporaryResumeState();
  StateType current_state = GetState();
  if (resume_state == current_state)
    return Status();

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
      // ResumeThread returns -1 on error, or the thread's *previous* suspend
      // count on success. This means that the return value is 1 when the thread
      // was restarted. Note that DWORD is an unsigned int, so we need to
      // explicitly compare with -1.
      previous_suspend_count = ::ResumeThread(thread_handle);

      if (previous_suspend_count == (DWORD)-1)
        return Status(::GetLastError(), eErrorTypeWin32);

    } while (previous_suspend_count > 1);
  }

  return Status();
}
