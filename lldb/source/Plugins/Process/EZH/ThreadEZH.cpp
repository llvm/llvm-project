//===-- ThreadEZH.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadEZH.h"
#include "RegisterContextEZH.h"

#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Core/Debugger.h"

using namespace lldb;
using namespace lldb_private;

ThreadEZH::ThreadEZH(Process &process, tid_t tid)
    : ThreadGDBRemote(process, tid) {}

RegisterContextSP ThreadEZH::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);
  return m_reg_context_sp;
}

RegisterContextSP ThreadEZH::CreateRegisterContextForFrame(StackFrame *frame) {
  uint32_t concrete_frame_idx = 0;
  if (frame)
    concrete_frame_idx = frame->GetConcreteFrameIndex();
  return std::make_shared<RegisterContextEZH>(*this, concrete_frame_idx);
}

bool ThreadEZH::CalculateStopInfo() {

  if (m_stop_info_sp)
    return true;

  // Set EZH's stop reason cleanly to Signal 2 (SIGINT) representing a debugger interrupt halt!
  m_stop_info_sp = lldb_private::StopInfo::CreateStopReasonWithSignal(*this, 2);
  SetStopInfo(m_stop_info_sp);
  return true;
}

void ThreadEZH::RefreshStateAfterStop() {
  // 1. Call base class to invalidate register context
  ThreadGDBRemote::RefreshStateAfterStop();
  if (m_reg_context_sp)
    m_reg_context_sp->InvalidateAllRegisters();
  // 2. Forcefully clear cached stack frames so UnwindLLDB re-fetches them using fresh JTAG memory reads!
  ClearStackFrames();
}
