//===-- ThreadPlanStepThroughGenericTrampoline.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepThroughGenericTrampoline.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

uint32_t ThreadPlanStepThroughGenericTrampoline::s_default_flag_values =
    ThreadPlanShouldStopHere::eStepInAvoidNoDebug;

ThreadPlanStepThroughGenericTrampoline::ThreadPlanStepThroughGenericTrampoline(
    Thread &thread, lldb::RunMode stop_others)
    : ThreadPlanStepRange(ThreadPlan::eKindStepThroughGenericTrampoline,
                          "Step through generic trampoline", thread, {}, {},
                          stop_others),
      ThreadPlanShouldStopHere(this) {

  SetFlagsToDefault();
  auto frame = GetThread().GetFrameWithStackID(m_stack_id);
  if (!frame)
    return;
  SymbolContext sc = frame->GetSymbolContext(eSymbolContextFunction);

  if (!sc.function)
    return;
  AddRange(sc.function->GetAddressRange());
}

ThreadPlanStepThroughGenericTrampoline::
    ~ThreadPlanStepThroughGenericTrampoline() = default;

void ThreadPlanStepThroughGenericTrampoline::GetDescription(
    Stream *s, lldb::DescriptionLevel level) {

  auto PrintFailureIfAny = [&]() {
    if (m_status.Success())
      return;
    s->Printf(" failed (%s)", m_status.AsCString());
  };

  if (level == lldb::eDescriptionLevelBrief) {
    s->Printf("step through generic trampoline");
    PrintFailureIfAny();
    return;
  }

  auto frame = GetThread().GetFrameWithStackID(m_stack_id);
  if (!frame) {
    s->Printf("<error, frame not available>");
    return;
  }

  SymbolContext sc = frame->GetSymbolContext(eSymbolContextFunction);
  if (!sc.function) {
    s->Printf("<error, function not available>");
    return;
  }

  s->Printf("Stepping through generic trampoline %s",
            sc.function->GetName().AsCString());

  lldb::StackFrameSP curr_frame = GetThread().GetStackFrameAtIndex(0);
  if (!curr_frame)
    return;

  SymbolContext curr_frame_sc =
      curr_frame->GetSymbolContext(eSymbolContextFunction);
  if (!curr_frame_sc.function)
    return;
  s->Printf(", current function: %s",
            curr_frame_sc.function->GetName().GetCString());

  PrintFailureIfAny();

  s->PutChar('.');
}

bool ThreadPlanStepThroughGenericTrampoline::ShouldStop(Event *event_ptr) {
  Log *log = GetLog(LLDBLog::Step);

  if (log) {
    StreamString s;
    DumpAddress(s.AsRawOstream(), GetThread().GetRegisterContext()->GetPC(),
                GetTarget().GetArchitecture().GetAddressByteSize());
    LLDB_LOGF(log, "ThreadPlanStepThroughGenericTrampoline reached %s.",
              s.GetData());
  }

  if (IsPlanComplete())
    return true;

  m_no_more_plans = false;

  Thread &thread = GetThread();
  lldb::StackFrameSP curr_frame = thread.GetStackFrameAtIndex(0);
  if (!curr_frame)
    return false;

  SymbolContext sc = curr_frame->GetSymbolContext(eSymbolContextFunction);

  if (sc.function && sc.function->IsGenericTrampoline() &&
      SetNextBranchBreakpoint()) {
    // While whatever frame we're in is a generic trampoline,
    // continue stepping to the next branch, until we
    // end up in a function which isn't a trampoline.
    return false;
  }

  m_no_more_plans = true;
  SetPlanComplete();
  return true;
}

bool ThreadPlanStepThroughGenericTrampoline::ValidatePlan(Stream *error) {
  // If trampoline support is disabled, there's nothing for us to do.
  if (!Target::GetGlobalProperties().GetEnableTrampolineSupport())
    return false;

  auto frame = GetThread().GetFrameWithStackID(m_stack_id);
  if (!frame)
    return false;

  SymbolContext sc = frame->GetSymbolContext(eSymbolContextFunction);
  return sc.function && sc.function->IsGenericTrampoline();
}
