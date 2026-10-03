//===-- ThreadPlanStepUntil.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepUntil.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

// ThreadPlanStepUntil: Run until we reach a given line number or step out of
// the current frame

ThreadPlanStepUntil::ThreadPlanStepUntil(Thread &thread,
                                         llvm::ArrayRef<addr_t> address_list,
                                         bool stop_others, uint32_t frame_idx)
    : ThreadPlanStepOut(ThreadPlan::eKindStepUntil, "Step until", thread,
                        /*addr_context=*/nullptr, /*first_insn=*/false,
                        stop_others, eVoteNoOpinion, eVoteNoOpinion, frame_idx,
                        /*step_out_avoids_code_without_debug_info=*/eLazyBoolNo,
                        /*continue_to_next_branch=*/false,
                        /*gather_return_value=*/false),
      m_step_from_insn(LLDB_INVALID_ADDRESS) {
  ClearShouldStopHereCallbacks();

  StackFrameSP frame_sp(thread.GetStackFrameAtIndex(frame_idx));
  if (!frame_sp)
    return;

  m_step_from_insn = frame_sp->GetStackID().GetPC();
  m_stack_id = frame_sp->GetStackID();

  Target &target = GetTarget();
  for (addr_t address : address_list) {
    Breakpoint *until_bp = target.CreateBreakpoint(address, true, false).get();
    if (until_bp != nullptr) {
      until_bp->SetThreadID(m_tid);
      m_until_points[address] = until_bp->GetID();
      until_bp->SetBreakpointKind("until-target");
    } else {
      m_until_points[address] = LLDB_INVALID_BREAK_ID;
    }
  }
}

ThreadPlanStepUntil::~ThreadPlanStepUntil() { Clear(); }

void ThreadPlanStepUntil::Clear() {
  Target &target = GetTarget();
  for (const auto &[address, break_id] : m_until_points)
    target.RemoveBreakpointByID(break_id);
  m_until_points.clear();
}

void ThreadPlanStepUntil::GetDescription(Stream *s,
                                         lldb::DescriptionLevel level) {
  if (level == lldb::eDescriptionLevelBrief) {
    s->PutCString("step until");
    if (IsPlanComplete() && !m_reached_until_point)
      s->PutCString(" - stepped out");
    return;
  }

  if (m_until_points.size() == 1) {
    s->Printf("Stepping from address 0x%" PRIx64 " until we reach 0x%" PRIx64
              " using breakpoint %d",
              (uint64_t)m_step_from_insn,
              (uint64_t)(*m_until_points.begin()).first,
              (*m_until_points.begin()).second);
  } else {
    s->Printf("Stepping from address 0x%" PRIx64 " until we reach one of:",
              (uint64_t)m_step_from_insn);
    for (const auto &[address, break_id] : m_until_points)
      s->Printf("\n\t0x%" PRIx64 " (bp: %d)", (uint64_t)address, break_id);
  }
  s->PutCString("\n");
  ThreadPlanStepOut::GetDescription(s, level);
}

bool ThreadPlanStepUntil::ValidatePlan(Stream *error) {
  for (const auto &[address, break_id] : m_until_points)
    if (!LLDB_BREAK_ID_IS_VALID(break_id))
      return false;
  return ThreadPlanStepOut::ValidatePlan(error);
}

BreakpointSiteSP ThreadPlanStepUntil::GetUntilPointSite() {
  StopInfoSP stop_info_sp = GetPrivateStopInfo();
  if (!stop_info_sp || stop_info_sp->GetStopReason() != eStopReasonBreakpoint)
    return nullptr;

  BreakpointSiteSP site_sp =
      m_process.GetBreakpointSiteList().FindByID(stop_info_sp->GetValue());
  if (!site_sp)
    return nullptr;

  for (const auto &[address, break_id] : m_until_points)
    if (site_sp->IsBreakpointAtThisSite(break_id))
      return site_sp;
  return nullptr;
}

void ThreadPlanStepUntil::CompleteIfInUntilFrame() {
  StackID frame_zero_id = GetThread().GetStackFrameAtIndex(0)->GetStackID();
  // Hits in younger frames, e.g. recursive calls, do not count.
  if (frame_zero_id.IsYoungerThan(m_stack_id))
    return;
  m_reached_until_point = true;
  SetPlanComplete();
}

bool ThreadPlanStepUntil::DoPlanExplainsStop(Event *event_ptr) {
  BreakpointSiteSP site_sp = GetUntilPointSite();
  if (!site_sp)
    return ThreadPlanStepOut::DoPlanExplainsStop(event_ptr);

  CompleteIfInUntilFrame();
  // A user breakpoint at the same site takes precedence in the stop report.
  return !site_sp->ContainsUserBreakpointForThread(GetThread());
}

bool ThreadPlanStepUntil::ShouldStop(Event *event_ptr) {
  if (GetUntilPointSite()) {
    CompleteIfInUntilFrame();
    return IsPlanComplete();
  }
  return ThreadPlanStepOut::ShouldStop(event_ptr);
}

void ThreadPlanStepUntil::SetUntilPointsEnabled(bool enabled) {
  Target &target = GetTarget();
  for (const auto &[address, break_id] : m_until_points)
    if (BreakpointSP until_bp_sp = target.GetBreakpointByID(break_id))
      until_bp_sp->SetEnabled(enabled);
}

bool ThreadPlanStepUntil::DoWillResume(StateType resume_state,
                                       bool current_plan) {
  SetUntilPointsEnabled(true);
  return ThreadPlanStepOut::DoWillResume(resume_state, current_plan);
}

bool ThreadPlanStepUntil::WillStop() {
  SetUntilPointsEnabled(false);
  return ThreadPlanStepOut::WillStop();
}

bool ThreadPlanStepUntil::IsPlanStale() {
  return IsPlanComplete() || ThreadPlanStepOut::IsPlanStale();
}

bool ThreadPlanStepUntil::MischiefManaged() {
  // StepOut::MischiefManaged returns true if and only the plan is completed, so
  // it also checks StepUntil's status.
  if (!ThreadPlanStepOut::MischiefManaged())
    return false;

  LLDB_LOGF(GetLog(LLDBLog::Step), "Completed step until plan.");
  Clear();
  return true;
}
