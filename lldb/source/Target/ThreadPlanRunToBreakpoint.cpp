//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanRunToBreakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

ThreadPlanRunToBreakpoint::ThreadPlanRunToBreakpoint(Thread &thread,
                                                     BreakpointSP bkpt_sp,
                                                     bool stop_others)
    : ThreadPlan(ThreadPlan::eKindRunToBreakpoint, "Run to breakpoint plan",
                 thread, eVoteNoOpinion, eVoteNoOpinion),
      m_stop_others(stop_others) {
  if (bkpt_sp)
    m_breakpoints.push_back(std::move(bkpt_sp));
}

ThreadPlanRunToBreakpoint::ThreadPlanRunToBreakpoint(
    Thread &thread, const std::vector<BreakpointSP> &breakpoints,
    bool stop_others)
    : ThreadPlan(ThreadPlan::eKindRunToBreakpoint, "Run to breakpoint plan",
                 thread, eVoteNoOpinion, eVoteNoOpinion),
      m_stop_others(stop_others) {
  for (auto bkpt_sp : breakpoints) {
    // Make the breakpoints thread specific:
    if (bkpt_sp) {
      bkpt_sp->SetThreadID(thread.GetID());
      m_breakpoints.push_back(bkpt_sp);
    }
  }
}

ThreadPlanRunToBreakpoint::~ThreadPlanRunToBreakpoint() {
  for (auto bkpt_sp : m_breakpoints)
    GetTarget().RemoveBreakpointByID(bkpt_sp->GetID());
}

void ThreadPlanRunToBreakpoint::GetDescription(Stream *s,
                                               lldb::DescriptionLevel level) {
  size_t num_bkpts = m_breakpoints.size();
  if (level == lldb::eDescriptionLevelBrief) {
    if (num_bkpts == 0) {
      s->PutCString("run to breakpoint with no breakpoints given.");
      return;
    } else if (num_bkpts == 1)
      s->PutCString("run to breakpoint: ");
    else
      s->PutCString("run to breakpoints: ");

    for (auto bkpt_sp : m_breakpoints) {
      *s << llvm::formatv("{0} ", bkpt_sp->GetID());
    }
  } else {
    if (num_bkpts == 0) {
      s->PutCString("run to breakpoint with no breakpoints given.");
      return;
    } else if (num_bkpts == 1)
      s->PutCString("Run to breakpoint: ");
    else {
      s->PutCString("Run to breakpoints: ");
    }

    // FIXME: I'm not preserving breakpoint ID's
    // in case I'm asked GetDescription after I've removed them.
    for (auto bkpt_sp : m_breakpoints) {
      s->PutCString("\n");
      s->Indent();
      bkpt_sp->Dump(s);
    }
  }
}

bool ThreadPlanRunToBreakpoint::ValidatePlan(Stream *error) {
  // We don't require that the breakpoints we were given actually
  // resolve to something when the plan was pushed.  The only error
  // was not to have passed any breakpoints.
  return m_breakpoints.size() != 0;
}

bool ThreadPlanRunToBreakpoint::DoPlanExplainsStop(Event *event_ptr) {
  return AtOurBreakpoint();
}

bool ThreadPlanRunToBreakpoint::ShouldStop(Event *event_ptr) {
  return AtOurBreakpoint();
}

bool ThreadPlanRunToBreakpoint::StopOthers() { return m_stop_others; }

void ThreadPlanRunToBreakpoint::SetStopOthers(bool new_value) {
  m_stop_others = new_value;
}

StateType ThreadPlanRunToBreakpoint::GetPlanRunState() { return eStateRunning; }

bool ThreadPlanRunToBreakpoint::WillStop() { return true; }

bool ThreadPlanRunToBreakpoint::MischiefManaged() {
  Log *log = GetLog(LLDBLog::Step);

  if (AtOurBreakpoint()) {
    // Remove the breakpoints
    for (auto bkpt_sp : m_breakpoints) {
      GetTarget().RemoveBreakpointByID(bkpt_sp->GetID());
    }
    m_breakpoints.clear();
    LLDB_LOGF(log, "Completed run to breakpoint plan.");
    ThreadPlan::MischiefManaged();
    return true;
  }
  return false;
}

bool ThreadPlanRunToBreakpoint::AtOurBreakpoint() {
  lldb::addr_t current_addr = GetThread().GetRegisterContext()->GetPC();
  Address current_address;
  current_address.SetLoadAddress(current_addr, &GetTarget());
  bool found_it = false;
  for (auto bkpt_sp : m_breakpoints) {
    if (auto location_sp = bkpt_sp->FindLocationByAddress(current_address)) {
      StreamString s;
      location_sp->GetDescription(&s, eDescriptionLevelBrief);
      LLDB_LOG(GetLog(LLDBLog::Step), "RunToBreakpoint hit breakpoint: {0}", s);
      found_it = true;
      break;
    }
  }
  return found_it;
}
