//===-- ThreadPlanStepOverBreakpoint.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepOverBreakpoint.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepOverBreakpoint: Single steps over a breakpoint bp_site_sp at the pc.
//----------------------------------------------------------------------

ThreadPlanStepOverBreakpoint::ThreadPlanStepOverBreakpoint (Thread &thread) :
    ThreadPlan (ThreadPlan::eKindStepOverBreakpoint, "Step over breakpoint trap",
                thread,
                eVoteNo,
                eVoteNoOpinion),  // We need to report the run since this happens
                            // first in the thread plan stack when stepping
                            // over a breakpoint
    m_breakpoint_addr (LLDB_INVALID_ADDRESS),
    m_auto_continue(false),
    m_reenabled_breakpoint_site (false)

{
    m_breakpoint_addr = m_thread.GetRegisterContext()->GetPC();
    m_breakpoint_site_id =  m_thread.GetProcess()->GetBreakpointSiteList().FindIDByAddress (m_breakpoint_addr);
}

ThreadPlanStepOverBreakpoint::~ThreadPlanStepOverBreakpoint ()
{
}

void
ThreadPlanStepOverBreakpoint::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    s->Printf("Single stepping past breakpoint site %" PRIu64 " at 0x%" PRIx64, m_breakpoint_site_id, (uint64_t)m_breakpoint_addr);
}

bool
ThreadPlanStepOverBreakpoint::ValidatePlan (Stream *error)
{
    return true;
}

bool
ThreadPlanStepOverBreakpoint::DoPlanExplainsStop (Event *event_ptr)
{
    StopInfoSP stop_info_sp = GetPrivateStopInfo ();
    if (stop_info_sp)
    {
        StopReason reason = stop_info_sp->GetStopReason();
        if (reason == eStopReasonTrace || reason == eStopReasonNone)
            return true;
        else
            return false;
    }
    return false;
}

bool
ThreadPlanStepOverBreakpoint::ShouldStop (Event *event_ptr)
{
    return false;
}

bool
ThreadPlanStepOverBreakpoint::StopOthers ()
{
    return true;
}

StateType
ThreadPlanStepOverBreakpoint::GetPlanRunState ()
{
    return eStateStepping;
}

bool
ThreadPlanStepOverBreakpoint::DoWillResume (StateType resume_state, bool current_plan)
{
    if (current_plan)
    {
        BreakpointSiteSP bp_site_sp (m_thread.GetProcess()->GetBreakpointSiteList().FindByAddress (m_breakpoint_addr));
        if (bp_site_sp  && bp_site_sp->IsEnabled())
            m_thread.GetProcess()->DisableBreakpointSite (bp_site_sp.get());
    }
    return true;
}

bool
ThreadPlanStepOverBreakpoint::WillStop ()
{
    ReenableBreakpointSite ();
    return true;
}

bool
ThreadPlanStepOverBreakpoint::MischiefManaged ()
{
    lldb::addr_t pc_addr = m_thread.GetRegisterContext()->GetPC();

    if (pc_addr == m_breakpoint_addr)
    {
        // If we are still at the PC of our breakpoint, then for some reason we didn't
        // get a chance to run.
        return false;
    }
    else
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
            log->Printf("Completed step over breakpoint plan.");
        // Otherwise, re-enable the breakpoint we were stepping over, and we're done.
        ReenableBreakpointSite ();
        ThreadPlan::MischiefManaged ();
        return true;
    }
}

void
ThreadPlanStepOverBreakpoint::ReenableBreakpointSite ()
{
    if (!m_reenabled_breakpoint_site)
    {
        m_reenabled_breakpoint_site = true;
        BreakpointSiteSP bp_site_sp (m_thread.GetProcess()->GetBreakpointSiteList().FindByAddress (m_breakpoint_addr));
        if (bp_site_sp)
        {
            m_thread.GetProcess()->EnableBreakpointSite (bp_site_sp.get());
        }
    }
}
void
ThreadPlanStepOverBreakpoint::ThreadDestroyed ()
{
    ReenableBreakpointSite ();
}

void
ThreadPlanStepOverBreakpoint::SetAutoContinue (bool do_it)
{
    m_auto_continue = do_it;
}

bool
ThreadPlanStepOverBreakpoint::ShouldAutoContinue (Event *event_ptr)
{
    return m_auto_continue;
}
