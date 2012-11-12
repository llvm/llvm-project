//===-- ThreadPlanStepInstruction.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Target/ThreadPlanStepInstruction.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepInstruction: Step over the current instruction
//----------------------------------------------------------------------

ThreadPlanStepInstruction::ThreadPlanStepInstruction
(
    Thread &thread,
    bool step_over,
    bool stop_other_threads,
    Vote stop_vote,
    Vote run_vote
) :
    ThreadPlan (ThreadPlan::eKindStepInstruction, "Step over single instruction", thread, stop_vote, run_vote),
    m_instruction_addr (0),
    m_stop_other_threads (stop_other_threads),
    m_step_over (step_over)
{
    m_instruction_addr = m_thread.GetRegisterContext()->GetPC(0);
    m_stack_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
}

ThreadPlanStepInstruction::~ThreadPlanStepInstruction ()
{
}

void
ThreadPlanStepInstruction::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
    {
        if (m_step_over)
            s->Printf ("instruction step over");
        else
            s->Printf ("instruction step into");
    }
    else
    {
        s->Printf ("Stepping one instruction past ");
        s->Address(m_instruction_addr, sizeof (addr_t));
        if (m_step_over)
            s->Printf(" stepping over calls");
        else
            s->Printf(" stepping into calls");
    }
}

bool
ThreadPlanStepInstruction::ValidatePlan (Stream *error)
{
    // Since we read the instruction we're stepping over from the thread,
    // this plan will always work.
    return true;
}

bool
ThreadPlanStepInstruction::PlanExplainsStop ()
{
    StopInfoSP stop_info_sp = GetPrivateStopReason();
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
ThreadPlanStepInstruction::ShouldStop (Event *event_ptr)
{
    if (m_step_over)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        
        StackID cur_frame_zero_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
        
        if (cur_frame_zero_id == m_stack_id || m_stack_id < cur_frame_zero_id)
        {
            if (m_thread.GetRegisterContext()->GetPC(0) != m_instruction_addr)
            {
                SetPlanComplete();
                return true;
            }
            else
                return false;
        }
        else
        {
            // We've stepped in, step back out again:
            StackFrame *return_frame = m_thread.GetStackFrameAtIndex(1).get();
            if (return_frame)
            {
                if (log)
                {
                    StreamString s;
                    s.PutCString ("Stepped in to: ");
                    addr_t stop_addr = m_thread.GetStackFrameAtIndex(0)->GetRegisterContext()->GetPC();
                    s.Address (stop_addr, m_thread.CalculateTarget()->GetArchitecture().GetAddressByteSize());
                    s.PutCString (" stepping out to: ");
                    addr_t return_addr = return_frame->GetRegisterContext()->GetPC();
                    s.Address (return_addr, m_thread.CalculateTarget()->GetArchitecture().GetAddressByteSize());
                    log->Printf("%s.", s.GetData());
                }
                
                // StepInstruction should probably have the tri-state RunMode, but for now it is safer to
                // run others.
                const bool stop_others = false;
                m_thread.QueueThreadPlanForStepOut(false,
                                                   NULL,
                                                   true,
                                                   stop_others,
                                                   eVoteNo,
                                                   eVoteNoOpinion,
                                                   0);
                return false;
            }
            else
            {
                if (log)
                    log->Printf("Could not find previous frame, stopping.");
                SetPlanComplete();
                return true;
            }

        }

    }
    else
    {
        if (m_thread.GetRegisterContext()->GetPC(0) != m_instruction_addr)
        {
            SetPlanComplete();
            return true;
        }
        else
            return false;
    }
}

bool
ThreadPlanStepInstruction::StopOthers ()
{
    return m_stop_other_threads;
}

StateType
ThreadPlanStepInstruction::GetPlanRunState ()
{
    return eStateStepping;
}

bool
ThreadPlanStepInstruction::WillStop ()
{
    return true;
}

bool
ThreadPlanStepInstruction::MischiefManaged ()
{
    if (IsPlanComplete())
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
            log->Printf("Completed single instruction step plan.");
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }
}

