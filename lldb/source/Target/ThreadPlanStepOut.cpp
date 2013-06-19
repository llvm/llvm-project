//===-- ThreadPlanStepOut.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepOut.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlanStepOverRange.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepOut: Step out of the current frame
//----------------------------------------------------------------------
ThreadPlanStepOut::ThreadPlanStepOut
(
    Thread &thread,
    SymbolContext *context,
    bool first_insn,
    bool stop_others,
    Vote stop_vote,
    Vote run_vote,
    uint32_t frame_idx
) :
    ThreadPlan (ThreadPlan::eKindStepOut, "Step out", thread, stop_vote, run_vote),
    m_step_from_context (context),
    m_step_from_insn (LLDB_INVALID_ADDRESS),
    m_return_bp_id (LLDB_INVALID_BREAK_ID),
    m_return_addr (LLDB_INVALID_ADDRESS),
    m_first_insn (first_insn),
    m_stop_others (stop_others),
    m_step_through_inline_plan_sp(),
    m_step_out_plan_sp (),
    m_immediate_step_from_function(NULL)

{
    m_step_from_insn = m_thread.GetRegisterContext()->GetPC(0);

    StackFrameSP return_frame_sp (m_thread.GetStackFrameAtIndex(frame_idx + 1));
    StackFrameSP immediate_return_from_sp (m_thread.GetStackFrameAtIndex (frame_idx));
    
    if (!return_frame_sp || !immediate_return_from_sp)
        return; // we can't do anything here.  ValidatePlan() will return false.
    
    m_step_out_to_id = return_frame_sp->GetStackID();
    m_immediate_step_from_id = immediate_return_from_sp->GetStackID();
    
    StackID frame_zero_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();

    // If the frame directly below the one we are returning to is inlined, we have to be
    // a little more careful.  It is non-trivial to determine the real "return code address" for
    // an inlined frame, so we have to work our way to that frame and then step out.
    if (immediate_return_from_sp && immediate_return_from_sp->IsInlined())
    {
        if (frame_idx > 0)
        {
            // First queue a plan that gets us to this inlined frame, and when we get there we'll queue a second
            // plan that walks us out of this frame.
            m_step_out_plan_sp.reset (new ThreadPlanStepOut(m_thread, 
                                                            NULL, 
                                                            false,
                                                            stop_others, 
                                                            eVoteNoOpinion, 
                                                            eVoteNoOpinion, 
                                                            frame_idx - 1));
        }
        else
        {
            // If we're already at the inlined frame we're stepping through, then just do that now.
            QueueInlinedStepPlan(false);
        }
        
    }
    else if (return_frame_sp)
    {
        // Find the return address and set a breakpoint there:
        // FIXME - can we do this more securely if we know first_insn?

        m_return_addr = return_frame_sp->GetFrameCodeAddress().GetLoadAddress(&m_thread.GetProcess()->GetTarget());
        
        if (m_return_addr == LLDB_INVALID_ADDRESS)
            return;
        
        Breakpoint *return_bp = m_thread.CalculateTarget()->CreateBreakpoint (m_return_addr, true).get();
        if (return_bp != NULL)
        {
            return_bp->SetThreadID(m_thread.GetID());
            m_return_bp_id = return_bp->GetID();
            return_bp->SetBreakpointKind ("step-out");
        }
        
        if (immediate_return_from_sp)
        {
            const SymbolContext &sc = immediate_return_from_sp->GetSymbolContext(eSymbolContextFunction);
            if (sc.function)
            {
                m_immediate_step_from_function = sc.function; 
            }
        }
    }

}

void
ThreadPlanStepOut::DidPush()
{
    if (m_step_out_plan_sp)
        m_thread.QueueThreadPlan(m_step_out_plan_sp, false);
    else if (m_step_through_inline_plan_sp)
        m_thread.QueueThreadPlan(m_step_through_inline_plan_sp, false);
}

ThreadPlanStepOut::~ThreadPlanStepOut ()
{
    if (m_return_bp_id != LLDB_INVALID_BREAK_ID)
        m_thread.CalculateTarget()->RemoveBreakpointByID(m_return_bp_id);
}

void
ThreadPlanStepOut::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf ("step out");
    else
    {
        if (m_step_out_plan_sp)
            s->Printf ("Stepping out to inlined frame so we can walk through it.");
        else if (m_step_through_inline_plan_sp)
            s->Printf ("Stepping out by stepping through inlined function.");
        else
            s->Printf ("Stepping out from address 0x%" PRIx64 " to return address 0x%" PRIx64 " using breakpoint site %d",
                       (uint64_t)m_step_from_insn,
                       (uint64_t)m_return_addr,
                       m_return_bp_id);
    }
}

bool
ThreadPlanStepOut::ValidatePlan (Stream *error)
{
    if (m_step_out_plan_sp)
        return m_step_out_plan_sp->ValidatePlan (error);
    else if (m_step_through_inline_plan_sp)
        return m_step_through_inline_plan_sp->ValidatePlan (error);
    else if (m_return_bp_id == LLDB_INVALID_BREAK_ID)
    {
        if (error)
            error->PutCString("Could not create return address breakpoint.");
        return false;
    }
    else
        return true;
}

bool
ThreadPlanStepOut::DoPlanExplainsStop (Event *event_ptr)
{
    // If one of our child plans just finished, then we do explain the stop.
    if (m_step_out_plan_sp)
    {
        if (m_step_out_plan_sp->MischiefManaged())
        {
            // If this one is done, then we are all done.
            CalculateReturnValue();
            SetPlanComplete();
            return true;
        }
        else
            return false;
    }
    else if (m_step_through_inline_plan_sp)
    {
        if (m_step_through_inline_plan_sp->MischiefManaged())
            return true;
        else
            return false;
    }
        
    // We don't explain signals or breakpoints (breakpoints that handle stepping in or
    // out will be handled by a child plan.
    
    StopInfoSP stop_info_sp = GetPrivateStopInfo ();
    if (stop_info_sp)
    {
        StopReason reason = stop_info_sp->GetStopReason();
        switch (reason)
        {
        case eStopReasonBreakpoint:
        {
            // If this is OUR breakpoint, we're fine, otherwise we don't know why this happened...
            BreakpointSiteSP site_sp (m_thread.GetProcess()->GetBreakpointSiteList().FindByID (stop_info_sp->GetValue()));
            if (site_sp && site_sp->IsBreakpointAtThisSite (m_return_bp_id))
            {
                bool done;
                
                StackID frame_zero_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
                
                if (m_step_out_to_id == frame_zero_id)
                    done = true;
                else if (m_step_out_to_id < frame_zero_id)
                {
                    // Either we stepped past the breakpoint, or the stack ID calculation
                    // was incorrect and we should probably stop.
                    done = true;
                }
                else
                {
                    if (m_immediate_step_from_id < frame_zero_id)
                        done = true;
                    else
                        done = false;
                }
                    
                if (done)
                {
                    CalculateReturnValue();
                    SetPlanComplete();
                }

                // If there was only one owner, then we're done.  But if we also hit some
                // user breakpoint on our way out, we should mark ourselves as done, but
                // also not claim to explain the stop, since it is more important to report
                // the user breakpoint than the step out completion.

                if (site_sp->GetNumberOfOwners() == 1)
                    return true;
                
            }
            return false;
        }
        case eStopReasonWatchpoint:
        case eStopReasonSignal:
        case eStopReasonException:
        case eStopReasonExec:
        case eStopReasonThreadExiting:
            return false;

        default:
            return true;
        }
    }
    return true;
}

bool
ThreadPlanStepOut::ShouldStop (Event *event_ptr)
{
        if (IsPlanComplete())
            return true;
        
        bool done;
        
        StackID frame_zero_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
        if (frame_zero_id < m_step_out_to_id)
            done = false;
        else
            done = true;
            
        if (done)
        {
            CalculateReturnValue();
            SetPlanComplete();
            return true;
        }
        else
        {
            if (m_step_out_plan_sp)
            {
                if (m_step_out_plan_sp->MischiefManaged())
                {
                    // Now step through the inlined stack we are in:
                    if (QueueInlinedStepPlan(true))
                    {
                        return false;
                    }
                    else
                    {
                        CalculateReturnValue();
                        SetPlanComplete ();
                        return true;
                    }
                }
                else
                    return m_step_out_plan_sp->ShouldStop(event_ptr);
            }
            else if (m_step_through_inline_plan_sp)
            {
                if (m_step_through_inline_plan_sp->MischiefManaged())
                {
                    // We don't calculate the return value here because we don't know how to.
                    // But in case we had a return value sitting around from our process in
                    // getting here, let's clear it out.
                    m_return_valobj_sp.reset();
                    SetPlanComplete();
                    return true;
                }
                else
                    return m_step_through_inline_plan_sp->ShouldStop(event_ptr);
            }
            else
                return false;
        }
}

bool
ThreadPlanStepOut::StopOthers ()
{
    return m_stop_others;
}

StateType
ThreadPlanStepOut::GetPlanRunState ()
{
    return eStateRunning;
}

bool
ThreadPlanStepOut::DoWillResume (StateType resume_state, bool current_plan)
{
    if (m_step_out_plan_sp || m_step_through_inline_plan_sp)
        return true;
        
    if (m_return_bp_id == LLDB_INVALID_BREAK_ID)
        return false;

    if (current_plan)
    {
        Breakpoint *return_bp = m_thread.CalculateTarget()->GetBreakpointByID(m_return_bp_id).get();
        if (return_bp != NULL)
            return_bp->SetEnabled (true);
    }
    return true;
}

bool
ThreadPlanStepOut::WillStop ()
{
    if (m_return_bp_id != LLDB_INVALID_BREAK_ID)
    {
        Breakpoint *return_bp = m_thread.CalculateTarget()->GetBreakpointByID(m_return_bp_id).get();
        if (return_bp != NULL)
            return_bp->SetEnabled (false);
    }
    
    return true;
}

bool
ThreadPlanStepOut::MischiefManaged ()
{
    if (IsPlanComplete())
    {
        // Did I reach my breakpoint?  If so I'm done.
        //
        // I also check the stack depth, since if we've blown past the breakpoint for some
        // reason and we're now stopping for some other reason altogether, then we're done
        // with this step out operation.

        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
            log->Printf("Completed step out plan.");
        if (m_return_bp_id != LLDB_INVALID_BREAK_ID)
        {
            m_thread.CalculateTarget()->RemoveBreakpointByID (m_return_bp_id);
            m_return_bp_id = LLDB_INVALID_BREAK_ID;
        }
        
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }
}

bool
ThreadPlanStepOut::QueueInlinedStepPlan (bool queue_now)
{
    // Now figure out the range of this inlined block, and set up a "step through range"
    // plan for that.  If we've been provided with a context, then use the block in that
    // context.  
    StackFrameSP immediate_return_from_sp (m_thread.GetStackFrameAtIndex (0));
    if (!immediate_return_from_sp)
        return false;
        
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {   
        StreamString s;
        immediate_return_from_sp->Dump(&s, true, false);
        log->Printf("Queuing inlined frame to step past: %s.", s.GetData());
    }
        
    Block *from_block = immediate_return_from_sp->GetFrameBlock();
    if (from_block)
    {
        Block *inlined_block = from_block->GetContainingInlinedBlock();
        if (inlined_block)
        {
            size_t num_ranges = inlined_block->GetNumRanges();
            AddressRange inline_range;
            if (inlined_block->GetRangeAtIndex(0, inline_range))
            {
                SymbolContext inlined_sc;
                inlined_block->CalculateSymbolContext(&inlined_sc);
                inlined_sc.target_sp = GetTarget().shared_from_this();
                RunMode run_mode = m_stop_others ? lldb::eOnlyThisThread : lldb::eAllThreads;
                ThreadPlanStepOverRange *step_through_inline_plan_ptr = new ThreadPlanStepOverRange(m_thread, 
                                                                                                    inline_range, 
                                                                                                    inlined_sc, 
                                                                                                    run_mode);
                step_through_inline_plan_ptr->SetOkayToDiscard(true);                                                                                    
                StreamString errors;
                if (!step_through_inline_plan_ptr->ValidatePlan(&errors))
                {
                    //FIXME: Log this failure.
                    delete step_through_inline_plan_ptr;
                    return false;
                }
                
                for (size_t i = 1; i < num_ranges; i++)
                {
                    if (inlined_block->GetRangeAtIndex (i, inline_range))
                        step_through_inline_plan_ptr->AddRange (inline_range);
                }
                m_step_through_inline_plan_sp.reset (step_through_inline_plan_ptr);
                if (queue_now)
                    m_thread.QueueThreadPlan (m_step_through_inline_plan_sp, false);
                return true;
            }
        }
    }
        
    return false;
}

void
ThreadPlanStepOut::CalculateReturnValue ()
{
    if (m_return_valobj_sp)
        return;
        
    if (m_immediate_step_from_function != NULL)
    {
        Type *return_type = m_immediate_step_from_function->GetType();
        lldb::clang_type_t return_clang_type = m_immediate_step_from_function->GetReturnClangType();
        if (return_type && return_clang_type)
        {
            ClangASTType ast_type (return_type->GetClangAST(), return_clang_type);
            
            lldb::ABISP abi_sp = m_thread.GetProcess()->GetABI();
            if (abi_sp)
            {
                m_return_valobj_sp = abi_sp->GetReturnValueObject(m_thread, ast_type);
            }
        }
    }
}

bool
ThreadPlanStepOut::IsPlanStale()
{
    // If we are still lower on the stack than the frame we are returning to, then
    // there's something for us to do.  Otherwise, we're stale.
    
    StackID frame_zero_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
    if (frame_zero_id < m_step_out_to_id)
        return false;
    else
        return true;
}

