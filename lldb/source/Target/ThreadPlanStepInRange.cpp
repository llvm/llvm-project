//===-- ThreadPlanStepInRange.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepInRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"
#include "lldb/Core/RegularExpression.h"

using namespace lldb;
using namespace lldb_private;

uint32_t ThreadPlanStepInRange::s_default_flag_values = ThreadPlanShouldStopHere::eAvoidNoDebug;

//----------------------------------------------------------------------
// ThreadPlanStepInRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepInRange::ThreadPlanStepInRange
(
    Thread &thread,
    const AddressRange &range,
    const SymbolContext &addr_context,
    lldb::RunMode stop_others
) :
    ThreadPlanStepRange (ThreadPlan::eKindStepInRange, "Step Range stepping in", thread, range, addr_context, stop_others),
    ThreadPlanShouldStopHere (this, ThreadPlanStepInRange::DefaultShouldStopHereCallback, NULL),
    m_step_past_prologue (true),
    m_virtual_step (false)
{
    SetFlagsToDefault ();
}

ThreadPlanStepInRange::~ThreadPlanStepInRange ()
{
}

void
ThreadPlanStepInRange::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("step in");
    else
    {
        s->Printf ("Stepping through range (stepping into functions): ");
        DumpRanges(s);
    }
}

bool
ThreadPlanStepInRange::ShouldStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    m_no_more_plans = false;
    
    if (log)
    {
        StreamString s;
        s.Address (m_thread.GetRegisterContext()->GetPC(), 
                   m_thread.CalculateTarget()->GetArchitecture().GetAddressByteSize());
        log->Printf("ThreadPlanStepInRange reached %s.", s.GetData());
    }

    if (IsPlanComplete())
        return true;
        
    ThreadPlan* new_plan = NULL;

    if (m_virtual_step)
    {
        // If we've just completed a virtual step, all we need to do is check for a ShouldStopHere plan, and otherwise
        // we're done.
        new_plan = InvokeShouldStopHereCallback();
    }
    else
    {
        // Stepping through should be done running other threads in general, since we're setting a breakpoint and
        // continuing.  So only stop others if we are explicitly told to do so.
        
        bool stop_others;
        if (m_stop_others == lldb::eOnlyThisThread)
            stop_others = false;
        else
            stop_others = true;
            
        FrameComparison frame_order = CompareCurrentFrameToStartFrame();
        
        if (frame_order == eFrameCompareOlder)
        {
            // If we're in an older frame then we should stop.
            //
            // A caveat to this is if we think the frame is older but we're actually in a trampoline.
            // I'm going to make the assumption that you wouldn't RETURN to a trampoline.  So if we are
            // in a trampoline we think the frame is older because the trampoline confused the backtracer.
            new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);
            if (new_plan == NULL)
                return true;
            else if (log)
            {
                log->Printf("Thought I stepped out, but in fact arrived at a trampoline.");
            }

        }
        else if (frame_order == eFrameCompareEqual && InSymbol())
        {
            // If we are not in a place we should step through, we're done.
            // One tricky bit here is that some stubs don't push a frame, so we have to check
            // both the case of a frame that is younger, or the same as this frame.  
            // However, if the frame is the same, and we are still in the symbol we started
            // in, the we don't need to do this.  This first check isn't strictly necessary,
            // but it is more efficient.
            
            // If we're still in the range, keep going, either by running to the next branch breakpoint, or by
            // stepping.
            if (InRange())
            {
                SetNextBranchBreakpoint();
                return false;
            }
        
            SetPlanComplete();
            return true;
        }
        
        // If we get to this point, we're not going to use a previously set "next branch" breakpoint, so delete it:
        ClearNextBranchBreakpoint();
        
        // We may have set the plan up above in the FrameIsOlder section:
        
        if (new_plan == NULL)
            new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);
        
        if (log)
        {
            if (new_plan != NULL)
                log->Printf ("Found a step through plan: %s", new_plan->GetName());
            else
                log->Printf ("No step through plan found.");
        }
        
        // If not, give the "should_stop" callback a chance to push a plan to get us out of here.
        // But only do that if we actually have stepped in.
        if (!new_plan && frame_order == eFrameCompareYounger)
            new_plan = InvokeShouldStopHereCallback();

        // If we've stepped in and we are going to stop here, check to see if we were asked to
        // run past the prologue, and if so do that.
        
        if (new_plan == NULL && frame_order == eFrameCompareYounger && m_step_past_prologue)
        {
            lldb::StackFrameSP curr_frame = m_thread.GetStackFrameAtIndex(0);
            if (curr_frame)
            {
                size_t bytes_to_skip = 0;
                lldb::addr_t curr_addr = m_thread.GetRegisterContext()->GetPC();
                Address func_start_address;
                
                SymbolContext sc = curr_frame->GetSymbolContext (eSymbolContextFunction | eSymbolContextSymbol);
                
                if (sc.function)
                {
                    func_start_address = sc.function->GetAddressRange().GetBaseAddress();
                    if (curr_addr == func_start_address.GetLoadAddress(m_thread.CalculateTarget().get()))
                        bytes_to_skip = sc.function->GetPrologueByteSize();
                }
                else if (sc.symbol)
                {
                    func_start_address = sc.symbol->GetAddress();
                    if (curr_addr == func_start_address.GetLoadAddress(m_thread.CalculateTarget().get()))
                        bytes_to_skip = sc.symbol->GetPrologueByteSize();
                }
                
                if (bytes_to_skip != 0)
                {
                    func_start_address.Slide (bytes_to_skip);
                    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
                    if (log)
                        log->Printf ("Pushing past prologue ");
                        
                    new_plan = m_thread.QueueThreadPlanForRunToAddress(false, func_start_address,true);
                }
            }
        }
     }
    
     if (new_plan == NULL)
     {
        m_no_more_plans = true;
        SetPlanComplete();
        return true;
    }
    else
    {
        m_no_more_plans = false;
        return false;
    }
}

void
ThreadPlanStepInRange::SetFlagsToDefault ()
{
    GetFlags().Set(ThreadPlanStepInRange::s_default_flag_values);
}

void 
ThreadPlanStepInRange::SetAvoidRegexp(const char *name)
{
    if (m_avoid_regexp_ap.get() == NULL)
        m_avoid_regexp_ap.reset (new RegularExpression(name));

    m_avoid_regexp_ap->Compile (name);
}

void
ThreadPlanStepInRange::SetDefaultFlagValue (uint32_t new_value)
{
    // TODO: Should we test this for sanity?
    ThreadPlanStepInRange::s_default_flag_values = new_value;
}

bool
ThreadPlanStepInRange::FrameMatchesAvoidRegexp ()
{
    StackFrame *frame = GetThread().GetStackFrameAtIndex(0).get();

    const RegularExpression *avoid_regexp_to_use = m_avoid_regexp_ap.get();
    if (avoid_regexp_to_use == NULL)
        avoid_regexp_to_use = GetThread().GetSymbolsToAvoidRegexp();
        
    if (avoid_regexp_to_use != NULL)
    {
        SymbolContext sc = frame->GetSymbolContext(eSymbolContextFunction|eSymbolContextBlock|eSymbolContextSymbol);
        if (sc.symbol != NULL)
        {
            const char *frame_function_name = sc.GetFunctionName().GetCString();
            if (frame_function_name)
               return avoid_regexp_to_use->Execute(frame_function_name);
        }
    }
    return false;
}

ThreadPlan *
ThreadPlanStepInRange::DefaultShouldStopHereCallback (ThreadPlan *current_plan, Flags &flags, void *baton)
{
    bool should_step_out = false;
    StackFrame *frame = current_plan->GetThread().GetStackFrameAtIndex(0).get();

    if (flags.Test(eAvoidNoDebug))
    {
        if (!frame->HasDebugInformation())
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
            if (log)
                log->Printf ("Stepping out of frame with no debug info");

            should_step_out = true;
        }
    }
    
    if (!should_step_out)
    {
        if (current_plan->GetKind() == eKindStepInRange)
        {
            ThreadPlanStepInRange *step_in_range_plan = static_cast<ThreadPlanStepInRange *> (current_plan);
            should_step_out = step_in_range_plan->FrameMatchesAvoidRegexp ();
        }
    }
    
    if (should_step_out)
    {
        // FIXME: Make sure the ThreadPlanForStepOut does the right thing with inlined functions.
        // We really should have all plans take the tri-state for "stop others" so we can do the right
        // thing.  For now let's be safe and always run others when we are likely to run arbitrary code.
        const bool stop_others = false;
        return current_plan->GetThread().QueueThreadPlanForStepOut (false, 
                                                                    NULL, 
                                                                    true, 
                                                                    stop_others,
                                                                    eVoteNo, 
                                                                    eVoteNoOpinion,
                                                                    0); // Frame index
    }

    return NULL;
}

bool
ThreadPlanStepInRange::PlanExplainsStop ()
{
    // We always explain a stop.  Either we've just done a single step, in which
    // case we'll do our ordinary processing, or we stopped for some
    // reason that isn't handled by our sub-plans, in which case we want to just stop right
    // away.
    // We also set ourselves complete when we stop for this sort of unintended reason, but mark
    // success as false so we don't end up being the reason for the stop.
    //
    // The only variation is that if we are doing "step by running to next branch" in which case
    // if we hit our branch breakpoint we don't set the plan to complete.
    
    if (m_virtual_step)
        return true;
    
    StopInfoSP stop_info_sp = GetPrivateStopReason();
    if (stop_info_sp)
    {
        StopReason reason = stop_info_sp->GetStopReason();

        switch (reason)
        {
        case eStopReasonBreakpoint:
            if (NextRangeBreakpointExplainsStop(stop_info_sp))
                return true;
        case eStopReasonWatchpoint:
        case eStopReasonSignal:
        case eStopReasonException:
            {
                LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
                if (log)
                    log->PutCString ("ThreadPlanStepInRange got asked if it explains the stop for some reason other than step.");
                SetPlanComplete(false);
            }
            break;
        default:
            break;
        }
    }
    return true;
}

bool
ThreadPlanStepInRange::WillResume (lldb::StateType resume_state, bool current_plan)
{
    if (resume_state == eStateStepping && current_plan)
    {
        // See if we are about to step over a virtual inlined call.
        bool step_without_resume = m_thread.DecrementCurrentInlinedDepth();
        if (step_without_resume)
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
            if (log)
                log->Printf ("ThreadPlanStepInRange::WillResume: returning false, inline_depth: %d",
                             m_thread.GetCurrentInlinedDepth());
            SetStopInfo(StopInfo::CreateStopReasonToTrace(m_thread));
            
            // FIXME: Maybe it would be better to create a InlineStep stop reason, but then
            // the whole rest of the world would have to handle that stop reason.
            m_virtual_step = true;
        }
        return !step_without_resume;
    }
    else
        return ThreadPlan::WillResume(resume_state, current_plan);
    
}
