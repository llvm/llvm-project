//===-- ThreadPlanStepOverRange.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepOverRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"

using namespace lldb_private;
using namespace lldb;


//----------------------------------------------------------------------
// ThreadPlanStepOverRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepOverRange::ThreadPlanStepOverRange
(
    Thread &thread,
    const AddressRange &range,
    const SymbolContext &addr_context,
    lldb::RunMode stop_others
) :
    ThreadPlanStepRange (ThreadPlan::eKindStepOverRange, "Step range stepping over", thread, range, addr_context, stop_others),
    m_first_resume(true)
{
}

ThreadPlanStepOverRange::~ThreadPlanStepOverRange ()
{
}

void
ThreadPlanStepOverRange::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("step over");
    else
    {
        s->Printf ("stepping through range (stepping over functions): ");
        DumpRanges(s);    
    }
}

bool
ThreadPlanStepOverRange::ShouldStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (log)
    {
        StreamString s;
        s.Address (m_thread.GetRegisterContext()->GetPC(), 
                   m_thread.CalculateTarget()->GetArchitecture().GetAddressByteSize());
        log->Printf("ThreadPlanStepOverRange reached %s.", s.GetData());
    }
    
    // If we're out of the range but in the same frame or in our caller's frame
    // then we should stop.
    // When stepping out we only stop others if we are forcing running one thread.
    bool stop_others;
    if (m_stop_others == lldb::eOnlyThisThread)
        stop_others = true;
    else 
        stop_others = false;

    ThreadPlan* new_plan = NULL;
    
    FrameComparison frame_order = CompareCurrentFrameToStartFrame();
    
    if (frame_order == eFrameCompareOlder)
    {
        // If we're in an older frame then we should stop.
        //
        // A caveat to this is if we think the frame is older but we're actually in a trampoline.
        // I'm going to make the assumption that you wouldn't RETURN to a trampoline.  So if we are
        // in a trampoline we think the frame is older because the trampoline confused the backtracer.
        // As below, we step through first, and then try to figure out how to get back out again.
        
        new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);

        if (new_plan != NULL && log)
            log->Printf("Thought I stepped out, but in fact arrived at a trampoline.");
    }
    else if (frame_order == eFrameCompareYounger)
    {
        // Make sure we really are in a new frame.  Do that by unwinding and seeing if the
        // start function really is our start function...
        StackFrameSP older_frame_sp = m_thread.GetStackFrameAtIndex(1);
        
        // But if we can't even unwind one frame we should just get out of here & stop...
        if (older_frame_sp)
        {
            const SymbolContext &older_context = older_frame_sp->GetSymbolContext(eSymbolContextEverything);
            
            // Match as much as is specified in the m_addr_context:
            // This is a fairly loose sanity check.  Note, sometimes the target doesn't get filled
            // in so I left out the target check.  And sometimes the module comes in as the .o file from the
            // inlined range, so I left that out too...
            
            bool older_ctx_is_equivalent = true;
            if (m_addr_context.comp_unit)
            {
                if (m_addr_context.comp_unit == older_context.comp_unit)
                {
                    if (m_addr_context.function && m_addr_context.function == older_context.function)
                    {
                        if (m_addr_context.block && m_addr_context.block == older_context.block)
                        {
                            older_ctx_is_equivalent = true;
                        }
                    }
                }
            }
            else if (m_addr_context.symbol && m_addr_context.symbol == older_context.symbol)
            {
                older_ctx_is_equivalent = true;
            }
        
            if (older_ctx_is_equivalent)
            {
                new_plan = m_thread.QueueThreadPlanForStepOut (false, 
                                                           NULL, 
                                                           true, 
                                                           stop_others, 
                                                           eVoteNo, 
                                                           eVoteNoOpinion,
                                                           0);
            }
            else 
            {
                new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);
                
            }
        }
    }
    else
    {
        // If we're still in the range, keep going.
        if (InRange())
        {
            SetNextBranchBreakpoint();
            return false;
        }


        if (!InSymbol())
        {
            // This one is a little tricky.  Sometimes we may be in a stub or something similar,
            // in which case we need to get out of there.  But if we are in a stub then it's 
            // likely going to be hard to get out from here.  It is probably easiest to step into the
            // stub, and then it will be straight-forward to step out.        
            new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);
        }
        else
        {
            // The current clang (at least through 424) doesn't always get the address range for the 
            // DW_TAG_inlined_subroutines right, so that when you leave the inlined range the line table says 
            // you are still in the source file of the inlining function.  This is bad, because now you are missing 
            // the stack frame for the function containing the inlining, and if you sensibly do "finish" to get
            // out of this function you will instead exit the containing function.
            // To work around this, we check whether we are still in the source file we started in, and if not assume
            // it is an error, and push a plan to get us out of this line and back to the containing file.

            if (m_addr_context.line_entry.IsValid())
            {
                SymbolContext sc;
                StackFrameSP frame_sp = m_thread.GetStackFrameAtIndex(0);
                sc = frame_sp->GetSymbolContext (eSymbolContextEverything);
                if (sc.line_entry.IsValid())
                {
                    if (sc.line_entry.file != m_addr_context.line_entry.file
                         && sc.comp_unit == m_addr_context.comp_unit
                         && sc.function == m_addr_context.function)
                    {
                        // Okay, find the next occurance of this file in the line table:
                        LineTable *line_table = m_addr_context.comp_unit->GetLineTable();
                        if (line_table)
                        {
                            Address cur_address = frame_sp->GetFrameCodeAddress();
                            uint32_t entry_idx;
                            LineEntry line_entry;
                            if (line_table->FindLineEntryByAddress (cur_address, line_entry, &entry_idx))
                            {
                                LineEntry next_line_entry;
                                bool step_past_remaining_inline = false;
                                if (entry_idx > 0)
                                {
                                    // We require the the previous line entry and the current line entry come
                                    // from the same file.
                                    // The other requirement is that the previous line table entry be part of an
                                    // inlined block, we don't want to step past cases where people have inlined
                                    // some code fragment by using #include <source-fragment.c> directly.
                                    LineEntry prev_line_entry;
                                    if (line_table->GetLineEntryAtIndex(entry_idx - 1, prev_line_entry)
                                        && prev_line_entry.file == line_entry.file)
                                    {
                                        SymbolContext prev_sc;
                                        Address prev_address = prev_line_entry.range.GetBaseAddress();
                                        prev_address.CalculateSymbolContext(&prev_sc);
                                        if (prev_sc.block)
                                        {
                                            Block *inlined_block = prev_sc.block->GetContainingInlinedBlock();
                                            if (inlined_block)
                                            {
                                                AddressRange inline_range;
                                                inlined_block->GetRangeContainingAddress(prev_address, inline_range);
                                                if (!inline_range.ContainsFileAddress(cur_address))
                                                {
                                                    
                                                    step_past_remaining_inline = true;
                                                }
                                                
                                            }
                                        }
                                    }
                                }
                                
                                if (step_past_remaining_inline)
                                {
                                    uint32_t look_ahead_step = 1;
                                    while (line_table->GetLineEntryAtIndex(entry_idx + look_ahead_step, next_line_entry))
                                    {
                                        // Make sure we haven't wandered out of the function we started from...
                                        Address next_line_address = next_line_entry.range.GetBaseAddress();
                                        Function *next_line_function = next_line_address.CalculateSymbolContextFunction();
                                        if (next_line_function != m_addr_context.function)
                                            break;
                                        
                                        if (next_line_entry.file == m_addr_context.line_entry.file)
                                        {
                                            const bool abort_other_plans = false;
                                            const bool stop_other_threads = false;
                                            new_plan = m_thread.QueueThreadPlanForRunToAddress(abort_other_plans,
                                                                                               next_line_address,
                                                                                               stop_other_threads);
                                            break;
                                        }
                                        look_ahead_step++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // If we get to this point, we're not going to use a previously set "next branch" breakpoint, so delete it:
    ClearNextBranchBreakpoint();
    
    if (new_plan == NULL)
        m_no_more_plans = true;
    else
        m_no_more_plans = false;

    if (new_plan == NULL)
    {
        // For efficiencies sake, we know we're done here so we don't have to do this
        // calculation again in MischiefManaged.
        SetPlanComplete();
        return true;
    }
    else
        return false;
}

bool
ThreadPlanStepOverRange::PlanExplainsStop ()
{
    // For crashes, breakpoint hits, signals, etc, let the base plan (or some plan above us)
    // handle the stop.  That way the user can see the stop, step around, and then when they
    // are done, continue and have their step complete.  The exception is if we've hit our
    // "run to next branch" breakpoint.
    // Note, unlike the step in range plan, we don't mark ourselves complete if we hit an
    // unexplained breakpoint/crash.
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    StopInfoSP stop_info_sp = GetPrivateStopReason();
    if (stop_info_sp)
    {
        StopReason reason = stop_info_sp->GetStopReason();

        switch (reason)
        {
        case eStopReasonTrace:
            return true;
            break;
        case eStopReasonBreakpoint:
            if (NextRangeBreakpointExplainsStop(stop_info_sp))
                return true;
            else
                return false;
            break;
        case eStopReasonWatchpoint:
        case eStopReasonSignal:
        case eStopReasonException:
        default:
            if (log)
                log->PutCString ("ThreadPlanStepInRange got asked if it explains the stop for some reason other than step.");
            return false;
            break;
        }
    }
    return true;
}

bool
ThreadPlanStepOverRange::WillResume (lldb::StateType resume_state, bool current_plan)
{
    if (resume_state != eStateSuspended && m_first_resume)
    {
        m_first_resume = false;
        if (resume_state == eStateStepping && current_plan)
        {
            // See if we are about to step over an inlined call in the middle of the inlined stack, if so figure
            // out its extents and reset our range to step over that.
            bool in_inlined_stack = m_thread.DecrementCurrentInlinedDepth();
            if (in_inlined_stack)
            {
                LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
                if (log)
                    log->Printf ("ThreadPlanStepInRange::WillResume: adjusting range to the frame at inlined depth %d.",
                                 m_thread.GetCurrentInlinedDepth());
                StackFrameSP stack_sp = m_thread.GetStackFrameAtIndex(0);
                if (stack_sp)
                {
                    Block *frame_block = stack_sp->GetFrameBlock();
                    lldb::addr_t curr_pc = m_thread.GetRegisterContext()->GetPC();
                    AddressRange my_range;
                    if (frame_block->GetRangeContainingLoadAddress(curr_pc, m_thread.GetProcess()->GetTarget(), my_range))
                    {
                        m_address_ranges.clear();
                        m_address_ranges.push_back(my_range);
                        if (log)
                        {
                            StreamString s;
                            const InlineFunctionInfo *inline_info = frame_block->GetInlinedFunctionInfo();
                            const char *name;
                            if (inline_info)
                                name = inline_info->GetName().AsCString();
                            else
                                name = "<unknown-notinlined>";
                            
                            s.Printf ("Stepping over inlined function \"%s\" in inlined stack: ", name);
                            DumpRanges(&s);
                            log->PutCString(s.GetData());
                        }
                    }
                    
                }
            }
        }
    }
    
    return ThreadPlan::WillResume(resume_state, current_plan);
}
