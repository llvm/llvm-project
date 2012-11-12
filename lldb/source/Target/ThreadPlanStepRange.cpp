//===-- ThreadPlanStepRange.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

using namespace lldb;
using namespace lldb_private;


//----------------------------------------------------------------------
// ThreadPlanStepRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepRange::ThreadPlanStepRange (ThreadPlanKind kind, 
                                          const char *name, 
                                          Thread &thread, 
                                          const AddressRange &range, 
                                          const SymbolContext &addr_context, 
                                          lldb::RunMode stop_others) :
    ThreadPlan (kind, name, thread, eVoteNoOpinion, eVoteNoOpinion),
    m_addr_context (addr_context),
    m_address_ranges (),
    m_stop_others (stop_others),
    m_stack_id (),
    m_no_more_plans (false),
    m_first_run_event (true)
{
    AddRange(range);
    m_stack_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
}

ThreadPlanStepRange::~ThreadPlanStepRange ()
{
    ClearNextBranchBreakpoint();
}

void
ThreadPlanStepRange::DidPush ()
{
    // See if we can find a "next range" breakpoint:
    SetNextBranchBreakpoint();
}

bool
ThreadPlanStepRange::ValidatePlan (Stream *error)
{
    return true;
}

Vote
ThreadPlanStepRange::ShouldReportStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    const Vote vote = IsPlanComplete() ? eVoteYes : eVoteNo;
    if (log)
        log->Printf ("ThreadPlanStepRange::ShouldReportStop() returning vote %i\n", vote);
    return vote;
}

void
ThreadPlanStepRange::AddRange(const AddressRange &new_range)
{
    // For now I'm just adding the ranges.  At some point we may want to 
    // condense the ranges if they overlap, though I don't think it is likely
    // to be very important.
    m_address_ranges.push_back (new_range);
    m_instruction_ranges.push_back (DisassemblerSP());
}

void
ThreadPlanStepRange::DumpRanges(Stream *s)
{
    size_t num_ranges = m_address_ranges.size();
    if (num_ranges == 1)
    {
        m_address_ranges[0].Dump (s, m_thread.CalculateTarget().get(), Address::DumpStyleLoadAddress);
    }
    else
    {
        for (size_t i = 0; i < num_ranges; i++)
        {
            s->PutCString("%d: ");
            m_address_ranges[i].Dump (s, m_thread.CalculateTarget().get(), Address::DumpStyleLoadAddress);
        }
    }
}

bool
ThreadPlanStepRange::InRange ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    bool ret_value = false;

    lldb::addr_t pc_load_addr = m_thread.GetRegisterContext()->GetPC();

    size_t num_ranges = m_address_ranges.size();
    for (size_t i = 0; i < num_ranges; i++)
    {
        ret_value = m_address_ranges[i].ContainsLoadAddress(pc_load_addr, m_thread.CalculateTarget().get());
        if (ret_value)
            break;
    }
    
    if (!ret_value)
    {
        // See if we've just stepped to another part of the same line number...
        StackFrame *frame = m_thread.GetStackFrameAtIndex(0).get();
        
        SymbolContext new_context(frame->GetSymbolContext(eSymbolContextEverything));
        if (m_addr_context.line_entry.IsValid() && new_context.line_entry.IsValid())
        {
            if (m_addr_context.line_entry.file == new_context.line_entry.file)
            {
                if (m_addr_context.line_entry.line == new_context.line_entry.line)
                {
                    m_addr_context = new_context;
                    AddRange(m_addr_context.line_entry.range);
                    ret_value = true;
                    if (log)
                    {
                        StreamString s;
                        m_addr_context.line_entry.Dump (&s,
                                                        m_thread.CalculateTarget().get(),
                                                        true,
                                                        Address::DumpStyleLoadAddress,
                                                        Address::DumpStyleLoadAddress,
                                                        true);

                        log->Printf ("Step range plan stepped to another range of same line: %s", s.GetData());
                    }
                }
                else if (new_context.line_entry.range.GetBaseAddress().GetLoadAddress(m_thread.CalculateTarget().get())
                         != pc_load_addr)
                {
                    // Another thing that sometimes happens here is that we step out of one line into the MIDDLE of another
                    // line.  So far I mostly see this due to bugs in the debug information.
                    // But we probably don't want to be in the middle of a line range, so in that case reset the stepping
                    // range to the line we've stepped into the middle of and continue.
                    m_addr_context = new_context;
                    m_address_ranges.clear();
                    AddRange(m_addr_context.line_entry.range);
                    ret_value = true;
                    if (log)
                    {
                        StreamString s;
                        m_addr_context.line_entry.Dump (&s, 
                                                        m_thread.CalculateTarget().get(),
                                                        true,
                                                        Address::DumpStyleLoadAddress,
                                                        Address::DumpStyleLoadAddress,
                                                        true);

                        log->Printf ("Step range plan stepped to the middle of new line(%d): %s, continuing to clear this line.", 
                                     new_context.line_entry.line, 
                                     s.GetData());
                    }
                
                }
            }
            
        }
        
    }

    if (!ret_value && log)
        log->Printf ("Step range plan out of range to 0x%llx", pc_load_addr);

    return ret_value;
}

bool
ThreadPlanStepRange::InSymbol()
{
    lldb::addr_t cur_pc = m_thread.GetRegisterContext()->GetPC();
    if (m_addr_context.function != NULL)
    {
        return m_addr_context.function->GetAddressRange().ContainsLoadAddress (cur_pc, m_thread.CalculateTarget().get());
    }
    else if (m_addr_context.symbol)
    {
        AddressRange range(m_addr_context.symbol->GetAddress(), m_addr_context.symbol->GetByteSize());
        return range.ContainsLoadAddress (cur_pc, m_thread.CalculateTarget().get());
    }
    return false;
}

// FIXME: This should also handle inlining if we aren't going to do inlining in the
// main stack.
//
// Ideally we should remember the whole stack frame list, and then compare that
// to the current list.

lldb::FrameComparison
ThreadPlanStepRange::CompareCurrentFrameToStartFrame()
{
    FrameComparison frame_order;
    
    StackID cur_frame_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
    
    if (cur_frame_id == m_stack_id)
    {
        frame_order = eFrameCompareEqual;
    }
    else if (cur_frame_id < m_stack_id)
    {
        frame_order = eFrameCompareYounger;
    }
    else
    {
        frame_order = eFrameCompareOlder;
    }
    return frame_order;
}

bool
ThreadPlanStepRange::StopOthers ()
{
    if (m_stop_others == lldb::eOnlyThisThread
        || m_stop_others == lldb::eOnlyDuringStepping)
        return true;
    else
        return false;
}

InstructionList *
ThreadPlanStepRange::GetInstructionsForAddress(lldb::addr_t addr, size_t &range_index, size_t &insn_offset)
{
    size_t num_ranges = m_address_ranges.size();
    for (size_t i = 0; i < num_ranges; i++)
    {
        if (m_address_ranges[i].ContainsLoadAddress(addr, &GetTarget()))
        {
            // Some joker added a zero size range to the stepping range...
            if (m_address_ranges[i].GetByteSize() == 0)
                return NULL;

            if (!m_instruction_ranges[i])
            {
                //Disassemble the address range given:
                ExecutionContext exe_ctx (m_thread.GetProcess());
                m_instruction_ranges[i] = Disassembler::DisassembleRange(GetTarget().GetArchitecture(),
                                                                         NULL,
                                                                         exe_ctx,
                                                                         m_address_ranges[i]);
                
            }
            if (!m_instruction_ranges[i])
                return NULL;
            else
            {
                // Find where we are in the instruction list as well.  If we aren't at an instruction,
                // return NULL.  In this case, we're probably lost, and shouldn't try to do anything fancy.
                
                insn_offset = m_instruction_ranges[i]->GetInstructionList().GetIndexOfInstructionAtLoadAddress(addr, GetTarget());
                if (insn_offset == UINT32_MAX)
                    return NULL;
                else
                {
                    range_index = i;
                    return &m_instruction_ranges[i]->GetInstructionList();
                }
            }
        }
    }
    return NULL;
}

void
ThreadPlanStepRange::ClearNextBranchBreakpoint()
{
    if (m_next_branch_bp_sp)
    {
        GetTarget().RemoveBreakpointByID (m_next_branch_bp_sp->GetID());
        m_next_branch_bp_sp.reset();
    }
}

bool
ThreadPlanStepRange::SetNextBranchBreakpoint ()
{
    // Stepping through ranges using breakpoints doesn't work yet, but with this off we fall back to instruction
    // single stepping.
    return false;
    // Always clear the next branch breakpoint, we don't want to leave one of these stranded.
    ClearNextBranchBreakpoint();
    lldb::addr_t cur_addr = GetThread().GetRegisterContext()->GetPC();
    // Find the current address in our address ranges, and fetch the disassembly if we haven't already:
    size_t pc_index;
    size_t range_index;
    InstructionList *instructions = GetInstructionsForAddress (cur_addr, range_index, pc_index);
    if (instructions == NULL)
        return false;
    else
    {
        uint32_t branch_index;
        branch_index = instructions->GetIndexOfNextBranchInstruction (pc_index);
        
        Address run_to_address;
        
        // If we didn't find a branch, run to the end of the range.
        if (branch_index == UINT32_MAX)
        {
            branch_index = instructions->GetSize() - 2;
        }
        if (branch_index - pc_index > 1)
        {
            const bool is_internal = true;
            run_to_address = instructions->GetInstructionAtIndex(branch_index)->GetAddress();
            m_next_branch_bp_sp = GetTarget().CreateBreakpoint(run_to_address, is_internal);
            m_next_branch_bp_sp->SetThreadID(m_thread.GetID());
            return true;
        }
    }
    return false;
}

bool
ThreadPlanStepRange::NextRangeBreakpointExplainsStop (lldb::StopInfoSP stop_info_sp)
{
    if (!m_next_branch_bp_sp)
        return false;
    
    break_id_t bp_site_id = stop_info_sp->GetValue();
    BreakpointSiteSP bp_site_sp = m_thread.GetProcess()->GetBreakpointSiteList().FindByID(bp_site_id);
    if (!bp_site_sp->IsBreakpointAtThisSite (m_next_branch_bp_sp->GetID()))
        return false;
    else
        return bp_site_sp->GetNumberOfOwners() == 1;
}

bool
ThreadPlanStepRange::WillStop ()
{
    return true;
}

StateType
ThreadPlanStepRange::GetPlanRunState ()
{
    if (m_next_branch_bp_sp)
        return eStateRunning;
    else
        return eStateStepping;
}

bool
ThreadPlanStepRange::MischiefManaged ()
{
    // If we have pushed some plans between ShouldStop & MischiefManaged, then we're not done...
    // I do this check first because we might have stepped somewhere that will fool InRange into
    // thinking it needs to step past the end of that line.  This happens, for instance, when stepping
    // over inlined code that is in the middle of the current line.
    
    if (!m_no_more_plans)
        return false;
    
    bool done = true;
    if (!IsPlanComplete())
    {
        if (InRange())
        {
            done = false;
        }
        else 
        {
            FrameComparison frame_order = CompareCurrentFrameToStartFrame();
            if (frame_order != eFrameCompareOlder)
            {
                if (m_no_more_plans)
                    done = true;
                else
                    done = false;
            }
            else
                done = true;
        }
    }

    if (done)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
            log->Printf("Completed step through range plan.");
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }

}

bool
ThreadPlanStepRange::IsPlanStale ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    FrameComparison frame_order = CompareCurrentFrameToStartFrame();
    
    if (frame_order == eFrameCompareOlder)
    {
        if (log)
        {
            log->Printf("ThreadPlanStepRange::IsPlanStale returning true, we've stepped out.");
        }
        return true;
    }
    else if (frame_order == eFrameCompareEqual && InSymbol())
    {
        // If we are not in a place we should step through, we've gotten stale.
        // One tricky bit here is that some stubs don't push a frame, so we should.  
        // check that we are in the same symbol.          
        if (!InRange())
        {
            return true;
        }
    }
    return false;
}
