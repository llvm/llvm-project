//===-- StackFrameList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackFrameList.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Unwind.h"

//#define DEBUG_STACK_FRAMES 1

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StackFrameList constructor
//----------------------------------------------------------------------
StackFrameList::StackFrameList
(
    Thread &thread, 
    const lldb::StackFrameListSP &prev_frames_sp, 
    bool show_inline_frames
) :
    m_thread (thread),
    m_prev_frames_sp (prev_frames_sp),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_frames (),
    m_selected_frame_idx (0),
    m_concrete_frames_fetched (0),
    m_current_inlined_depth (UINT32_MAX),
    m_current_inlined_pc (LLDB_INVALID_ADDRESS),
    m_show_inlined_frames (show_inline_frames)
{
    if (prev_frames_sp)
    {
        m_current_inlined_depth = prev_frames_sp->m_current_inlined_depth;
        m_current_inlined_pc =    prev_frames_sp->m_current_inlined_pc;
    }
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StackFrameList::~StackFrameList()
{
}

void
StackFrameList::CalculateCurrentInlinedDepth()
{
    uint32_t cur_inlined_depth = GetCurrentInlinedDepth();
    if (cur_inlined_depth == UINT32_MAX)
    {
        ResetCurrentInlinedDepth();
    }
}

uint32_t
StackFrameList::GetCurrentInlinedDepth ()
{
    if (m_show_inlined_frames && m_current_inlined_pc != LLDB_INVALID_ADDRESS)
    {
        lldb::addr_t cur_pc = m_thread.GetRegisterContext()->GetPC();
        if (cur_pc != m_current_inlined_pc)
        {
            m_current_inlined_pc = LLDB_INVALID_ADDRESS;
            m_current_inlined_depth = UINT32_MAX;
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
            if (log && log->GetVerbose())
                log->Printf ("GetCurrentInlinedDepth: invalidating current inlined depth.\n");
        }
        return m_current_inlined_depth;
    }
    else
    {
        return UINT32_MAX;
    }
}

void
StackFrameList::ResetCurrentInlinedDepth ()
{
    if (m_show_inlined_frames)
    {        
        GetFramesUpTo(0);
        if (!m_frames[0]->IsInlined())
        {
            m_current_inlined_depth = UINT32_MAX;
            m_current_inlined_pc = LLDB_INVALID_ADDRESS;
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
            if (log && log->GetVerbose())
                log->Printf ("ResetCurrentInlinedDepth: Invalidating current inlined depth.\n");
        }
        else
        {
            // We only need to do something special about inlined blocks when we
            // are at the beginning of an inlined function:
            // FIXME: We probably also have to do something special if the PC is at the END
            // of an inlined function, which coincides with the end of either its containing
            // function or another inlined function.
            
            lldb::addr_t curr_pc = m_thread.GetRegisterContext()->GetPC();
            Block *block_ptr = m_frames[0]->GetFrameBlock();
            if (block_ptr)
            {
                Address pc_as_address;
                pc_as_address.SetLoadAddress(curr_pc, &(m_thread.GetProcess()->GetTarget()));
                AddressRange containing_range;
                if (block_ptr->GetRangeContainingAddress(pc_as_address, containing_range))
                {
                    if (pc_as_address == containing_range.GetBaseAddress())
                    {
                        // If we got here because of a breakpoint hit, then set the inlined depth depending on where
                        // the breakpoint was set.
                        // If we got here because of a crash, then set the inlined depth to the deepest most block.
                        // Otherwise, we stopped here naturally as the result of a step, so set ourselves in the
                        // containing frame of the whole set of nested inlines, so the user can then "virtually"
                        // step into the frames one by one, or next over the whole mess.
                        // Note: We don't have to handle being somewhere in the middle of the stack here, since
                        // ResetCurrentInlinedDepth doesn't get called if there is a valid inlined depth set.
                        StopInfoSP stop_info_sp = m_thread.GetStopInfo();
                        if (stop_info_sp)
                        {
                            switch (stop_info_sp->GetStopReason())
                            {
                            case eStopReasonWatchpoint:
                            case eStopReasonException:
                            case eStopReasonSignal:
                                // In all these cases we want to stop in the deepest most frame.
                                m_current_inlined_pc = curr_pc;
                                m_current_inlined_depth = 0;
                                break;
                            case eStopReasonBreakpoint:
                                {
                                    // FIXME: Figure out what this break point is doing, and set the inline depth
                                    // appropriately.  Be careful to take into account breakpoints that implement
                                    // step over prologue, since that should do the default calculation.
                                    // For now, if the breakpoints corresponding to this hit are all internal,
                                    // I set the stop location to the top of the inlined stack, since that will make
                                    // things like stepping over prologues work right.  But if there are any non-internal
                                    // breakpoints I do to the bottom of the stack, since that was the old behavior.
                                    uint32_t bp_site_id = stop_info_sp->GetValue();
                                    BreakpointSiteSP bp_site_sp(m_thread.GetProcess()->GetBreakpointSiteList().FindByID(bp_site_id));
                                    bool all_internal = true;
                                    if (bp_site_sp)
                                    {
                                        uint32_t num_owners = bp_site_sp->GetNumberOfOwners();
                                        for (uint32_t i = 0; i < num_owners; i++)
                                        {
                                            Breakpoint &bp_ref = bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint();
                                            if (!bp_ref.IsInternal())
                                            {
                                                all_internal = false;
                                            }
                                        }
                                    }
                                    if (!all_internal)
                                    {
                                        m_current_inlined_pc = curr_pc;
                                        m_current_inlined_depth = 0;
                                        break;
                                    }
                                }
                            default:
                                {
                                    // Otherwise, we should set ourselves at the container of the inlining, so that the
                                    // user can descend into them.
                                    // So first we check whether we have more than one inlined block sharing this PC:
                                    int num_inlined_functions = 0;
                                    
                                    for  (Block *container_ptr = block_ptr->GetInlinedParent();
                                              container_ptr != NULL;
                                              container_ptr = container_ptr->GetInlinedParent())
                                    {
                                        if (!container_ptr->GetRangeContainingAddress(pc_as_address, containing_range))
                                            break;
                                        if (pc_as_address != containing_range.GetBaseAddress())
                                            break;
                                        
                                        num_inlined_functions++;
                                    }
                                    m_current_inlined_pc = curr_pc;
                                    m_current_inlined_depth = num_inlined_functions + 1;
                                    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
                                    if (log && log->GetVerbose())
                                        log->Printf ("ResetCurrentInlinedDepth: setting inlined depth: %d 0x%llx.\n", m_current_inlined_depth, curr_pc);
                                    
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

bool
StackFrameList::DecrementCurrentInlinedDepth ()
{
    if (m_show_inlined_frames)
    {
        uint32_t current_inlined_depth = GetCurrentInlinedDepth();
        if (current_inlined_depth != UINT32_MAX)
        {
            if (current_inlined_depth > 0)
            {
                m_current_inlined_depth--;
                return true;
            }
        }
    }
    return false;
}

void
StackFrameList::SetCurrentInlinedDepth (uint32_t new_depth)
{
    m_current_inlined_depth = new_depth;
    if (new_depth == UINT32_MAX)
        m_current_inlined_pc = LLDB_INVALID_ADDRESS;
    else
        m_current_inlined_pc = m_thread.GetRegisterContext()->GetPC();
}

void
StackFrameList::GetFramesUpTo(uint32_t end_idx)
{
    // We've already gotten more frames than asked for, or we've already finished unwinding, return.
    if (m_frames.size() > end_idx || GetAllFramesFetched())
        return;
        
    Unwind *unwinder = m_thread.GetUnwinder ();

    if (m_show_inlined_frames)
    {
#if defined (DEBUG_STACK_FRAMES)
        StreamFile s(stdout, false);
#endif
        // If we are hiding some frames from the outside world, we need to add those onto the total count of
        // frames to fetch.  However, we don't need ot do that if end_idx is 0 since in that case we always
        // get the first concrete frame and all the inlined frames below it...  And of course, if end_idx is
        // UINT32_MAX that means get all, so just do that...
        
        uint32_t inlined_depth = 0;
        if (end_idx > 0 && end_idx != UINT32_MAX)
        {
            inlined_depth = GetCurrentInlinedDepth();
            if (inlined_depth != UINT32_MAX)
            {
                if (end_idx > 0)
                    end_idx += inlined_depth;
            }
        }
        
        StackFrameSP unwind_frame_sp;
        do
        {
            uint32_t idx = m_concrete_frames_fetched++;
            lldb::addr_t pc;
            lldb::addr_t cfa;
            if (idx == 0)
            {
                // We might have already created frame zero, only create it
                // if we need to
                if (m_frames.empty())
                {
                    m_thread.GetRegisterContext();
                    assert (m_thread.m_reg_context_sp.get());

                    const bool success = unwinder->GetFrameInfoAtIndex(idx, cfa, pc);
                    // There shouldn't be any way not to get the frame info for frame 0.
                    // But if the unwinder can't make one, lets make one by hand with the
                    // SP as the CFA and see if that gets any further.
                    if (!success)
                    {
                        cfa = m_thread.GetRegisterContext()->GetSP();
                        pc = m_thread.GetRegisterContext()->GetPC();
                    }
                    
                    unwind_frame_sp.reset (new StackFrame (m_thread.shared_from_this(),
                                                           m_frames.size(), 
                                                           idx,
                                                           m_thread.m_reg_context_sp,
                                                           cfa,
                                                           pc,
                                                           NULL));
                    m_frames.push_back (unwind_frame_sp);
                }
                else
                {
                    unwind_frame_sp = m_frames.front();
                    cfa = unwind_frame_sp->m_id.GetCallFrameAddress();
                }
            }
            else
            {
                const bool success = unwinder->GetFrameInfoAtIndex(idx, cfa, pc);
                if (!success)
                {
                    // We've gotten to the end of the stack.
                    SetAllFramesFetched();
                    break;
                }
                unwind_frame_sp.reset (new StackFrame (m_thread.shared_from_this(), m_frames.size(), idx, cfa, pc, NULL));
                m_frames.push_back (unwind_frame_sp);
            }
            
            SymbolContext unwind_sc = unwind_frame_sp->GetSymbolContext (eSymbolContextBlock | eSymbolContextFunction);
            Block *unwind_block = unwind_sc.block;
            if (unwind_block)
            {
                Address curr_frame_address (unwind_frame_sp->GetFrameCodeAddress());
                // Be sure to adjust the frame address to match the address
                // that was used to lookup the symbol context above. If we are
                // in the first concrete frame, then we lookup using the current
                // address, else we decrement the address by one to get the correct
                // location.
                if (idx > 0)
                    curr_frame_address.Slide(-1);
                    
                SymbolContext next_frame_sc;
                Address next_frame_address;
                
                while (unwind_sc.GetParentOfInlinedScope(curr_frame_address, next_frame_sc, next_frame_address))
                {
                        StackFrameSP frame_sp(new StackFrame (m_thread.shared_from_this(),
                                                              m_frames.size(),
                                                              idx,
                                                              unwind_frame_sp->GetRegisterContextSP (),
                                                              cfa,
                                                              next_frame_address,
                                                              &next_frame_sc));  
                                                    
                        m_frames.push_back (frame_sp);
                        unwind_sc = next_frame_sc;
                        curr_frame_address = next_frame_address;
                }
            }
        } while (m_frames.size() - 1 < end_idx);

        // Don't try to merge till you've calculated all the frames in this stack.
        if (GetAllFramesFetched() && m_prev_frames_sp)
        {
            StackFrameList *prev_frames = m_prev_frames_sp.get();
            StackFrameList *curr_frames = this;
            
            //curr_frames->m_current_inlined_depth = prev_frames->m_current_inlined_depth;
            //curr_frames->m_current_inlined_pc = prev_frames->m_current_inlined_pc;
            //printf ("GetFramesUpTo: Copying current inlined depth: %d 0x%llx.\n", curr_frames->m_current_inlined_depth, curr_frames->m_current_inlined_pc);

#if defined (DEBUG_STACK_FRAMES)
            s.PutCString("\nprev_frames:\n");
            prev_frames->Dump (&s);
            s.PutCString("\ncurr_frames:\n");
            curr_frames->Dump (&s);
            s.EOL();
#endif
            size_t curr_frame_num, prev_frame_num;
            
            for (curr_frame_num = curr_frames->m_frames.size(), prev_frame_num = prev_frames->m_frames.size();
                 curr_frame_num > 0 && prev_frame_num > 0;
                 --curr_frame_num, --prev_frame_num)
            {
                const size_t curr_frame_idx = curr_frame_num-1;
                const size_t prev_frame_idx = prev_frame_num-1;
                StackFrameSP curr_frame_sp (curr_frames->m_frames[curr_frame_idx]);
                StackFrameSP prev_frame_sp (prev_frames->m_frames[prev_frame_idx]);

#if defined (DEBUG_STACK_FRAMES)
                s.Printf("\n\nCurr frame #%u ", curr_frame_idx);
                if (curr_frame_sp)
                    curr_frame_sp->Dump (&s, true, false);
                else
                    s.PutCString("NULL");
                s.Printf("\nPrev frame #%u ", prev_frame_idx);
                if (prev_frame_sp)
                    prev_frame_sp->Dump (&s, true, false);
                else
                    s.PutCString("NULL");
#endif

                StackFrame *curr_frame = curr_frame_sp.get();
                StackFrame *prev_frame = prev_frame_sp.get();
                
                if (curr_frame == NULL || prev_frame == NULL)
                    break;

                // Check the stack ID to make sure they are equal
                if (curr_frame->GetStackID() != prev_frame->GetStackID())
                    break;

                prev_frame->UpdatePreviousFrameFromCurrentFrame (*curr_frame);
                // Now copy the fixed up previous frame into the current frames
                // so the pointer doesn't change
                m_frames[curr_frame_idx] = prev_frame_sp;
                //curr_frame->UpdateCurrentFrameFromPreviousFrame (*prev_frame);
                
#if defined (DEBUG_STACK_FRAMES)
                s.Printf("\n    Copying previous frame to current frame");
#endif
            }
            // We are done with the old stack frame list, we can release it now
            m_prev_frames_sp.reset();
        }
        
#if defined (DEBUG_STACK_FRAMES)
            s.PutCString("\n\nNew frames:\n");
            Dump (&s);
            s.EOL();
#endif
    }
    else
    {
        if (end_idx < m_concrete_frames_fetched)
            return;
            
        uint32_t num_frames = unwinder->GetFramesUpTo(end_idx);
        if (num_frames <= end_idx + 1)
        {
            //Done unwinding.
            m_concrete_frames_fetched = UINT32_MAX;
        }
        m_frames.resize(num_frames);
    }
}

uint32_t
StackFrameList::GetNumFrames (bool can_create)
{
    Mutex::Locker locker (m_mutex);

    if (can_create)
        GetFramesUpTo (UINT32_MAX);

    uint32_t inlined_depth = GetCurrentInlinedDepth();
    if (inlined_depth == UINT32_MAX)
        return m_frames.size();
    else
        return m_frames.size() - inlined_depth;
}

void
StackFrameList::Dump (Stream *s)
{
    if (s == NULL)
        return;
    Mutex::Locker locker (m_mutex);

    const_iterator pos, begin = m_frames.begin(), end = m_frames.end();
    for (pos = begin; pos != end; ++pos)
    {
        StackFrame *frame = (*pos).get();
        s->Printf("%p: ", frame);
        if (frame)
        {
            frame->GetStackID().Dump (s);
            frame->DumpUsingSettingsFormat (s);
        }
        else
            s->Printf("frame #%u", (uint32_t)std::distance (begin, pos));
        s->EOL();
    }
    s->EOL();
}

StackFrameSP
StackFrameList::GetFrameAtIndex (uint32_t idx)
{
    StackFrameSP frame_sp;
    Mutex::Locker locker (m_mutex);
    uint32_t inlined_depth = GetCurrentInlinedDepth();
    if (inlined_depth != UINT32_MAX)
        idx += inlined_depth;
    
    if (idx < m_frames.size())
        frame_sp = m_frames[idx];

    if (frame_sp)
        return frame_sp;
        
        // GetFramesUpTo will fill m_frames with as many frames as you asked for,
        // if there are that many.  If there weren't then you asked for too many
        // frames.
        GetFramesUpTo (idx);
        if (idx < m_frames.size())
        {
            if (m_show_inlined_frames)
            {
                // When inline frames are enabled we actually create all the frames in GetFramesUpTo.
                frame_sp = m_frames[idx];
            }
            else
            {
                Unwind *unwinder = m_thread.GetUnwinder ();
                if (unwinder)
                {
                    addr_t pc, cfa;
                    if (unwinder->GetFrameInfoAtIndex(idx, cfa, pc))
                    {
                        frame_sp.reset (new StackFrame (m_thread.shared_from_this(), idx, idx, cfa, pc, NULL));
                        
                        Function *function = frame_sp->GetSymbolContext (eSymbolContextFunction).function;
                        if (function)
                        {
                            // When we aren't showing inline functions we always use
                            // the top most function block as the scope.
                            frame_sp->SetSymbolContextScope (&function->GetBlock(false));
                        }
                        else 
                        {
                            // Set the symbol scope from the symbol regardless if it is NULL or valid.
                            frame_sp->SetSymbolContextScope (frame_sp->GetSymbolContext (eSymbolContextSymbol).symbol);
                        }
                        SetFrameAtIndex(idx, frame_sp);
                    }
                }
            }
        }
    return frame_sp;
}

StackFrameSP
StackFrameList::GetFrameWithConcreteFrameIndex (uint32_t unwind_idx)
{
    // First try assuming the unwind index is the same as the frame index. The 
    // unwind index is always greater than or equal to the frame index, so it
    // is a good place to start. If we have inlined frames we might have 5
    // concrete frames (frame unwind indexes go from 0-4), but we might have 15
    // frames after we make all the inlined frames. Most of the time the unwind
    // frame index (or the concrete frame index) is the same as the frame index.
    uint32_t frame_idx = unwind_idx;
    StackFrameSP frame_sp (GetFrameAtIndex (frame_idx));
    while (frame_sp)
    {
        if (frame_sp->GetFrameIndex() == unwind_idx)
            break;
        frame_sp = GetFrameAtIndex (++frame_idx);
    }
    return frame_sp;
}

StackFrameSP
StackFrameList::GetFrameWithStackID (const StackID &stack_id)
{
    uint32_t frame_idx = 0;
    StackFrameSP frame_sp;
    do
    {
        frame_sp = GetFrameAtIndex (frame_idx);
        if (frame_sp && frame_sp->GetStackID() == stack_id)
            break;
        frame_idx++;
    }
    while (frame_sp);
    return frame_sp;
}

bool
StackFrameList::SetFrameAtIndex (uint32_t idx, StackFrameSP &frame_sp)
{
    if (idx >= m_frames.size())
        m_frames.resize(idx + 1);
    // Make sure allocation succeeded by checking bounds again
    if (idx < m_frames.size())
    {
        m_frames[idx] = frame_sp;
        return true;
    }
    return false;   // resize failed, out of memory?
}

uint32_t
StackFrameList::GetSelectedFrameIndex () const
{
    Mutex::Locker locker (m_mutex);
    return m_selected_frame_idx;
}


uint32_t
StackFrameList::SetSelectedFrame (lldb_private::StackFrame *frame)
{
    Mutex::Locker locker (m_mutex);
    const_iterator pos;
    const_iterator begin = m_frames.begin();
    const_iterator end = m_frames.end();
    m_selected_frame_idx = 0;
    for (pos = begin; pos != end; ++pos)
    {
        if (pos->get() == frame)
        {
            m_selected_frame_idx = std::distance (begin, pos);
            uint32_t inlined_depth = GetCurrentInlinedDepth();
            if (inlined_depth != UINT32_MAX)
                m_selected_frame_idx -= inlined_depth;
            break;
        }
    }
    SetDefaultFileAndLineToSelectedFrame();
    return m_selected_frame_idx;
}

// Mark a stack frame as the current frame using the frame index
bool
StackFrameList::SetSelectedFrameByIndex (uint32_t idx)
{
    Mutex::Locker locker (m_mutex);
    StackFrameSP frame_sp (GetFrameAtIndex (idx));
    if (frame_sp)
    {
        SetSelectedFrame(frame_sp.get());
        return true;
    }
    else
        return false;
}

void
StackFrameList::SetDefaultFileAndLineToSelectedFrame()
{
    if (m_thread.GetID() == m_thread.GetProcess()->GetThreadList().GetSelectedThread()->GetID())
    {
        StackFrameSP frame_sp (GetFrameAtIndex (GetSelectedFrameIndex()));
        if (frame_sp)
        {
            SymbolContext sc = frame_sp->GetSymbolContext(eSymbolContextLineEntry);
            if (sc.line_entry.file)
                m_thread.CalculateTarget()->GetSourceManager().SetDefaultFileAndLine (sc.line_entry.file, 
                                                                                            sc.line_entry.line);
        }
    }
}

// The thread has been run, reset the number stack frames to zero so we can
// determine how many frames we have lazily.
void
StackFrameList::Clear ()
{
    Mutex::Locker locker (m_mutex);
    m_frames.clear();
    m_concrete_frames_fetched = 0;
}

void
StackFrameList::InvalidateFrames (uint32_t start_idx)
{
    Mutex::Locker locker (m_mutex);
    if (m_show_inlined_frames)
    {
        Clear();
    }
    else
    {
        const size_t num_frames = m_frames.size();
        while (start_idx < num_frames)
        {
            m_frames[start_idx].reset();
            ++start_idx;
        }
    }
}

void
StackFrameList::Merge (std::auto_ptr<StackFrameList>& curr_ap, lldb::StackFrameListSP& prev_sp)
{
    Mutex::Locker curr_locker (curr_ap.get() ? &curr_ap->m_mutex : NULL);
    Mutex::Locker prev_locker (prev_sp.get() ? &prev_sp->m_mutex : NULL);

#if defined (DEBUG_STACK_FRAMES)
    StreamFile s(stdout, false);
    s.PutCString("\n\nStackFrameList::Merge():\nPrev:\n");
    if (prev_sp.get())
        prev_sp->Dump (&s);
    else
        s.PutCString ("NULL");
    s.PutCString("\nCurr:\n");
    if (curr_ap.get())
        curr_ap->Dump (&s);
    else
        s.PutCString ("NULL");
    s.EOL();
#endif

    if (curr_ap.get() == NULL || curr_ap->GetNumFrames (false) == 0)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("No current frames, leave previous frames alone...\n");
#endif
        curr_ap.release();
        return;
    }

    if (prev_sp.get() == NULL || prev_sp->GetNumFrames (false) == 0)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("No previous frames, so use current frames...\n");
#endif
        // We either don't have any previous frames, or since we have more than
        // one current frames it means we have all the frames and can safely
        // replace our previous frames.
        prev_sp.reset (curr_ap.release());
        return;
    }

    const uint32_t num_curr_frames = curr_ap->GetNumFrames (false);
    
    if (num_curr_frames > 1)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("We have more than one current frame, so use current frames...\n");
#endif
        // We have more than one current frames it means we have all the frames 
        // and can safely replace our previous frames.
        prev_sp.reset (curr_ap.release());

#if defined (DEBUG_STACK_FRAMES)
        s.PutCString("\nMerged:\n");
        prev_sp->Dump (&s);
#endif
        return;
    }

    StackFrameSP prev_frame_zero_sp(prev_sp->GetFrameAtIndex (0));
    StackFrameSP curr_frame_zero_sp(curr_ap->GetFrameAtIndex (0));
    StackID curr_stack_id (curr_frame_zero_sp->GetStackID());
    StackID prev_stack_id (prev_frame_zero_sp->GetStackID());

#if defined (DEBUG_STACK_FRAMES)
    const uint32_t num_prev_frames = prev_sp->GetNumFrames (false);
    s.Printf("\n%u previous frames with one current frame\n", num_prev_frames);
#endif

    // We have only a single current frame
    // Our previous stack frames only had a single frame as well...
    if (curr_stack_id == prev_stack_id)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.Printf("\nPrevious frame #0 is same as current frame #0, merge the cached data\n");
#endif

        curr_frame_zero_sp->UpdateCurrentFrameFromPreviousFrame (*prev_frame_zero_sp);
//        prev_frame_zero_sp->UpdatePreviousFrameFromCurrentFrame (*curr_frame_zero_sp);
//        prev_sp->SetFrameAtIndex (0, prev_frame_zero_sp);
    }
    else if (curr_stack_id < prev_stack_id)
    {
#if defined (DEBUG_STACK_FRAMES)
        s.Printf("\nCurrent frame #0 has a stack ID that is less than the previous frame #0, insert current frame zero in front of previous\n");
#endif
        prev_sp->m_frames.insert (prev_sp->m_frames.begin(), curr_frame_zero_sp);
    }
    
    curr_ap.release();

#if defined (DEBUG_STACK_FRAMES)
    s.PutCString("\nMerged:\n");
    prev_sp->Dump (&s);
#endif


}

lldb::StackFrameSP
StackFrameList::GetStackFrameSPForStackFramePtr (StackFrame *stack_frame_ptr)
{
    const_iterator pos;
    const_iterator begin = m_frames.begin();
    const_iterator end = m_frames.end();
    lldb::StackFrameSP ret_sp;
    
    for (pos = begin; pos != end; ++pos)
    {
        if (pos->get() == stack_frame_ptr)
        {
            ret_sp = (*pos);
            break;
        }
    }
    return ret_sp;
}

size_t
StackFrameList::GetStatus (Stream& strm,
                           uint32_t first_frame,
                           uint32_t num_frames,
                           bool show_frame_info,
                           uint32_t num_frames_with_source)
{
    size_t num_frames_displayed = 0;
    
    if (num_frames == 0)
        return 0;
    
    StackFrameSP frame_sp;
    uint32_t frame_idx = 0;
    uint32_t last_frame;
    
    // Don't let the last frame wrap around...
    if (num_frames == UINT32_MAX)
        last_frame = UINT32_MAX;
    else
        last_frame = first_frame + num_frames;
    
    for (frame_idx = first_frame; frame_idx < last_frame; ++frame_idx)
    {
        frame_sp = GetFrameAtIndex(frame_idx);
        if (frame_sp.get() == NULL)
            break;
        
        if (!frame_sp->GetStatus (strm,
                                  show_frame_info,
                                  num_frames_with_source > (first_frame - frame_idx)))
            break;
        ++num_frames_displayed;
    }
    
    strm.IndentLess();
    return num_frames_displayed;
}

