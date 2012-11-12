//===-- UnwindLLDB.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/Log.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"

#include "UnwindLLDB.h"
#include "RegisterContextLLDB.h"

using namespace lldb;
using namespace lldb_private;

UnwindLLDB::UnwindLLDB (Thread &thread) :
    Unwind (thread),
    m_frames(),
    m_unwind_complete(false)
{
}

uint32_t
UnwindLLDB::DoGetFrameCount()
{
    if (!m_unwind_complete)
    {
//#define DEBUG_FRAME_SPEED 1
#if DEBUG_FRAME_SPEED
#define FRAME_COUNT 10000
        TimeValue time_value (TimeValue::Now());
#endif
        if (!AddFirstFrame ())
            return 0;

        ProcessSP process_sp (m_thread.GetProcess());
        ABI *abi = process_sp ? process_sp->GetABI().get() : NULL;

        while (AddOneMoreFrame (abi))
        {
#if DEBUG_FRAME_SPEED
            if ((m_frames.size() % FRAME_COUNT) == 0)
            {
                TimeValue now(TimeValue::Now());
                uint64_t delta_t = now - time_value;
                printf ("%u frames in %llu.%09llu ms (%g frames/sec)\n", 
                        FRAME_COUNT,
                        delta_t / TimeValue::NanoSecPerSec, 
                        delta_t % TimeValue::NanoSecPerSec,
                        (float)FRAME_COUNT / ((float)delta_t / (float)TimeValue::NanoSecPerSec));
                time_value = now;
            }
#endif
        }
    }
    return m_frames.size ();
}

bool
UnwindLLDB::AddFirstFrame ()
{
    if (m_frames.size() > 0)
        return true;
        
    // First, set up the 0th (initial) frame
    CursorSP first_cursor_sp(new Cursor ());
    RegisterContextLLDBSP reg_ctx_sp (new RegisterContextLLDB (m_thread, 
                                                               RegisterContextLLDBSP(), 
                                                               first_cursor_sp->sctx, 
                                                               0, *this));
    if (reg_ctx_sp.get() == NULL)
        goto unwind_done;
    
    if (!reg_ctx_sp->IsValid())
        goto unwind_done;

    if (!reg_ctx_sp->GetCFA (first_cursor_sp->cfa))
        goto unwind_done;

    if (!reg_ctx_sp->ReadPC (first_cursor_sp->start_pc))
        goto unwind_done;

    // Everything checks out, so release the auto pointer value and let the
    // cursor own it in its shared pointer
    first_cursor_sp->reg_ctx_lldb_sp = reg_ctx_sp;
    m_frames.push_back (first_cursor_sp);
    return true;
unwind_done:
    m_unwind_complete = true;
    return false;
}

// For adding a non-zero stack frame to m_frames.
bool
UnwindLLDB::AddOneMoreFrame (ABI *abi)
{
    // If we've already gotten to the end of the stack, don't bother to try again...
    if (m_unwind_complete)
        return false;
        
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    CursorSP cursor_sp(new Cursor ());

    // Frame zero is a little different
    if (m_frames.size() == 0)
        return false;

    uint32_t cur_idx = m_frames.size ();
    RegisterContextLLDBSP reg_ctx_sp(new RegisterContextLLDB (m_thread, 
                                                              m_frames[cur_idx - 1]->reg_ctx_lldb_sp, 
                                                              cursor_sp->sctx, 
                                                              cur_idx, 
                                                              *this));
    if (reg_ctx_sp.get() == NULL)
        goto unwind_done;

    if (!reg_ctx_sp->IsValid())
    {
        if (log)
        {
            log->Printf("%*sFrame %d invalid RegisterContext for this frame, stopping stack walk", 
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        goto unwind_done;
    }
    if (!reg_ctx_sp->GetCFA (cursor_sp->cfa))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get CFA for this frame, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        goto unwind_done;
    }
    if (abi && !abi->CallFrameAddressIsValid(cursor_sp->cfa))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get a valid CFA for this frame, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        goto unwind_done;
    }
    if (!reg_ctx_sp->ReadPC (cursor_sp->start_pc))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get PC for this frame, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        goto unwind_done;
    }
    if (abi && !abi->CodeAddressIsValid (cursor_sp->start_pc))
    {
        if (log)
        {
            log->Printf("%*sFrame %d did not get a valid PC, stopping stack walk",
                        cur_idx < 100 ? cur_idx : 100, "", cur_idx);
        }
        goto unwind_done;
    }
    if (!m_frames.empty())
    {
        if (m_frames.back()->start_pc == cursor_sp->start_pc)
        {
            if (m_frames.back()->cfa == cursor_sp->cfa)
                goto unwind_done; // Infinite loop where the current cursor is the same as the previous one...
            else if (abi && abi->StackUsesFrames())
            {
                // We might have a CFA that is not using the frame pointer and
                // we want to validate that the frame pointer is valid.
                if (reg_ctx_sp->GetFP() == 0)
                    goto unwind_done;
            }
        }
    }
    cursor_sp->reg_ctx_lldb_sp = reg_ctx_sp;
    m_frames.push_back (cursor_sp);
    return true;
    
unwind_done:
    m_unwind_complete = true;
    return false;
}

bool
UnwindLLDB::DoGetFrameInfoAtIndex (uint32_t idx, addr_t& cfa, addr_t& pc)
{
    if (m_frames.size() == 0)
    {
        if (!AddFirstFrame())
            return false;
    }

    ProcessSP process_sp (m_thread.GetProcess());
    ABI *abi = process_sp ? process_sp->GetABI().get() : NULL;

    while (idx >= m_frames.size() && AddOneMoreFrame (abi))
        ;

    if (idx < m_frames.size ())
    {
        cfa = m_frames[idx]->cfa;
        pc = m_frames[idx]->start_pc;
        return true;
    }
    return false;
}

lldb::RegisterContextSP
UnwindLLDB::DoCreateRegisterContextForFrame (StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t idx = frame->GetConcreteFrameIndex ();

    if (idx == 0)
    {
        return m_thread.GetRegisterContext();
    }

    if (m_frames.size() == 0)
    {
        if (!AddFirstFrame())
            return reg_ctx_sp;
    }

    ProcessSP process_sp (m_thread.GetProcess());
    ABI *abi = process_sp ? process_sp->GetABI().get() : NULL;

    while (idx >= m_frames.size())
    {
        if (!AddOneMoreFrame (abi))
            break;
    }

    const uint32_t num_frames = m_frames.size();
    if (idx < num_frames)
    {
        Cursor *frame_cursor = m_frames[idx].get();
        reg_ctx_sp = frame_cursor->reg_ctx_lldb_sp;
    }
    return reg_ctx_sp;
}

UnwindLLDB::RegisterContextLLDBSP
UnwindLLDB::GetRegisterContextForFrameNum (uint32_t frame_num)
{
    RegisterContextLLDBSP reg_ctx_sp;
    if (frame_num < m_frames.size())
        reg_ctx_sp = m_frames[frame_num]->reg_ctx_lldb_sp;
    return reg_ctx_sp;
}

bool
UnwindLLDB::SearchForSavedLocationForRegister (uint32_t lldb_regnum, lldb_private::UnwindLLDB::RegisterLocation &regloc, uint32_t starting_frame_num, bool pc_or_return_address_reg)
{
    int64_t frame_num = starting_frame_num;
    if (frame_num >= m_frames.size())
        return false;

    // Never interrogate more than one level while looking for the saved pc value.  If the value
    // isn't saved by frame_num, none of the frames lower on the stack will have a useful value.
    if (pc_or_return_address_reg)
    {
        if (m_frames[frame_num]->reg_ctx_lldb_sp->SavedLocationForRegister (lldb_regnum, regloc))
          return true;
        else
          return false;
    }
    while (frame_num >= 0)
    {
        if (m_frames[frame_num]->reg_ctx_lldb_sp->SavedLocationForRegister (lldb_regnum, regloc))
            return true;
        frame_num--;
    }
    return false;
}
