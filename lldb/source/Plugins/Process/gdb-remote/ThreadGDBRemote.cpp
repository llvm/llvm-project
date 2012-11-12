//===-- ThreadGDBRemote.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ThreadGDBRemote.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/State.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Breakpoint/Watchpoint.h"

#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "Utility/StringExtractorGDBRemote.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Thread Registers
//----------------------------------------------------------------------

ThreadGDBRemote::ThreadGDBRemote (Process &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_thread_name (),
    m_dispatch_queue_name (),
    m_thread_dispatch_qaddr (LLDB_INVALID_ADDRESS)
{
    ProcessGDBRemoteLog::LogIf(GDBR_LOG_THREAD, "%p: ThreadGDBRemote::ThreadGDBRemote (pid = %i, tid = 0x%4.4x)", 
                               this, 
                               process.GetID(),
                               GetID());
}

ThreadGDBRemote::~ThreadGDBRemote ()
{
    ProcessSP process_sp(GetProcess());
    ProcessGDBRemoteLog::LogIf(GDBR_LOG_THREAD, "%p: ThreadGDBRemote::~ThreadGDBRemote (pid = %i, tid = 0x%4.4x)", 
                               this, 
                               process_sp ? process_sp->GetID() : LLDB_INVALID_PROCESS_ID, 
                               GetID());
    DestroyThread();
}

const char *
ThreadGDBRemote::GetName ()
{
    if (m_thread_name.empty())
        return NULL;
    return m_thread_name.c_str();
}


const char *
ThreadGDBRemote::GetQueueName ()
{
    // Always re-fetch the dispatch queue name since it can change

    if (m_thread_dispatch_qaddr != 0 || m_thread_dispatch_qaddr != LLDB_INVALID_ADDRESS)
    {
        ProcessSP process_sp (GetProcess());
        if (process_sp)
        {
            ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(process_sp.get());
            return gdb_process->GetDispatchQueueNameForThread (m_thread_dispatch_qaddr, m_dispatch_queue_name);
        }
    }
    return NULL;
}

bool
ThreadGDBRemote::WillResume (StateType resume_state)
{
    // Call the Thread::WillResume first. If we stop at a signal, the stop info
    // class for signal will set the resume signal that we need below. The signal
    // stuff obeys the Process::UnixSignal defaults. 
    // If the thread's WillResume returns false, that means that we aren't going to actually resume,
    // in which case we should not do the rest of our "resume" work.
    
    if (!Thread::WillResume(resume_state))
        return false;
    
    ClearStackFrames();

    int signo = GetResumeSignal();
    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (GDBR_LOG_THREAD));
    if (log)
        log->Printf ("Resuming thread: %4.4llx with state: %s.", GetID(), StateAsCString(resume_state));

    ProcessSP process_sp (GetProcess());
    if (process_sp)
    {
        ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(process_sp.get());
        switch (resume_state)
        {
        case eStateSuspended:
        case eStateStopped:
            // Don't append anything for threads that should stay stopped.
            break;

        case eStateRunning:
            if (gdb_process->GetUnixSignals().SignalIsValid (signo))
                gdb_process->m_continue_C_tids.push_back(std::make_pair(GetID(), signo));
            else
                gdb_process->m_continue_c_tids.push_back(GetID());
            break;

        case eStateStepping:
            if (gdb_process->GetUnixSignals().SignalIsValid (signo))
                gdb_process->m_continue_S_tids.push_back(std::make_pair(GetID(), signo));
            else
                gdb_process->m_continue_s_tids.push_back(GetID());
            break;

        default:
            break;
        }
        return true;
    }
    return false;
}

void
ThreadGDBRemote::RefreshStateAfterStop()
{
    // Invalidate all registers in our register context. We don't set "force" to
    // true because the stop reply packet might have had some register values
    // that were expedited and these will already be copied into the register
    // context by the time this function gets called. The GDBRemoteRegisterContext
    // class has been made smart enough to detect when it needs to invalidate
    // which registers are valid by putting hooks in the register read and 
    // register supply functions where they check the process stop ID and do
    // the right thing.
    const bool force = false;
    GetRegisterContext()->InvalidateIfNeeded (force);
}

void
ThreadGDBRemote::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}


bool
ThreadGDBRemote::ThreadIDIsValid (lldb::tid_t thread)
{
    return thread != 0;
}

void
ThreadGDBRemote::Dump(Log *log, uint32_t index)
{
}


bool
ThreadGDBRemote::ShouldStop (bool &step_more)
{
    return true;
}
lldb::RegisterContextSP
ThreadGDBRemote::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL)
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadGDBRemote::CreateRegisterContextForFrame (StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    const bool read_all_registers_at_once = false;
    uint32_t concrete_frame_idx = 0;
    
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();

    
    if (concrete_frame_idx == 0)
    {
        ProcessSP process_sp (GetProcess());
        if (process_sp)
        {
            ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(process_sp.get());
            reg_ctx_sp.reset (new GDBRemoteRegisterContext (*this, concrete_frame_idx, gdb_process->m_register_info, read_all_registers_at_once));
        }
    }
    else if (m_unwinder_ap.get())
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    return reg_ctx_sp;
}

bool
ThreadGDBRemote::PrivateSetRegisterValue (uint32_t reg, StringExtractor &response)
{
    GDBRemoteRegisterContext *gdb_reg_ctx = static_cast<GDBRemoteRegisterContext *>(GetRegisterContext ().get());
    assert (gdb_reg_ctx);
    return gdb_reg_ctx->PrivateSetRegisterValue (reg, response);
}

lldb::StopInfoSP
ThreadGDBRemote::GetPrivateStopReason ()
{
    ProcessSP process_sp (GetProcess());
    if (process_sp)
    {
        const uint32_t process_stop_id = process_sp->GetStopID();
        if (m_thread_stop_reason_stop_id != process_stop_id ||
            (m_actual_stop_info_sp && !m_actual_stop_info_sp->IsValid()))
        {
            if (IsStillAtLastBreakpointHit())
                return m_actual_stop_info_sp;

            // If GetGDBProcess().SetThreadStopInfo() doesn't find a stop reason
            // for this thread, then m_actual_stop_info_sp will not ever contain
            // a valid stop reason and the "m_actual_stop_info_sp->IsValid() == false"
            // check will never be able to tell us if we have the correct stop info
            // for this thread and we will continually send qThreadStopInfo packets
            // down to the remote GDB server, so we need to keep our own notion
            // of the stop ID that m_actual_stop_info_sp is valid for (even if it
            // contains nothing). We use m_thread_stop_reason_stop_id for this below.
            m_thread_stop_reason_stop_id = process_stop_id;
            m_actual_stop_info_sp.reset();

            StringExtractorGDBRemote stop_packet;
            ProcessGDBRemote *gdb_process = static_cast<ProcessGDBRemote *>(process_sp.get());
            if (gdb_process->GetGDBRemote().GetThreadStopInfo(GetID(), stop_packet))
                gdb_process->SetThreadStopInfo (stop_packet);
        }
    }
    return m_actual_stop_info_sp;
}


