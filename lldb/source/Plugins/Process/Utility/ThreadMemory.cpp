//===-- ThreadMemory.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Utility/ThreadMemory.h"
#include "lldb/Target/OperatingSystem.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Unwind.h"

using namespace lldb;
using namespace lldb_private;

ThreadMemory::ThreadMemory (Process &process,
                            tid_t tid,
                            const ValueObjectSP &thread_info_valobj_sp) :
    Thread (process, tid),
    m_thread_info_valobj_sp (thread_info_valobj_sp),
    m_name(),
    m_queue()
{
}


ThreadMemory::ThreadMemory (Process &process,
                            lldb::tid_t tid,
                            const char *name,
                            const char *queue,
                            lldb::addr_t register_data_addr) :
    Thread (process, tid),
    m_thread_info_valobj_sp (),
    m_name(),
    m_queue(),
    m_register_data_addr (register_data_addr)
{
    if (name)
        m_name = name;
    if (queue)
        m_queue = queue;
}


ThreadMemory::~ThreadMemory()
{
    DestroyThread();
}

bool
ThreadMemory::WillResume (StateType resume_state)
{
    ClearStackFrames();
    // Call the Thread::WillResume first. If we stop at a signal, the stop info
    // class for signal will set the resume signal that we need below. The signal
    // stuff obeys the Process::UnixSignal defaults. 
    Thread::WillResume(resume_state);
    return true;
}

RegisterContextSP
ThreadMemory::GetRegisterContext ()
{
    if (!m_reg_context_sp)
    {
        ProcessSP process_sp (GetProcess());
        if (process_sp)
        {
            OperatingSystem *os = process_sp->GetOperatingSystem ();
            if (os)
                m_reg_context_sp = os->CreateRegisterContextForThread (this, m_register_data_addr);
        }
    }
    return m_reg_context_sp;
}

RegisterContextSP
ThreadMemory::CreateRegisterContextForFrame (StackFrame *frame)
{
    RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();
    
    if (concrete_frame_idx == 0)
    {
        reg_ctx_sp = GetRegisterContext ();
    }
    else if (m_unwinder_ap.get())
    {
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    }
    return reg_ctx_sp;
}

lldb::StopInfoSP
ThreadMemory::GetPrivateStopReason ()
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
            
            OperatingSystem *os = process_sp->GetOperatingSystem ();
            if (os)
                m_actual_stop_info_sp = os->CreateThreadStopReason (this);
        }
    }
    return m_actual_stop_info_sp;
    
}

void
ThreadMemory::RefreshStateAfterStop()
{
    RegisterContextSP reg_ctx_sp(GetRegisterContext());
    if (reg_ctx_sp)
    {
        const bool force = true;
        reg_ctx_sp->InvalidateIfNeeded (force);
    }
}
