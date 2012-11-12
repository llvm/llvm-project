//===-- ThreadMachCore.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ThreadMachCore.h"

#include "llvm/Support/MachO.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/State.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Breakpoint/Watchpoint.h"

#include "ProcessMachCore.h"
//#include "RegisterContextKDP_arm.h"
//#include "RegisterContextKDP_i386.h"
//#include "RegisterContextKDP_x86_64.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Thread Registers
//----------------------------------------------------------------------

ThreadMachCore::ThreadMachCore (Process &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_thread_name (),
    m_dispatch_queue_name (),
    m_thread_dispatch_qaddr (LLDB_INVALID_ADDRESS),
    m_thread_reg_ctx_sp ()
{
}

ThreadMachCore::~ThreadMachCore ()
{
    DestroyThread();
}

const char *
ThreadMachCore::GetName ()
{
    if (m_thread_name.empty())
        return NULL;
    return m_thread_name.c_str();
}

void
ThreadMachCore::RefreshStateAfterStop()
{
    // Invalidate all registers in our register context. We don't set "force" to
    // true because the stop reply packet might have had some register values
    // that were expedited and these will already be copied into the register
    // context by the time this function gets called. The KDPRegisterContext
    // class has been made smart enough to detect when it needs to invalidate
    // which registers are valid by putting hooks in the register read and 
    // register supply functions where they check the process stop ID and do
    // the right thing.
    const bool force = false;
    GetRegisterContext()->InvalidateIfNeeded (force);
}

void
ThreadMachCore::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}


bool
ThreadMachCore::ThreadIDIsValid (lldb::tid_t thread)
{
    return thread != 0;
}

lldb::RegisterContextSP
ThreadMachCore::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL)
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadMachCore::CreateRegisterContextForFrame (StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();

    if (concrete_frame_idx == 0)
    {
        if (!m_thread_reg_ctx_sp)
        {
            ProcessSP process_sp (GetProcess());
            
            ObjectFile *core_objfile = static_cast<ProcessMachCore *>(process_sp.get())->GetCoreObjectFile ();
            if (core_objfile)
                m_thread_reg_ctx_sp = core_objfile->GetThreadContextAtIndex (GetID(), *this);
        }
        reg_ctx_sp = m_thread_reg_ctx_sp;
    }
    else if (m_unwinder_ap.get())
    {
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    }
    return reg_ctx_sp;
}

lldb::StopInfoSP
ThreadMachCore::GetPrivateStopReason ()
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

            // TODO: can we query the initial state of the thread here?
            // For now I am just going to pretend that a SIGSTOP happened.

            SetStopInfo(StopInfo::CreateStopReasonWithSignal (*this, SIGSTOP));

            // If GetKDPProcess().SetThreadStopInfo() doesn't find a stop reason
            // for this thread, then m_actual_stop_info_sp will not ever contain
            // a valid stop reason and the "m_actual_stop_info_sp->IsValid() == false"
            // check will never be able to tell us if we have the correct stop info
            // for this thread and we will continually send qThreadStopInfo packets
            // down to the remote KDP server, so we need to keep our own notion
            // of the stop ID that m_actual_stop_info_sp is valid for (even if it
            // contains nothing). We use m_thread_stop_reason_stop_id for this below.
    //        m_thread_stop_reason_stop_id = process_stop_id;
    //        m_actual_stop_info_sp.reset();

        }
    }
    return m_actual_stop_info_sp;
}


