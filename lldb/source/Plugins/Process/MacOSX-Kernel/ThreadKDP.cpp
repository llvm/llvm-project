//===-- ThreadKDP.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ThreadKDP.h"

#include "llvm/Support/MachO.h"

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

#include "ProcessKDP.h"
#include "ProcessKDPLog.h"
#include "RegisterContextKDP_arm.h"
#include "RegisterContextKDP_i386.h"
#include "RegisterContextKDP_x86_64.h"
#include "Plugins/Process/Utility/StopInfoMachException.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Thread Registers
//----------------------------------------------------------------------

ThreadKDP::ThreadKDP (Process &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_thread_name (),
    m_dispatch_queue_name (),
    m_thread_dispatch_qaddr (LLDB_INVALID_ADDRESS)
{
    ProcessKDPLog::LogIf(KDP_LOG_THREAD, "%p: ThreadKDP::ThreadKDP (tid = 0x%4.4x)", this, GetID());
}

ThreadKDP::~ThreadKDP ()
{
    ProcessKDPLog::LogIf(KDP_LOG_THREAD, "%p: ThreadKDP::~ThreadKDP (tid = 0x%4.4x)", this, GetID());
    DestroyThread();
}

const char *
ThreadKDP::GetName ()
{
    if (m_thread_name.empty())
        return NULL;
    return m_thread_name.c_str();
}

const char *
ThreadKDP::GetQueueName ()
{
    return NULL;
}

bool
ThreadKDP::WillResume (StateType resume_state)
{
    // Call the Thread::WillResume first. If we stop at a signal, the stop info
    // class for signal will set the resume signal that we need below. The signal
    // stuff obeys the Process::UnixSignal defaults. 
    Thread::WillResume(resume_state);

    ClearStackFrames();

    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf ("Resuming thread: %4.4llx with state: %s.", GetID(), StateAsCString(resume_state));

    return true;
}

void
ThreadKDP::RefreshStateAfterStop()
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
ThreadKDP::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}


bool
ThreadKDP::ThreadIDIsValid (lldb::tid_t thread)
{
    return thread != 0;
}

void
ThreadKDP::Dump(Log *log, uint32_t index)
{
}


bool
ThreadKDP::ShouldStop (bool &step_more)
{
    return true;
}
lldb::RegisterContextSP
ThreadKDP::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL)
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadKDP::CreateRegisterContextForFrame (StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();

    if (concrete_frame_idx == 0)
    {
        ProcessSP process_sp (CalculateProcess());
        if (process_sp)
        {
            switch (static_cast<ProcessKDP *>(process_sp.get())->GetCommunication().GetCPUType())
            {
                case llvm::MachO::CPUTypeARM:
                    reg_ctx_sp.reset (new RegisterContextKDP_arm (*this, concrete_frame_idx));
                    break;
                case llvm::MachO::CPUTypeI386:
                    reg_ctx_sp.reset (new RegisterContextKDP_i386 (*this, concrete_frame_idx));
                    break;
                case llvm::MachO::CPUTypeX86_64:
                    reg_ctx_sp.reset (new RegisterContextKDP_x86_64 (*this, concrete_frame_idx));
                    break;
                default:
                    assert (!"Add CPU type support in KDP");
                    break;
            }
        }
    }
    else if (m_unwinder_ap.get())
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    return reg_ctx_sp;
}

lldb::StopInfoSP
ThreadKDP::GetPrivateStopReason ()
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

            if (m_cached_stop_info_sp)
                SetStopInfo (m_cached_stop_info_sp);
            else
                SetStopInfo(StopInfo::CreateStopReasonWithSignal (*this, SIGSTOP));
        }
    }
    return m_actual_stop_info_sp;
}

void
ThreadKDP::SetStopInfoFrom_KDP_EXCEPTION (const DataExtractor &exc_reply_packet)
{
    uint32_t offset = 0;
    uint8_t reply_command = exc_reply_packet.GetU8(&offset);
    if (reply_command == CommunicationKDP::KDP_EXCEPTION)
    {
        offset = 8;
        const uint32_t count = exc_reply_packet.GetU32 (&offset);
        if (count >= 1)
        {
            //const uint32_t cpu = exc_reply_packet.GetU32 (&offset);
            offset += 4; // Skip the useless CPU field
            const uint32_t exc_type = exc_reply_packet.GetU32 (&offset);
            const uint32_t exc_code = exc_reply_packet.GetU32 (&offset);
            const uint32_t exc_subcode = exc_reply_packet.GetU32 (&offset);
            // We have to make a copy of the stop info because the thread list
            // will iterate through the threads and clear all stop infos..
            
            // Let the StopInfoMachException::CreateStopReasonWithMachException()
            // function update the PC if needed as we might hit a software breakpoint
            // and need to decrement the PC (i386 and x86_64 need this) and KDP
            // doesn't do this for us.
            const bool pc_already_adjusted = false;
            const bool adjust_pc_if_needed = true;

            m_cached_stop_info_sp = StopInfoMachException::CreateStopReasonWithMachException (*this,
                                                                                              exc_type,
                                                                                              2,
                                                                                              exc_code,
                                                                                              exc_subcode,
                                                                                              0,
                                                                                              pc_already_adjusted,
                                                                                              adjust_pc_if_needed);            
        }
    }
}

