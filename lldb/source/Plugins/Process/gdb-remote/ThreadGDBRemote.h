//===-- ThreadGDBRemote.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadGDBRemote_h_
#define liblldb_ThreadGDBRemote_h_

#include <string>

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

class StringExtractor;
class ProcessGDBRemote;

class ThreadGDBRemote : public lldb_private::Thread
{
public:
    ThreadGDBRemote (lldb_private::Process &process, lldb::tid_t tid);

    virtual
    ~ThreadGDBRemote ();

    virtual bool
    WillResume (lldb::StateType resume_state);

    virtual void
    RefreshStateAfterStop();

    virtual const char *
    GetName ();

    virtual const char *
    GetQueueName ();

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual void
    ClearStackFrames ();

    void
    Dump (lldb_private::Log *log, uint32_t index);

    static bool
    ThreadIDIsValid (lldb::tid_t thread);

    bool
    ShouldStop (bool &step_more);

    const char *
    GetBasicInfoAsString ();

    void
    SetName (const char *name)
    {
        if (name && name[0])
            m_thread_name.assign (name);
        else
            m_thread_name.clear();
    }

    lldb::addr_t
    GetThreadDispatchQAddr ()
    {
        return m_thread_dispatch_qaddr;
    }

    void
    SetThreadDispatchQAddr (lldb::addr_t thread_dispatch_qaddr)
    {
        m_thread_dispatch_qaddr = thread_dispatch_qaddr;
    }

protected:
    
    friend class ProcessGDBRemote;

    bool
    PrivateSetRegisterValue (uint32_t reg, 
                             StringExtractor &response);
                             
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    std::string m_thread_name;
    std::string m_dispatch_queue_name;
    lldb::addr_t m_thread_dispatch_qaddr;
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------

    void
    SetStopInfoFromPacket (StringExtractor &stop_packet, uint32_t stop_id);

    virtual lldb::StopInfoSP
    GetPrivateStopReason ();


};

#endif  // liblldb_ThreadGDBRemote_h_
