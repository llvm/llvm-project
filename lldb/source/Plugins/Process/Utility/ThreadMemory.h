//===-- ThreadMemory.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadMemory_h_
#define liblldb_ThreadMemory_h_

#include "lldb/Target/Thread.h"

class ThreadMemory :
    public lldb_private::Thread
{
public:

    ThreadMemory (lldb_private::Process &process,
                  lldb::tid_t tid,
                  const lldb::ValueObjectSP &thread_info_valobj_sp);

    ThreadMemory (lldb_private::Process &process,
                  lldb::tid_t tid,
                  const char *name,
                  const char *queue,
                  lldb::addr_t register_data_addr);

    virtual
    ~ThreadMemory();

    //------------------------------------------------------------------
    // lldb_private::Thread methods
    //------------------------------------------------------------------
    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual bool
    CalculateStopInfo ();

    virtual const char *
    GetInfo ()
    {
        if (m_backing_thread_sp)
            m_backing_thread_sp->GetInfo();
        return NULL;
    }

    virtual const char *
    GetName ()
    {
        if (!m_name.empty())
            return m_name.c_str();
        if (m_backing_thread_sp)
            m_backing_thread_sp->GetName();
        return NULL;
    }
    
    virtual const char *
    GetQueueName ()
    {
        if (!m_queue.empty())
            return m_queue.c_str();
        if (m_backing_thread_sp)
            m_backing_thread_sp->GetQueueName();
        return NULL;
    }

    virtual void
    WillResume (lldb::StateType resume_state);

    virtual void
    DidResume ()
    {
        if (m_backing_thread_sp)
            m_backing_thread_sp->DidResume();
    }
    
    virtual lldb::user_id_t
    GetProtocolID () const
    {
        if (m_backing_thread_sp)
            return m_backing_thread_sp->GetProtocolID();
        return Thread::GetProtocolID();
    }

    virtual void
    RefreshStateAfterStop();
    
    lldb::ValueObjectSP &
    GetValueObject ()
    {
        return m_thread_info_valobj_sp;
    }
    
    virtual void
    ClearStackFrames ();

    virtual void
    ClearBackingThread ()
    {
        m_backing_thread_sp.reset();
    }

    virtual bool
    SetBackingThread (const lldb::ThreadSP &thread_sp)
    {
        //printf ("Thread 0x%llx is being backed by thread 0x%llx\n", GetID(), thread_sp->GetID());
        m_backing_thread_sp = thread_sp;
        return (bool)thread_sp;
    }
    
    virtual lldb::ThreadSP
    GetBackingThread () const
    {
        return m_backing_thread_sp;
    }

protected:
    
    virtual bool
    IsOperatingSystemPluginThread () const
    {
        return true;
    }
    

    //------------------------------------------------------------------
    // For ThreadMemory and subclasses
    //------------------------------------------------------------------
    // If this memory thread is actually represented by a thread from the
    // lldb_private::Process subclass, then fill in the thread here and
    // all APIs will be routed through this thread object. If m_backing_thread_sp
    // is empty, then this thread is simply in memory with no representation
    // through the process plug-in.
    lldb::ThreadSP m_backing_thread_sp;
    lldb::ValueObjectSP m_thread_info_valobj_sp;
    std::string m_name;
    std::string m_queue;
    lldb::addr_t m_register_data_addr;
private:
    //------------------------------------------------------------------
    // For ThreadMemory only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ThreadMemory);
};

#endif  // liblldb_ThreadMemory_h_
