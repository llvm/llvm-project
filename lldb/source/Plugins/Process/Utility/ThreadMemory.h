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
    virtual void
    RefreshStateAfterStop();

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual lldb::StopInfoSP
    GetPrivateStopReason ();

    virtual const char *
    GetName ()
    {
        return m_name.c_str();
    }
    
    virtual const char *
    GetQueueName ()
    {
        return m_queue.c_str();
    }

    virtual bool
    WillResume (lldb::StateType resume_state);

    lldb::ValueObjectSP &
    GetValueObject ()
    {
        return m_thread_info_valobj_sp;
    }

protected:
    //------------------------------------------------------------------
    // For ThreadMemory and subclasses
    //------------------------------------------------------------------
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
