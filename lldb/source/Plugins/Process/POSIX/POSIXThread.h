//===-- POSIXThread.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_POSIXThread_H_
#define liblldb_POSIXThread_H_

// C Includes
// C++ Includes
#include <memory>

// Other libraries and framework includes
#include "lldb/Target/Thread.h"
#include "RegisterContextPOSIX.h"

class ProcessMessage;
class ProcessMonitor;
class RegisterContextPOSIX;

//------------------------------------------------------------------------------
// @class POSIXThread
// @brief Abstraction of a linux process (thread).
class POSIXThread
    : public lldb_private::Thread
{
public:
    POSIXThread(lldb_private::Process &process, lldb::tid_t tid);

    virtual ~POSIXThread();

    void
    RefreshStateAfterStop();

    bool
    WillResume(lldb::StateType resume_state);

    const char *
    GetInfo();

    virtual lldb::RegisterContextSP
    GetRegisterContext();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    //--------------------------------------------------------------------------
    // These static functions provide a mapping from the register offset
    // back to the register index or name for use in debugging or log
    // output.

    static unsigned
    GetRegisterIndexFromOffset(unsigned offset);

    static const char *
    GetRegisterName(unsigned reg);

    static const char *
    GetRegisterNameFromOffset(unsigned offset);

    //--------------------------------------------------------------------------
    // These methods form a specialized interface to linux threads.
    //
    bool Resume();

    void Notify(const ProcessMessage &message);

private:
    RegisterContextPOSIX *
    GetRegisterContextPOSIX ()
    {
        if (!m_reg_context_sp)
            m_reg_context_sp = GetRegisterContext();
#if 0
        return dynamic_cast<RegisterContextPOSIX*>(m_reg_context_sp.get());
#endif
        return (RegisterContextPOSIX *)m_reg_context_sp.get();
    }
    
    std::auto_ptr<lldb_private::StackFrame> m_frame_ap;

    lldb::BreakpointSiteSP m_breakpoint;
    lldb::StopInfoSP m_stop_info;

    ProcessMonitor &
    GetMonitor();

    lldb::StopInfoSP
    GetPrivateStopReason();

    void BreakNotify(const ProcessMessage &message);
    void TraceNotify(const ProcessMessage &message);
    void LimboNotify(const ProcessMessage &message);
    void SignalNotify(const ProcessMessage &message);
    void SignalDeliveredNotify(const ProcessMessage &message);
    void CrashNotify(const ProcessMessage &message);

    lldb_private::Unwind *
    GetUnwinder();
};

#endif // #ifndef liblldb_POSIXThread_H_
