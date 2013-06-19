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

    virtual void
    WillResume(lldb::StateType resume_state);

    // This notifies the thread when a private stop occurs.
    virtual void
    DidStop ();

    const char *
    GetInfo();

    virtual lldb::RegisterContextSP
    GetRegisterContext();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    //--------------------------------------------------------------------------
    // These functions provide a mapping from the register offset
    // back to the register index or name for use in debugging or log
    // output.

    unsigned
    GetRegisterIndexFromOffset(unsigned offset);

    const char *
    GetRegisterName(unsigned reg);

    const char *
    GetRegisterNameFromOffset(unsigned offset);

    //--------------------------------------------------------------------------
    // These methods form a specialized interface to linux threads.
    //
    bool Resume();

    void Notify(const ProcessMessage &message);

    //--------------------------------------------------------------------------
    // These methods provide an interface to watchpoints
    //
    bool EnableHardwareWatchpoint(lldb_private::Watchpoint *wp);

    bool DisableHardwareWatchpoint(lldb_private::Watchpoint *wp);

    uint32_t NumSupportedHardwareWatchpoints();

    uint32_t FindVacantWatchpointIndex();

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
    
    std::unique_ptr<lldb_private::StackFrame> m_frame_ap;

    lldb::BreakpointSiteSP m_breakpoint;

    ProcessMonitor &
    GetMonitor();

    virtual bool
    CalculateStopInfo();

    void BreakNotify(const ProcessMessage &message);
    void WatchNotify(const ProcessMessage &message);
    void TraceNotify(const ProcessMessage &message);
    void LimboNotify(const ProcessMessage &message);
    void SignalNotify(const ProcessMessage &message);
    void SignalDeliveredNotify(const ProcessMessage &message);
    void CrashNotify(const ProcessMessage &message);
    void ThreadNotify(const ProcessMessage &message);
    void ExitNotify(const ProcessMessage &message);

    lldb_private::Unwind *
    GetUnwinder();
};

#endif // #ifndef liblldb_POSIXThread_H_
