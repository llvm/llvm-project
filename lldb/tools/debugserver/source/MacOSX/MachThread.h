//===-- MachThread.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/19/07.
//
//===----------------------------------------------------------------------===//

#ifndef __MachThread_h__
#define __MachThread_h__

#include <string>
#include <vector>

#include <libproc.h>
#include <mach/mach.h>
#include <pthread.h>
#include <sys/signal.h>

#include "PThreadCondition.h"
#include "PThreadMutex.h"
#include "MachException.h"
#include "DNBArch.h"
#include "DNBRegisterInfo.h"

class DNBBreakpoint;
class MachProcess;
class MachThreadList;

class MachThread
{
public:

                    MachThread (MachProcess *process, uint64_t unique_thread_id = 0, thread_t mach_port_number = 0);
                    ~MachThread ();

    MachProcess *   Process() { return m_process; }
    const MachProcess *
                    Process() const { return m_process; }
    nub_process_t   ProcessID() const;
    void            Dump(uint32_t index);
    uint64_t        ThreadID() const { return m_unique_id; }
    thread_t        MachPortNumber() const { return m_mach_port_number; }
    thread_t        InferiorThreadID() const;

    uint32_t        SequenceID() const { return m_seq_id; }
    static bool     ThreadIDIsValid(uint64_t thread);       // The 64-bit system-wide unique thread identifier
    static bool     MachPortNumberIsValid(thread_t thread); // The mach port # for this thread in debugserver namespace
    void            Resume(bool others_stopped);
    void            Suspend();
    bool            SetSuspendCountBeforeResume(bool others_stopped);
    bool            RestoreSuspendCountAfterStop();

    bool            GetRegisterState(int flavor, bool force);
    bool            SetRegisterState(int flavor);
    uint64_t        GetPC(uint64_t failValue = INVALID_NUB_ADDRESS);    // Get program counter
    bool            SetPC(uint64_t value);                              // Set program counter
    uint64_t        GetSP(uint64_t failValue = INVALID_NUB_ADDRESS);    // Get stack pointer

    nub_break_t     CurrentBreakpoint();
    uint32_t        EnableHardwareBreakpoint (const DNBBreakpoint *breakpoint);
    uint32_t        EnableHardwareWatchpoint (const DNBBreakpoint *watchpoint);
    bool            DisableHardwareBreakpoint (const DNBBreakpoint *breakpoint);
    bool            DisableHardwareWatchpoint (const DNBBreakpoint *watchpoint);
    uint32_t        NumSupportedHardwareWatchpoints () const;
    bool            RollbackTransForHWP();
    bool            FinishTransForHWP();

    nub_state_t     GetState();
    void            SetState(nub_state_t state);

    void            ThreadWillResume (const DNBThreadResumeAction *thread_action, bool others_stopped = false);
    bool            ShouldStop(bool &step_more);
    bool            IsStepping();
    bool            ThreadDidStop();
    bool            NotifyException(MachException::Data& exc);
    const MachException::Data& GetStopException() { return m_stop_exception; }

    uint32_t        GetNumRegistersInSet(int regSet) const;
    const char *    GetRegisterSetName(int regSet) const;
    const DNBRegisterInfo *
                    GetRegisterInfo(int regSet, int regIndex) const;
    void            DumpRegisterState(int regSet);
    const DNBRegisterSetInfo *
                    GetRegisterSetInfo(nub_size_t *num_reg_sets ) const;
    bool            GetRegisterValue ( uint32_t reg_set_idx, uint32_t reg_idx, DNBRegisterValue *reg_value );
    bool            SetRegisterValue ( uint32_t reg_set_idx, uint32_t reg_idx, const DNBRegisterValue *reg_value );
    nub_size_t      GetRegisterContext (void *buf, nub_size_t buf_len);
    nub_size_t      SetRegisterContext (const void *buf, nub_size_t buf_len);
    void            NotifyBreakpointChanged (const DNBBreakpoint *bp)
                    {
                    }

    bool            IsUserReady();
    struct thread_basic_info *
                    GetBasicInfo ();
    const char *    GetBasicInfoAsString () const;
    const char *    GetName ();
    
    DNBArchProtocol* 
    GetArchProtocol()
    {
        return m_arch_ap.get();
    }

    static uint64_t GetGloballyUniqueThreadIDForMachPortID (thread_t mach_port_id);

protected:
    static bool     GetBasicInfo(thread_t threadID, struct thread_basic_info *basic_info);

    bool
    GetIdentifierInfo ();

//    const char *
//    GetDispatchQueueName();
//
    MachProcess *                   m_process;      // The process that owns this thread
    uint64_t                        m_unique_id;    // The globally unique ID for this thread (nub_thread_t)
    thread_t                        m_mach_port_number;  // The mach port # for this thread in debugserver namesp.
    uint32_t                        m_seq_id;       // A Sequential ID that increments with each new thread
    nub_state_t                     m_state;        // The state of our process
    PThreadMutex                    m_state_mutex;  // Multithreaded protection for m_state
    nub_break_t                     m_break_id;     // Breakpoint that this thread is (stopped)/was(running) at (NULL for none)
    struct thread_basic_info        m_basic_info;   // Basic information for a thread used to see if a thread is valid
    int32_t                         m_suspend_count; // The current suspend count > 0 means we have suspended m_suspendCount times,
                                                    //                           < 0 means we have resumed it m_suspendCount times.
    MachException::Data             m_stop_exception; // The best exception that describes why this thread is stopped
    std::unique_ptr<DNBArchProtocol> m_arch_ap;      // Arch specific information for register state and more
    const DNBRegisterSetInfo *      m_reg_sets;      // Register set information for this thread
    nub_size_t                      m_num_reg_sets;
    thread_identifier_info_data_t   m_ident_info;
    struct proc_threadinfo          m_proc_threadinfo;
    std::string                     m_dispatch_queue_name;

private:
    friend class MachThreadList;
    void HardwareWatchpointStateChanged(); // Provide a chance to update the global view of the hardware watchpoint state
};

typedef std::shared_ptr<MachThread> MachThreadSP;

#endif
