//===-- MachThreadList.h ----------------------------------------*- C++ -*-===//
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

#ifndef __MachThreadList_h__
#define __MachThreadList_h__

#include "MachThread.h"

class DNBThreadResumeActions;

class MachThreadList
{
public:
                    MachThreadList ();
                    ~MachThreadList ();

    void            Clear ();
    void            Dump () const;
    bool            GetRegisterValue (nub_thread_t tid, uint32_t reg_set_idx, uint32_t reg_idx, DNBRegisterValue *reg_value) const;
    bool            SetRegisterValue (nub_thread_t tid, uint32_t reg_set_idx, uint32_t reg_idx, const DNBRegisterValue *reg_value) const;
    nub_size_t      GetRegisterContext (nub_thread_t tid, void *buf, size_t buf_len);
    nub_size_t      SetRegisterContext (nub_thread_t tid, const void *buf, size_t buf_len);
    const char *    GetThreadInfo (nub_thread_t tid) const;
    void            ProcessWillResume (MachProcess *process, const DNBThreadResumeActions &thread_actions);
    uint32_t        ProcessDidStop (MachProcess *process);
    bool            NotifyException (MachException::Data& exc);
    bool            ShouldStop (bool &step_more);
    const char *    GetName (nub_thread_t tid);
    nub_state_t     GetState (nub_thread_t tid);
    nub_thread_t    SetCurrentThread (nub_thread_t tid);
    bool            GetThreadStoppedReason (nub_thread_t tid, struct DNBThreadStopInfo *stop_info) const;
    void            DumpThreadStoppedReason (nub_thread_t tid) const;
    bool            GetIdentifierInfo (nub_thread_t tid, thread_identifier_info_data_t *ident_info);
    nub_size_t      NumThreads () const;
    nub_thread_t    ThreadIDAtIndex (nub_size_t idx) const;
    nub_thread_t    CurrentThreadID ();
    void            CurrentThread (MachThreadSP& threadSP);
    void            NotifyBreakpointChanged (const DNBBreakpoint *bp);
    uint32_t        EnableHardwareBreakpoint (const DNBBreakpoint *bp) const;
    bool            DisableHardwareBreakpoint (const DNBBreakpoint *bp) const;
    uint32_t        EnableHardwareWatchpoint (const DNBBreakpoint *wp) const;
    bool            DisableHardwareWatchpoint (const DNBBreakpoint *wp) const;
    uint32_t        NumSupportedHardwareWatchpoints () const;

    uint32_t        GetThreadIndexForThreadStoppedWithSignal (const int signo) const;

    MachThreadSP    GetThreadByID (nub_thread_t tid) const;

    MachThreadSP    GetThreadByMachPortNumber (thread_t mach_port_number) const;
    nub_thread_t    GetThreadIDByMachPortNumber (thread_t mach_port_number) const;
    thread_t        GetMachPortNumberByThreadID (nub_thread_t globally_unique_id) const;

protected:
    typedef std::vector<MachThreadSP>   collection;
    typedef collection::iterator        iterator;
    typedef collection::const_iterator  const_iterator;

    uint32_t        UpdateThreadList (MachProcess *process, bool update, collection *num_threads = NULL);
//  const_iterator  FindThreadByID (thread_t tid) const;

    collection      m_threads;
    mutable PThreadMutex m_threads_mutex;
    MachThreadSP    m_current_thread;
};

#endif // #ifndef __MachThreadList_h__

