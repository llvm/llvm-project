//===-- MachProcess.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/15/07.
//
//===----------------------------------------------------------------------===//

#include "DNB.h"
#include <inttypes.h>
#include <mach/mach.h>
#include <signal.h>
#include <spawn.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include "MacOSX/CFUtils.h"
#include "SysSignal.h"

#include <algorithm>
#include <map>

#include "DNBDataRef.h"
#include "DNBLog.h"
#include "DNBThreadResumeActions.h"
#include "DNBTimer.h"
#include "MachProcess.h"
#include "PseudoTerminal.h"

#include "CFBundle.h"
#include "CFData.h"
#include "CFString.h"

static CFStringRef CopyBundleIDForPath (const char *app_buncle_path, DNBError &err_str);

#ifdef WITH_SPRINGBOARD

#include <CoreFoundation/CoreFoundation.h>
#include <SpringBoardServices/SpringBoardServer.h>
#include <SpringBoardServices/SBSWatchdogAssertion.h>

static bool
IsSBProcess (nub_process_t pid)
{
    CFReleaser<CFArrayRef> appIdsForPID (::SBSCopyDisplayIdentifiersForProcessID(pid));
    return appIdsForPID.get() != NULL;
}

#endif

#if 0
#define DEBUG_LOG(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

#ifndef MACH_PROCESS_USE_POSIX_SPAWN
#define MACH_PROCESS_USE_POSIX_SPAWN 1
#endif

#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR       0x0100
#endif

MachProcess::MachProcess() :
    m_pid               (0),
    m_cpu_type          (0),
    m_child_stdin       (-1),
    m_child_stdout      (-1),
    m_child_stderr      (-1),
    m_path              (),
    m_args              (),
    m_task              (this),
    m_flags             (eMachProcessFlagsNone),
    m_stdio_thread      (0),
    m_stdio_mutex       (PTHREAD_MUTEX_RECURSIVE),
    m_stdout_data       (),
    m_thread_actions    (),
    m_profile_enabled   (false),
    m_profile_interval_usec (0),
    m_profile_thread    (0),
    m_profile_data_mutex(PTHREAD_MUTEX_RECURSIVE),
    m_profile_data      (),
    m_thread_list        (),
    m_exception_messages (),
    m_exception_messages_mutex (PTHREAD_MUTEX_RECURSIVE),
    m_state             (eStateUnloaded),
    m_state_mutex       (PTHREAD_MUTEX_RECURSIVE),
    m_events            (0, kAllEventsMask),
    m_breakpoints       (),
    m_watchpoints       (),
    m_name_to_addr_callback(NULL),
    m_name_to_addr_baton(NULL),
    m_image_infos_callback(NULL),
    m_image_infos_baton(NULL)
{
    DNBLogThreadedIf(LOG_PROCESS | LOG_VERBOSE, "%s", __PRETTY_FUNCTION__);
}

MachProcess::~MachProcess()
{
    DNBLogThreadedIf(LOG_PROCESS | LOG_VERBOSE, "%s", __PRETTY_FUNCTION__);
    Clear();
}

pid_t
MachProcess::SetProcessID(pid_t pid)
{
    // Free any previous process specific data or resources
    Clear();
    // Set the current PID appropriately
    if (pid == 0)
        m_pid = ::getpid ();
    else
        m_pid = pid;
    return m_pid;    // Return actualy PID in case a zero pid was passed in
}

nub_state_t
MachProcess::GetState()
{
    // If any other threads access this we will need a mutex for it
    PTHREAD_MUTEX_LOCKER(locker, m_state_mutex);
    return m_state;
}

const char *
MachProcess::ThreadGetName(nub_thread_t tid)
{
    return m_thread_list.GetName(tid);
}

nub_state_t
MachProcess::ThreadGetState(nub_thread_t tid)
{
    return m_thread_list.GetState(tid);
}


nub_size_t
MachProcess::GetNumThreads () const
{
    return m_thread_list.NumThreads();
}

nub_thread_t
MachProcess::GetThreadAtIndex (nub_size_t thread_idx) const
{
    return m_thread_list.ThreadIDAtIndex(thread_idx);
}

nub_thread_t
MachProcess::GetThreadIDForMachPortNumber (thread_t mach_port_number) const
{
    return m_thread_list.GetThreadIDByMachPortNumber (mach_port_number);
}

nub_bool_t
MachProcess::SyncThreadState (nub_thread_t tid)
{
    MachThreadSP thread_sp(m_thread_list.GetThreadByID(tid));
    if (!thread_sp)
        return false;
    kern_return_t kret = ::thread_abort_safely(thread_sp->MachPortNumber());
    DNBLogThreadedIf (LOG_THREAD, "thread = 0x%8.8" PRIx32 " calling thread_abort_safely (tid) => %u (GetGPRState() for stop_count = %u)", thread_sp->MachPortNumber(), kret, thread_sp->Process()->StopCount());

    if (kret == KERN_SUCCESS)
        return true;
    else
        return false;
    
}

nub_thread_t
MachProcess::GetCurrentThread ()
{
    return m_thread_list.CurrentThreadID();
}

nub_thread_t
MachProcess::GetCurrentThreadMachPort ()
{
    return m_thread_list.GetMachPortNumberByThreadID(m_thread_list.CurrentThreadID());
}

nub_thread_t
MachProcess::SetCurrentThread(nub_thread_t tid)
{
    return m_thread_list.SetCurrentThread(tid);
}

bool
MachProcess::GetThreadStoppedReason(nub_thread_t tid, struct DNBThreadStopInfo *stop_info) const
{
    return m_thread_list.GetThreadStoppedReason(tid, stop_info);
}

void
MachProcess::DumpThreadStoppedReason(nub_thread_t tid) const
{
    return m_thread_list.DumpThreadStoppedReason(tid);
}

const char *
MachProcess::GetThreadInfo(nub_thread_t tid) const
{
    return m_thread_list.GetThreadInfo(tid);
}

uint32_t
MachProcess::GetCPUType ()
{
    if (m_cpu_type == 0 && m_pid != 0)
        m_cpu_type = MachProcess::GetCPUTypeForLocalProcess (m_pid);
    return m_cpu_type;
}

const DNBRegisterSetInfo *
MachProcess::GetRegisterSetInfo (nub_thread_t tid, nub_size_t *num_reg_sets) const
{
    MachThreadSP thread_sp (m_thread_list.GetThreadByID (tid));
    if (thread_sp)
    {
        DNBArchProtocol *arch = thread_sp->GetArchProtocol();
        if (arch)
            return arch->GetRegisterSetInfo (num_reg_sets);
    }
    *num_reg_sets = 0;
    return NULL;
}

bool
MachProcess::GetRegisterValue ( nub_thread_t tid, uint32_t set, uint32_t reg, DNBRegisterValue *value ) const
{
    return m_thread_list.GetRegisterValue(tid, set, reg, value);
}

bool
MachProcess::SetRegisterValue ( nub_thread_t tid, uint32_t set, uint32_t reg, const DNBRegisterValue *value ) const
{
    return m_thread_list.SetRegisterValue(tid, set, reg, value);
}

void
MachProcess::SetState(nub_state_t new_state)
{
    // If any other threads access this we will need a mutex for it
    uint32_t event_mask = 0;

    // Scope for mutex locker
    {
        PTHREAD_MUTEX_LOCKER(locker, m_state_mutex);
        DNBLogThreadedIf(LOG_PROCESS, "MachProcess::SetState ( %s )", DNBStateAsString(new_state));

        const nub_state_t old_state = m_state;

        if (old_state != new_state)
        {
            if (NUB_STATE_IS_STOPPED(new_state))
                event_mask = eEventProcessStoppedStateChanged;
            else
                event_mask = eEventProcessRunningStateChanged;

            m_state = new_state;
            if (new_state == eStateStopped)
                m_stop_count++;
        }
    }

    if (event_mask != 0)
    {
        m_events.SetEvents (event_mask);

        // Wait for the event bit to reset if a reset ACK is requested
        m_events.WaitForResetAck(event_mask);
    }

}

void
MachProcess::Clear()
{
    // Clear any cached thread list while the pid and task are still valid

    m_task.Clear();
    // Now clear out all member variables
    m_pid = INVALID_NUB_PROCESS;
    CloseChildFileDescriptors();
    m_path.clear();
    m_args.clear();
    SetState(eStateUnloaded);
    m_flags = eMachProcessFlagsNone;
    m_stop_count = 0;
    m_thread_list.Clear();
    {
        PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);
        m_exception_messages.clear();
    }
    if (m_profile_thread)
    {
        pthread_join(m_profile_thread, NULL);
        m_profile_thread = NULL;
    }
}


bool
MachProcess::StartSTDIOThread()
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( )", __FUNCTION__);
    // Create the thread that watches for the child STDIO
    return ::pthread_create (&m_stdio_thread, NULL, MachProcess::STDIOThread, this) == 0;
}

void
MachProcess::SetEnableAsyncProfiling(bool enable, uint64_t interval_usec, DNBProfileDataScanType scan_type)
{
    m_profile_enabled = enable;
    m_profile_interval_usec = interval_usec;
    m_profile_scan_type = scan_type;
    
    if (m_profile_enabled && (m_profile_thread == NULL))
    {
        StartProfileThread();
    }
    else if (!m_profile_enabled && m_profile_thread)
    {
        pthread_join(m_profile_thread, NULL);
        m_profile_thread = NULL;
    }
}

bool
MachProcess::StartProfileThread()
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( )", __FUNCTION__);
    // Create the thread that profiles the inferior and reports back if enabled
    return ::pthread_create (&m_profile_thread, NULL, MachProcess::ProfileThread, this) == 0;
}


nub_addr_t
MachProcess::LookupSymbol(const char *name, const char *shlib)
{
    if (m_name_to_addr_callback != NULL && name && name[0])
        return m_name_to_addr_callback(ProcessID(), name, shlib, m_name_to_addr_baton);
    return INVALID_NUB_ADDRESS;
}

bool
MachProcess::Resume (const DNBThreadResumeActions& thread_actions)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Resume ()");
    nub_state_t state = GetState();

    if (CanResume(state))
    {
        m_thread_actions = thread_actions;
        PrivateResume();
        return true;
    }
    else if (state == eStateRunning)
    {
        DNBLog("Resume() - task 0x%x is already running, ignoring...", m_task.TaskPort());
        return true;
    }
    DNBLog("Resume() - task 0x%x has state %s, can't continue...", m_task.TaskPort(), DNBStateAsString(state));
    return false;
}

bool
MachProcess::Kill (const struct timespec *timeout_abstime)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Kill ()");
    nub_state_t state = DoSIGSTOP(true, false, NULL);
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Kill() DoSIGSTOP() state = %s", DNBStateAsString(state));
    errno = 0;
    DNBLog ("Sending ptrace PT_KILL to terminate inferior process.");
    ::ptrace (PT_KILL, m_pid, 0, 0);
    DNBError err;
    err.SetErrorToErrno();
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Kill() DoSIGSTOP() ::ptrace (PT_KILL, pid=%u, 0, 0) => 0x%8.8x (%s)", m_pid, err.Error(), err.AsString());
    m_thread_actions = DNBThreadResumeActions (eStateRunning, 0);
    PrivateResume ();
    
    // Try and reap the process without touching our m_events since
    // we want the code above this to still get the eStateExited event
    const uint32_t reap_timeout_usec = 1000000;    // Wait 1 second and try to reap the process
    const uint32_t reap_interval_usec = 10000;  //
    uint32_t reap_time_elapsed;
    for (reap_time_elapsed = 0;
         reap_time_elapsed < reap_timeout_usec;
         reap_time_elapsed += reap_interval_usec)
    {
        if (GetState() == eStateExited)
            break;
        usleep(reap_interval_usec);
    }
    DNBLog ("Waited %u ms for process to be reaped (state = %s)", reap_time_elapsed/1000, DNBStateAsString(GetState()));
    return true;
}

bool
MachProcess::Signal (int signal, const struct timespec *timeout_abstime)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Signal (signal = %d, timeout = %p)", signal, timeout_abstime);
    nub_state_t state = GetState();
    if (::kill (ProcessID(), signal) == 0)
    {
        // If we were running and we have a timeout, wait for the signal to stop
        if (IsRunning(state) && timeout_abstime)
        {
            DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Signal (signal = %d, timeout = %p) waiting for signal to stop process...", signal, timeout_abstime);
            m_events.WaitForSetEvents(eEventProcessStoppedStateChanged, timeout_abstime);
            state = GetState();
            DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Signal (signal = %d, timeout = %p) state = %s", signal, timeout_abstime, DNBStateAsString(state));
            return !IsRunning (state);
        }
        DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Signal (signal = %d, timeout = %p) not waiting...", signal, timeout_abstime);
        return true;
    }
    DNBError err(errno, DNBError::POSIX);
    err.LogThreadedIfError("kill (pid = %d, signo = %i)", ProcessID(), signal);
    return false;

}

nub_state_t
MachProcess::DoSIGSTOP (bool clear_bps_and_wps, bool allow_running, uint32_t *thread_idx_ptr)
{
    nub_state_t state = GetState();
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::DoSIGSTOP() state = %s", DNBStateAsString (state));

    if (!IsRunning(state))
    {
        if (clear_bps_and_wps)
        {
            DisableAllBreakpoints (true);
            DisableAllWatchpoints (true);
            clear_bps_and_wps = false;
        }

        // If we already have a thread stopped due to a SIGSTOP, we don't have
        // to do anything...
        uint32_t thread_idx = m_thread_list.GetThreadIndexForThreadStoppedWithSignal (SIGSTOP);
        if (thread_idx_ptr)
            *thread_idx_ptr = thread_idx;
        if (thread_idx != UINT32_MAX)
            return GetState();

        // No threads were stopped with a SIGSTOP, we need to run and halt the
        // process with a signal
        DNBLogThreadedIf(LOG_PROCESS, "MachProcess::DoSIGSTOP() state = %s -- resuming process", DNBStateAsString (state));
        if (allow_running)
            m_thread_actions = DNBThreadResumeActions (eStateRunning, 0);
        else
            m_thread_actions = DNBThreadResumeActions (eStateSuspended, 0);
            
        PrivateResume ();

        // Reset the event that says we were indeed running
        m_events.ResetEvents(eEventProcessRunningStateChanged);
        state = GetState();
    }

    // We need to be stopped in order to be able to detach, so we need
    // to send ourselves a SIGSTOP

    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::DoSIGSTOP() state = %s -- sending SIGSTOP", DNBStateAsString (state));
    struct timespec sigstop_timeout;
    DNBTimer::OffsetTimeOfDay(&sigstop_timeout, 2, 0);
    Signal (SIGSTOP, &sigstop_timeout);
    if (clear_bps_and_wps)
    {
        DisableAllBreakpoints (true);
        DisableAllWatchpoints (true);
        //clear_bps_and_wps = false;
    }
    uint32_t thread_idx = m_thread_list.GetThreadIndexForThreadStoppedWithSignal (SIGSTOP);
    if (thread_idx_ptr)
        *thread_idx_ptr = thread_idx;
    return GetState();
}

bool
MachProcess::Detach()
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Detach()");

    uint32_t thread_idx = UINT32_MAX;
    nub_state_t state = DoSIGSTOP(true, true, &thread_idx);
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Detach() DoSIGSTOP() returned %s", DNBStateAsString(state));

    {
        m_thread_actions.Clear();
        DNBThreadResumeAction thread_action;
        thread_action.tid = m_thread_list.ThreadIDAtIndex (thread_idx);
        thread_action.state = eStateRunning;
        thread_action.signal = -1;
        thread_action.addr = INVALID_NUB_ADDRESS;
        
        m_thread_actions.Append (thread_action);
        m_thread_actions.SetDefaultThreadActionIfNeeded (eStateRunning, 0);
        
        PTHREAD_MUTEX_LOCKER (locker, m_exception_messages_mutex);

        ReplyToAllExceptions ();

    }

    m_task.ShutDownExcecptionThread();

    // Detach from our process
    errno = 0;
    nub_process_t pid = m_pid;
    int ret = ::ptrace (PT_DETACH, pid, (caddr_t)1, 0);
    DNBError err(errno, DNBError::POSIX);
    if (DNBLogCheckLogBit(LOG_PROCESS) || err.Fail() || (ret != 0))
        err.LogThreaded("::ptrace (PT_DETACH, %u, (caddr_t)1, 0)", pid);

    // Resume our task
    m_task.Resume();

    // NULL our task out as we have already retored all exception ports
    m_task.Clear();

    // Clear out any notion of the process we once were
    Clear();

    SetState(eStateDetached);

    return true;
}

nub_size_t
MachProcess::RemoveTrapsFromBuffer (nub_addr_t addr, nub_size_t size, uint8_t *buf) const
{
    nub_size_t bytes_removed = 0;
    const DNBBreakpoint *bp;
    nub_addr_t intersect_addr;
    nub_size_t intersect_size;
    nub_size_t opcode_offset;
    nub_size_t idx;
    for (idx = 0; (bp = m_breakpoints.GetByIndex(idx)) != NULL; ++idx)
    {
        if (bp->IntersectsRange(addr, size, &intersect_addr, &intersect_size, &opcode_offset))
        {
            assert(addr <= intersect_addr && intersect_addr < addr + size);
            assert(addr < intersect_addr + intersect_size && intersect_addr + intersect_size <= addr + size);
            assert(opcode_offset + intersect_size <= bp->ByteSize());
            nub_size_t buf_offset = intersect_addr - addr;
            ::memcpy(buf + buf_offset, bp->SavedOpcodeBytes() + opcode_offset, intersect_size);
        }
    }
    return bytes_removed;
}

//----------------------------------------------------------------------
// ReadMemory from the MachProcess level will always remove any software
// breakpoints from the memory buffer before returning. If you wish to
// read memory and see those traps, read from the MachTask
// (m_task.ReadMemory()) as that version will give you what is actually
// in inferior memory.
//----------------------------------------------------------------------
nub_size_t
MachProcess::ReadMemory (nub_addr_t addr, nub_size_t size, void *buf)
{
    // We need to remove any current software traps (enabled software
    // breakpoints) that we may have placed in our tasks memory.

    // First just read the memory as is
    nub_size_t bytes_read = m_task.ReadMemory(addr, size, buf);

    // Then place any opcodes that fall into this range back into the buffer
    // before we return this to callers.
    if (bytes_read > 0)
        RemoveTrapsFromBuffer (addr, size, (uint8_t *)buf);
    return bytes_read;
}

//----------------------------------------------------------------------
// WriteMemory from the MachProcess level will always write memory around
// any software breakpoints. Any software breakpoints will have their
// opcodes modified if they are enabled. Any memory that doesn't overlap
// with software breakpoints will be written to. If you wish to write to
// inferior memory without this interference, then write to the MachTask
// (m_task.WriteMemory()) as that version will always modify inferior
// memory.
//----------------------------------------------------------------------
nub_size_t
MachProcess::WriteMemory (nub_addr_t addr, nub_size_t size, const void *buf)
{
    // We need to write any data that would go where any current software traps
    // (enabled software breakpoints) any software traps (breakpoints) that we
    // may have placed in our tasks memory.

    std::map<nub_addr_t, DNBBreakpoint *> addr_to_bp_map;
    DNBBreakpoint *bp;
    nub_size_t idx;
    for (idx = 0; (bp = m_breakpoints.GetByIndex(idx)) != NULL; ++idx)
    {
        if (bp->IntersectsRange(addr, size, NULL, NULL, NULL))
            addr_to_bp_map[bp->Address()] = bp;
    }

    // If we don't have any software breakpoints that are in this buffer, then
    // we can just write memory and be done with it.
    if (addr_to_bp_map.empty())
        return m_task.WriteMemory(addr, size, buf);

    // If we make it here, we have some breakpoints that overlap and we need
    // to work around them.

    nub_size_t bytes_written = 0;
    nub_addr_t intersect_addr;
    nub_size_t intersect_size;
    nub_size_t opcode_offset;
    const uint8_t *ubuf = (const uint8_t *)buf;
    std::map<nub_addr_t, DNBBreakpoint *>::iterator pos, end = addr_to_bp_map.end();
    for (pos = addr_to_bp_map.begin(); pos != end; ++pos)
    {
        bp = pos->second;

        assert(bp->IntersectsRange(addr, size, &intersect_addr, &intersect_size, &opcode_offset));
        assert(addr <= intersect_addr && intersect_addr < addr + size);
        assert(addr < intersect_addr + intersect_size && intersect_addr + intersect_size <= addr + size);
        assert(opcode_offset + intersect_size <= bp->ByteSize());

        // Check for bytes before this breakpoint
        const nub_addr_t curr_addr = addr + bytes_written;
        if (intersect_addr > curr_addr)
        {
            // There are some bytes before this breakpoint that we need to
            // just write to memory
            nub_size_t curr_size = intersect_addr - curr_addr;
            nub_size_t curr_bytes_written = m_task.WriteMemory(curr_addr, curr_size, ubuf + bytes_written);
            bytes_written += curr_bytes_written;
            if (curr_bytes_written != curr_size)
            {
                // We weren't able to write all of the requested bytes, we
                // are done looping and will return the number of bytes that
                // we have written so far.
                break;
            }
        }

        // Now write any bytes that would cover up any software breakpoints
        // directly into the breakpoint opcode buffer
        ::memcpy(bp->SavedOpcodeBytes() + opcode_offset, ubuf + bytes_written, intersect_size);
        bytes_written += intersect_size;
    }

    // Write any remaining bytes after the last breakpoint if we have any left
    if (bytes_written < size)
        bytes_written += m_task.WriteMemory(addr + bytes_written, size - bytes_written, ubuf + bytes_written);

    return bytes_written;
}

void
MachProcess::ReplyToAllExceptions ()
{
    PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);
    if (m_exception_messages.empty() == false)
    {
        MachException::Message::iterator pos;
        MachException::Message::iterator begin = m_exception_messages.begin();
        MachException::Message::iterator end = m_exception_messages.end();
        for (pos = begin; pos != end; ++pos)
        {
            DNBLogThreadedIf(LOG_EXCEPTIONS, "Replying to exception %u...", (uint32_t)std::distance(begin, pos));
            int thread_reply_signal = 0;

            nub_thread_t tid = m_thread_list.GetThreadIDByMachPortNumber (pos->state.thread_port);
            const DNBThreadResumeAction *action = NULL;
            if (tid != INVALID_NUB_THREAD)
            {
                action = m_thread_actions.GetActionForThread (tid, false);
            }

            if (action)
            {
                thread_reply_signal = action->signal;
                if (thread_reply_signal)
                    m_thread_actions.SetSignalHandledForThread (tid);
            }

            DNBError err (pos->Reply(this, thread_reply_signal));
            if (DNBLogCheckLogBit(LOG_EXCEPTIONS))
                err.LogThreadedIfError("Error replying to exception");
        }

        // Erase all exception message as we should have used and replied
        // to them all already.
        m_exception_messages.clear();
    }
}
void
MachProcess::PrivateResume ()
{
    PTHREAD_MUTEX_LOCKER (locker, m_exception_messages_mutex);

    ReplyToAllExceptions ();
//    bool stepOverBreakInstruction = step;

    // Let the thread prepare to resume and see if any threads want us to
    // step over a breakpoint instruction (ProcessWillResume will modify
    // the value of stepOverBreakInstruction).
    m_thread_list.ProcessWillResume (this, m_thread_actions);

    // Set our state accordingly
    if (m_thread_actions.NumActionsWithState(eStateStepping))
        SetState (eStateStepping);
    else
        SetState (eStateRunning);

    // Now resume our task.
    m_task.Resume();
}

nub_break_t
MachProcess::CreateBreakpoint(nub_addr_t addr, nub_size_t length, bool hardware, thread_t tid)
{
    DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::CreateBreakpoint ( addr = 0x%8.8llx, length = %llu, hardware = %i, tid = 0x%4.4x )", (uint64_t)addr, (uint64_t)length, hardware, tid);
    if (hardware && tid == INVALID_NUB_THREAD)
        tid = GetCurrentThread();

    DNBBreakpoint bp(addr, length, tid, hardware);
    nub_break_t breakID = m_breakpoints.Add(bp);
    if (EnableBreakpoint(breakID))
    {
        DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::CreateBreakpoint ( addr = 0x%8.8llx, length = %llu, tid = 0x%4.4x ) => %u", (uint64_t)addr, (uint64_t)length, tid, breakID);
        return breakID;
    }
    else
    {
        m_breakpoints.Remove(breakID);
    }
    // We failed to enable the breakpoint
    return INVALID_NUB_BREAK_ID;
}

nub_watch_t
MachProcess::CreateWatchpoint(nub_addr_t addr, nub_size_t length, uint32_t watch_flags, bool hardware, thread_t tid)
{
    DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::CreateWatchpoint ( addr = 0x%8.8llx, length = %llu, flags = 0x%8.8x, hardware = %i, tid = 0x%4.4x )", (uint64_t)addr, (uint64_t)length, watch_flags, hardware, tid);
    if (hardware && tid == INVALID_NUB_THREAD)
        tid = GetCurrentThread();

    DNBBreakpoint watch(addr, length, tid, hardware);
    watch.SetIsWatchpoint(watch_flags);

    nub_watch_t watchID = m_watchpoints.Add(watch);
    if (EnableWatchpoint(watchID))
    {
        DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::CreateWatchpoint ( addr = 0x%8.8llx, length = %llu, tid = 0x%x) => %u", (uint64_t)addr, (uint64_t)length, tid, watchID);
        return watchID;
    }
    else
    {
        DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::CreateWatchpoint ( addr = 0x%8.8llx, length = %llu, tid = 0x%x) => FAILED (%u)", (uint64_t)addr, (uint64_t)length, tid, watchID);
        m_watchpoints.Remove(watchID);
    }
    // We failed to enable the watchpoint
    return INVALID_NUB_BREAK_ID;
}

nub_size_t
MachProcess::DisableAllBreakpoints(bool remove)
{
    DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::%s (remove = %d )", __FUNCTION__, remove);
    DNBBreakpoint *bp;
    nub_size_t disabled_count = 0;
    nub_size_t idx = 0;
    while ((bp = m_breakpoints.GetByIndex(idx)) != NULL)
    {
        bool success = DisableBreakpoint(bp->GetID(), remove);

        if (success)
            disabled_count++;
        // If we failed to disable the breakpoint or we aren't removing the breakpoint
        // increment the breakpoint index. Otherwise DisableBreakpoint will have removed
        // the breakpoint at this index and we don't need to change it.
        if ((success == false) || (remove == false))
            idx++;
    }
    return disabled_count;
}

nub_size_t
MachProcess::DisableAllWatchpoints(bool remove)
{
    DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::%s (remove = %d )", __FUNCTION__, remove);
    DNBBreakpoint *wp;
    nub_size_t disabled_count = 0;
    nub_size_t idx = 0;
    while ((wp = m_watchpoints.GetByIndex(idx)) != NULL)
    {
        bool success = DisableWatchpoint(wp->GetID(), remove);

        if (success)
            disabled_count++;
        // If we failed to disable the watchpoint or we aren't removing the watchpoint
        // increment the watchpoint index. Otherwise DisableWatchpoint will have removed
        // the watchpoint at this index and we don't need to change it.
        if ((success == false) || (remove == false))
            idx++;
    }
    return disabled_count;
}

bool
MachProcess::DisableBreakpoint(nub_break_t breakID, bool remove)
{
    DNBBreakpoint *bp = m_breakpoints.FindByID (breakID);
    if (bp)
    {
        nub_addr_t addr = bp->Address();
        DNBLogThreadedIf(LOG_BREAKPOINTS | LOG_VERBOSE, "MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx", breakID, remove, (uint64_t)addr);

        if (bp->IsHardware())
        {
            bool hw_disable_result = m_thread_list.DisableHardwareBreakpoint (bp);

            if (hw_disable_result == true)
            {
                bp->SetEnabled(false);
                // Let the thread list know that a breakpoint has been modified
                if (remove)
                {
                    m_thread_list.NotifyBreakpointChanged(bp);
                    m_breakpoints.Remove(breakID);
                }
                DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx (hardware) => success", breakID, remove, (uint64_t)addr);
                return true;
            }

            return false;
        }

        const nub_size_t break_op_size = bp->ByteSize();
        assert (break_op_size > 0);
        const uint8_t * const break_op = DNBArchProtocol::GetBreakpointOpcode (bp->ByteSize());
        if (break_op_size > 0)
        {
            // Clear a software breakoint instruction
            uint8_t curr_break_op[break_op_size];
            bool break_op_found = false;

            // Read the breakpoint opcode
            if (m_task.ReadMemory(addr, break_op_size, curr_break_op) == break_op_size)
            {
                bool verify = false;
                if (bp->IsEnabled())
                {
                    // Make sure we have the a breakpoint opcode exists at this address
                    if (memcmp(curr_break_op, break_op, break_op_size) == 0)
                    {
                        break_op_found = true;
                        // We found a valid breakpoint opcode at this address, now restore
                        // the saved opcode.
                        if (m_task.WriteMemory(addr, break_op_size, bp->SavedOpcodeBytes()) == break_op_size)
                        {
                            verify = true;
                        }
                        else
                        {
                            DNBLogError("MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx memory write failed when restoring original opcode", breakID, remove, (uint64_t)addr);
                        }
                    }
                    else
                    {
                        DNBLogWarning("MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx expected a breakpoint opcode but didn't find one.", breakID, remove, (uint64_t)addr);
                        // Set verify to true and so we can check if the original opcode has already been restored
                        verify = true;
                    }
                }
                else
                {
                    DNBLogThreadedIf(LOG_BREAKPOINTS | LOG_VERBOSE, "MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx is not enabled", breakID, remove, (uint64_t)addr);
                    // Set verify to true and so we can check if the original opcode is there
                    verify = true;
                }

                if (verify)
                {
                    uint8_t verify_opcode[break_op_size];
                    // Verify that our original opcode made it back to the inferior
                    if (m_task.ReadMemory(addr, break_op_size, verify_opcode) == break_op_size)
                    {
                        // compare the memory we just read with the original opcode
                        if (memcmp(bp->SavedOpcodeBytes(), verify_opcode, break_op_size) == 0)
                        {
                            // SUCCESS
                            bp->SetEnabled(false);
                            // Let the thread list know that a breakpoint has been modified
                            if (remove)
                            {
                                m_thread_list.NotifyBreakpointChanged(bp);
                                m_breakpoints.Remove(breakID);
                            }
                            DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx => success", breakID, remove, (uint64_t)addr);
                            return true;
                        }
                        else
                        {
                            if (break_op_found)
                                DNBLogError("MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx: failed to restore original opcode", breakID, remove, (uint64_t)addr);
                            else
                                DNBLogError("MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) addr = 0x%8.8llx: opcode changed", breakID, remove, (uint64_t)addr);
                        }
                    }
                    else
                    {
                        DNBLogWarning("MachProcess::DisableBreakpoint: unable to disable breakpoint 0x%8.8llx", (uint64_t)addr);
                    }
                }
            }
            else
            {
                DNBLogWarning("MachProcess::DisableBreakpoint: unable to read memory at 0x%8.8llx", (uint64_t)addr);
            }
        }
    }
    else
    {
        DNBLogError("MachProcess::DisableBreakpoint ( breakID = %d, remove = %d ) invalid breakpoint ID", breakID, remove);
    }
    return false;
}

bool
MachProcess::DisableWatchpoint(nub_watch_t watchID, bool remove)
{
    DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::%s(watchID = %d, remove = %d)", __FUNCTION__, watchID, remove);
    DNBBreakpoint *wp = m_watchpoints.FindByID (watchID);
    if (wp)
    {
        nub_addr_t addr = wp->Address();
        DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::DisableWatchpoint ( watchID = %d, remove = %d ) addr = 0x%8.8llx", watchID, remove, (uint64_t)addr);

        if (wp->IsHardware())
        {
            bool hw_disable_result = m_thread_list.DisableHardwareWatchpoint (wp);

            if (hw_disable_result == true)
            {
                wp->SetEnabled(false);
                if (remove)
                    m_watchpoints.Remove(watchID);
                DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::Disablewatchpoint ( watchID = %d, remove = %d ) addr = 0x%8.8llx (hardware) => success", watchID, remove, (uint64_t)addr);
                return true;
            }
        }

        // TODO: clear software watchpoints if we implement them
    }
    else
    {
        DNBLogError("MachProcess::DisableWatchpoint ( watchID = %d, remove = %d ) invalid watchpoint ID", watchID, remove);
    }
    return false;
}


void
MachProcess::DumpBreakpoint(nub_break_t breakID) const
{
    DNBLogThreaded("MachProcess::DumpBreakpoint(breakID = %d)", breakID);

    if (NUB_BREAK_ID_IS_VALID(breakID))
    {
        const DNBBreakpoint *bp = m_breakpoints.FindByID(breakID);
        if (bp)
            bp->Dump();
        else
            DNBLog("MachProcess::DumpBreakpoint(breakID = %d): invalid breakID", breakID);
    }
    else
    {
        m_breakpoints.Dump();
    }
}

void
MachProcess::DumpWatchpoint(nub_watch_t watchID) const
{
    DNBLogThreaded("MachProcess::DumpWatchpoint(watchID = %d)", watchID);

    if (NUB_BREAK_ID_IS_VALID(watchID))
    {
        const DNBBreakpoint *wp = m_watchpoints.FindByID(watchID);
        if (wp)
            wp->Dump();
        else
            DNBLog("MachProcess::DumpWatchpoint(watchID = %d): invalid watchID", watchID);
    }
    else
    {
        m_watchpoints.Dump();
    }
}

uint32_t
MachProcess::GetNumSupportedHardwareWatchpoints () const
{
    return m_thread_list.NumSupportedHardwareWatchpoints();
}

bool
MachProcess::EnableBreakpoint(nub_break_t breakID)
{
    DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::EnableBreakpoint ( breakID = %d )", breakID);
    DNBBreakpoint *bp = m_breakpoints.FindByID (breakID);
    if (bp)
    {
        nub_addr_t addr = bp->Address();
        if (bp->IsEnabled())
        {
            DNBLogWarning("MachProcess::EnableBreakpoint ( breakID = %d ) addr = 0x%8.8llx: breakpoint already enabled.", breakID, (uint64_t)addr);
            return true;
        }
        else
        {
            if (bp->HardwarePreferred())
            {
                bp->SetHardwareIndex(m_thread_list.EnableHardwareBreakpoint(bp));
                if (bp->IsHardware())
                {
                    bp->SetEnabled(true);
                    return true;
                }
            }

            const nub_size_t break_op_size = bp->ByteSize();
            assert (break_op_size != 0);
            const uint8_t * const break_op = DNBArchProtocol::GetBreakpointOpcode (break_op_size);
            if (break_op_size > 0)
            {
                // Save the original opcode by reading it
                if (m_task.ReadMemory(addr, break_op_size, bp->SavedOpcodeBytes()) == break_op_size)
                {
                    // Write a software breakpoint in place of the original opcode
                    if (m_task.WriteMemory(addr, break_op_size, break_op) == break_op_size)
                    {
                        uint8_t verify_break_op[4];
                        if (m_task.ReadMemory(addr, break_op_size, verify_break_op) == break_op_size)
                        {
                            if (memcmp(break_op, verify_break_op, break_op_size) == 0)
                            {
                                bp->SetEnabled(true);
                                // Let the thread list know that a breakpoint has been modified
                                m_thread_list.NotifyBreakpointChanged(bp);
                                DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::EnableBreakpoint ( breakID = %d ) addr = 0x%8.8llx: SUCCESS.", breakID, (uint64_t)addr);
                                return true;
                            }
                            else
                            {
                                DNBLogError("MachProcess::EnableBreakpoint ( breakID = %d ) addr = 0x%8.8llx: breakpoint opcode verification failed.", breakID, (uint64_t)addr);
                            }
                        }
                        else
                        {
                            DNBLogError("MachProcess::EnableBreakpoint ( breakID = %d ) addr = 0x%8.8llx: unable to read memory to verify breakpoint opcode.", breakID, (uint64_t)addr);
                        }
                    }
                    else
                    {
                        DNBLogError("MachProcess::EnableBreakpoint ( breakID = %d ) addr = 0x%8.8llx: unable to write breakpoint opcode to memory.", breakID, (uint64_t)addr);
                    }
                }
                else
                {
                    DNBLogError("MachProcess::EnableBreakpoint ( breakID = %d ) addr = 0x%8.8llx: unable to read memory at breakpoint address.", breakID, (uint64_t)addr);
                }
            }
            else
            {
                DNBLogError("MachProcess::EnableBreakpoint ( breakID = %d ) no software breakpoint opcode for current architecture.", breakID);
            }
        }
    }
    return false;
}

bool
MachProcess::EnableWatchpoint(nub_watch_t watchID)
{
    DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::EnableWatchpoint(watchID = %d)", watchID);
    DNBBreakpoint *wp = m_watchpoints.FindByID (watchID);
    if (wp)
    {
        nub_addr_t addr = wp->Address();
        if (wp->IsEnabled())
        {
            DNBLogWarning("MachProcess::EnableWatchpoint(watchID = %d) addr = 0x%8.8llx: watchpoint already enabled.", watchID, (uint64_t)addr);
            return true;
        }
        else
        {
            // Currently only try and set hardware watchpoints.
            wp->SetHardwareIndex(m_thread_list.EnableHardwareWatchpoint(wp));
            if (wp->IsHardware())
            {
                wp->SetEnabled(true);
                return true;
            }
            // TODO: Add software watchpoints by doing page protection tricks.
        }
    }
    return false;
}

// Called by the exception thread when an exception has been received from
// our process. The exception message is completely filled and the exception
// data has already been copied.
void
MachProcess::ExceptionMessageReceived (const MachException::Message& exceptionMessage)
{
    PTHREAD_MUTEX_LOCKER (locker, m_exception_messages_mutex);

    if (m_exception_messages.empty())
        m_task.Suspend();

    DNBLogThreadedIf(LOG_EXCEPTIONS, "MachProcess::ExceptionMessageReceived ( )");

    // Use a locker to automatically unlock our mutex in case of exceptions
    // Add the exception to our internal exception stack
    m_exception_messages.push_back(exceptionMessage);
}

void
MachProcess::ExceptionMessageBundleComplete()
{
    // We have a complete bundle of exceptions for our child process.
    PTHREAD_MUTEX_LOCKER (locker, m_exception_messages_mutex);
    DNBLogThreadedIf(LOG_EXCEPTIONS, "%s: %llu exception messages.", __PRETTY_FUNCTION__, (uint64_t)m_exception_messages.size());
    if (!m_exception_messages.empty())
    {
        // Let all threads recover from stopping and do any clean up based
        // on the previous thread state (if any).
        m_thread_list.ProcessDidStop(this);

        // Let each thread know of any exceptions
        task_t task = m_task.TaskPort();
        size_t i;
        for (i=0; i<m_exception_messages.size(); ++i)
        {
            // Let the thread list figure use the MachProcess to forward all exceptions
            // on down to each thread.
            if (m_exception_messages[i].state.task_port == task)
                m_thread_list.NotifyException(m_exception_messages[i].state);
            if (DNBLogCheckLogBit(LOG_EXCEPTIONS))
                m_exception_messages[i].Dump();
        }

        if (DNBLogCheckLogBit(LOG_THREAD))
            m_thread_list.Dump();

        bool step_more = false;
        if (m_thread_list.ShouldStop(step_more))
        {
            // Wait for the eEventProcessRunningStateChanged event to be reset
            // before changing state to stopped to avoid race condition with
            // very fast start/stops
            struct timespec timeout;
            //DNBTimer::OffsetTimeOfDay(&timeout, 0, 250 * 1000);   // Wait for 250 ms
            DNBTimer::OffsetTimeOfDay(&timeout, 1, 0);  // Wait for 250 ms
            m_events.WaitForEventsToReset(eEventProcessRunningStateChanged, &timeout);
            SetState(eStateStopped);
        }
        else
        {
            // Resume without checking our current state.
            PrivateResume ();
        }
    }
    else
    {
        DNBLogThreadedIf(LOG_EXCEPTIONS, "%s empty exception messages bundle (%llu exceptions).", __PRETTY_FUNCTION__, (uint64_t)m_exception_messages.size());
    }
}

nub_size_t
MachProcess::CopyImageInfos ( struct DNBExecutableImageInfo **image_infos, bool only_changed)
{
    if (m_image_infos_callback != NULL)
        return m_image_infos_callback(ProcessID(), image_infos, only_changed, m_image_infos_baton);
    return 0;
}

void
MachProcess::SharedLibrariesUpdated ( )
{
    uint32_t event_bits = eEventSharedLibsStateChange;
    // Set the shared library event bit to let clients know of shared library
    // changes
    m_events.SetEvents(event_bits);
    // Wait for the event bit to reset if a reset ACK is requested
    m_events.WaitForResetAck(event_bits);
}

void
MachProcess::AppendSTDOUT (char* s, size_t len)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (<%llu> %s) ...", __FUNCTION__, (uint64_t)len, s);
    PTHREAD_MUTEX_LOCKER (locker, m_stdio_mutex);
    m_stdout_data.append(s, len);
    m_events.SetEvents(eEventStdioAvailable);

    // Wait for the event bit to reset if a reset ACK is requested
    m_events.WaitForResetAck(eEventStdioAvailable);
}

size_t
MachProcess::GetAvailableSTDOUT (char *buf, size_t buf_size)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (&%p[%llu]) ...", __FUNCTION__, buf, (uint64_t)buf_size);
    PTHREAD_MUTEX_LOCKER (locker, m_stdio_mutex);
    size_t bytes_available = m_stdout_data.size();
    if (bytes_available > 0)
    {
        if (bytes_available > buf_size)
        {
            memcpy(buf, m_stdout_data.data(), buf_size);
            m_stdout_data.erase(0, buf_size);
            bytes_available = buf_size;
        }
        else
        {
            memcpy(buf, m_stdout_data.data(), bytes_available);
            m_stdout_data.clear();
        }
    }
    return bytes_available;
}

nub_addr_t
MachProcess::GetDYLDAllImageInfosAddress ()
{
    DNBError err;
    return m_task.GetDYLDAllImageInfosAddress(err);
}

size_t
MachProcess::GetAvailableSTDERR (char *buf, size_t buf_size)
{
    return 0;
}

void *
MachProcess::STDIOThread(void *arg)
{
    MachProcess *proc = (MachProcess*) arg;
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( arg = %p ) thread starting...", __FUNCTION__, arg);

    // We start use a base and more options so we can control if we
    // are currently using a timeout on the mach_msg. We do this to get a
    // bunch of related exceptions on our exception port so we can process
    // then together. When we have multiple threads, we can get an exception
    // per thread and they will come in consecutively. The main thread loop
    // will start by calling mach_msg to without having the MACH_RCV_TIMEOUT
    // flag set in the options, so we will wait forever for an exception on
    // our exception port. After we get one exception, we then will use the
    // MACH_RCV_TIMEOUT option with a zero timeout to grab all other current
    // exceptions for our process. After we have received the last pending
    // exception, we will get a timeout which enables us to then notify
    // our main thread that we have an exception bundle avaiable. We then wait
    // for the main thread to tell this exception thread to start trying to get
    // exceptions messages again and we start again with a mach_msg read with
    // infinite timeout.
    DNBError err;
    int stdout_fd = proc->GetStdoutFileDescriptor();
    int stderr_fd = proc->GetStderrFileDescriptor();
    if (stdout_fd == stderr_fd)
        stderr_fd = -1;

    while (stdout_fd >= 0 || stderr_fd >= 0)
    {
        ::pthread_testcancel ();

        fd_set read_fds;
        FD_ZERO (&read_fds);
        if (stdout_fd >= 0)
            FD_SET (stdout_fd, &read_fds);
        if (stderr_fd >= 0)
            FD_SET (stderr_fd, &read_fds);
        int nfds = std::max<int>(stdout_fd, stderr_fd) + 1;

        int num_set_fds = select (nfds, &read_fds, NULL, NULL, NULL);
        DNBLogThreadedIf(LOG_PROCESS, "select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);

        if (num_set_fds < 0)
        {
            int select_errno = errno;
            if (DNBLogCheckLogBit(LOG_PROCESS))
            {
                err.SetError (select_errno, DNBError::POSIX);
                err.LogThreadedIfError("select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);
            }

            switch (select_errno)
            {
            case EAGAIN:    // The kernel was (perhaps temporarily) unable to allocate the requested number of file descriptors, or we have non-blocking IO
                break;
            case EBADF:     // One of the descriptor sets specified an invalid descriptor.
                return NULL;
                break;
            case EINTR:     // A signal was delivered before the time limit expired and before any of the selected events occurred.
            case EINVAL:    // The specified time limit is invalid. One of its components is negative or too large.
            default:        // Other unknown error
                break;
            }
        }
        else if (num_set_fds == 0)
        {
        }
        else
        {
            char s[1024];
            s[sizeof(s)-1] = '\0';  // Ensure we have NULL termination
            int bytes_read = 0;
            if (stdout_fd >= 0 && FD_ISSET (stdout_fd, &read_fds))
            {
                do
                {
                    bytes_read = ::read (stdout_fd, s, sizeof(s)-1);
                    if (bytes_read < 0)
                    {
                        int read_errno = errno;
                        DNBLogThreadedIf(LOG_PROCESS, "read (stdout_fd, ) => %d   errno: %d (%s)", bytes_read, read_errno, strerror(read_errno));
                    }
                    else if (bytes_read == 0)
                    {
                        // EOF...
                        DNBLogThreadedIf(LOG_PROCESS, "read (stdout_fd, ) => %d  (reached EOF for child STDOUT)", bytes_read);
                        stdout_fd = -1;
                    }
                    else if (bytes_read > 0)
                    {
                        proc->AppendSTDOUT(s, bytes_read);
                    }

                } while (bytes_read > 0);
            }

            if (stderr_fd >= 0 && FD_ISSET (stderr_fd, &read_fds))
            {
                do
                {
                    bytes_read = ::read (stderr_fd, s, sizeof(s)-1);
                    if (bytes_read < 0)
                    {
                        int read_errno = errno;
                        DNBLogThreadedIf(LOG_PROCESS, "read (stderr_fd, ) => %d   errno: %d (%s)", bytes_read, read_errno, strerror(read_errno));
                    }
                    else if (bytes_read == 0)
                    {
                        // EOF...
                        DNBLogThreadedIf(LOG_PROCESS, "read (stderr_fd, ) => %d  (reached EOF for child STDERR)", bytes_read);
                        stderr_fd = -1;
                    }
                    else if (bytes_read > 0)
                    {
                        proc->AppendSTDOUT(s, bytes_read);
                    }

                } while (bytes_read > 0);
            }
        }
    }
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (%p): thread exiting...", __FUNCTION__, arg);
    return NULL;
}


void
MachProcess::SignalAsyncProfileData (const char *info)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (%s) ...", __FUNCTION__, info);
    PTHREAD_MUTEX_LOCKER (locker, m_profile_data_mutex);
    m_profile_data.push_back(info);
    m_events.SetEvents(eEventProfileDataAvailable);
    
    // Wait for the event bit to reset if a reset ACK is requested
    m_events.WaitForResetAck(eEventProfileDataAvailable);
}


size_t
MachProcess::GetAsyncProfileData (char *buf, size_t buf_size)
{
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (&%p[%llu]) ...", __FUNCTION__, buf, (uint64_t)buf_size);
    PTHREAD_MUTEX_LOCKER (locker, m_profile_data_mutex);
    if (m_profile_data.empty())
        return 0;
    
    size_t bytes_available = m_profile_data.front().size();
    if (bytes_available > 0)
    {
        if (bytes_available > buf_size)
        {
            memcpy(buf, m_profile_data.front().data(), buf_size);
            m_profile_data.front().erase(0, buf_size);
            bytes_available = buf_size;
        }
        else
        {
            memcpy(buf, m_profile_data.front().data(), bytes_available);
            m_profile_data.erase(m_profile_data.begin());
        }
    }
    return bytes_available;
}


void *
MachProcess::ProfileThread(void *arg)
{
    MachProcess *proc = (MachProcess*) arg;
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( arg = %p ) thread starting...", __FUNCTION__, arg);

    while (proc->IsProfilingEnabled())
    {
        nub_state_t state = proc->GetState();
        if (state == eStateRunning)
        {
            std::string data = proc->Task().GetProfileData(proc->GetProfileScanType());
            if (!data.empty())
            {
                proc->SignalAsyncProfileData(data.c_str());
            }
        }
        else if ((state == eStateUnloaded) || (state == eStateDetached) || (state == eStateUnloaded))
        {
            // Done. Get out of this thread.
            break;
        }
        
        // A simple way to set up the profile interval. We can also use select() or dispatch timer source if necessary.
        usleep(proc->ProfileInterval());
    }
    return NULL;
}


pid_t
MachProcess::AttachForDebug (pid_t pid, char *err_str, size_t err_len)
{
    // Clear out and clean up from any current state
    Clear();
    if (pid != 0)
    {
        DNBError err;
        // Make sure the process exists...
        if (::getpgid (pid) < 0)
        {
            err.SetErrorToErrno();
            const char *err_cstr = err.AsString();
            ::snprintf (err_str, err_len, "%s", err_cstr ? err_cstr : "No such process");
            return INVALID_NUB_PROCESS;
        }

        SetState(eStateAttaching);
        m_pid = pid;
        // Let ourselves know we are going to be using SBS if the correct flag bit is set...
#ifdef WITH_SPRINGBOARD
        if (IsSBProcess(pid))
            m_flags |= eMachProcessFlagsUsingSBS;
#endif
        if (!m_task.StartExceptionThread(err))
        {
            const char *err_cstr = err.AsString();
            ::snprintf (err_str, err_len, "%s", err_cstr ? err_cstr : "unable to start the exception thread");
            DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to pid %d", pid);
            m_pid = INVALID_NUB_PROCESS;
            return INVALID_NUB_PROCESS;
        }

        errno = 0;
        if (::ptrace (PT_ATTACHEXC, pid, 0, 0))
            err.SetError(errno);
        else
            err.Clear();

        if (err.Success())
        {
            m_flags |= eMachProcessFlagsAttached;
            // Sleep a bit to let the exception get received and set our process status
            // to stopped.
            ::usleep(250000);
            DNBLogThreadedIf(LOG_PROCESS, "successfully attached to pid %d", pid);
            return m_pid;
        }
        else
        {
            ::snprintf (err_str, err_len, "%s", err.AsString());
            DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to pid %d", pid);
        }
    }
    return INVALID_NUB_PROCESS;
}

// Do the process specific setup for attach.  If this returns NULL, then there's no
// platform specific stuff to be done to wait for the attach.  If you get non-null,
// pass that token to the CheckForProcess method, and then to CleanupAfterAttach.

//  Call PrepareForAttach before attaching to a process that has not yet launched
// This returns a token that can be passed to CheckForProcess, and to CleanupAfterAttach.
// You should call CleanupAfterAttach to free the token, and do whatever other
// cleanup seems good.

const void *
MachProcess::PrepareForAttach (const char *path, nub_launch_flavor_t launch_flavor, bool waitfor, DNBError &err_str)
{
#ifdef WITH_SPRINGBOARD
    // Tell SpringBoard to halt the next launch of this application on startup.

    if (!waitfor)
        return NULL;

    const char *app_ext = strstr(path, ".app");
    const bool is_app = app_ext != NULL && (app_ext[4] == '\0' || app_ext[4] == '/');
    if (!is_app)
    {
        DNBLogThreadedIf(LOG_PROCESS, "MachProcess::PrepareForAttach(): path '%s' doesn't contain .app, we can't tell springboard to wait for launch...", path);
        return NULL;
    }

    if (launch_flavor != eLaunchFlavorSpringBoard
        && launch_flavor != eLaunchFlavorDefault)
        return NULL;

    std::string app_bundle_path(path, app_ext + strlen(".app"));

    CFStringRef bundleIDCFStr = CopyBundleIDForPath (app_bundle_path.c_str (), err_str);
    std::string bundleIDStr;
    CFString::UTF8(bundleIDCFStr, bundleIDStr);
    DNBLogThreadedIf(LOG_PROCESS, "CopyBundleIDForPath (%s, err_str) returned @\"%s\"", app_bundle_path.c_str (), bundleIDStr.c_str());

    if (bundleIDCFStr == NULL)
    {
        return NULL;
    }

    SBSApplicationLaunchError sbs_error = 0;

    const char *stdout_err = "/dev/null";
    CFString stdio_path;
    stdio_path.SetFileSystemRepresentation (stdout_err);

    DNBLogThreadedIf(LOG_PROCESS, "SBSLaunchApplicationForDebugging ( @\"%s\" , NULL, NULL, NULL, @\"%s\", @\"%s\", SBSApplicationDebugOnNextLaunch | SBSApplicationLaunchWaitForDebugger )", bundleIDStr.c_str(), stdout_err, stdout_err);
    sbs_error = SBSLaunchApplicationForDebugging (bundleIDCFStr,
                                                  (CFURLRef)NULL,         // openURL
                                                  NULL, // launch_argv.get(),
                                                  NULL, // launch_envp.get(),  // CFDictionaryRef environment
                                                  stdio_path.get(),
                                                  stdio_path.get(),
                                                  SBSApplicationDebugOnNextLaunch | SBSApplicationLaunchWaitForDebugger);

    if (sbs_error != SBSApplicationLaunchErrorSuccess)
    {
        err_str.SetError(sbs_error, DNBError::SpringBoard);
        return NULL;
    }

    DNBLogThreadedIf(LOG_PROCESS, "Successfully set DebugOnNextLaunch.");
    return bundleIDCFStr;
# else
  return NULL;
#endif
}

// Pass in the token you got from PrepareForAttach.  If there is a process
// for that token, then the pid will be returned, otherwise INVALID_NUB_PROCESS
// will be returned.

nub_process_t
MachProcess::CheckForProcess (const void *attach_token)
{
    if (attach_token == NULL)
        return INVALID_NUB_PROCESS;

#ifdef WITH_SPRINGBOARD
    CFStringRef bundleIDCFStr = (CFStringRef) attach_token;
    Boolean got_it;
    nub_process_t attach_pid;
    got_it = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &attach_pid);
    if (got_it)
        return attach_pid;
    else
        return INVALID_NUB_PROCESS;
#endif
    return INVALID_NUB_PROCESS;
}

// Call this to clean up after you have either attached or given up on the attach.
// Pass true for success if you have attached, false if you have not.
// The token will also be freed at this point, so you can't use it after calling
// this method.

void
MachProcess::CleanupAfterAttach (const void *attach_token, bool success, DNBError &err_str)
{
#ifdef WITH_SPRINGBOARD
    if (attach_token == NULL)
        return;

    // Tell SpringBoard to cancel the debug on next launch of this application
    // if we failed to attach
    if (!success)
    {
        SBSApplicationLaunchError sbs_error = 0;
        CFStringRef bundleIDCFStr = (CFStringRef) attach_token;

        sbs_error = SBSLaunchApplicationForDebugging (bundleIDCFStr,
                                                      (CFURLRef)NULL,
                                                      NULL,
                                                      NULL,
                                                      NULL,
                                                      NULL,
                                                      SBSApplicationCancelDebugOnNextLaunch);

        if (sbs_error != SBSApplicationLaunchErrorSuccess)
        {
            err_str.SetError(sbs_error, DNBError::SpringBoard);
            return;
        }
    }

    CFRelease((CFStringRef) attach_token);
#endif
}

pid_t
MachProcess::LaunchForDebug
(
    const char *path,
    char const *argv[],
    char const *envp[],
    const char *working_directory, // NULL => dont' change, non-NULL => set working directory for inferior to this
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    bool no_stdio,
    nub_launch_flavor_t launch_flavor,
    int disable_aslr,
    DNBError &launch_err
)
{
    // Clear out and clean up from any current state
    Clear();

    DNBLogThreadedIf(LOG_PROCESS, "%s( path = '%s', argv = %p, envp = %p, launch_flavor = %u, disable_aslr = %d )", __FUNCTION__, path, argv, envp, launch_flavor, disable_aslr);

    // Fork a child process for debugging
    SetState(eStateLaunching);

    switch (launch_flavor)
    {
    case eLaunchFlavorForkExec:
        m_pid = MachProcess::ForkChildForPTraceDebugging (path, argv, envp, this, launch_err);
        break;

#ifdef WITH_SPRINGBOARD

    case eLaunchFlavorSpringBoard:
        {
            //  .../whatever.app/whatever ?  
            //  Or .../com.apple.whatever.app/whatever -- be careful of ".app" in "com.apple.whatever" here
            const char *app_ext = strstr (path, ".app/");
            if (app_ext == NULL)
            {
                // .../whatever.app ?
                int len = strlen (path);
                if (len > 5)
                {
                    if (strcmp (path + len - 4, ".app") == 0)
                    {
                        app_ext = path + len - 4;
                    }
                }
            }
            if (app_ext)
            {
                std::string app_bundle_path(path, app_ext + strlen(".app"));
                if (SBLaunchForDebug (app_bundle_path.c_str(), argv, envp, no_stdio, launch_err) != 0)
                    return m_pid; // A successful SBLaunchForDebug() returns and assigns a non-zero m_pid.
                else
                    break; // We tried a springboard launch, but didn't succeed lets get out
            }
        }
        // In case the executable name has a ".app" fragment which confuses our debugserver,
        // let's do an intentional fallthrough here...
        launch_flavor = eLaunchFlavorPosixSpawn;

#endif

    case eLaunchFlavorPosixSpawn:
        m_pid = MachProcess::PosixSpawnChildForPTraceDebugging (path, 
                                                                DNBArchProtocol::GetArchitecture (),
                                                                argv, 
                                                                envp, 
                                                                working_directory,
                                                                stdin_path,
                                                                stdout_path,
                                                                stderr_path,
                                                                no_stdio, 
                                                                this, 
                                                                disable_aslr, 
                                                                launch_err);
        break;

    default:
        // Invalid  launch
        launch_err.SetError(NUB_GENERIC_ERROR, DNBError::Generic);
        return INVALID_NUB_PROCESS;
    }

    if (m_pid == INVALID_NUB_PROCESS)
    {
        // If we don't have a valid process ID and no one has set the error,
        // then return a generic error
        if (launch_err.Success())
            launch_err.SetError(NUB_GENERIC_ERROR, DNBError::Generic);
    }
    else
    {
        m_path = path;
        size_t i;
        char const *arg;
        for (i=0; (arg = argv[i]) != NULL; i++)
            m_args.push_back(arg);

        m_task.StartExceptionThread(launch_err);
        if (launch_err.Fail())
        {
            if (launch_err.AsString() == NULL)
                launch_err.SetErrorString("unable to start the exception thread");
            DNBLog ("Could not get inferior's Mach exception port, sending ptrace PT_KILL and exiting.");
            ::ptrace (PT_KILL, m_pid, 0, 0);
            m_pid = INVALID_NUB_PROCESS;
            return INVALID_NUB_PROCESS;
        }

        StartSTDIOThread();

        if (launch_flavor == eLaunchFlavorPosixSpawn)
        {

            SetState (eStateAttaching);
            errno = 0;
            int err = ::ptrace (PT_ATTACHEXC, m_pid, 0, 0);
            if (err == 0)
            {
                m_flags |= eMachProcessFlagsAttached;
                DNBLogThreadedIf(LOG_PROCESS, "successfully spawned pid %d", m_pid);
                launch_err.Clear();
            }
            else
            {
                SetState (eStateExited);
                DNBError ptrace_err(errno, DNBError::POSIX);
                DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to spawned pid %d (err = %i, errno = %i (%s))", m_pid, err, ptrace_err.Error(), ptrace_err.AsString());
                launch_err.SetError(NUB_GENERIC_ERROR, DNBError::Generic);
            }
        }
        else
        {
            launch_err.Clear();
        }
    }
    return m_pid;
}

pid_t
MachProcess::PosixSpawnChildForPTraceDebugging
(
    const char *path,
    cpu_type_t cpu_type,
    char const *argv[],
    char const *envp[],
    const char *working_directory,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    bool no_stdio,
    MachProcess* process,
    int disable_aslr,
    DNBError& err
)
{
    posix_spawnattr_t attr;
    short flags;
    DNBLogThreadedIf(LOG_PROCESS, "%s ( path='%s', argv=%p, envp=%p, working_dir=%s, stdin=%s, stdout=%s stderr=%s, no-stdio=%i)", 
                     __FUNCTION__, 
                     path, 
                     argv, 
                     envp,
                     working_directory,
                     stdin_path,
                     stdout_path,
                     stderr_path,
                     no_stdio);

    err.SetError( ::posix_spawnattr_init (&attr), DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
        err.LogThreaded("::posix_spawnattr_init ( &attr )");
    if (err.Fail())
        return INVALID_NUB_PROCESS;

    flags = POSIX_SPAWN_START_SUSPENDED | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;
    if (disable_aslr)
        flags |= _POSIX_SPAWN_DISABLE_ASLR;

    sigset_t no_signals;
    sigset_t all_signals;
    sigemptyset (&no_signals);
    sigfillset (&all_signals);
    ::posix_spawnattr_setsigmask(&attr, &no_signals);
    ::posix_spawnattr_setsigdefault(&attr, &all_signals);

    err.SetError( ::posix_spawnattr_setflags (&attr, flags), DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
        err.LogThreaded("::posix_spawnattr_setflags ( &attr, POSIX_SPAWN_START_SUSPENDED%s )", flags & _POSIX_SPAWN_DISABLE_ASLR ? " | _POSIX_SPAWN_DISABLE_ASLR" : "");
    if (err.Fail())
        return INVALID_NUB_PROCESS;

    // Don't do this on SnowLeopard, _sometimes_ the TASK_BASIC_INFO will fail
    // and we will fail to continue with our process...
    
    // On SnowLeopard we should set "DYLD_NO_PIE" in the inferior environment....
     
#if !defined(__arm__)

    // We don't need to do this for ARM, and we really shouldn't now that we
    // have multiple CPU subtypes and no posix_spawnattr call that allows us
    // to set which CPU subtype to launch...
    if (cpu_type != 0)
    {
        size_t ocount = 0;
        err.SetError( ::posix_spawnattr_setbinpref_np (&attr, 1, &cpu_type, &ocount), DNBError::POSIX);
        if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
            err.LogThreaded("::posix_spawnattr_setbinpref_np ( &attr, 1, cpu_type = 0x%8.8x, count => %llu )", cpu_type, (uint64_t)ocount);

        if (err.Fail() != 0 || ocount != 1)
            return INVALID_NUB_PROCESS;
    }
#endif

    PseudoTerminal pty;

    posix_spawn_file_actions_t file_actions;
    err.SetError( ::posix_spawn_file_actions_init (&file_actions), DNBError::POSIX);
    int file_actions_valid = err.Success();
    if (!file_actions_valid || DNBLogCheckLogBit(LOG_PROCESS))
        err.LogThreaded("::posix_spawn_file_actions_init ( &file_actions )");
    int pty_error = -1;
    pid_t pid = INVALID_NUB_PROCESS;
    if (file_actions_valid)
    {
        if (stdin_path == NULL && stdout_path == NULL && stderr_path == NULL && !no_stdio)
        {
            pty_error = pty.OpenFirstAvailableMaster(O_RDWR|O_NOCTTY);
            if (pty_error == PseudoTerminal::success)
            {
                stdin_path = stdout_path = stderr_path = pty.SlaveName();
            }
        }

        // if no_stdio or std paths not supplied, then route to "/dev/null".
        if (no_stdio || stdin_path == NULL || stdin_path[0] == '\0')
            stdin_path = "/dev/null";
        if (no_stdio || stdout_path == NULL || stdout_path[0] == '\0')
            stdout_path = "/dev/null";
        if (no_stdio || stderr_path == NULL || stderr_path[0] == '\0')
            stderr_path = "/dev/null";

        err.SetError( ::posix_spawn_file_actions_addopen (&file_actions,
                                                          STDIN_FILENO,
                                                          stdin_path,
                                                          O_RDONLY | O_NOCTTY,
                                                          0),
                     DNBError::POSIX);
        if (err.Fail() || DNBLogCheckLogBit (LOG_PROCESS))
            err.LogThreaded ("::posix_spawn_file_actions_addopen (&file_actions, filedes=STDIN_FILENO, path='%s')", stdin_path);
        
        err.SetError( ::posix_spawn_file_actions_addopen (&file_actions,
                                                          STDOUT_FILENO,
                                                          stdout_path,
                                                          O_WRONLY | O_NOCTTY | O_CREAT,
                                                          0640),
                     DNBError::POSIX);
        if (err.Fail() || DNBLogCheckLogBit (LOG_PROCESS))
            err.LogThreaded ("::posix_spawn_file_actions_addopen (&file_actions, filedes=STDOUT_FILENO, path='%s')", stdout_path);
        
        err.SetError( ::posix_spawn_file_actions_addopen (&file_actions,
                                                          STDERR_FILENO,
                                                          stderr_path,
                                                          O_WRONLY | O_NOCTTY | O_CREAT,
                                                          0640),
                     DNBError::POSIX);
        if (err.Fail() || DNBLogCheckLogBit (LOG_PROCESS))
            err.LogThreaded ("::posix_spawn_file_actions_addopen (&file_actions, filedes=STDERR_FILENO, path='%s')", stderr_path);

        // TODO: Verify if we can set the working directory back immediately
        // after the posix_spawnp call without creating a race condition???
        if (working_directory)
            ::chdir (working_directory);
        
        err.SetError( ::posix_spawnp (&pid, path, &file_actions, &attr, (char * const*)argv, (char * const*)envp), DNBError::POSIX);
        if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
            err.LogThreaded("::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, attr = %p, argv = %p, envp = %p )", pid, path, &file_actions, &attr, argv, envp);
    }
    else
    {
        // TODO: Verify if we can set the working directory back immediately
        // after the posix_spawnp call without creating a race condition???
        if (working_directory)
            ::chdir (working_directory);
        
        err.SetError( ::posix_spawnp (&pid, path, NULL, &attr, (char * const*)argv, (char * const*)envp), DNBError::POSIX);
        if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
            err.LogThreaded("::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, attr = %p, argv = %p, envp = %p )", pid, path, NULL, &attr, argv, envp);
    }

    // We have seen some cases where posix_spawnp was returning a valid
    // looking pid even when an error was returned, so clear it out
    if (err.Fail())
        pid = INVALID_NUB_PROCESS;

    if (pty_error == 0)
    {
        if (process != NULL)
        {
            int master_fd = pty.ReleaseMasterFD();
            process->SetChildFileDescriptors(master_fd, master_fd, master_fd);
        }
    }
    ::posix_spawnattr_destroy (&attr);

    if (pid != INVALID_NUB_PROCESS)
    {
        cpu_type_t pid_cpu_type = MachProcess::GetCPUTypeForLocalProcess (pid);
        DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( ) pid=%i, cpu_type=0x%8.8x", __FUNCTION__, pid, pid_cpu_type);
        if (pid_cpu_type)
            DNBArchProtocol::SetArchitecture (pid_cpu_type);
    }

    if (file_actions_valid)
    {
        DNBError err2;
        err2.SetError( ::posix_spawn_file_actions_destroy (&file_actions), DNBError::POSIX);
        if (err2.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
            err2.LogThreaded("::posix_spawn_file_actions_destroy ( &file_actions )");
    }

    return pid;
}

uint32_t
MachProcess::GetCPUTypeForLocalProcess (pid_t pid)
{
    int mib[CTL_MAXNAME]={0,};
    size_t len = CTL_MAXNAME;
    if (::sysctlnametomib("sysctl.proc_cputype", mib, &len)) 
        return 0;

    mib[len] = pid;
    len++;
            
    cpu_type_t cpu;
    size_t cpu_len = sizeof(cpu);
    if (::sysctl (mib, len, &cpu, &cpu_len, 0, 0))
        cpu = 0;
    return cpu;
}

pid_t
MachProcess::ForkChildForPTraceDebugging
(
    const char *path,
    char const *argv[],
    char const *envp[],
    MachProcess* process,
    DNBError& launch_err
)
{
    PseudoTerminal::Error pty_error = PseudoTerminal::success;

    // Use a fork that ties the child process's stdin/out/err to a pseudo
    // terminal so we can read it in our MachProcess::STDIOThread
    // as unbuffered io.
    PseudoTerminal pty;
    pid_t pid = pty.Fork(pty_error);

    if (pid < 0)
    {
        //--------------------------------------------------------------
        // Error during fork.
        //--------------------------------------------------------------
        return pid;
    }
    else if (pid == 0)
    {
        //--------------------------------------------------------------
        // Child process
        //--------------------------------------------------------------
        ::ptrace (PT_TRACE_ME, 0, 0, 0);    // Debug this process
        ::ptrace (PT_SIGEXC, 0, 0, 0);    // Get BSD signals as mach exceptions

        // If our parent is setgid, lets make sure we don't inherit those
        // extra powers due to nepotism.
        if (::setgid (getgid ()) == 0)
        {

            // Let the child have its own process group. We need to execute
            // this call in both the child and parent to avoid a race condition
            // between the two processes.
            ::setpgid (0, 0);    // Set the child process group to match its pid

            // Sleep a bit to before the exec call
            ::sleep (1);

            // Turn this process into
            ::execv (path, (char * const *)argv);
        }
        // Exit with error code. Child process should have taken
        // over in above exec call and if the exec fails it will
        // exit the child process below.
        ::exit (127);
    }
    else
    {
        //--------------------------------------------------------------
        // Parent process
        //--------------------------------------------------------------
        // Let the child have its own process group. We need to execute
        // this call in both the child and parent to avoid a race condition
        // between the two processes.
        ::setpgid (pid, pid);    // Set the child process group to match its pid

        if (process != NULL)
        {
            // Release our master pty file descriptor so the pty class doesn't
            // close it and so we can continue to use it in our STDIO thread
            int master_fd = pty.ReleaseMasterFD();
            process->SetChildFileDescriptors(master_fd, master_fd, master_fd);
        }
    }
    return pid;
}

#ifdef WITH_SPRINGBOARD

pid_t
MachProcess::SBLaunchForDebug (const char *path, char const *argv[], char const *envp[], bool no_stdio, DNBError &launch_err)
{
    // Clear out and clean up from any current state
    Clear();

    DNBLogThreadedIf(LOG_PROCESS, "%s( '%s', argv)", __FUNCTION__, path);

    // Fork a child process for debugging
    SetState(eStateLaunching);
    m_pid = MachProcess::SBForkChildForPTraceDebugging(path, argv, envp, no_stdio, this, launch_err);
    if (m_pid != 0)
    {
        m_flags |= eMachProcessFlagsUsingSBS;
        m_path = path;
        size_t i;
        char const *arg;
        for (i=0; (arg = argv[i]) != NULL; i++)
            m_args.push_back(arg);
        m_task.StartExceptionThread(launch_err);
        
        if (launch_err.Fail())
        {
            if (launch_err.AsString() == NULL)
                launch_err.SetErrorString("unable to start the exception thread");
            DNBLog ("Could not get inferior's Mach exception port, sending ptrace PT_KILL and exiting.");
            ::ptrace (PT_KILL, m_pid, 0, 0);
            m_pid = INVALID_NUB_PROCESS;
            return INVALID_NUB_PROCESS;
        }

        StartSTDIOThread();
        SetState (eStateAttaching);
        int err = ::ptrace (PT_ATTACHEXC, m_pid, 0, 0);
        if (err == 0)
        {
            m_flags |= eMachProcessFlagsAttached;
            DNBLogThreadedIf(LOG_PROCESS, "successfully attached to pid %d", m_pid);
        }
        else
        {
            SetState (eStateExited);
            DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to pid %d", m_pid);
        }
    }
    return m_pid;
}

#include <servers/bootstrap.h>

// This returns a CFRetained pointer to the Bundle ID for app_bundle_path,
// or NULL if there was some problem getting the bundle id.
static CFStringRef
CopyBundleIDForPath (const char *app_bundle_path, DNBError &err_str)
{
    CFBundle bundle(app_bundle_path);
    CFStringRef bundleIDCFStr = bundle.GetIdentifier();
    std::string bundleID;
    if (CFString::UTF8(bundleIDCFStr, bundleID) == NULL)
    {
        struct stat app_bundle_stat;
        char err_msg[PATH_MAX];

        if (::stat (app_bundle_path, &app_bundle_stat) < 0)
        {
            err_str.SetError(errno, DNBError::POSIX);
            snprintf(err_msg, sizeof(err_msg), "%s: \"%s\"", err_str.AsString(), app_bundle_path);
            err_str.SetErrorString(err_msg);
            DNBLogThreadedIf(LOG_PROCESS, "%s() error: %s", __FUNCTION__, err_msg);
        }
        else
        {
            err_str.SetError(-1, DNBError::Generic);
            snprintf(err_msg, sizeof(err_msg), "failed to extract CFBundleIdentifier from %s", app_bundle_path);
            err_str.SetErrorString(err_msg);
            DNBLogThreadedIf(LOG_PROCESS, "%s() error: failed to extract CFBundleIdentifier from '%s'", __FUNCTION__, app_bundle_path);
        }
        return NULL;
    }

    DNBLogThreadedIf(LOG_PROCESS, "%s() extracted CFBundleIdentifier: %s", __FUNCTION__, bundleID.c_str());
    CFRetain (bundleIDCFStr);

    return bundleIDCFStr;
}

pid_t
MachProcess::SBForkChildForPTraceDebugging (const char *app_bundle_path, char const *argv[], char const *envp[], bool no_stdio, MachProcess* process, DNBError &launch_err)
{
    DNBLogThreadedIf(LOG_PROCESS, "%s( '%s', argv, %p)", __FUNCTION__, app_bundle_path, process);
    CFAllocatorRef alloc = kCFAllocatorDefault;

    if (argv[0] == NULL)
        return INVALID_NUB_PROCESS;

    size_t argc = 0;
    // Count the number of arguments
    while (argv[argc] != NULL)
        argc++;

    // Enumerate the arguments
    size_t first_launch_arg_idx = 1;
    CFReleaser<CFMutableArrayRef> launch_argv;

    if (argv[first_launch_arg_idx])
    {
        size_t launch_argc = argc > 0 ? argc - 1 : 0;
        launch_argv.reset (::CFArrayCreateMutable (alloc, launch_argc, &kCFTypeArrayCallBacks));
        size_t i;
        char const *arg;
        CFString launch_arg;
        for (i=first_launch_arg_idx; (i < argc) && ((arg = argv[i]) != NULL); i++)
        {
            launch_arg.reset(::CFStringCreateWithCString (alloc, arg, kCFStringEncodingUTF8));
            if (launch_arg.get() != NULL)
                CFArrayAppendValue(launch_argv.get(), launch_arg.get());
            else
                break;
        }
    }

    // Next fill in the arguments dictionary.  Note, the envp array is of the form
    // Variable=value but SpringBoard wants a CF dictionary.  So we have to convert
    // this here.

    CFReleaser<CFMutableDictionaryRef> launch_envp;

    if (envp[0])
    {
        launch_envp.reset(::CFDictionaryCreateMutable(alloc, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        const char *value;
        int name_len;
        CFString name_string, value_string;

        for (int i = 0; envp[i] != NULL; i++)
        {
            value = strstr (envp[i], "=");

            // If the name field is empty or there's no =, skip it.  Somebody's messing with us.
            if (value == NULL || value == envp[i])
                continue;

            name_len = value - envp[i];

            // Now move value over the "="
            value++;

            name_string.reset(::CFStringCreateWithBytes(alloc, (const UInt8 *) envp[i], name_len, kCFStringEncodingUTF8, false));
            value_string.reset(::CFStringCreateWithCString(alloc, value, kCFStringEncodingUTF8));
            CFDictionarySetValue (launch_envp.get(), name_string.get(), value_string.get());
        }
    }

    CFString stdio_path;

    PseudoTerminal pty;
    if (!no_stdio)
    {
        PseudoTerminal::Error pty_err = pty.OpenFirstAvailableMaster(O_RDWR|O_NOCTTY);
        if (pty_err == PseudoTerminal::success)
        {
            const char* slave_name = pty.SlaveName();
            DNBLogThreadedIf(LOG_PROCESS, "%s() successfully opened master pty, slave is %s", __FUNCTION__, slave_name);
            if (slave_name && slave_name[0])
            {
                ::chmod (slave_name, S_IRWXU | S_IRWXG | S_IRWXO);
                stdio_path.SetFileSystemRepresentation (slave_name);
            }
        }
    }
    
    if (stdio_path.get() == NULL)
    {
        stdio_path.SetFileSystemRepresentation ("/dev/null");
    }

    CFStringRef bundleIDCFStr = CopyBundleIDForPath (app_bundle_path, launch_err);
    if (bundleIDCFStr == NULL)
        return INVALID_NUB_PROCESS;

    std::string bundleID;
    CFString::UTF8(bundleIDCFStr, bundleID);

    // Find SpringBoard
    SBSApplicationLaunchError sbs_error = 0;
    sbs_error = SBSLaunchApplicationForDebugging (bundleIDCFStr,
                                                  (CFURLRef)NULL,         // openURL
                                                  launch_argv.get(),
                                                  launch_envp.get(),  // CFDictionaryRef environment
                                                  stdio_path.get(),
                                                  stdio_path.get(),
                                                  SBSApplicationLaunchWaitForDebugger | SBSApplicationLaunchUnlockDevice);


    launch_err.SetError(sbs_error, DNBError::SpringBoard);

    if (sbs_error == SBSApplicationLaunchErrorSuccess)
    {
        static const useconds_t pid_poll_interval = 200000;
        static const useconds_t pid_poll_timeout = 30000000;

        useconds_t pid_poll_total = 0;

        nub_process_t pid = INVALID_NUB_PROCESS;
        Boolean pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
        // Poll until the process is running, as long as we are getting valid responses and the timeout hasn't expired
        // A return PID of 0 means the process is not running, which may be because it hasn't been (asynchronously) started
        // yet, or that it died very quickly (if you weren't using waitForDebugger).
        while (!pid_found && pid_poll_total < pid_poll_timeout)
        {
            usleep (pid_poll_interval);
            pid_poll_total += pid_poll_interval;
            DNBLogThreadedIf(LOG_PROCESS, "%s() polling Springboard for pid for %s...", __FUNCTION__, bundleID.c_str());
            pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
        }

        CFRelease (bundleIDCFStr);
        if (pid_found)
        {
            if (process != NULL)
            {
                // Release our master pty file descriptor so the pty class doesn't
                // close it and so we can continue to use it in our STDIO thread
                int master_fd = pty.ReleaseMasterFD();
                process->SetChildFileDescriptors(master_fd, master_fd, master_fd);
            }
            DNBLogThreadedIf(LOG_PROCESS, "%s() => pid = %4.4x", __FUNCTION__, pid);
        }
        else
        {
            DNBLogError("failed to lookup the process ID for CFBundleIdentifier %s.", bundleID.c_str());
        }
        return pid;
    }

    DNBLogError("unable to launch the application with CFBundleIdentifier '%s' sbs_error = %u", bundleID.c_str(), sbs_error);
    return INVALID_NUB_PROCESS;
}

#endif // #ifdef WITH_SPRINGBOARD


