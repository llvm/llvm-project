//===-- NativeProcessProtocol.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeProcessProtocol.h"

#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Host/common/SoftwareBreakpoint.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBAssert.h"

using namespace lldb;
using namespace lldb_private;

// -----------------------------------------------------------------------------
// NativeProcessProtocol Members
// -----------------------------------------------------------------------------

NativeProcessProtocol::NativeProcessProtocol(lldb::pid_t pid)
    : m_pid(pid),
      m_threads(),
      m_current_thread_id(LLDB_INVALID_THREAD_ID),
      m_threads_mutex(),
      m_state(lldb::eStateInvalid),
      m_state_mutex(),
      m_exit_type(eExitTypeInvalid),
      m_exit_status(0),
      m_exit_description(),
      m_delegates_mutex(),
      m_delegates(),
      m_breakpoint_list(),
      m_watchpoint_list(),
      m_terminal_fd(-1),
      m_stop_id(0)
{
}

lldb_private::Error
NativeProcessProtocol::Interrupt ()
{
    Error error;
#if !defined (SIGSTOP)
    error.SetErrorString ("local host does not support signaling");
    return error;
#else
    return Signal (SIGSTOP);
#endif
}

lldb_private::Error
NativeProcessProtocol::GetMemoryRegionInfo (lldb::addr_t load_addr, MemoryRegionInfo &range_info)
{
    // Default: not implemented.
    return Error ("not implemented");
}

bool
NativeProcessProtocol::GetExitStatus (ExitType *exit_type, int *status, std::string &exit_description)
{
    if (m_state == lldb::eStateExited)
    {
        *exit_type = m_exit_type;
        *status = m_exit_status;
        exit_description = m_exit_description;
        return true;
    }

    *status = 0;
    return false;
}

bool
NativeProcessProtocol::SetExitStatus (ExitType exit_type, int status, const char *exit_description, bool bNotifyStateChange)
{
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessProtocol::%s(%d, %d, %s, %s) called",
                __FUNCTION__,
                exit_type,
                status,
                exit_description ? exit_description : "nullptr",
                bNotifyStateChange ? "true" : "false");

    // Exit status already set
    if (m_state == lldb::eStateExited)
    {
        if (log)
            log->Printf ("NativeProcessProtocol::%s exit status already set to %d, ignoring new set to %d", __FUNCTION__, m_exit_status, status);
        return false;
    }

    m_state = lldb::eStateExited;

    m_exit_type = exit_type;
    m_exit_status = status;
    if (exit_description && exit_description[0])
        m_exit_description = exit_description;
    else
        m_exit_description.clear();

    if (bNotifyStateChange)
        SynchronouslyNotifyProcessStateChanged (lldb::eStateExited);

    return true;
}

NativeThreadProtocolSP
NativeProcessProtocol::GetThreadAtIndex (uint32_t idx)
{
    std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
    if (idx < m_threads.size ())
        return m_threads[idx];
    return NativeThreadProtocolSP ();
}

NativeThreadProtocolSP
NativeProcessProtocol::GetThreadByIDUnlocked (lldb::tid_t tid)
{
    for (auto thread_sp : m_threads)
    {
        if (thread_sp->GetID() == tid)
            return thread_sp;
    }
    return NativeThreadProtocolSP ();
}

NativeThreadProtocolSP
NativeProcessProtocol::GetThreadByID (lldb::tid_t tid)
{
    std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
    return GetThreadByIDUnlocked (tid);
}

bool
NativeProcessProtocol::IsAlive () const
{
    return m_state != eStateDetached
        && m_state != eStateExited
        && m_state != eStateInvalid
        && m_state != eStateUnloaded;
}

bool
NativeProcessProtocol::GetByteOrder (lldb::ByteOrder &byte_order) const
{
    ArchSpec process_arch;
    if (!GetArchitecture (process_arch))
        return false;
    byte_order = process_arch.GetByteOrder ();
    return true;
}

const NativeWatchpointList::WatchpointMap&
NativeProcessProtocol::GetWatchpointMap () const
{
    return m_watchpoint_list.GetWatchpointMap();
}

uint32_t
NativeProcessProtocol::GetMaxWatchpoints () const
{
    // This default implementation will return the number of
    // *hardware* breakpoints available.  MacOSX and other OS
    // implementations that support software breakpoints will want to
    // override this correctly for their implementation.
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // get any thread
    NativeThreadProtocolSP thread_sp (const_cast<NativeProcessProtocol*> (this)->GetThreadAtIndex (0));
    if (!thread_sp)
    {
        if (log)
            log->Warning ("NativeProcessProtocol::%s (): failed to find a thread to grab a NativeRegisterContext!", __FUNCTION__);
        return 0;
    }

    NativeRegisterContextSP reg_ctx_sp (thread_sp->GetRegisterContext ());
    if (!reg_ctx_sp)
    {
        if (log)
            log->Warning ("NativeProcessProtocol::%s (): failed to get a RegisterContextNativeProcess from the first thread!", __FUNCTION__);
        return 0;
    }

    return reg_ctx_sp->NumSupportedHardwareWatchpoints ();
}

Error
NativeProcessProtocol::SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware)
{
    // This default implementation assumes setting the watchpoint for
    // the process will require setting the watchpoint for each of the
    // threads.  Furthermore, it will track watchpoints set for the
    // process and will add them to each thread that is attached to
    // via the (FIXME implement) OnThreadAttached () method.

    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Update the thread list
    UpdateThreads ();

    // Keep track of the threads we successfully set the watchpoint
    // for.  If one of the thread watchpoint setting operations fails,
    // back off and remove the watchpoint for all the threads that
    // were successfully set so we get back to a consistent state.
    std::vector<NativeThreadProtocolSP> watchpoint_established_threads;

    // Tell each thread to set a watchpoint.  In the event that
    // hardware watchpoints are requested but the SetWatchpoint fails,
    // try to set a software watchpoint as a fallback.  It's
    // conceivable that if there are more threads than hardware
    // watchpoints available, some of the threads will fail to set
    // hardware watchpoints while software ones may be available.
    std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not have a NULL thread!");
        if (!thread_sp)
            continue;

        Error thread_error = thread_sp->SetWatchpoint (addr, size, watch_flags, hardware);
        if (thread_error.Fail () && hardware)
        {
            // Try software watchpoints since we failed on hardware watchpoint setting
            // and we may have just run out of hardware watchpoints.
            thread_error = thread_sp->SetWatchpoint (addr, size, watch_flags, false);
            if (thread_error.Success ())
            {
                if (log)
                    log->Warning ("hardware watchpoint requested but software watchpoint set"); 
            }
        }

        if (thread_error.Success ())
        {
            // Remember that we set this watchpoint successfully in
            // case we need to clear it later.
            watchpoint_established_threads.push_back (thread_sp);
        }
        else
        {
            // Unset the watchpoint for each thread we successfully
            // set so that we get back to a consistent state of "not
            // set" for the watchpoint.
            for (auto unwatch_thread_sp : watchpoint_established_threads)
            {
                Error remove_error = unwatch_thread_sp->RemoveWatchpoint (addr);
                if (remove_error.Fail () && log)
                {
                    log->Warning ("NativeProcessProtocol::%s (): RemoveWatchpoint failed for pid=%" PRIu64 ", tid=%" PRIu64 ": %s",
                            __FUNCTION__, GetID (), unwatch_thread_sp->GetID (), remove_error.AsCString ());
                }
            }

            return thread_error;
        }
    }
    return m_watchpoint_list.Add (addr, size, watch_flags, hardware);
}

Error
NativeProcessProtocol::RemoveWatchpoint (lldb::addr_t addr)
{
    // Update the thread list
    UpdateThreads ();

    Error overall_error;

    std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not have a NULL thread!");
        if (!thread_sp)
            continue;

        const Error thread_error = thread_sp->RemoveWatchpoint (addr);
        if (thread_error.Fail ())
        {
            // Keep track of the first thread error if any threads
            // fail. We want to try to remove the watchpoint from
            // every thread, though, even if one or more have errors.
            if (!overall_error.Fail ())
                overall_error = thread_error;
        }
    }
    const Error error = m_watchpoint_list.Remove(addr);
    return overall_error.Fail() ? overall_error : error;
}

bool
NativeProcessProtocol::RegisterNativeDelegate (NativeDelegate &native_delegate)
{
    std::lock_guard<std::recursive_mutex> guard(m_delegates_mutex);
    if (std::find (m_delegates.begin (), m_delegates.end (), &native_delegate) != m_delegates.end ())
        return false;

    m_delegates.push_back (&native_delegate);
    native_delegate.InitializeDelegate (this);
    return true;
}

bool
NativeProcessProtocol::UnregisterNativeDelegate (NativeDelegate &native_delegate)
{
    std::lock_guard<std::recursive_mutex> guard(m_delegates_mutex);

    const auto initial_size = m_delegates.size ();
    m_delegates.erase (remove (m_delegates.begin (), m_delegates.end (), &native_delegate), m_delegates.end ());

    // We removed the delegate if the count of delegates shrank after
    // removing all copies of the given native_delegate from the vector.
    return m_delegates.size () < initial_size;
}

void
NativeProcessProtocol::SynchronouslyNotifyProcessStateChanged (lldb::StateType state)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    std::lock_guard<std::recursive_mutex> guard(m_delegates_mutex);
    for (auto native_delegate: m_delegates)
        native_delegate->ProcessStateChanged (this, state);

    if (log)
    {
        if (!m_delegates.empty ())
        {
            log->Printf ("NativeProcessProtocol::%s: sent state notification [%s] from process %" PRIu64,
                    __FUNCTION__, lldb_private::StateAsCString (state),  GetID ());
        }
        else
        {
            log->Printf ("NativeProcessProtocol::%s: would send state notification [%s] from process %" PRIu64 ", but no delegates",
                    __FUNCTION__, lldb_private::StateAsCString (state),  GetID ());
        }
    }
}

void
NativeProcessProtocol::NotifyDidExec ()
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessProtocol::%s - preparing to call delegates", __FUNCTION__);

    {
        std::lock_guard<std::recursive_mutex> guard(m_delegates_mutex);
        for (auto native_delegate: m_delegates)
            native_delegate->DidExec (this);
    }
}


Error
NativeProcessProtocol::SetSoftwareBreakpoint (lldb::addr_t addr, uint32_t size_hint)
{
    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("NativeProcessProtocol::%s addr = 0x%" PRIx64, __FUNCTION__, addr);

    return m_breakpoint_list.AddRef (addr, size_hint, false,
            [this] (lldb::addr_t addr, size_t size_hint, bool /* hardware */, NativeBreakpointSP &breakpoint_sp)->Error
            { return SoftwareBreakpoint::CreateSoftwareBreakpoint (*this, addr, size_hint, breakpoint_sp); });
}

Error
NativeProcessProtocol::RemoveBreakpoint (lldb::addr_t addr)
{
    return m_breakpoint_list.DecRef (addr);
}

Error
NativeProcessProtocol::EnableBreakpoint (lldb::addr_t addr)
{
    return m_breakpoint_list.EnableBreakpoint (addr);
}

Error
NativeProcessProtocol::DisableBreakpoint (lldb::addr_t addr)
{
    return m_breakpoint_list.DisableBreakpoint (addr);
}

lldb::StateType
NativeProcessProtocol::GetState () const
{
    std::lock_guard<std::recursive_mutex> guard(m_state_mutex);
    return m_state;
}

void
NativeProcessProtocol::SetState (lldb::StateType state, bool notify_delegates)
{
    std::lock_guard<std::recursive_mutex> guard(m_state_mutex);

    if (state == m_state)
        return;

    m_state = state;

    if (StateIsStoppedState (state, false))
    {
        ++m_stop_id;

        // Give process a chance to do any stop id bump processing, such as
        // clearing cached data that is invalidated each time the process runs.
        // Note if/when we support some threads running, we'll end up needing
        // to manage this per thread and per process.
        DoStopIDBumped (m_stop_id);
    }

    // Optionally notify delegates of the state change.
    if (notify_delegates)
        SynchronouslyNotifyProcessStateChanged (state);
}

uint32_t NativeProcessProtocol::GetStopID () const
{
    std::lock_guard<std::recursive_mutex> guard(m_state_mutex);
    return m_stop_id;
}

void
NativeProcessProtocol::DoStopIDBumped (uint32_t /* newBumpId */)
{
    // Default implementation does nothing.
}

Error
NativeProcessProtocol::ResolveProcessArchitecture(lldb::pid_t pid,
                                                  ArchSpec &arch)
{
    // Grab process info for the running process.
    ProcessInstanceInfo process_info;
    if (!Host::GetProcessInfo(pid, process_info))
        return Error("failed to get process info");

    // Resolve the executable module.
    ModuleSpecList module_specs;
    if (!ObjectFile::GetModuleSpecifications(process_info.GetExecutableFile(),
                                             0, 0, module_specs))
        return Error("failed to get module specifications");
    lldbassert(module_specs.GetSize() == 1);

    arch = module_specs.GetModuleSpecRefAtIndex(0).GetArchitecture();
    if (arch.IsValid())
        return Error();
    else
        return Error("failed to retrieve a valid architecture from the exe module");
}

#ifndef __linux__
// These need to be implemented to support lldb-gdb-server on a given platform. Stubs are
// provided to make the rest of the code link on non-supported platforms.

Error
NativeProcessProtocol::Launch (ProcessLaunchInfo &launch_info,
        NativeDelegate &native_delegate,
        MainLoop &mainloop,
        NativeProcessProtocolSP &process_sp)
{
    llvm_unreachable("Platform has no NativeProcessProtocol support");
}

Error
NativeProcessProtocol::Attach (lldb::pid_t pid,
        NativeDelegate &native_delegate,
        MainLoop &mainloop,
        NativeProcessProtocolSP &process_sp)
{
    llvm_unreachable("Platform has no NativeProcessProtocol support");
}

#endif
