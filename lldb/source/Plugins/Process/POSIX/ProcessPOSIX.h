//===-- ProcessPOSIX.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessPOSIX_H_
#define liblldb_ProcessPOSIX_H_

// C Includes

// C++ Includes
#include <queue>

// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "lldb/Target/UnixSignals.h"
#include "ProcessMessage.h"

class ProcessMonitor;

class ProcessPOSIX :
    public lldb_private::Process
{
public:

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    ProcessPOSIX(lldb_private::Target& target,
                 lldb_private::Listener &listener);

    virtual
    ~ProcessPOSIX();

    //------------------------------------------------------------------
    // Process protocol.
    //------------------------------------------------------------------
    virtual bool
    CanDebug(lldb_private::Target &target, bool plugin_specified_by_name);

    virtual lldb_private::Error
    WillLaunch(lldb_private::Module *module);

    virtual lldb_private::Error
    DoAttachToProcessWithID(lldb::pid_t pid);

    virtual lldb_private::Error
    DoAttachToProcessWithID (lldb::pid_t pid, const lldb_private::ProcessAttachInfo &attach_info);

    virtual lldb_private::Error
    DoLaunch (lldb_private::Module *exe_module, 
              const lldb_private::ProcessLaunchInfo &launch_info);

    virtual void
    DidLaunch();

    virtual lldb_private::Error
    DoResume();

    virtual lldb_private::Error
    DoHalt(bool &caused_stop);

    virtual lldb_private::Error
    DoDetach(bool keep_stopped);

    virtual lldb_private::Error
    DoSignal(int signal);

    virtual lldb_private::Error
    DoDestroy();

    virtual void
    RefreshStateAfterStop();

    virtual bool
    IsAlive();

    virtual size_t
    DoReadMemory(lldb::addr_t vm_addr,
                 void *buf,
                 size_t size,
                 lldb_private::Error &error);

    virtual size_t
    DoWriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                  lldb_private::Error &error);

    virtual lldb::addr_t
    DoAllocateMemory(size_t size, uint32_t permissions,
                     lldb_private::Error &error);

    virtual lldb_private::Error
    DoDeallocateMemory(lldb::addr_t ptr);

    virtual lldb::addr_t
    ResolveIndirectFunction(const lldb_private::Address *address, lldb_private::Error &error);

    virtual size_t
    GetSoftwareBreakpointTrapOpcode(lldb_private::BreakpointSite* bp_site);

    virtual lldb_private::Error
    EnableBreakpointSite(lldb_private::BreakpointSite *bp_site);

    virtual lldb_private::Error
    DisableBreakpointSite(lldb_private::BreakpointSite *bp_site);

    virtual lldb_private::Error
    EnableWatchpoint(lldb_private::Watchpoint *wp, bool notify = true);

    virtual lldb_private::Error
    DisableWatchpoint(lldb_private::Watchpoint *wp, bool notify = true);

    virtual lldb_private::Error
    GetWatchpointSupportInfo(uint32_t &num);

    virtual lldb_private::Error
    GetWatchpointSupportInfo(uint32_t &num, bool &after);

    virtual uint32_t
    UpdateThreadListIfNeeded();

    virtual bool
    UpdateThreadList(lldb_private::ThreadList &old_thread_list, 
                     lldb_private::ThreadList &new_thread_list) = 0;

    virtual lldb::ByteOrder
    GetByteOrder() const;

    virtual lldb::addr_t
    GetImageInfoAddress();

    virtual size_t
    PutSTDIN(const char *buf, size_t len, lldb_private::Error &error);

    //--------------------------------------------------------------------------
    // ProcessPOSIX internal API.

    /// Registers the given message with this process.
    void SendMessage(const ProcessMessage &message);

    ProcessMonitor &
    GetMonitor() { assert(m_monitor); return *m_monitor; }

    lldb_private::UnixSignals &
    GetUnixSignals();

    const char *
    GetFilePath(const lldb_private::ProcessLaunchInfo::FileAction *file_action,
                const char *default_path);

    /// Stops all threads in the process.
    /// The \p stop_tid parameter indicates the thread which initiated the stop.
    virtual void
    StopAllThreads(lldb::tid_t stop_tid);

protected:
    /// Target byte order.
    lldb::ByteOrder m_byte_order;

    /// Process monitor;
    ProcessMonitor *m_monitor;

    /// The module we are executing.
    lldb_private::Module *m_module;

    /// Message queue notifying this instance of inferior process state changes.
    lldb_private::Mutex m_message_mutex;
    std::queue<ProcessMessage> m_message_queue;

    /// True when the process has entered a state of "limbo".
    ///
    /// This flag qualifies eStateStopped.  It lets us know that when we
    /// continue from this state the process will exit.  Also, when true,
    /// Process::m_exit_status is set.
    bool m_in_limbo;

    /// Drive any exit events to completion.
    bool m_exit_now;

    /// OS-specific signal set.
    lldb_private::UnixSignals m_signals;

    /// Returns true if the process has exited.
    bool HasExited();

    /// Returns true if the process is stopped.
    bool IsStopped();

    typedef std::map<lldb::addr_t, lldb::addr_t> MMapMap;
    MMapMap m_addr_to_mmap_size;
};

#endif  // liblldb_MacOSXProcess_H_
