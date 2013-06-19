//===-- ProcessLinux.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessLinux_H_
#define liblldb_ProcessLinux_H_

// C Includes

// C++ Includes
#include <queue>

// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "LinuxSignals.h"
#include "ProcessMessage.h"
#include "ProcessPOSIX.h"

class ProcessMonitor;

class ProcessLinux :
    public ProcessPOSIX
{
public:
    //------------------------------------------------------------------
    // Static functions.
    //------------------------------------------------------------------
    static lldb::ProcessSP
    CreateInstance(lldb_private::Target& target,
                   lldb_private::Listener &listener,
                   const lldb_private::FileSpec *);

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    ProcessLinux(lldb_private::Target& target,
                 lldb_private::Listener &listener);

    virtual bool
    UpdateThreadList(lldb_private::ThreadList &old_thread_list, lldb_private::ThreadList &new_thread_list);
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    GetPluginCommandHelp(const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand(lldb_private::Args &command,
                         lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging(lldb_private::Stream *strm,
                        lldb_private::Args &command);

    //------------------------------------------------------------------
    // Plug-in process overrides
    //------------------------------------------------------------------
    virtual lldb_private::UnixSignals &
    GetUnixSignals ()
    {
        return m_linux_signals;
    }

    //------------------------------------------------------------------
    // ProcessPOSIX overrides
    //------------------------------------------------------------------
    virtual void
    StopAllThreads(lldb::tid_t stop_tid);

private:

    /// Linux-specific signal set.
    LinuxSignals m_linux_signals;

    // Flag to avoid recursion when stopping all threads.
    bool m_stopping_threads;
};

#endif  // liblldb_MacOSXProcess_H_
