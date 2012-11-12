//===-- ProcessKDP.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessKDP_h_
#define liblldb_ProcessKDP_h_

// C Includes

// C++ Includes
#include <list>
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

#include "CommunicationKDP.h"
#include "Utility/StringExtractor.h"

class ThreadKDP;

class ProcessKDP : public lldb_private::Process
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    static lldb::ProcessSP
    CreateInstance (lldb_private::Target& target, 
                    lldb_private::Listener &listener,
                    const lldb_private::FileSpec *crash_file_path);
    
    static void
    Initialize();
    
    static void
    Terminate();
    
    static const char *
    GetPluginNameStatic();
    
    static const char *
    GetPluginDescriptionStatic();
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ProcessKDP(lldb_private::Target& target, lldb_private::Listener &listener);
    
    virtual
    ~ProcessKDP();
    
    //------------------------------------------------------------------
    // Check if a given Process
    //------------------------------------------------------------------
    virtual bool
    CanDebug (lldb_private::Target &target,
              bool plugin_specified_by_name);
    
    virtual lldb_private::CommandObject *
    GetPluginCommandObject();
    
    //------------------------------------------------------------------
    // Creating a new process, or attaching to an existing one
    //------------------------------------------------------------------
    virtual lldb_private::Error
    WillLaunch (lldb_private::Module* module);
    
    virtual lldb_private::Error
    DoLaunch (lldb_private::Module *exe_module, 
              const lldb_private::ProcessLaunchInfo &launch_info);
    
    virtual lldb_private::Error
    WillAttachToProcessWithID (lldb::pid_t pid);
    
    virtual lldb_private::Error
    WillAttachToProcessWithName (const char *process_name, bool wait_for_launch);
    
    virtual lldb_private::Error
    DoConnectRemote (lldb_private::Stream *strm, const char *remote_url);
    
    virtual lldb_private::Error
    DoAttachToProcessWithID (lldb::pid_t pid);
    
    virtual lldb_private::Error
    DoAttachToProcessWithID (lldb::pid_t pid, const lldb_private::ProcessAttachInfo &attach_info);
    
    virtual lldb_private::Error
    DoAttachToProcessWithName (const char *process_name, bool wait_for_launch, const lldb_private::ProcessAttachInfo &attach_info);
    
    virtual void
    DidAttach ();
    
    lldb::addr_t
    GetImageInfoAddress();

    lldb_private::DynamicLoader *
    GetDynamicLoader ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    //------------------------------------------------------------------
    // Process Control
    //------------------------------------------------------------------
    virtual lldb_private::Error
    WillResume ();
    
    virtual lldb_private::Error
    DoResume ();
    
    virtual lldb_private::Error
    DoHalt (bool &caused_stop);
    
    virtual lldb_private::Error
    DoDetach ();
    
    virtual lldb_private::Error
    DoSignal (int signal);
    
    virtual lldb_private::Error
    WillDestroy ();
    
    virtual lldb_private::Error
    DoDestroy ();
    
    virtual void
    RefreshStateAfterStop();
    
    //------------------------------------------------------------------
    // Process Queries
    //------------------------------------------------------------------
    virtual bool
    IsAlive ();
    
    //------------------------------------------------------------------
    // Process Memory
    //------------------------------------------------------------------
    virtual size_t
    DoReadMemory (lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error);
    
    virtual size_t
    DoWriteMemory (lldb::addr_t addr, const void *buf, size_t size, lldb_private::Error &error);
    
    virtual lldb::addr_t
    DoAllocateMemory (size_t size, uint32_t permissions, lldb_private::Error &error);
    
    virtual lldb_private::Error
    DoDeallocateMemory (lldb::addr_t ptr);

    //----------------------------------------------------------------------
    // Process Breakpoints
    //----------------------------------------------------------------------
    virtual lldb_private::Error
    EnableBreakpoint (lldb_private::BreakpointSite *bp_site);
    
    virtual lldb_private::Error
    DisableBreakpoint (lldb_private::BreakpointSite *bp_site);
    
    //----------------------------------------------------------------------
    // Process Watchpoints
    //----------------------------------------------------------------------
    virtual lldb_private::Error
    EnableWatchpoint (lldb_private::Watchpoint *wp);
    
    virtual lldb_private::Error
    DisableWatchpoint (lldb_private::Watchpoint *wp);
    
    CommunicationKDP &
    GetCommunication()
    {
        return m_comm;
    }

protected:
    friend class ThreadKDP;
    friend class CommunicationKDP;
    
    //----------------------------------------------------------------------
    // Accessors
    //----------------------------------------------------------------------
    bool
    IsRunning ( lldb::StateType state )
    {
        return    state == lldb::eStateRunning || IsStepping(state);
    }
    
    bool
    IsStepping ( lldb::StateType state)
    {
        return    state == lldb::eStateStepping;
    }

    bool
    CanResume ( lldb::StateType state)
    {
        return state == lldb::eStateStopped;
    }
    
    bool
    HasExited (lldb::StateType state)
    {
        return state == lldb::eStateExited;
    }
    
    bool
    ProcessIDIsValid ( ) const;
    
    //    static void
    //    STDIOReadThreadBytesReceived (void *baton, const void *src, size_t src_len);
    
    //    void
    //    AppendSTDOUT (const char* s, size_t len);
    
    void
    Clear ( );
    
    virtual bool
    UpdateThreadList (lldb_private::ThreadList &old_thread_list, 
                      lldb_private::ThreadList &new_thread_list);
    
    enum
    {
        eBroadcastBitAsyncContinue                  = (1 << 0),
        eBroadcastBitAsyncThreadShouldExit          = (1 << 1)
    };
    
    lldb::ThreadSP
    GetKernelThread (lldb_private::ThreadList &old_thread_list,
                     lldb_private::ThreadList &new_thread_list);

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    CommunicationKDP m_comm;
    lldb_private::Broadcaster m_async_broadcaster;
    lldb::thread_t m_async_thread;
    bool m_destroy_in_process;
    std::string m_dyld_plugin_name;
    lldb::addr_t m_kernel_load_addr;
    lldb::CommandObjectSP m_command_sp;


    bool
    StartAsyncThread ();
    
    void
    StopAsyncThread ();
    
    static void *
    AsyncThread (void *arg);
    
private:
    //------------------------------------------------------------------
    // For ProcessKDP only
    //------------------------------------------------------------------
    
    DISALLOW_COPY_AND_ASSIGN (ProcessKDP);
    
};

#endif  // liblldb_ProcessKDP_h_
