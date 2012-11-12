//===-- ProcessKDP.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/State.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionGroupString.h"
#include "lldb/Interpreter/OptionGroupUInt64.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

// Project includes
#include "ProcessKDP.h"
#include "ProcessKDPLog.h"
#include "ThreadKDP.h"
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"
#include "Utility/StringExtractor.h"

using namespace lldb;
using namespace lldb_private;

const char *
ProcessKDP::GetPluginNameStatic()
{
    return "kdp-remote";
}

const char *
ProcessKDP::GetPluginDescriptionStatic()
{
    return "KDP Remote protocol based debugging plug-in for darwin kernel debugging.";
}

void
ProcessKDP::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessKDP::CreateInstance);
}


lldb::ProcessSP
ProcessKDP::CreateInstance (Target &target, 
                            Listener &listener,
                            const FileSpec *crash_file_path)
{
    lldb::ProcessSP process_sp;
    if (crash_file_path == NULL)
        process_sp.reset(new ProcessKDP (target, listener));
    return process_sp;
}

bool
ProcessKDP::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    Module *exe_module = target.GetExecutableModulePointer();
    if (exe_module)
    {
        const llvm::Triple &triple_ref = target.GetArchitecture().GetTriple();
        switch (triple_ref.getOS())
        {
            case llvm::Triple::Darwin:  // Should use "macosx" for desktop and "ios" for iOS, but accept darwin just in case
            case llvm::Triple::MacOSX:  // For desktop targets
            case llvm::Triple::IOS:     // For arm targets
                if (triple_ref.getVendor() == llvm::Triple::Apple)
                {
                    ObjectFile *exe_objfile = exe_module->GetObjectFile();
                    if (exe_objfile->GetType() == ObjectFile::eTypeExecutable && 
                        exe_objfile->GetStrata() == ObjectFile::eStrataKernel)
                        return true;
                }
                break;

            default:
                break;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// ProcessKDP constructor
//----------------------------------------------------------------------
ProcessKDP::ProcessKDP(Target& target, Listener &listener) :
    Process (target, listener),
    m_comm("lldb.process.kdp-remote.communication"),
    m_async_broadcaster (NULL, "lldb.process.kdp-remote.async-broadcaster"),
    m_async_thread (LLDB_INVALID_HOST_THREAD),
    m_destroy_in_process (false),
    m_dyld_plugin_name (),
    m_kernel_load_addr (LLDB_INVALID_ADDRESS),
    m_command_sp()
{
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncThreadShouldExit,   "async thread should exit");
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncContinue,           "async thread continue");
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessKDP::~ProcessKDP()
{
    Clear();
    // We need to call finalize on the process before destroying ourselves
    // to make sure all of the broadcaster cleanup goes as planned. If we
    // destruct this class, then Process::~Process() might have problems
    // trying to fully destroy the broadcaster.
    Finalize();
}

//----------------------------------------------------------------------
// PluginInterface
//----------------------------------------------------------------------
const char *
ProcessKDP::GetPluginName()
{
    return "Process debugging plug-in that uses the Darwin KDP remote protocol";
}

const char *
ProcessKDP::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessKDP::GetPluginVersion()
{
    return 1;
}

Error
ProcessKDP::WillLaunch (Module* module)
{
    Error error;
    error.SetErrorString ("launching not supported in kdp-remote plug-in");
    return error;
}

Error
ProcessKDP::WillAttachToProcessWithID (lldb::pid_t pid)
{
    Error error;
    error.SetErrorString ("attaching to a by process ID not supported in kdp-remote plug-in");
    return error;
}

Error
ProcessKDP::WillAttachToProcessWithName (const char *process_name, bool wait_for_launch)
{
    Error error;
    error.SetErrorString ("attaching to a by process name not supported in kdp-remote plug-in");
    return error;
}

Error
ProcessKDP::DoConnectRemote (Stream *strm, const char *remote_url)
{
    Error error;

    // Don't let any JIT happen when doing KDP as we can't allocate
    // memory and we don't want to be mucking with threads that might
    // already be handling exceptions
    SetCanJIT(false);

    if (remote_url == NULL || remote_url[0] == '\0')
    {
        error.SetErrorStringWithFormat ("invalid connection URL '%s'", remote_url);
        return error;
    }

    std::auto_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
    if (conn_ap.get())
    {
        // Only try once for now.
        // TODO: check if we should be retrying?
        const uint32_t max_retry_count = 1;
        for (uint32_t retry_count = 0; retry_count < max_retry_count; ++retry_count)
        {
            if (conn_ap->Connect(remote_url, &error) == eConnectionStatusSuccess)
                break;
            usleep (100000);
        }
    }

    if (conn_ap->IsConnected())
    {
        const uint16_t reply_port = conn_ap->GetReadPort ();

        if (reply_port != 0)
        {
            m_comm.SetConnection(conn_ap.release());

            if (m_comm.SendRequestReattach(reply_port))
            {
                if (m_comm.SendRequestConnect(reply_port, reply_port, "Greetings from LLDB..."))
                {
                    m_comm.GetVersion();
                    uint32_t cpu = m_comm.GetCPUType();
                    uint32_t sub = m_comm.GetCPUSubtype();
                    ArchSpec kernel_arch;
                    kernel_arch.SetArchitecture(eArchTypeMachO, cpu, sub);
                    m_target.SetArchitecture(kernel_arch);

                    /* Get the kernel's UUID and load address via KDP_KERNELVERSION packet.  */
                    /* An EFI kdp session has neither UUID nor load address. */

                    UUID kernel_uuid = m_comm.GetUUID ();
                    addr_t kernel_load_addr = m_comm.GetLoadAddress ();

                    if (m_comm.RemoteIsEFI ())
                    {
                        m_dyld_plugin_name = DynamicLoaderStatic::GetPluginNameStatic();
                    }
                    else if (kernel_load_addr != LLDB_INVALID_ADDRESS)
                    {
                        m_kernel_load_addr = kernel_load_addr;
                        m_dyld_plugin_name = DynamicLoaderDarwinKernel::GetPluginNameStatic();
                    }

                    // Set the thread ID
                    UpdateThreadListIfNeeded ();
                    SetID (1);
                    GetThreadList ();
                    SetPrivateState (eStateStopped);
                    StreamSP async_strm_sp(m_target.GetDebugger().GetAsyncOutputStream());
                    if (async_strm_sp)
                    {
                        const char *cstr;
                        if ((cstr = m_comm.GetKernelVersion ()) != NULL)
                        {
                            async_strm_sp->Printf ("Version: %s\n", cstr);
                            async_strm_sp->Flush();
                        }
//                      if ((cstr = m_comm.GetImagePath ()) != NULL)
//                      {
//                          async_strm_sp->Printf ("Image Path: %s\n", cstr);
//                          async_strm_sp->Flush();
//                      }            
                    }
                }
                else
                {
                    error.SetErrorString("KDP_REATTACH failed");
                }
            }
            else
            {
                error.SetErrorString("KDP_REATTACH failed");
            }
        }
        else
        {
            error.SetErrorString("invalid reply port from UDP connection");
        }
    }
    else
    {
        if (error.Success())
            error.SetErrorStringWithFormat ("failed to connect to '%s'", remote_url);
    }
    if (error.Fail())
        m_comm.Disconnect();

    return error;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessKDP::DoLaunch (Module *exe_module, 
                      const ProcessLaunchInfo &launch_info)
{
    Error error;
    error.SetErrorString ("launching not supported in kdp-remote plug-in");
    return error;
}


Error
ProcessKDP::DoAttachToProcessWithID (lldb::pid_t attach_pid)
{
    Error error;
    error.SetErrorString ("attach to process by ID is not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DoAttachToProcessWithID (lldb::pid_t attach_pid, const ProcessAttachInfo &attach_info)
{
    Error error;
    error.SetErrorString ("attach to process by ID is not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DoAttachToProcessWithName (const char *process_name, bool wait_for_launch, const ProcessAttachInfo &attach_info)
{
    Error error;
    error.SetErrorString ("attach to process by name is not suppported in kdp remote debugging");
    return error;
}


void
ProcessKDP::DidAttach ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::DidAttach()");
    if (GetID() != LLDB_INVALID_PROCESS_ID)
    {
        // TODO: figure out the register context that we will use
    }
}

addr_t
ProcessKDP::GetImageInfoAddress()
{
    return m_kernel_load_addr;
}

lldb_private::DynamicLoader *
ProcessKDP::GetDynamicLoader ()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, m_dyld_plugin_name.empty() ? NULL : m_dyld_plugin_name.c_str()));
    return m_dyld_ap.get();
}

Error
ProcessKDP::WillResume ()
{
    return Error();
}

Error
ProcessKDP::DoResume ()
{
    Error error;
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    // Only start the async thread if we try to do any process control
    if (!IS_VALID_LLDB_HOST_THREAD(m_async_thread))
        StartAsyncThread ();

    bool resume = false;
    
    // With KDP there is only one thread we can tell what to do
    ThreadSP kernel_thread_sp (GetKernelThread(m_thread_list, m_thread_list));
    if (kernel_thread_sp)
    {
        const StateType thread_resume_state = kernel_thread_sp->GetTemporaryResumeState();
        switch (thread_resume_state)
        {
            case eStateSuspended:
                // Nothing to do here when a thread will stay suspended
                // we just leave the CPU mask bit set to zero for the thread
                break;
                
            case eStateStepping:
                kernel_thread_sp->GetRegisterContext()->HardwareSingleStep (true);
                resume = true;
                break;
    
            case eStateRunning:
                kernel_thread_sp->GetRegisterContext()->HardwareSingleStep (false);
                resume = true;
                break;

            default:
                // The only valid thread resume states are listed above
                assert (!"invalid thread resume state");
                break;
        }
    }

    if (resume)
    {
        if (log)
            log->Printf ("ProcessKDP::DoResume () sending resume");
        
        if (m_comm.SendRequestResume ())
        {
            m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncContinue);
            SetPrivateState(eStateRunning);
        }
        else
            error.SetErrorString ("KDP resume failed");
    }
    else
    {
        error.SetErrorString ("kernel thread is suspended");        
    }
    
    return error;
}

lldb::ThreadSP
ProcessKDP::GetKernelThread(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // KDP only tells us about one thread/core. Any other threads will usually
    // be the ones that are read from memory by the OS plug-ins.
    const lldb::tid_t kernel_tid = 1;
    ThreadSP thread_sp (old_thread_list.FindThreadByID (kernel_tid, false));
    if (!thread_sp)
    {
        thread_sp.reset(new ThreadKDP (*this, kernel_tid));
        new_thread_list.AddThread(thread_sp);
    }
    return thread_sp;
}




bool
ProcessKDP::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // locker will keep a mutex locked until it goes out of scope
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_THREAD));
    if (log && log->GetMask().Test(KDP_LOG_VERBOSE))
        log->Printf ("ProcessKDP::%s (pid = %llu)", __FUNCTION__, GetID());
    
    // Even though there is a CPU mask, it doesn't mean to can see each CPU
    // indivudually, there is really only one. Lets call this thread 1.
    GetKernelThread (old_thread_list, new_thread_list);

    return new_thread_list.GetSize(false) > 0;
}

void
ProcessKDP::RefreshStateAfterStop ()
{
    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();
}

Error
ProcessKDP::DoHalt (bool &caused_stop)
{
    Error error;
    
    if (m_comm.IsRunning())
    {
        if (m_destroy_in_process)
        {
            // If we are attemping to destroy, we need to not return an error to
            // Halt or DoDestroy won't get called.
            // We are also currently running, so send a process stopped event
            SetPrivateState (eStateStopped);
        }
        else
        {
            error.SetErrorString ("KDP cannot interrupt a running kernel");
        }
    }
    return error;
}

Error
ProcessKDP::DoDetach()
{
    Error error;
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::DoDetach()");
    
    if (m_comm.IsRunning())
    {
        // We are running and we can't interrupt a running kernel, so we need
        // to just close the connection to the kernel and hope for the best
    }
    else
    {
        DisableAllBreakpointSites ();
        
        m_thread_list.DiscardThreadPlans();
        
        if (m_comm.IsConnected())
        {

            m_comm.SendRequestDisconnect();

            size_t response_size = m_comm.Disconnect ();
            if (log)
            {
                if (response_size)
                    log->PutCString ("ProcessKDP::DoDetach() detach packet sent successfully");
                else
                    log->PutCString ("ProcessKDP::DoDetach() detach packet send failed");
            }
        }
    }
    StopAsyncThread ();    
    m_comm.Clear();
    
    SetPrivateState (eStateDetached);
    ResumePrivateStateThread();
    
    //KillDebugserverProcess ();
    return error;
}

Error
ProcessKDP::WillDestroy ()
{
    Error error;
    m_destroy_in_process = true;
    return error;
}

Error
ProcessKDP::DoDestroy ()
{
    // For KDP there really is no difference between destroy and detach
    return DoDetach();
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessKDP::IsAlive ()
{
    return m_comm.IsConnected() && m_private_state.GetValue() != eStateExited;
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------
size_t
ProcessKDP::DoReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    if (m_comm.IsConnected())
        return m_comm.SendRequestReadMemory (addr, buf, size, error);
    error.SetErrorString ("not connected");
    return 0;
}

size_t
ProcessKDP::DoWriteMemory (addr_t addr, const void *buf, size_t size, Error &error)
{
    if (m_comm.IsConnected())
        return m_comm.SendRequestWriteMemory (addr, buf, size, error);
    error.SetErrorString ("not connected");
    return 0;
}

lldb::addr_t
ProcessKDP::DoAllocateMemory (size_t size, uint32_t permissions, Error &error)
{
    error.SetErrorString ("memory allocation not suppported in kdp remote debugging");
    return LLDB_INVALID_ADDRESS;
}

Error
ProcessKDP::DoDeallocateMemory (lldb::addr_t addr)
{
    Error error;
    error.SetErrorString ("memory deallocation not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::EnableBreakpoint (BreakpointSite *bp_site)
{
    if (m_comm.LocalBreakpointsAreSupported ())
    {
        Error error;
        if (!bp_site->IsEnabled())
        {
            if (m_comm.SendRequestBreakpoint(true, bp_site->GetLoadAddress()))
            {
                bp_site->SetEnabled(true);
                bp_site->SetType (BreakpointSite::eExternal);
            }
            else
            {
                error.SetErrorString ("KDP set breakpoint failed");
            }
        }
        return error;
    }
    return EnableSoftwareBreakpoint (bp_site);
}

Error
ProcessKDP::DisableBreakpoint (BreakpointSite *bp_site)
{
    if (m_comm.LocalBreakpointsAreSupported ())
    {
        Error error;
        if (bp_site->IsEnabled())
        {
            BreakpointSite::Type bp_type = bp_site->GetType();
            if (bp_type == BreakpointSite::eExternal)
            {
                if (m_destroy_in_process && m_comm.IsRunning())
                {
                    // We are trying to destroy our connection and we are running
                    bp_site->SetEnabled(false);
                }
                else
                {
                    if (m_comm.SendRequestBreakpoint(false, bp_site->GetLoadAddress()))
                        bp_site->SetEnabled(false);
                    else
                        error.SetErrorString ("KDP remove breakpoint failed");
                }
            }
            else
            {
                error = DisableSoftwareBreakpoint (bp_site);
            }
        }
        return error;
    }
    return DisableSoftwareBreakpoint (bp_site);
}

Error
ProcessKDP::EnableWatchpoint (Watchpoint *wp)
{
    Error error;
    error.SetErrorString ("watchpoints are not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DisableWatchpoint (Watchpoint *wp)
{
    Error error;
    error.SetErrorString ("watchpoints are not suppported in kdp remote debugging");
    return error;
}

void
ProcessKDP::Clear()
{
    m_thread_list.Clear();
}

Error
ProcessKDP::DoSignal (int signo)
{
    Error error;
    error.SetErrorString ("sending signals is not suppported in kdp remote debugging");
    return error;
}

void
ProcessKDP::Initialize()
{
    static bool g_initialized = false;
    
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);
        
        Log::Callbacks log_callbacks = {
            ProcessKDPLog::DisableLog,
            ProcessKDPLog::EnableLog,
            ProcessKDPLog::ListLogCategories
        };
        
        Log::RegisterLogChannel (ProcessKDP::GetPluginNameStatic(), log_callbacks);
    }
}

bool
ProcessKDP::StartAsyncThread ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    
    if (log)
        log->Printf ("ProcessKDP::StartAsyncThread ()");
    
    if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
        return true;

    m_async_thread = Host::ThreadCreate ("<lldb.process.kdp-remote.async>", ProcessKDP::AsyncThread, this, NULL);
    return IS_VALID_LLDB_HOST_THREAD(m_async_thread);
}

void
ProcessKDP::StopAsyncThread ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    
    if (log)
        log->Printf ("ProcessKDP::StopAsyncThread ()");
    
    m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncThreadShouldExit);
    
    // Stop the stdio thread
    if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
    {
        Host::ThreadJoin (m_async_thread, NULL, NULL);
        m_async_thread = LLDB_INVALID_HOST_THREAD;
    }
}


void *
ProcessKDP::AsyncThread (void *arg)
{
    ProcessKDP *process = (ProcessKDP*) arg;
    
    const lldb::pid_t pid = process->GetID();

    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::AsyncThread (arg = %p, pid = %llu) thread starting...", arg, pid);
    
    Listener listener ("ProcessKDP::AsyncThread");
    EventSP event_sp;
    const uint32_t desired_event_mask = eBroadcastBitAsyncContinue |
                                        eBroadcastBitAsyncThreadShouldExit;
    
    
    if (listener.StartListeningForEvents (&process->m_async_broadcaster, desired_event_mask) == desired_event_mask)
    {
        bool done = false;
        while (!done)
        {
            if (log)
                log->Printf ("ProcessKDP::AsyncThread (pid = %llu) listener.WaitForEvent (NULL, event_sp)...",
                             pid);
            if (listener.WaitForEvent (NULL, event_sp))
            {
                uint32_t event_type = event_sp->GetType();
                if (log)
                    log->Printf ("ProcessKDP::AsyncThread (pid = %llu) Got an event of type: %d...",
                                 pid,
                                 event_type);
                
                // When we are running, poll for 1 second to try and get an exception
                // to indicate the process has stopped. If we don't get one, check to
                // make sure no one asked us to exit
                bool is_running = false;
                DataExtractor exc_reply_packet;
                do
                {
                    switch (event_type)
                    {
                    case eBroadcastBitAsyncContinue:
                        {
                            is_running = true;
                            if (process->m_comm.WaitForPacketWithTimeoutMicroSeconds (exc_reply_packet, 1 * USEC_PER_SEC))
                            {
                                ThreadSP thread_sp (process->GetKernelThread(process->GetThreadList(), process->GetThreadList()));
                                thread_sp->GetRegisterContext()->InvalidateAllRegisters();
                                static_cast<ThreadKDP *>(thread_sp.get())->SetStopInfoFrom_KDP_EXCEPTION (exc_reply_packet);

                                // TODO: parse the stop reply packet
                                is_running = false;                                
                                process->SetPrivateState(eStateStopped);
                            }
                            else
                            {
                                // Check to see if we are supposed to exit. There is no way to
                                // interrupt a running kernel, so all we can do is wait for an
                                // exception or detach...
                                if (listener.GetNextEvent(event_sp))
                                {
                                    // We got an event, go through the loop again
                                    event_type = event_sp->GetType();
                                }
                            }
                        }
                        break;
                            
                    case eBroadcastBitAsyncThreadShouldExit:
                        if (log)
                            log->Printf ("ProcessKDP::AsyncThread (pid = %llu) got eBroadcastBitAsyncThreadShouldExit...",
                                         pid);
                        done = true;
                        is_running = false;
                        break;
                            
                    default:
                        if (log)
                            log->Printf ("ProcessKDP::AsyncThread (pid = %llu) got unknown event 0x%8.8x",
                                         pid,
                                         event_type);
                        done = true;
                        is_running = false;
                        break;
                    }
                } while (is_running);
            }
            else
            {
                if (log)
                    log->Printf ("ProcessKDP::AsyncThread (pid = %llu) listener.WaitForEvent (NULL, event_sp) => false",
                                 pid);
                done = true;
            }
        }
    }
    
    if (log)
        log->Printf ("ProcessKDP::AsyncThread (arg = %p, pid = %llu) thread exiting...",
                     arg,
                     pid);
    
    process->m_async_thread = LLDB_INVALID_HOST_THREAD;
    return NULL;
}


class CommandObjectProcessKDPPacketSend : public CommandObjectParsed
{
private:
    
    OptionGroupOptions m_option_group;
    OptionGroupUInt64 m_command_byte;
    OptionGroupString m_packet_data;
    
    virtual Options *
    GetOptions ()
    {
        return &m_option_group;
    }
    

public:
    CommandObjectProcessKDPPacketSend(CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "process plugin packet send",
                             "Send a custom packet through the KDP protocol by specifying the command byte and the packet payload data. A packet will be sent with a correct header and payload, and the raw result bytes will be displayed as a string value. ",
                             NULL),
        m_option_group (interpreter),
        m_command_byte(LLDB_OPT_SET_1, true , "command", 'c', 0, eArgTypeNone, "Specify the command byte to use when sending the KDP request packet.", 0),
        m_packet_data (LLDB_OPT_SET_1, false, "payload", 'p', 0, eArgTypeNone, "Specify packet payload bytes as a hex ASCII string with no spaces or hex prefixes.", NULL)
    {
        m_option_group.Append (&m_command_byte, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_packet_data , LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Finalize();
    }
    
    ~CommandObjectProcessKDPPacketSend ()
    {
    }
    
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        if (argc == 0)
        {
            if (!m_command_byte.GetOptionValue().OptionWasSet())
            {
                result.AppendError ("the --command option must be set to a valid command byte");
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                const uint64_t command_byte = m_command_byte.GetOptionValue().GetUInt64Value(0);
                if (command_byte > 0 && command_byte <= UINT8_MAX)
                {
                    ProcessKDP *process = (ProcessKDP *)m_interpreter.GetExecutionContext().GetProcessPtr();
                    if (process)
                    {
                        const StateType state = process->GetState();
                        
                        if (StateIsStoppedState (state, true))
                        {
                            std::vector<uint8_t> payload_bytes;
                            const char *ascii_hex_bytes_cstr = m_packet_data.GetOptionValue().GetCurrentValue();
                            if (ascii_hex_bytes_cstr && ascii_hex_bytes_cstr[0])
                            {
                                StringExtractor extractor(ascii_hex_bytes_cstr);
                                const size_t ascii_hex_bytes_cstr_len = extractor.GetStringRef().size();
                                if (ascii_hex_bytes_cstr_len & 1)
                                {
                                    result.AppendErrorWithFormat ("payload data must contain an even number of ASCII hex characters: '%s'", ascii_hex_bytes_cstr);
                                    result.SetStatus (eReturnStatusFailed);
                                    return false;
                                }
                                payload_bytes.resize(ascii_hex_bytes_cstr_len/2);
                                if (extractor.GetHexBytes(&payload_bytes[0], payload_bytes.size(), '\xdd') != payload_bytes.size())
                                {
                                    result.AppendErrorWithFormat ("payload data must only contain ASCII hex characters (no spaces or hex prefixes): '%s'", ascii_hex_bytes_cstr);
                                    result.SetStatus (eReturnStatusFailed);
                                    return false;
                                }
                            }
                            Error error;
                            DataExtractor reply;
                            process->GetCommunication().SendRawRequest (command_byte,
                                                                        payload_bytes.empty() ? NULL : payload_bytes.data(),
                                                                        payload_bytes.size(),
                                                                        reply,
                                                                        error);
                            
                            if (error.Success())
                            {
                                // Copy the binary bytes into a hex ASCII string for the result
                                StreamString packet;
                                packet.PutBytesAsRawHex8(reply.GetDataStart(),
                                                         reply.GetByteSize(),
                                                         lldb::endian::InlHostByteOrder(),
                                                         lldb::endian::InlHostByteOrder());
                                result.AppendMessage(packet.GetString().c_str());
                                result.SetStatus (eReturnStatusSuccessFinishResult);
                                return true;
                            }
                            else
                            {
                                const char *error_cstr = error.AsCString();
                                if (error_cstr && error_cstr[0])
                                    result.AppendError (error_cstr);
                                else
                                    result.AppendErrorWithFormat ("unknown error 0x%8.8x", error.GetError());
                                result.SetStatus (eReturnStatusFailed);
                                return false;
                            }
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("process must be stopped in order to send KDP packets, state is %s", StateAsCString (state));
                            result.SetStatus (eReturnStatusFailed);
                        }
                    }
                    else
                    {
                        result.AppendError ("invalid process");
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    result.AppendErrorWithFormat ("invalid command byte 0x%llx, valid values are 1 - 255", command_byte);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        else
        {
            result.AppendErrorWithFormat ("'%s' takes no arguments, only options.", m_cmd_name.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        return false;
    }
};

class CommandObjectProcessKDPPacket : public CommandObjectMultiword
{
private:

public:
    CommandObjectProcessKDPPacket(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "process plugin packet",
                            "Commands that deal with KDP remote packets.",
                            NULL)
    {
        LoadSubCommand ("send", CommandObjectSP (new CommandObjectProcessKDPPacketSend (interpreter)));
    }
    
    ~CommandObjectProcessKDPPacket ()
    {
    }
};

class CommandObjectMultiwordProcessKDP : public CommandObjectMultiword
{
public:
    CommandObjectMultiwordProcessKDP (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "process plugin",
                            "A set of commands for operating on a ProcessKDP process.",
                            "process plugin <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("packet", CommandObjectSP (new CommandObjectProcessKDPPacket    (interpreter)));
    }
    
    ~CommandObjectMultiwordProcessKDP ()
    {
    }
};

CommandObject *
ProcessKDP::GetPluginCommandObject()
{
    if (!m_command_sp)
        m_command_sp.reset (new CommandObjectMultiwordProcessKDP (GetTarget().GetDebugger().GetCommandInterpreter()));
    return m_command_sp.get();
}

