//===-- ProcessPOSIX.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Target.h"

#include "ProcessPOSIX.h"
#include "ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "ProcessMonitor.h"
#include "POSIXThread.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Static functions.
#if 0
Process*
ProcessPOSIX::CreateInstance(Target& target, Listener &listener)
{
    return new ProcessPOSIX(target, listener);
}


void
ProcessPOSIX::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);

        Log::Callbacks log_callbacks = {
            ProcessPOSIXLog::DisableLog,
            ProcessPOSIXLog::EnableLog,
            ProcessPOSIXLog::ListLogCategories
        };
        
        Log::RegisterLogChannel (ProcessPOSIX::GetPluginNameStatic(), log_callbacks);
    }
}
#endif

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessPOSIX::ProcessPOSIX(Target& target, Listener &listener)
    : Process(target, listener),
      m_byte_order(lldb::endian::InlHostByteOrder()),
      m_monitor(NULL),
      m_module(NULL),
      m_in_limbo(false),
      m_exit_now(false)
{
    // FIXME: Putting this code in the ctor and saving the byte order in a
    // member variable is a hack to avoid const qual issues in GetByteOrder.
	lldb::ModuleSP module = GetTarget().GetExecutableModule();
	if (module != NULL && module->GetObjectFile() != NULL)
		m_byte_order = module->GetObjectFile()->GetByteOrder();
}

ProcessPOSIX::~ProcessPOSIX()
{
    delete m_monitor;
}

//------------------------------------------------------------------------------
// Process protocol.

bool
ProcessPOSIX::CanDebug(Target &target, bool plugin_specified_by_name)
{
    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
        return exe_module_sp->GetFileSpec().Exists();
    return false;
}

Error
ProcessPOSIX::DoAttachToProcessWithID(lldb::pid_t pid)
{
    Error error;
    assert(m_monitor == NULL);

    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));
    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessPOSIX::%s(pid = %i)", __FUNCTION__, GetID());

    m_monitor = new ProcessMonitor(this, pid, error);

    if (!error.Success())
        return error;

    SetID(pid);
    return error;
}

Error
ProcessPOSIX::DoAttachToProcessWithID (lldb::pid_t pid,  const ProcessAttachInfo &attach_info)
{
    return DoAttachToProcessWithID(pid);
}

Error
ProcessPOSIX::WillLaunch(Module* module)
{
    Error error;
    return error;
}

const char *
ProcessPOSIX::GetFilePath(
    const lldb_private::ProcessLaunchInfo::FileAction *file_action,
    const char *default_path)
{
    const char *pts_name = "/dev/pts/";
    const char *path = NULL;

    if (file_action)
    {
        if (file_action->GetAction () == ProcessLaunchInfo::FileAction::eFileActionOpen)
            path = file_action->GetPath();
            // By default the stdio paths passed in will be pseudo-terminal
            // (/dev/pts). If so, convert to using a different default path
            // instead to redirect I/O to the debugger console. This should
            //  also handle user overrides to /dev/null or a different file.
            if (::strncmp(path, pts_name, ::strlen(pts_name)) == 0)
                path = default_path;
    }

    return path;
}

Error
ProcessPOSIX::DoLaunch (Module *module,
                       const ProcessLaunchInfo &launch_info)
{
    Error error;
    assert(m_monitor == NULL);

    SetPrivateState(eStateLaunching);

    const lldb_private::ProcessLaunchInfo::FileAction *file_action;

    // Default of NULL will mean to use existing open file descriptors
    const char *stdin_path = NULL;
    const char *stdout_path = NULL;
    const char *stderr_path = NULL;
    
    file_action = launch_info.GetFileActionForFD (STDIN_FILENO);
    stdin_path = GetFilePath(file_action, stdin_path);

    file_action = launch_info.GetFileActionForFD (STDOUT_FILENO);
    stdout_path = GetFilePath(file_action, stdout_path);

    file_action = launch_info.GetFileActionForFD (STDERR_FILENO);
    stderr_path = GetFilePath(file_action, stderr_path);

    m_monitor = new ProcessMonitor (this, 
                                    module,
                                    launch_info.GetArguments().GetConstArgumentVector(), 
                                    launch_info.GetEnvironmentEntries().GetConstArgumentVector(),
                                    stdin_path, 
                                    stdout_path, 
                                    stderr_path,
                                    error);

    m_module = module;

    if (!error.Success())
        return error;

    SetID(m_monitor->GetPID());
    return error;
}

void
ProcessPOSIX::DidLaunch()
{
}

Error
ProcessPOSIX::DoResume()
{
    StateType state = GetPrivateState();

    assert(state == eStateStopped || state == eStateCrashed);

    // We are about to resume a thread that will cause the process to exit so
    // set our exit status now.  Do not change our state if the inferior
    // crashed.
    if (state == eStateStopped) 
    {
        if (m_in_limbo)
            SetExitStatus(m_exit_status, NULL);
        else
            SetPrivateState(eStateRunning);
    }

    bool did_resume = false;
    uint32_t thread_count = m_thread_list.GetSize(false);
    for (uint32_t i = 0; i < thread_count; ++i)
    {
        POSIXThread *thread = static_cast<POSIXThread*>(
            m_thread_list.GetThreadAtIndex(i, false).get());
        did_resume = thread->Resume() || did_resume;
    }
    assert(did_resume && "Process resume failed!");

    return Error();
}

addr_t
ProcessPOSIX::GetImageInfoAddress()
{
    Target *target = &GetTarget();
    ObjectFile *obj_file = target->GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress();

    if (addr.IsValid()) 
        return addr.GetLoadAddress(target);
    else
        return LLDB_INVALID_ADDRESS;
}

Error
ProcessPOSIX::DoHalt(bool &caused_stop)
{
    Error error;

    if (IsStopped())
    {
        caused_stop = false;
    }
    else if (kill(GetID(), SIGSTOP))
    {
        caused_stop = false;
        error.SetErrorToErrno();
    }
    else
    {
        caused_stop = true;
    }

    return error;
}

Error
ProcessPOSIX::DoDetach()
{
    Error error;

    error = m_monitor->Detach();
    if (error.Success())
        SetPrivateState(eStateDetached);

    return error;
}

Error
ProcessPOSIX::DoSignal(int signal)
{
    Error error;

    if (kill(GetID(), signal))
        error.SetErrorToErrno();

    return error;
}

Error
ProcessPOSIX::DoDestroy()
{
    Error error;

    if (!HasExited())
    {
        // Drive the exit event to completion (do not keep the inferior in
        // limbo).
        m_exit_now = true;

        if ((m_monitor == NULL || kill(m_monitor->GetPID(), SIGKILL)) && error.Success())
        {
            error.SetErrorToErrno();
            return error;
        }

        SetPrivateState(eStateExited);
    }

    return error;
}

void
ProcessPOSIX::SendMessage(const ProcessMessage &message)
{
    Mutex::Locker lock(m_message_mutex);

    switch (message.GetKind())
    {
    default:
        assert(false && "Unexpected process message!");
        break;

    case ProcessMessage::eInvalidMessage:
        return;

    case ProcessMessage::eLimboMessage:
        m_in_limbo = true;
        m_exit_status = message.GetExitStatus();
        if (m_exit_now)
        {
            SetPrivateState(eStateExited);
            m_monitor->Detach();
        }
        else
            SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eExitMessage:
        m_exit_status = message.GetExitStatus();
        SetExitStatus(m_exit_status, NULL);
        break;

    case ProcessMessage::eTraceMessage:
    case ProcessMessage::eBreakpointMessage:
        SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eSignalMessage:
    case ProcessMessage::eSignalDeliveredMessage:
        SetPrivateState(eStateStopped);
        break;

    case ProcessMessage::eCrashMessage:
        SetPrivateState(eStateCrashed);
        break;
    }

    m_message_queue.push(message);
}

void
ProcessPOSIX::RefreshStateAfterStop()
{
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));
    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessPOSIX::%s()", __FUNCTION__);

    Mutex::Locker lock(m_message_mutex);
    if (m_message_queue.empty())
        return;

    ProcessMessage &message = m_message_queue.front();

    // Resolve the thread this message corresponds to and pass it along.
    // FIXME: we're really dealing with the pid here.  This should get
    // fixed when this code is fixed to handle multiple threads.
    lldb::tid_t tid = message.GetTID();
    if (log)
        log->Printf ("ProcessPOSIX::%s() pid = %i", __FUNCTION__, tid);
    POSIXThread *thread = static_cast<POSIXThread*>(
        GetThreadList().FindThreadByID(tid, false).get());

    assert(thread);
    thread->Notify(message);

    m_message_queue.pop();
}

bool
ProcessPOSIX::IsAlive()
{
    StateType state = GetPrivateState();
    return state != eStateDetached && state != eStateExited && state != eStateInvalid;
}

size_t
ProcessPOSIX::DoReadMemory(addr_t vm_addr,
                           void *buf, size_t size, Error &error)
{
    assert(m_monitor);
    return m_monitor->ReadMemory(vm_addr, buf, size, error);
}

size_t
ProcessPOSIX::DoWriteMemory(addr_t vm_addr, const void *buf, size_t size,
                            Error &error)
{
    assert(m_monitor);
    return m_monitor->WriteMemory(vm_addr, buf, size, error);
}

addr_t
ProcessPOSIX::DoAllocateMemory(size_t size, uint32_t permissions,
                               Error &error)
{
    addr_t allocated_addr = LLDB_INVALID_ADDRESS;

    unsigned prot = 0;
    if (permissions & lldb::ePermissionsReadable)
        prot |= eMmapProtRead;
    if (permissions & lldb::ePermissionsWritable)
        prot |= eMmapProtWrite;
    if (permissions & lldb::ePermissionsExecutable)
        prot |= eMmapProtExec;

    if (InferiorCallMmap(this, allocated_addr, 0, size, prot,
                         eMmapFlagsAnon | eMmapFlagsPrivate, -1, 0)) {
        m_addr_to_mmap_size[allocated_addr] = size;
        error.Clear();
    } else {
        allocated_addr = LLDB_INVALID_ADDRESS;
        error.SetErrorStringWithFormat("unable to allocate %zu bytes of memory with permissions %s", size, GetPermissionsAsCString (permissions));
    }

    return allocated_addr;
}

Error
ProcessPOSIX::DoDeallocateMemory(lldb::addr_t addr)
{
    Error error;
    MMapMap::iterator pos = m_addr_to_mmap_size.find(addr);
    if (pos != m_addr_to_mmap_size.end() &&
        InferiorCallMunmap(this, addr, pos->second))
        m_addr_to_mmap_size.erase (pos);
    else
        error.SetErrorStringWithFormat("unable to deallocate memory at 0x%llx", addr);

    return error;
}

size_t
ProcessPOSIX::GetSoftwareBreakpointTrapOpcode(BreakpointSite* bp_site)
{
    static const uint8_t g_i386_opcode[] = { 0xCC };

    ArchSpec arch = GetTarget().GetArchitecture();
    const uint8_t *opcode = NULL;
    size_t opcode_size = 0;

    switch (arch.GetCore())
    {
    default:
        assert(false && "CPU type not supported!");
        break;

    case ArchSpec::eCore_x86_32_i386:
    case ArchSpec::eCore_x86_64_x86_64:
        opcode = g_i386_opcode;
        opcode_size = sizeof(g_i386_opcode);
        break;
    }

    bp_site->SetTrapOpcode(opcode, opcode_size);
    return opcode_size;
}

Error
ProcessPOSIX::EnableBreakpoint(BreakpointSite *bp_site)
{
    return EnableSoftwareBreakpoint(bp_site);
}

Error
ProcessPOSIX::DisableBreakpoint(BreakpointSite *bp_site)
{
    return DisableSoftwareBreakpoint(bp_site);
}

uint32_t
ProcessPOSIX::UpdateThreadListIfNeeded()
{
    // Do not allow recursive updates.
    return m_thread_list.GetSize(false);
}

bool
ProcessPOSIX::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_THREAD));
    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessPOSIX::%s() (pid = %i)", __FUNCTION__, GetID());

    // Update the process thread list with this new thread.
    // FIXME: We should be using tid, not pid.
    assert(m_monitor);
    ThreadSP thread_sp (old_thread_list.FindThreadByID (GetID(), false));
    if (!thread_sp) {
        thread_sp.reset(new POSIXThread(*this, GetID()));
    }

    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessPOSIX::%s() updated pid = %i", __FUNCTION__, GetID());
    new_thread_list.AddThread(thread_sp);

    return new_thread_list.GetSize(false) > 0;
}

ByteOrder
ProcessPOSIX::GetByteOrder() const
{
    // FIXME: We should be able to extract this value directly.  See comment in
    // ProcessPOSIX().
    return m_byte_order;
}

size_t
ProcessPOSIX::PutSTDIN(const char *buf, size_t len, Error &error)
{
    ssize_t status;
    if ((status = write(m_monitor->GetTerminalFD(), buf, len)) < 0) 
    {
        error.SetErrorToErrno();
        return 0;
    }
    return status;
}

size_t
ProcessPOSIX::GetSTDOUT(char *buf, size_t len, Error &error)
{
    ssize_t bytes_read;

    // The terminal file descriptor is always in non-block mode.
    if ((bytes_read = read(m_monitor->GetTerminalFD(), buf, len)) < 0) 
    {
        if (errno != EAGAIN)
            error.SetErrorToErrno();
        return 0;
    }
    return bytes_read;
}

size_t
ProcessPOSIX::GetSTDERR(char *buf, size_t len, Error &error)
{
    return GetSTDOUT(buf, len, error);
}

UnixSignals &
ProcessPOSIX::GetUnixSignals()
{
    return m_signals;
}

//------------------------------------------------------------------------------
// Utility functions.

bool
ProcessPOSIX::HasExited()
{
    switch (GetPrivateState())
    {
    default:
        break;

    case eStateDetached:
    case eStateExited:
        return true;
    }

    return false;
}

bool
ProcessPOSIX::IsStopped()
{
    switch (GetPrivateState())
    {
    default:
        break;

    case eStateStopped:
    case eStateCrashed:
    case eStateSuspended:
        return true;
    }

    return false;
}
