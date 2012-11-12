//===-- ProcessGDBRemote.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <spawn.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/mman.h>       // for mmap
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

// C++ Includes
#include <algorithm>
#include <map>

// Other libraries and framework includes

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Value.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Utility/PseudoTerminal.h"

// Project includes
#include "lldb/Host/Host.h"
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "Plugins/Process/Utility/StopInfoMachException.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "Utility/StringExtractorGDBRemote.h"
#include "GDBRemoteRegisterContext.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "ThreadGDBRemote.h"

#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"

namespace lldb
{
    // Provide a function that can easily dump the packet history if we know a
    // ProcessGDBRemote * value (which we can get from logs or from debugging).
    // We need the function in the lldb namespace so it makes it into the final
    // executable since the LLDB shared library only exports stuff in the lldb
    // namespace. This allows you to attach with a debugger and call this
    // function and get the packet history dumped to a file.
    void
    DumpProcessGDBRemotePacketHistory (void *p, const char *path)
    {
        lldb_private::StreamFile strm;
        lldb_private::Error error (strm.GetFile().Open(path, lldb_private::File::eOpenOptionWrite | lldb_private::File::eOpenOptionCanCreate));
        if (error.Success())
            ((ProcessGDBRemote *)p)->GetGDBRemote().DumpHistory (strm);
    }
}


#define DEBUGSERVER_BASENAME    "debugserver"
using namespace lldb;
using namespace lldb_private;

static bool rand_initialized = false;

// TODO Randomly assigning a port is unsafe.  We should get an unused
// ephemeral port from the kernel and make sure we reserve it before passing
// it to debugserver.

#if defined (__APPLE__)
#define LOW_PORT    (IPPORT_RESERVED)
#define HIGH_PORT   (IPPORT_HIFIRSTAUTO)
#else
#define LOW_PORT    (1024u)
#define HIGH_PORT   (49151u)
#endif

static inline uint16_t
get_random_port ()
{
    if (!rand_initialized)
    {
        time_t seed = time(NULL);

        rand_initialized = true;
        srand(seed);
    }
    return (rand() % (HIGH_PORT - LOW_PORT)) + LOW_PORT;
}


const char *
ProcessGDBRemote::GetPluginNameStatic()
{
    return "gdb-remote";
}

const char *
ProcessGDBRemote::GetPluginDescriptionStatic()
{
    return "GDB Remote protocol based debugging plug-in.";
}

void
ProcessGDBRemote::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessGDBRemote::CreateInstance);
}


lldb::ProcessSP
ProcessGDBRemote::CreateInstance (Target &target, Listener &listener, const FileSpec *crash_file_path)
{
    lldb::ProcessSP process_sp;
    if (crash_file_path == NULL)
        process_sp.reset (new ProcessGDBRemote (target, listener));
    return process_sp;
}

bool
ProcessGDBRemote::CanDebug (Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    Module *exe_module = target.GetExecutableModulePointer();
    if (exe_module)
    {
        ObjectFile *exe_objfile = exe_module->GetObjectFile();
        // We can't debug core files...
        switch (exe_objfile->GetType())
        {
            case ObjectFile::eTypeInvalid:
            case ObjectFile::eTypeCoreFile:
            case ObjectFile::eTypeDebugInfo:
            case ObjectFile::eTypeObjectFile:
            case ObjectFile::eTypeSharedLibrary:
            case ObjectFile::eTypeStubLibrary:
                return false;
            case ObjectFile::eTypeExecutable:
            case ObjectFile::eTypeDynamicLinker:
            case ObjectFile::eTypeUnknown:
                break;
        }
        return exe_module->GetFileSpec().Exists();
    }
    // However, if there is no executable module, we return true since we might be preparing to attach.
    return true;
}

//----------------------------------------------------------------------
// ProcessGDBRemote constructor
//----------------------------------------------------------------------
ProcessGDBRemote::ProcessGDBRemote(Target& target, Listener &listener) :
    Process (target, listener),
    m_flags (0),
    m_gdb_comm(false),
    m_debugserver_pid (LLDB_INVALID_PROCESS_ID),
    m_last_stop_packet (),
    m_last_stop_packet_mutex (Mutex::eMutexTypeNormal),
    m_register_info (),
    m_async_broadcaster (NULL, "lldb.process.gdb-remote.async-broadcaster"),
    m_async_thread (LLDB_INVALID_HOST_THREAD),
    m_async_thread_state(eAsyncThreadNotStarted),
    m_async_thread_state_mutex(Mutex::eMutexTypeRecursive),
    m_thread_ids (),
    m_continue_c_tids (),
    m_continue_C_tids (),
    m_continue_s_tids (),
    m_continue_S_tids (),
    m_dispatch_queue_offsets_addr (LLDB_INVALID_ADDRESS),
    m_max_memory_size (512),
    m_addr_to_mmap_size (),
    m_thread_create_bp_sp (),
    m_waiting_for_attach (false),
    m_destroy_tried_resuming (false),
    m_dyld_plugin_name(),
    m_kernel_load_addr (LLDB_INVALID_ADDRESS),
    m_command_sp ()
{
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncThreadShouldExit,   "async thread should exit");
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncContinue,           "async thread continue");
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncThreadDidExit,      "async thread did exit");
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessGDBRemote::~ProcessGDBRemote()
{
    //  m_mach_process.UnregisterNotificationCallbacks (this);
    Clear();
    // We need to call finalize on the process before destroying ourselves
    // to make sure all of the broadcaster cleanup goes as planned. If we
    // destruct this class, then Process::~Process() might have problems
    // trying to fully destroy the broadcaster.
    Finalize();
    
    // The general Finalize is going to try to destroy the process and that SHOULD
    // shut down the async thread.  However, if we don't kill it it will get stranded and
    // its connection will go away so when it wakes up it will crash.  So kill it for sure here.
    StopAsyncThread();
    KillDebugserverProcess();
}

//----------------------------------------------------------------------
// PluginInterface
//----------------------------------------------------------------------
const char *
ProcessGDBRemote::GetPluginName()
{
    return "Process debugging plug-in that uses the GDB remote protocol";
}

const char *
ProcessGDBRemote::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessGDBRemote::GetPluginVersion()
{
    return 1;
}

void
ProcessGDBRemote::BuildDynamicRegisterInfo (bool force)
{
    if (!force && m_register_info.GetNumRegisters() > 0)
        return;

    char packet[128];
    m_register_info.Clear();
    uint32_t reg_offset = 0;
    uint32_t reg_num = 0;
    for (StringExtractorGDBRemote::ResponseType response_type = StringExtractorGDBRemote::eResponse;
         response_type == StringExtractorGDBRemote::eResponse; 
         ++reg_num)
    {
        const int packet_len = ::snprintf (packet, sizeof(packet), "qRegisterInfo%x", reg_num);
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (m_gdb_comm.SendPacketAndWaitForResponse(packet, packet_len, response, false))
        {
            response_type = response.GetResponseType();
            if (response_type == StringExtractorGDBRemote::eResponse)
            {
                std::string name;
                std::string value;
                ConstString reg_name;
                ConstString alt_name;
                ConstString set_name;
                RegisterInfo reg_info = { NULL,                 // Name
                    NULL,                 // Alt name
                    0,                    // byte size
                    reg_offset,           // offset
                    eEncodingUint,        // encoding
                    eFormatHex,           // formate
                    {
                        LLDB_INVALID_REGNUM, // GCC reg num
                        LLDB_INVALID_REGNUM, // DWARF reg num
                        LLDB_INVALID_REGNUM, // generic reg num
                        reg_num,             // GDB reg num
                        reg_num           // native register number
                    },
                    NULL,
                    NULL
                };

                while (response.GetNameColonValue(name, value))
                {
                    if (name.compare("name") == 0)
                    {
                        reg_name.SetCString(value.c_str());
                    }
                    else if (name.compare("alt-name") == 0)
                    {
                        alt_name.SetCString(value.c_str());
                    }
                    else if (name.compare("bitsize") == 0)
                    {
                        reg_info.byte_size = Args::StringToUInt32(value.c_str(), 0, 0) / CHAR_BIT;
                    }
                    else if (name.compare("offset") == 0)
                    {
                        uint32_t offset = Args::StringToUInt32(value.c_str(), UINT32_MAX, 0);
                        if (reg_offset != offset)
                        {
                            reg_offset = offset;
                        }
                    }
                    else if (name.compare("encoding") == 0)
                    {
                        const Encoding encoding = Args::StringToEncoding (value.c_str());
                        if (encoding != eEncodingInvalid)
                            reg_info.encoding = encoding;
                    }
                    else if (name.compare("format") == 0)
                    {
                        Format format = eFormatInvalid;
                        if (Args::StringToFormat (value.c_str(), format, NULL).Success())
                            reg_info.format = format;
                        else if (value.compare("binary") == 0)
                            reg_info.format = eFormatBinary;
                        else if (value.compare("decimal") == 0)
                            reg_info.format = eFormatDecimal;
                        else if (value.compare("hex") == 0)
                            reg_info.format = eFormatHex;
                        else if (value.compare("float") == 0)
                            reg_info.format = eFormatFloat;
                        else if (value.compare("vector-sint8") == 0)
                            reg_info.format = eFormatVectorOfSInt8;
                        else if (value.compare("vector-uint8") == 0)
                            reg_info.format = eFormatVectorOfUInt8;
                        else if (value.compare("vector-sint16") == 0)
                            reg_info.format = eFormatVectorOfSInt16;
                        else if (value.compare("vector-uint16") == 0)
                            reg_info.format = eFormatVectorOfUInt16;
                        else if (value.compare("vector-sint32") == 0)
                            reg_info.format = eFormatVectorOfSInt32;
                        else if (value.compare("vector-uint32") == 0)
                            reg_info.format = eFormatVectorOfUInt32;
                        else if (value.compare("vector-float32") == 0)
                            reg_info.format = eFormatVectorOfFloat32;
                        else if (value.compare("vector-uint128") == 0)
                            reg_info.format = eFormatVectorOfUInt128;
                    }
                    else if (name.compare("set") == 0)
                    {
                        set_name.SetCString(value.c_str());
                    }
                    else if (name.compare("gcc") == 0)
                    {
                        reg_info.kinds[eRegisterKindGCC] = Args::StringToUInt32(value.c_str(), LLDB_INVALID_REGNUM, 0);
                    }
                    else if (name.compare("dwarf") == 0)
                    {
                        reg_info.kinds[eRegisterKindDWARF] = Args::StringToUInt32(value.c_str(), LLDB_INVALID_REGNUM, 0);
                    }
                    else if (name.compare("generic") == 0)
                    {
                        reg_info.kinds[eRegisterKindGeneric] = Args::StringToGenericRegister (value.c_str());
                    }
                }

                reg_info.byte_offset = reg_offset;
                assert (reg_info.byte_size != 0);
                reg_offset += reg_info.byte_size;
                m_register_info.AddRegister(reg_info, reg_name, alt_name, set_name);
            }
        }
        else
        {
            break;
        }
    }

    // We didn't get anything if the accumulated reg_num is zero.  See if we are
    // debugging ARM and fill with a hard coded register set until we can get an
    // updated debugserver down on the devices.
    // On the other hand, if the accumulated reg_num is positive, see if we can
    // add composite registers to the existing primordial ones.
    bool from_scratch = (reg_num == 0);

    const ArchSpec &target_arch = GetTarget().GetArchitecture();
    const ArchSpec &remote_arch = m_gdb_comm.GetHostArchitecture();
    if (!target_arch.IsValid())
    {
        if (remote_arch.IsValid()
              && remote_arch.GetMachine() == llvm::Triple::arm
              && remote_arch.GetTriple().getVendor() == llvm::Triple::Apple)
            m_register_info.HardcodeARMRegisters(from_scratch);
    }
    else if (target_arch.GetMachine() == llvm::Triple::arm)
    {
        m_register_info.HardcodeARMRegisters(from_scratch);
    }

    // Add some convenience registers (eax, ebx, ecx, edx, esi, edi, ebp, esp) to x86_64.
    if ((target_arch.IsValid() && target_arch.GetMachine() == llvm::Triple::x86_64)
        || (remote_arch.IsValid() && remote_arch.GetMachine() == llvm::Triple::x86_64))
        m_register_info.Addx86_64ConvenienceRegisters();

    // At this point, we can finalize our register info.
    m_register_info.Finalize ();
}

Error
ProcessGDBRemote::WillLaunch (Module* module)
{
    return WillLaunchOrAttach ();
}

Error
ProcessGDBRemote::WillAttachToProcessWithID (lldb::pid_t pid)
{
    return WillLaunchOrAttach ();
}

Error
ProcessGDBRemote::WillAttachToProcessWithName (const char *process_name, bool wait_for_launch)
{
    return WillLaunchOrAttach ();
}

Error
ProcessGDBRemote::DoConnectRemote (Stream *strm, const char *remote_url)
{
    Error error (WillLaunchOrAttach ());
    
    if (error.Fail())
        return error;

    error = ConnectToDebugserver (remote_url);

    if (error.Fail())
        return error;
    StartAsyncThread ();

    CheckForKernel (strm);

    lldb::pid_t pid = m_gdb_comm.GetCurrentProcessID ();
    if (pid == LLDB_INVALID_PROCESS_ID)
    {
        // We don't have a valid process ID, so note that we are connected
        // and could now request to launch or attach, or get remote process 
        // listings...
        SetPrivateState (eStateConnected);
    }
    else
    {
        // We have a valid process
        SetID (pid);
        GetThreadList();
        if (m_gdb_comm.SendPacketAndWaitForResponse("?", 1, m_last_stop_packet, false))
        {
            const StateType state = SetThreadStopInfo (m_last_stop_packet);
            if (state == eStateStopped)
            {
                SetPrivateState (state);
            }
            else
                error.SetErrorStringWithFormat ("Process %llu was reported after connecting to '%s', but state was not stopped: %s", pid, remote_url, StateAsCString (state));
        }
        else
            error.SetErrorStringWithFormat ("Process %llu was reported after connecting to '%s', but no stop reply packet was received", pid, remote_url);
    }

    if (error.Success() 
        && !GetTarget().GetArchitecture().IsValid()
        && m_gdb_comm.GetHostArchitecture().IsValid())
    {
        GetTarget().SetArchitecture(m_gdb_comm.GetHostArchitecture());
    }

    return error;
}

// When we are establishing a connection to a remote system and we have no executable specified,
// or the executable is a kernel, we may be looking at a KASLR situation (where the kernel has been
// slid in memory.)
//
// This function tries to locate the kernel in memory if this is possibly a kernel debug session.
//
// If a kernel is found, return the address of the kernel in GetImageInfoAddress() -- the 
// DynamicLoaderDarwinKernel plugin uses this address as the kernel load address and will load the
// binary, if needed, along with all the kexts.

void
ProcessGDBRemote::CheckForKernel (Stream *strm)
{
    // early return if this isn't an "unknown" system (kernel debugging doesn't have a system type)
    const ArchSpec &gdb_remote_arch = m_gdb_comm.GetHostArchitecture();
    if (!gdb_remote_arch.IsValid() || gdb_remote_arch.GetTriple().getVendor() != llvm::Triple::UnknownVendor)
        return;

    Module *exe_module = GetTarget().GetExecutableModulePointer();
    ObjectFile *exe_objfile = NULL;
    if (exe_module)
        exe_objfile = exe_module->GetObjectFile();

    // early return if we have an executable and it is not a kernel--this is very unlikely to be a kernel debug session.
    if (exe_objfile
        && (exe_objfile->GetType() != ObjectFile::eTypeExecutable 
            || exe_objfile->GetStrata() != ObjectFile::eStrataKernel))
        return;

    // See if the kernel is in memory at the File address (slide == 0) -- no work needed, if so.
    if (exe_objfile && exe_objfile->GetHeaderAddress().IsValid())
    {
        ModuleSP memory_module_sp;
        memory_module_sp = ReadModuleFromMemory (exe_module->GetFileSpec(), exe_objfile->GetHeaderAddress().GetFileAddress(), false, false);
        if (memory_module_sp.get() 
            && memory_module_sp->GetUUID().IsValid() 
            && memory_module_sp->GetUUID() == exe_module->GetUUID())
        {
            m_kernel_load_addr = exe_objfile->GetHeaderAddress().GetFileAddress();
            m_dyld_plugin_name = DynamicLoaderDarwinKernel::GetPluginNameStatic();
            SetCanJIT(false);
            return;
        }
    }

    // See if the kernel's load address is stored in the kernel's low globals page; this is
    // done when a debug boot-arg has been set.  

    Error error;
    uint8_t buf[24];
    ModuleSP memory_module_sp;
    addr_t kernel_addr = LLDB_INVALID_ADDRESS;
    
    // First try the 32-bit 
    if (memory_module_sp.get() == NULL)
    {
        DataExtractor data4 (buf, sizeof(buf), gdb_remote_arch.GetByteOrder(), 4);
        if (DoReadMemory (0xffff0110, buf, 4, error) == 4)
        {
            uint32_t offset = 0;
            kernel_addr = data4.GetU32(&offset);
            memory_module_sp = ReadModuleFromMemory (FileSpec("mach_kernel", false), kernel_addr, false, false);
            if (!memory_module_sp.get()
                || !memory_module_sp->GetUUID().IsValid()
                || memory_module_sp->GetObjectFile() == NULL
                || memory_module_sp->GetObjectFile()->GetType() != ObjectFile::eTypeExecutable
                || memory_module_sp->GetObjectFile()->GetStrata() != ObjectFile::eStrataKernel)
            {
                memory_module_sp.reset();
            }
        }
    }

    // Now try the 64-bit location
    if (memory_module_sp.get() == NULL)
    {
        DataExtractor data8 (buf, sizeof(buf), gdb_remote_arch.GetByteOrder(), 8);
        if (DoReadMemory (0xffffff8000002010ULL, buf, 8, error) == 8)
        {   
            uint32_t offset = 0; 
            kernel_addr = data8.GetU32(&offset);
            memory_module_sp = ReadModuleFromMemory (FileSpec("mach_kernel", false), kernel_addr, false, false);
            if (!memory_module_sp.get()
                || !memory_module_sp->GetUUID().IsValid()
                || memory_module_sp->GetObjectFile() == NULL
                || memory_module_sp->GetObjectFile()->GetType() != ObjectFile::eTypeExecutable
                || memory_module_sp->GetObjectFile()->GetStrata() != ObjectFile::eStrataKernel)
            {
                memory_module_sp.reset();
            }
        }
    }

    if (memory_module_sp.get() 
        && memory_module_sp->GetArchitecture().IsValid() 
        && memory_module_sp->GetArchitecture().GetTriple().getVendor() == llvm::Triple::Apple)
    {
        m_kernel_load_addr = kernel_addr;
        m_dyld_plugin_name = DynamicLoaderDarwinKernel::GetPluginNameStatic();
        SetCanJIT(false);
        return;
    }
}

Error
ProcessGDBRemote::WillLaunchOrAttach ()
{
    Error error;
    m_stdio_communication.Clear ();
    return error;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessGDBRemote::DoLaunch (Module *exe_module, const ProcessLaunchInfo &launch_info)
{
    Error error;

    uint32_t launch_flags = launch_info.GetFlags().Get();
    const char *stdin_path = NULL;
    const char *stdout_path = NULL;
    const char *stderr_path = NULL;
    const char *working_dir = launch_info.GetWorkingDirectory();

    const ProcessLaunchInfo::FileAction *file_action;
    file_action = launch_info.GetFileActionForFD (STDIN_FILENO);
    if (file_action)
    {
        if (file_action->GetAction () == ProcessLaunchInfo::FileAction::eFileActionOpen)
            stdin_path = file_action->GetPath();
    }
    file_action = launch_info.GetFileActionForFD (STDOUT_FILENO);
    if (file_action)
    {
        if (file_action->GetAction () == ProcessLaunchInfo::FileAction::eFileActionOpen)
            stdout_path = file_action->GetPath();
    }
    file_action = launch_info.GetFileActionForFD (STDERR_FILENO);
    if (file_action)
    {
        if (file_action->GetAction () == ProcessLaunchInfo::FileAction::eFileActionOpen)
            stderr_path = file_action->GetPath();
    }

    //  ::LogSetBitMask (GDBR_LOG_DEFAULT);
    //  ::LogSetOptions (LLDB_LOG_OPTION_THREADSAFE | LLDB_LOG_OPTION_PREPEND_TIMESTAMP | LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD);
    //  ::LogSetLogFile ("/dev/stdout");
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));

    ObjectFile * object_file = exe_module->GetObjectFile();
    if (object_file)
    {
        char host_port[128];
        snprintf (host_port, sizeof(host_port), "localhost:%u", get_random_port ());
        char connect_url[128];
        snprintf (connect_url, sizeof(connect_url), "connect://%s", host_port);

        // Make sure we aren't already connected?
        if (!m_gdb_comm.IsConnected())
        {
            error = StartDebugserverProcess (host_port, launch_info);
            if (error.Fail())
            {
                if (log)
                    log->Printf("failed to start debugserver process: %s", error.AsCString());
                return error;
            }

            error = ConnectToDebugserver (connect_url);
        }
        
        if (error.Success())
        {
            lldb_utility::PseudoTerminal pty;
            const bool disable_stdio = (launch_flags & eLaunchFlagDisableSTDIO) != 0;

            // If the debugserver is local and we aren't disabling STDIO, lets use
            // a pseudo terminal to instead of relying on the 'O' packets for stdio
            // since 'O' packets can really slow down debugging if the inferior 
            // does a lot of output.
            PlatformSP platform_sp (m_target.GetPlatform());
            if (platform_sp && platform_sp->IsHost() && !disable_stdio)
            {
                const char *slave_name = NULL;
                if (stdin_path == NULL || stdout_path == NULL || stderr_path == NULL)
                {
                    if (pty.OpenFirstAvailableMaster(O_RDWR|O_NOCTTY, NULL, 0))
                        slave_name = pty.GetSlaveName (NULL, 0);
                }
                if (stdin_path == NULL) 
                    stdin_path = slave_name;

                if (stdout_path == NULL)
                    stdout_path = slave_name;

                if (stderr_path == NULL)
                    stderr_path = slave_name;
            }

            // Set STDIN to /dev/null if we want STDIO disabled or if either
            // STDOUT or STDERR have been set to something and STDIN hasn't
            if (disable_stdio || (stdin_path == NULL && (stdout_path || stderr_path)))
                stdin_path = "/dev/null";
            
            // Set STDOUT to /dev/null if we want STDIO disabled or if either
            // STDIN or STDERR have been set to something and STDOUT hasn't
            if (disable_stdio || (stdout_path == NULL && (stdin_path || stderr_path)))
                stdout_path = "/dev/null";
            
            // Set STDERR to /dev/null if we want STDIO disabled or if either
            // STDIN or STDOUT have been set to something and STDERR hasn't
            if (disable_stdio || (stderr_path == NULL && (stdin_path || stdout_path)))
                stderr_path = "/dev/null";

            if (stdin_path) 
                m_gdb_comm.SetSTDIN (stdin_path);
            if (stdout_path)
                m_gdb_comm.SetSTDOUT (stdout_path);
            if (stderr_path)
                m_gdb_comm.SetSTDERR (stderr_path);

            m_gdb_comm.SetDisableASLR (launch_flags & eLaunchFlagDisableASLR);

            m_gdb_comm.SendLaunchArchPacket (m_target.GetArchitecture().GetArchitectureName());
            
            if (working_dir && working_dir[0])
            {
                m_gdb_comm.SetWorkingDir (working_dir);
            }

            // Send the environment and the program + arguments after we connect
            const Args &environment = launch_info.GetEnvironmentEntries();
            if (environment.GetArgumentCount())
            {
                size_t num_environment_entries = environment.GetArgumentCount();
                for (size_t i=0; i<num_environment_entries; ++i)
                {
                    const char *env_entry = environment.GetArgumentAtIndex(i);
                    if (env_entry == NULL || m_gdb_comm.SendEnvironmentPacket(env_entry) != 0)
                        break;
                }
            }

            const uint32_t old_packet_timeout = m_gdb_comm.SetPacketTimeout (10);
            int arg_packet_err = m_gdb_comm.SendArgumentsPacket (launch_info.GetArguments().GetConstArgumentVector());
            if (arg_packet_err == 0)
            {
                std::string error_str;
                if (m_gdb_comm.GetLaunchSuccess (error_str))
                {
                    SetID (m_gdb_comm.GetCurrentProcessID ());
                }
                else
                {
                    error.SetErrorString (error_str.c_str());
                }
            }
            else
            {
                error.SetErrorStringWithFormat("'A' packet returned an error: %i", arg_packet_err);
            }
            
            m_gdb_comm.SetPacketTimeout (old_packet_timeout);
                
            if (GetID() == LLDB_INVALID_PROCESS_ID)
            {
                if (log)
                    log->Printf("failed to connect to debugserver: %s", error.AsCString());
                KillDebugserverProcess ();
                return error;
            }

            if (m_gdb_comm.SendPacketAndWaitForResponse("?", 1, m_last_stop_packet, false))
            {
                SetPrivateState (SetThreadStopInfo (m_last_stop_packet));
                
                if (!disable_stdio)
                {
                    if (pty.GetMasterFileDescriptor() != lldb_utility::PseudoTerminal::invalid_fd)
                        SetSTDIOFileDescriptor (pty.ReleaseMasterFileDescriptor());
                }
            }
        }
        else
        {
            if (log)
                log->Printf("failed to connect to debugserver: %s", error.AsCString());
        }
    }
    else
    {
        // Set our user ID to an invalid process ID.
        SetID(LLDB_INVALID_PROCESS_ID);
        error.SetErrorStringWithFormat ("failed to get object file from '%s' for arch %s", 
                                        exe_module->GetFileSpec().GetFilename().AsCString(), 
                                        exe_module->GetArchitecture().GetArchitectureName());
    }
    return error;

}


Error
ProcessGDBRemote::ConnectToDebugserver (const char *connect_url)
{
    Error error;
    // Sleep and wait a bit for debugserver to start to listen...
    std::auto_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
    if (conn_ap.get())
    {
        const uint32_t max_retry_count = 50;
        uint32_t retry_count = 0;
        while (!m_gdb_comm.IsConnected())
        {
            if (conn_ap->Connect(connect_url, &error) == eConnectionStatusSuccess)
            {
                m_gdb_comm.SetConnection (conn_ap.release());
                break;
            }
            retry_count++;

            if (retry_count >= max_retry_count)
                break;

            usleep (100000);
        }
    }

    if (!m_gdb_comm.IsConnected())
    {
        if (error.Success())
            error.SetErrorString("not connected to remote gdb server");
        return error;
    }

    // We always seem to be able to open a connection to a local port
    // so we need to make sure we can then send data to it. If we can't
    // then we aren't actually connected to anything, so try and do the
    // handshake with the remote GDB server and make sure that goes 
    // alright.
    if (!m_gdb_comm.HandshakeWithServer (NULL))
    {
        m_gdb_comm.Disconnect();
        if (error.Success())
            error.SetErrorString("not connected to remote gdb server");
        return error;
    }
    m_gdb_comm.ResetDiscoverableSettings();
    m_gdb_comm.QueryNoAckModeSupported ();
    m_gdb_comm.GetThreadSuffixSupported ();
    m_gdb_comm.GetListThreadsInStopReplySupported ();
    m_gdb_comm.GetHostInfo ();
    m_gdb_comm.GetVContSupported ('c');
    m_gdb_comm.GetVAttachOrWaitSupported();
    
    size_t num_cmds = GetExtraStartupCommands().GetArgumentCount();
    for (size_t idx = 0; idx < num_cmds; idx++)
    {
        StringExtractorGDBRemote response;
        m_gdb_comm.SendPacketAndWaitForResponse (GetExtraStartupCommands().GetArgumentAtIndex(idx), response, false);
    }
    return error;
}

void
ProcessGDBRemote::DidLaunchOrAttach ()
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::DidLaunch()");
    if (GetID() != LLDB_INVALID_PROCESS_ID)
    {
        m_dispatch_queue_offsets_addr = LLDB_INVALID_ADDRESS;

        BuildDynamicRegisterInfo (false);

        // See if the GDB server supports the qHostInfo information

        const ArchSpec &gdb_remote_arch = m_gdb_comm.GetHostArchitecture();
        if (gdb_remote_arch.IsValid())
        {
            ArchSpec &target_arch = GetTarget().GetArchitecture();

            if (target_arch.IsValid())
            {
                // If the remote host is ARM and we have apple as the vendor, then 
                // ARM executables and shared libraries can have mixed ARM architectures.
                // You can have an armv6 executable, and if the host is armv7, then the
                // system will load the best possible architecture for all shared libraries
                // it has, so we really need to take the remote host architecture as our
                // defacto architecture in this case.

                if (gdb_remote_arch.GetMachine() == llvm::Triple::arm &&
                    gdb_remote_arch.GetTriple().getVendor() == llvm::Triple::Apple)
                {
                    target_arch = gdb_remote_arch;
                }
                else
                {
                    // Fill in what is missing in the triple
                    const llvm::Triple &remote_triple = gdb_remote_arch.GetTriple();
                    llvm::Triple &target_triple = target_arch.GetTriple();
                    if (target_triple.getVendorName().size() == 0)
                    {
                        target_triple.setVendor (remote_triple.getVendor());

                        if (target_triple.getOSName().size() == 0)
                        {
                            target_triple.setOS (remote_triple.getOS());

                            if (target_triple.getEnvironmentName().size() == 0)
                                target_triple.setEnvironment (remote_triple.getEnvironment());
                        }
                    }
                }
            }
            else
            {
                // The target doesn't have a valid architecture yet, set it from
                // the architecture we got from the remote GDB server
                target_arch = gdb_remote_arch;
            }
        }
    }
}

void
ProcessGDBRemote::DidLaunch ()
{
    DidLaunchOrAttach ();
}

Error
ProcessGDBRemote::DoAttachToProcessWithID (lldb::pid_t attach_pid)
{
    ProcessAttachInfo attach_info;
    return DoAttachToProcessWithID(attach_pid, attach_info);
}

Error
ProcessGDBRemote::DoAttachToProcessWithID (lldb::pid_t attach_pid, const ProcessAttachInfo &attach_info)
{
    Error error;
    // Clear out and clean up from any current state
    Clear();
    if (attach_pid != LLDB_INVALID_PROCESS_ID)
    {
        // Make sure we aren't already connected?
        if (!m_gdb_comm.IsConnected())
        {
            char host_port[128];
            snprintf (host_port, sizeof(host_port), "localhost:%u", get_random_port ());
            char connect_url[128];
            snprintf (connect_url, sizeof(connect_url), "connect://%s", host_port);

            error = StartDebugserverProcess (host_port, attach_info);
            
            if (error.Fail())
            {
                const char *error_string = error.AsCString();
                if (error_string == NULL)
                    error_string = "unable to launch " DEBUGSERVER_BASENAME;

                SetExitStatus (-1, error_string);
            }
            else
            {
                error = ConnectToDebugserver (connect_url);
            }
        }
    
        if (error.Success())
        {
            char packet[64];
            const int packet_len = ::snprintf (packet, sizeof(packet), "vAttach;%llx", attach_pid);
            SetID (attach_pid);            
            m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncContinue, new EventDataBytes (packet, packet_len));
        }
    }
    return error;
}

size_t
ProcessGDBRemote::AttachInputReaderCallback
(
    void *baton, 
    InputReader *reader, 
    lldb::InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    if (notification == eInputReaderGotToken)
    {
        ProcessGDBRemote *gdb_process = (ProcessGDBRemote *)baton;
        if (gdb_process->m_waiting_for_attach)
            gdb_process->m_waiting_for_attach = false;
        reader->SetIsDone(true);
        return 1;
    }
    return 0;
}

Error
ProcessGDBRemote::DoAttachToProcessWithName (const char *process_name, bool wait_for_launch, const ProcessAttachInfo &attach_info)
{
    Error error;
    // Clear out and clean up from any current state
    Clear();

    if (process_name && process_name[0])
    {
        // Make sure we aren't already connected?
        if (!m_gdb_comm.IsConnected())
        {
            char host_port[128];
            snprintf (host_port, sizeof(host_port), "localhost:%u", get_random_port ());
            char connect_url[128];
            snprintf (connect_url, sizeof(connect_url), "connect://%s", host_port);

            error = StartDebugserverProcess (host_port, attach_info);
            if (error.Fail())
            {
                const char *error_string = error.AsCString();
                if (error_string == NULL)
                    error_string = "unable to launch " DEBUGSERVER_BASENAME;

                SetExitStatus (-1, error_string);
            }
            else
            {
                error = ConnectToDebugserver (connect_url);
            }
        }

        if (error.Success())
        {
            StreamString packet;
            
            if (wait_for_launch)
            {
                if (!m_gdb_comm.GetVAttachOrWaitSupported())
                {
                    packet.PutCString ("vAttachWait");
                }
                else
                {
                    if (attach_info.GetIgnoreExisting())
                        packet.PutCString("vAttachWait");
                    else
                        packet.PutCString ("vAttachOrWait");
                }
            }
            else
                packet.PutCString("vAttachName");
            packet.PutChar(';');
            packet.PutBytesAsRawHex8(process_name, strlen(process_name), lldb::endian::InlHostByteOrder(), lldb::endian::InlHostByteOrder());
            
            m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncContinue, new EventDataBytes (packet.GetData(), packet.GetSize()));

        }
    }
    return error;
}


void
ProcessGDBRemote::DidAttach ()
{
    DidLaunchOrAttach ();
}

Error
ProcessGDBRemote::WillResume ()
{
    m_continue_c_tids.clear();
    m_continue_C_tids.clear();
    m_continue_s_tids.clear();
    m_continue_S_tids.clear();
    return Error();
}

Error
ProcessGDBRemote::DoResume ()
{
    Error error;
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::Resume()");
    
    Listener listener ("gdb-remote.resume-packet-sent");
    if (listener.StartListeningForEvents (&m_gdb_comm, GDBRemoteCommunication::eBroadcastBitRunPacketSent))
    {
        listener.StartListeningForEvents (&m_async_broadcaster, ProcessGDBRemote::eBroadcastBitAsyncThreadDidExit);
        
        StreamString continue_packet;
        bool continue_packet_error = false;
        if (m_gdb_comm.HasAnyVContSupport ())
        {
            continue_packet.PutCString ("vCont");
        
            if (!m_continue_c_tids.empty())
            {
                if (m_gdb_comm.GetVContSupported ('c'))
                {
                    for (tid_collection::const_iterator t_pos = m_continue_c_tids.begin(), t_end = m_continue_c_tids.end(); t_pos != t_end; ++t_pos)
                        continue_packet.Printf(";c:%4.4llx", *t_pos);
                }
                else 
                    continue_packet_error = true;
            }
            
            if (!continue_packet_error && !m_continue_C_tids.empty())
            {
                if (m_gdb_comm.GetVContSupported ('C'))
                {
                    for (tid_sig_collection::const_iterator s_pos = m_continue_C_tids.begin(), s_end = m_continue_C_tids.end(); s_pos != s_end; ++s_pos)
                        continue_packet.Printf(";C%2.2x:%4.4llx", s_pos->second, s_pos->first);
                }
                else 
                    continue_packet_error = true;
            }

            if (!continue_packet_error && !m_continue_s_tids.empty())
            {
                if (m_gdb_comm.GetVContSupported ('s'))
                {
                    for (tid_collection::const_iterator t_pos = m_continue_s_tids.begin(), t_end = m_continue_s_tids.end(); t_pos != t_end; ++t_pos)
                        continue_packet.Printf(";s:%4.4llx", *t_pos);
                }
                else 
                    continue_packet_error = true;
            }
            
            if (!continue_packet_error && !m_continue_S_tids.empty())
            {
                if (m_gdb_comm.GetVContSupported ('S'))
                {
                    for (tid_sig_collection::const_iterator s_pos = m_continue_S_tids.begin(), s_end = m_continue_S_tids.end(); s_pos != s_end; ++s_pos)
                        continue_packet.Printf(";S%2.2x:%4.4llx", s_pos->second, s_pos->first);
                }
                else
                    continue_packet_error = true;
            }
            
            if (continue_packet_error)
                continue_packet.GetString().clear();
        }
        else
            continue_packet_error = true;
        
        if (continue_packet_error)
        {
            // Either no vCont support, or we tried to use part of the vCont
            // packet that wasn't supported by the remote GDB server.
            // We need to try and make a simple packet that can do our continue
            const size_t num_threads = GetThreadList().GetSize();
            const size_t num_continue_c_tids = m_continue_c_tids.size();
            const size_t num_continue_C_tids = m_continue_C_tids.size();
            const size_t num_continue_s_tids = m_continue_s_tids.size();
            const size_t num_continue_S_tids = m_continue_S_tids.size();
            if (num_continue_c_tids > 0)
            {
                if (num_continue_c_tids == num_threads)
                {
                    // All threads are resuming...
                    m_gdb_comm.SetCurrentThreadForRun (-1);
                    continue_packet.PutChar ('c'); 
                    continue_packet_error = false;
                }
                else if (num_continue_c_tids == 1 &&
                         num_continue_C_tids == 0 && 
                         num_continue_s_tids == 0 && 
                         num_continue_S_tids == 0 )
                {
                    // Only one thread is continuing
                    m_gdb_comm.SetCurrentThreadForRun (m_continue_c_tids.front());
                    continue_packet.PutChar ('c');                
                    continue_packet_error = false;
                }
            }

            if (continue_packet_error && num_continue_C_tids > 0)
            {
                if ((num_continue_C_tids + num_continue_c_tids) == num_threads && 
                    num_continue_C_tids > 0 && 
                    num_continue_s_tids == 0 && 
                    num_continue_S_tids == 0 )
                {
                    const int continue_signo = m_continue_C_tids.front().second;
                    // Only one thread is continuing
                    if (num_continue_C_tids > 1)
                    {
                        // More that one thread with a signal, yet we don't have 
                        // vCont support and we are being asked to resume each
                        // thread with a signal, we need to make sure they are
                        // all the same signal, or we can't issue the continue
                        // accurately with the current support...
                        if (num_continue_C_tids > 1)
                        {
                            continue_packet_error = false;
                            for (size_t i=1; i<m_continue_C_tids.size(); ++i)
                            {
                                if (m_continue_C_tids[i].second != continue_signo)
                                    continue_packet_error = true;
                            }
                        }
                        if (!continue_packet_error)
                            m_gdb_comm.SetCurrentThreadForRun (-1);
                    }
                    else
                    {
                        // Set the continue thread ID
                        continue_packet_error = false;
                        m_gdb_comm.SetCurrentThreadForRun (m_continue_C_tids.front().first);
                    }
                    if (!continue_packet_error)
                    {
                        // Add threads continuing with the same signo...
                        continue_packet.Printf("C%2.2x", continue_signo);
                    }
                }
            }

            if (continue_packet_error && num_continue_s_tids > 0)
            {
                if (num_continue_s_tids == num_threads)
                {
                    // All threads are resuming...
                    m_gdb_comm.SetCurrentThreadForRun (-1);
                    continue_packet.PutChar ('s');
                    continue_packet_error = false;
                }
                else if (num_continue_c_tids == 0 &&
                         num_continue_C_tids == 0 && 
                         num_continue_s_tids == 1 && 
                         num_continue_S_tids == 0 )
                {
                    // Only one thread is stepping
                    m_gdb_comm.SetCurrentThreadForRun (m_continue_s_tids.front());
                    continue_packet.PutChar ('s');                
                    continue_packet_error = false;
                }
            }

            if (!continue_packet_error && num_continue_S_tids > 0)
            {
                if (num_continue_S_tids == num_threads)
                {
                    const int step_signo = m_continue_S_tids.front().second;
                    // Are all threads trying to step with the same signal?
                    continue_packet_error = false;
                    if (num_continue_S_tids > 1)
                    {
                        for (size_t i=1; i<num_threads; ++i)
                        {
                            if (m_continue_S_tids[i].second != step_signo)
                                continue_packet_error = true;
                        }
                    }
                    if (!continue_packet_error)
                    {
                        // Add threads stepping with the same signo...
                        m_gdb_comm.SetCurrentThreadForRun (-1);
                        continue_packet.Printf("S%2.2x", step_signo);
                    }
                }
                else if (num_continue_c_tids == 0 &&
                         num_continue_C_tids == 0 && 
                         num_continue_s_tids == 0 && 
                         num_continue_S_tids == 1 )
                {
                    // Only one thread is stepping with signal
                    m_gdb_comm.SetCurrentThreadForRun (m_continue_S_tids.front().first);
                    continue_packet.Printf("S%2.2x", m_continue_S_tids.front().second);
                    continue_packet_error = false;
                }
            }
        }

        if (continue_packet_error)
        {
            error.SetErrorString ("can't make continue packet for this resume");
        }
        else
        {
            EventSP event_sp;
            TimeValue timeout;
            timeout = TimeValue::Now();
            timeout.OffsetWithSeconds (5);
            if (!IS_VALID_LLDB_HOST_THREAD(m_async_thread))
            {
                error.SetErrorString ("Trying to resume but the async thread is dead.");
                if (log)
                    log->Printf ("ProcessGDBRemote::DoResume: Trying to resume but the async thread is dead.");
                return error;
            }
            
            m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncContinue, new EventDataBytes (continue_packet.GetData(), continue_packet.GetSize()));

            if (listener.WaitForEvent (&timeout, event_sp) == false)
            {
                error.SetErrorString("Resume timed out.");
                if (log)
                    log->Printf ("ProcessGDBRemote::DoResume: Resume timed out.");
            }
            else if (event_sp->BroadcasterIs (&m_async_broadcaster))
            {
                error.SetErrorString ("Broadcast continue, but the async thread was killed before we got an ack back.");
                if (log)
                    log->Printf ("ProcessGDBRemote::DoResume: Broadcast continue, but the async thread was killed before we got an ack back.");
                return error;
            }
        }
    }

    return error;
}

void
ProcessGDBRemote::ClearThreadIDList ()
{
    Mutex::Locker locker(m_thread_list.GetMutex());
    m_thread_ids.clear();
}

bool
ProcessGDBRemote::UpdateThreadIDList ()
{
    Mutex::Locker locker(m_thread_list.GetMutex());
    bool sequence_mutex_unavailable = false;
    m_gdb_comm.GetCurrentThreadIDs (m_thread_ids, sequence_mutex_unavailable);
    if (sequence_mutex_unavailable)
    {
        return false; // We just didn't get the list
    }
    return true;
}

bool
ProcessGDBRemote::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // locker will keep a mutex locked until it goes out of scope
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_THREAD));
    if (log && log->GetMask().Test(GDBR_LOG_VERBOSE))
        log->Printf ("ProcessGDBRemote::%s (pid = %llu)", __FUNCTION__, GetID());
    
    size_t num_thread_ids = m_thread_ids.size();
    // The "m_thread_ids" thread ID list should always be updated after each stop
    // reply packet, but in case it isn't, update it here.
    if (num_thread_ids == 0)
    {
        if (!UpdateThreadIDList ())
            return false;
        num_thread_ids = m_thread_ids.size();
    }

    if (num_thread_ids > 0)
    {
        for (size_t i=0; i<num_thread_ids; ++i)
        {
            tid_t tid = m_thread_ids[i];
            ThreadSP thread_sp (old_thread_list.FindThreadByID (tid, false));
            if (!thread_sp)
                thread_sp.reset (new ThreadGDBRemote (*this, tid));
            new_thread_list.AddThread(thread_sp);
        }
    }

    return true;
}


StateType
ProcessGDBRemote::SetThreadStopInfo (StringExtractor& stop_packet)
{
    stop_packet.SetFilePos (0);
    const char stop_type = stop_packet.GetChar();
    switch (stop_type)
    {
    case 'T':
    case 'S':
        {
            if (GetStopID() == 0)
            {
                // Our first stop, make sure we have a process ID, and also make
                // sure we know about our registers
                if (GetID() == LLDB_INVALID_PROCESS_ID)
                {
                    lldb::pid_t pid = m_gdb_comm.GetCurrentProcessID ();
                    if (pid != LLDB_INVALID_PROCESS_ID)
                        SetID (pid);
                }
                BuildDynamicRegisterInfo (true);
            }
            // Stop with signal and thread info
            const uint8_t signo = stop_packet.GetHexU8();
            std::string name;
            std::string value;
            std::string thread_name;
            std::string reason;
            std::string description;
            uint32_t exc_type = 0;
            std::vector<addr_t> exc_data;
            addr_t thread_dispatch_qaddr = LLDB_INVALID_ADDRESS;
            ThreadSP thread_sp;

            while (stop_packet.GetNameColonValue(name, value))
            {
                if (name.compare("metype") == 0)
                {
                    // exception type in big endian hex
                    exc_type = Args::StringToUInt32 (value.c_str(), 0, 16);
                }
                else if (name.compare("medata") == 0)
                {
                    // exception data in big endian hex
                    exc_data.push_back(Args::StringToUInt64 (value.c_str(), 0, 16));
                }
                else if (name.compare("thread") == 0)
                {
                    // thread in big endian hex
                    lldb::tid_t tid = Args::StringToUInt64 (value.c_str(), LLDB_INVALID_THREAD_ID, 16);
                    // m_thread_list does have its own mutex, but we need to
                    // hold onto the mutex between the call to m_thread_list.FindThreadByID(...)
                    // and the m_thread_list.AddThread(...) so it doesn't change on us
                    Mutex::Locker locker (m_thread_list.GetMutex ());
                    thread_sp = m_thread_list.FindThreadByID(tid, false);
                    if (!thread_sp)
                    {
                        // Create the thread if we need to
                        thread_sp.reset (new ThreadGDBRemote (*this, tid));
                        m_thread_list.AddThread(thread_sp);
                    }
                }
                else if (name.compare("threads") == 0)
                {
                    Mutex::Locker locker(m_thread_list.GetMutex());
                    m_thread_ids.clear();
                    // A comma separated list of all threads in the current
                    // process that includes the thread for this stop reply
                    // packet
                    size_t comma_pos;
                    lldb::tid_t tid;
                    while ((comma_pos = value.find(',')) != std::string::npos)
                    {
                        value[comma_pos] = '\0';
                        // thread in big endian hex
                        tid = Args::StringToUInt64 (value.c_str(), LLDB_INVALID_THREAD_ID, 16);
                        if (tid != LLDB_INVALID_THREAD_ID)
                            m_thread_ids.push_back (tid);
                        value.erase(0, comma_pos + 1);
                            
                    }
                    tid = Args::StringToUInt64 (value.c_str(), LLDB_INVALID_THREAD_ID, 16);
                    if (tid != LLDB_INVALID_THREAD_ID)
                        m_thread_ids.push_back (tid);
                }
                else if (name.compare("hexname") == 0)
                {
                    StringExtractor name_extractor;
                    // Swap "value" over into "name_extractor"
                    name_extractor.GetStringRef().swap(value);
                    // Now convert the HEX bytes into a string value
                    name_extractor.GetHexByteString (value);
                    thread_name.swap (value);
                }
                else if (name.compare("name") == 0)
                {
                    thread_name.swap (value);
                }
                else if (name.compare("qaddr") == 0)
                {
                    thread_dispatch_qaddr = Args::StringToUInt64 (value.c_str(), 0, 16);
                }
                else if (name.compare("reason") == 0)
                {
                    reason.swap(value);
                }
                else if (name.compare("description") == 0)
                {
                    StringExtractor desc_extractor;
                    // Swap "value" over into "name_extractor"
                    desc_extractor.GetStringRef().swap(value);
                    // Now convert the HEX bytes into a string value
                    desc_extractor.GetHexByteString (thread_name);
                }
                else if (name.size() == 2 && ::isxdigit(name[0]) && ::isxdigit(name[1]))
                {
                    // We have a register number that contains an expedited
                    // register value. Lets supply this register to our thread
                    // so it won't have to go and read it.
                    if (thread_sp)
                    {
                        uint32_t reg = Args::StringToUInt32 (name.c_str(), UINT32_MAX, 16);

                        if (reg != UINT32_MAX)
                        {
                            StringExtractor reg_value_extractor;
                            // Swap "value" over into "reg_value_extractor"
                            reg_value_extractor.GetStringRef().swap(value);
                            if (!static_cast<ThreadGDBRemote *> (thread_sp.get())->PrivateSetRegisterValue (reg, reg_value_extractor))
                            {
                                Host::SetCrashDescriptionWithFormat("Setting thread register '%s' (decoded to %u (0x%x)) with value '%s' for stop packet: '%s'", 
                                                                    name.c_str(), 
                                                                    reg, 
                                                                    reg, 
                                                                    reg_value_extractor.GetStringRef().c_str(), 
                                                                    stop_packet.GetStringRef().c_str());
                            }
                        }
                    }
                }
            }

            if (thread_sp)
            {
                ThreadGDBRemote *gdb_thread = static_cast<ThreadGDBRemote *> (thread_sp.get());

                gdb_thread->SetThreadDispatchQAddr (thread_dispatch_qaddr);
                gdb_thread->SetName (thread_name.empty() ? NULL : thread_name.c_str());
                if (exc_type != 0)
                {
                    const size_t exc_data_size = exc_data.size();

                    gdb_thread->SetStopInfo (StopInfoMachException::CreateStopReasonWithMachException (*thread_sp,
                                                                                                       exc_type, 
                                                                                                       exc_data_size,
                                                                                                       exc_data_size >= 1 ? exc_data[0] : 0,
                                                                                                       exc_data_size >= 2 ? exc_data[1] : 0,
                                                                                                       exc_data_size >= 3 ? exc_data[2] : 0));
                }
                else
                {
                    bool handled = false;
                    if (!reason.empty())
                    {
                        if (reason.compare("trace") == 0)
                        {
                            gdb_thread->SetStopInfo (StopInfo::CreateStopReasonToTrace (*thread_sp));
                            handled = true;
                        }
                        else if (reason.compare("breakpoint") == 0)
                        {
                            addr_t pc = gdb_thread->GetRegisterContext()->GetPC();
                            lldb::BreakpointSiteSP bp_site_sp = gdb_thread->GetProcess()->GetBreakpointSiteList().FindByAddress(pc);
                            if (bp_site_sp)
                            {
                                // If the breakpoint is for this thread, then we'll report the hit, but if it is for another thread,
                                // we can just report no reason.  We don't need to worry about stepping over the breakpoint here, that
                                // will be taken care of when the thread resumes and notices that there's a breakpoint under the pc.
                                handled = true;
                                if (bp_site_sp->ValidForThisThread (gdb_thread))
                                {
                                    gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithBreakpointSiteID (*thread_sp, bp_site_sp->GetID()));
                                }
                                else
                                {
                                    StopInfoSP invalid_stop_info_sp;
                                    gdb_thread->SetStopInfo (invalid_stop_info_sp);
                                }
                            }
                            
                        }
                        else if (reason.compare("trap") == 0)
                        {
                            // Let the trap just use the standard signal stop reason below...
                        }
                        else if (reason.compare("watchpoint") == 0)
                        {
                            break_id_t watch_id = LLDB_INVALID_WATCH_ID;
                            // TODO: locate the watchpoint somehow...
                            gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithWatchpointID (*thread_sp, watch_id));
                            handled = true;
                        }
                        else if (reason.compare("exception") == 0)
                        {
                            gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithException(*thread_sp, description.c_str()));
                            handled = true;
                        }
                    }
                    
                    if (signo)
                    {
                        if (signo == SIGTRAP)
                        {
                            // Currently we are going to assume SIGTRAP means we are either
                            // hitting a breakpoint or hardware single stepping. 
                            handled = true;
                            addr_t pc = gdb_thread->GetRegisterContext()->GetPC();
                            lldb::BreakpointSiteSP bp_site_sp = gdb_thread->GetProcess()->GetBreakpointSiteList().FindByAddress(pc);
                            
                            if (bp_site_sp)
                            {
                                // If the breakpoint is for this thread, then we'll report the hit, but if it is for another thread,
                                // we can just report no reason.  We don't need to worry about stepping over the breakpoint here, that
                                // will be taken care of when the thread resumes and notices that there's a breakpoint under the pc.
                                if (bp_site_sp->ValidForThisThread (gdb_thread))
                                {
                                    gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithBreakpointSiteID (*thread_sp, bp_site_sp->GetID()));
                                }
                                else
                                {
                                    StopInfoSP invalid_stop_info_sp;
                                    gdb_thread->SetStopInfo (invalid_stop_info_sp);
                                }
                            }
                            else
                            {
                                // If we were stepping then assume the stop was the result of the trace.  If we were
                                // not stepping then report the SIGTRAP.
                                // FIXME: We are still missing the case where we single step over a trap instruction.
                                if (gdb_thread->GetTemporaryResumeState() == eStateStepping)
                                    gdb_thread->SetStopInfo (StopInfo::CreateStopReasonToTrace (*thread_sp));
                                else
                                    gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithSignal(*thread_sp, signo));
                            }
                        }
                        if (!handled)
                            gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithSignal (*thread_sp, signo));
                }
                else
                {
                    StopInfoSP invalid_stop_info_sp;
                    gdb_thread->SetStopInfo (invalid_stop_info_sp);
                }
                    
                    if (!description.empty())
                    {
                        lldb::StopInfoSP stop_info_sp (gdb_thread->GetStopInfo ());
                        if (stop_info_sp)
                        {
                            stop_info_sp->SetDescription (description.c_str());
                        }
                        else
                        {
                            gdb_thread->SetStopInfo (StopInfo::CreateStopReasonWithException (*thread_sp, description.c_str()));
                        }
                    }
                }
            }
            return eStateStopped;
        }
        break;

    case 'W':
        // process exited
        return eStateExited;

    default:
        break;
    }
    return eStateInvalid;
}

void
ProcessGDBRemote::RefreshStateAfterStop ()
{
    Mutex::Locker locker(m_thread_list.GetMutex());
    m_thread_ids.clear();
    // Set the thread stop info. It might have a "threads" key whose value is
    // a list of all thread IDs in the current process, so m_thread_ids might
    // get set.
    SetThreadStopInfo (m_last_stop_packet);
    // Check to see if SetThreadStopInfo() filled in m_thread_ids?
    if (m_thread_ids.empty())
    {
        // No, we need to fetch the thread list manually
        UpdateThreadIDList();
    }

    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();
    
}

Error
ProcessGDBRemote::DoHalt (bool &caused_stop)
{
    Error error;

    bool timed_out = false;
    Mutex::Locker locker;
    
    if (m_public_state.GetValue() == eStateAttaching)
    {
        // We are being asked to halt during an attach. We need to just close
        // our file handle and debugserver will go away, and we can be done...
        m_gdb_comm.Disconnect();
    }
    else
    {
        if (!m_gdb_comm.SendInterrupt (locker, 2, timed_out))
        {
            if (timed_out)
                error.SetErrorString("timed out sending interrupt packet");
            else
                error.SetErrorString("unknown error sending interrupt packet");
        }
        
        caused_stop = m_gdb_comm.GetInterruptWasSent ();
    }
    return error;
}

Error
ProcessGDBRemote::InterruptIfRunning 
(
    bool discard_thread_plans, 
    bool catch_stop_event, 
    EventSP &stop_event_sp
)
{
    Error error;

    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));
    
    bool paused_private_state_thread = false;
    const bool is_running = m_gdb_comm.IsRunning();
    if (log)
        log->Printf ("ProcessGDBRemote::InterruptIfRunning(discard_thread_plans=%i, catch_stop_event=%i) is_running=%i", 
                     discard_thread_plans, 
                     catch_stop_event,
                     is_running);

    if (discard_thread_plans)
    {
        if (log)
            log->Printf ("ProcessGDBRemote::InterruptIfRunning() discarding all thread plans");
        m_thread_list.DiscardThreadPlans();
    }
    if (is_running)
    {
        if (catch_stop_event)
        {
            if (log)
                log->Printf ("ProcessGDBRemote::InterruptIfRunning() pausing private state thread");
            PausePrivateStateThread();
            paused_private_state_thread = true;
        }

        bool timed_out = false;
        Mutex::Locker locker;
        
        if (!m_gdb_comm.SendInterrupt (locker, 1, timed_out))
        {
            if (timed_out)
                error.SetErrorString("timed out sending interrupt packet");
            else
                error.SetErrorString("unknown error sending interrupt packet");
            if (paused_private_state_thread)
                ResumePrivateStateThread();
            return error;
        }
        
        if (catch_stop_event)
        {
            // LISTEN HERE
            TimeValue timeout_time;
            timeout_time = TimeValue::Now();
            timeout_time.OffsetWithSeconds(5);
            StateType state = WaitForStateChangedEventsPrivate (&timeout_time, stop_event_sp);
    
            timed_out = state == eStateInvalid;
            if (log)
                log->Printf ("ProcessGDBRemote::InterruptIfRunning() catch stop event: state = %s, timed-out=%i", StateAsCString(state), timed_out);

            if (timed_out)
                error.SetErrorString("unable to verify target stopped");
        }
        
        if (paused_private_state_thread)
        {
            if (log)
                log->Printf ("ProcessGDBRemote::InterruptIfRunning() resuming private state thread");
            ResumePrivateStateThread();
        }
    }
    return error;
}

Error
ProcessGDBRemote::WillDetach ()
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::WillDetach()");

    bool discard_thread_plans = true; 
    bool catch_stop_event = true;
    EventSP event_sp;
    
    // FIXME: InterruptIfRunning should be done in the Process base class, or better still make Halt do what is
    // needed.  This shouldn't be a feature of a particular plugin.
    
    return InterruptIfRunning (discard_thread_plans, catch_stop_event, event_sp);
}

Error
ProcessGDBRemote::DoDetach()
{
    Error error;
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::DoDetach()");

    DisableAllBreakpointSites ();

    m_thread_list.DiscardThreadPlans();

    bool success = m_gdb_comm.Detach ();
    if (log)
    {
        if (success)
            log->PutCString ("ProcessGDBRemote::DoDetach() detach packet sent successfully");
        else
            log->PutCString ("ProcessGDBRemote::DoDetach() detach packet send failed");
    }
    // Sleep for one second to let the process get all detached...
    StopAsyncThread ();

    SetPrivateState (eStateDetached);
    ResumePrivateStateThread();

    //KillDebugserverProcess ();
    return error;
}


Error
ProcessGDBRemote::DoDestroy ()
{
    Error error;
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::DoDestroy()");

    // There is a bug in older iOS debugservers where they don't shut down the process
    // they are debugging properly.  If the process is sitting at a breakpoint or an exception,
    // this can cause problems with restarting.  So we check to see if any of our threads are stopped
    // at a breakpoint, and if so we remove all the breakpoints, resume the process, and THEN
    // destroy it again.
    //
    // Note, we don't have a good way to test the version of debugserver, but I happen to know that
    // the set of all the iOS debugservers which don't support GetThreadSuffixSupported() and that of
    // the debugservers with this bug are equal.  There really should be a better way to test this!
    //
    // We also use m_destroy_tried_resuming to make sure we only do this once, if we resume and then halt and
    // get called here to destroy again and we're still at a breakpoint or exception, then we should
    // just do the straight-forward kill.
    //
    // And of course, if we weren't able to stop the process by the time we get here, it isn't
    // necessary (or helpful) to do any of this.

    if (!m_gdb_comm.GetThreadSuffixSupported() && m_public_state.GetValue() != eStateRunning)
    {
        PlatformSP platform_sp = GetTarget().GetPlatform();
        
        // FIXME: These should be ConstStrings so we aren't doing strcmp'ing.
        if (platform_sp
            && platform_sp->GetName()
            && strcmp (platform_sp->GetName(), PlatformRemoteiOS::GetShortPluginNameStatic()) == 0)
        {
            if (m_destroy_tried_resuming)
            {
                if (log)
                    log->PutCString ("ProcessGDBRemote::DoDestroy()Tried resuming to destroy once already, not doing it again.");
            }
            else
            {            
                // At present, the plans are discarded and the breakpoints disabled Process::Destroy,
                // but we really need it to happen here and it doesn't matter if we do it twice.
                m_thread_list.DiscardThreadPlans();
                DisableAllBreakpointSites();
                
                bool stop_looks_like_crash = false;
                ThreadList &threads = GetThreadList();
                
                {
                    Mutex::Locker locker(threads.GetMutex());
                    
                    size_t num_threads = threads.GetSize();
                    for (size_t i = 0; i < num_threads; i++)
                    {
                        ThreadSP thread_sp = threads.GetThreadAtIndex(i);
                        StopInfoSP stop_info_sp = thread_sp->GetPrivateStopReason();
                        StopReason reason = eStopReasonInvalid;
                        if (stop_info_sp)
                            reason = stop_info_sp->GetStopReason();
                        if (reason == eStopReasonBreakpoint
                            || reason == eStopReasonException)
                        {
                            if (log)
                                log->Printf ("ProcessGDBRemote::DoDestroy() - thread: %lld stopped with reason: %s.",
                                             thread_sp->GetID(),
                                             stop_info_sp->GetDescription());
                            stop_looks_like_crash = true;
                            break;
                        }
                    }
                }
                
                if (stop_looks_like_crash)
                {
                    if (log)
                        log->PutCString ("ProcessGDBRemote::DoDestroy() - Stopped at a breakpoint, continue and then kill.");
                    m_destroy_tried_resuming = true;
                    
                    // If we are going to run again before killing, it would be good to suspend all the threads 
                    // before resuming so they won't get into more trouble.  Sadly, for the threads stopped with
                    // the breakpoint or exception, the exception doesn't get cleared if it is suspended, so we do
                    // have to run the risk of letting those threads proceed a bit.
    
                    {
                        Mutex::Locker locker(threads.GetMutex());
                        
                        size_t num_threads = threads.GetSize();
                        for (size_t i = 0; i < num_threads; i++)
                        {
                            ThreadSP thread_sp = threads.GetThreadAtIndex(i);
                            StopInfoSP stop_info_sp = thread_sp->GetPrivateStopReason();
                            StopReason reason = eStopReasonInvalid;
                            if (stop_info_sp)
                                reason = stop_info_sp->GetStopReason();
                            if (reason != eStopReasonBreakpoint
                                && reason != eStopReasonException)
                            {
                                if (log)
                                    log->Printf ("ProcessGDBRemote::DoDestroy() - Suspending thread: %lld before running.",
                                                 thread_sp->GetID());
                                thread_sp->SetResumeState(eStateSuspended);
                            }
                        }
                    }
                    Resume ();
                    return Destroy();
                }
            }
        }
    }
    
    // Interrupt if our inferior is running...
    int exit_status = SIGABRT;
    std::string exit_string;

    if (m_gdb_comm.IsConnected())
    {
        if (m_public_state.GetValue() != eStateAttaching)
        {

            StringExtractorGDBRemote response;
            bool send_async = true;
            const uint32_t old_packet_timeout = m_gdb_comm.SetPacketTimeout (3);

            if (m_gdb_comm.SendPacketAndWaitForResponse("k", 1, response, send_async))
            {
                char packet_cmd = response.GetChar(0);

                if (packet_cmd == 'W' || packet_cmd == 'X')
                {
                    SetLastStopPacket (response);
                    ClearThreadIDList ();
                    exit_status = response.GetHexU8();
                }
                else
                {
                    if (log)
                        log->Printf ("ProcessGDBRemote::DoDestroy - got unexpected response to k packet: %s", response.GetStringRef().c_str());
                    exit_string.assign("got unexpected response to k packet: ");
                    exit_string.append(response.GetStringRef());
                }
            }
            else
            {
                if (log)
                    log->Printf ("ProcessGDBRemote::DoDestroy - failed to send k packet");
                exit_string.assign("failed to send the k packet");
            }

            m_gdb_comm.SetPacketTimeout(old_packet_timeout);
        }
        else
        {
            if (log)
                log->Printf ("ProcessGDBRemote::DoDestroy - failed to send k packet");
            exit_string.assign ("killed or interrupted while attaching.");
        }
    }
    else
    {
        // If we missed setting the exit status on the way out, do it here.
        // NB set exit status can be called multiple times, the first one sets the status.
        exit_string.assign("destroying when not connected to debugserver");
    }

    SetExitStatus(exit_status, exit_string.c_str());

    StopAsyncThread ();
    KillDebugserverProcess ();
    return error;
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessGDBRemote::IsAlive ()
{
    return m_gdb_comm.IsConnected() && m_private_state.GetValue() != eStateExited;
}

// For kernel debugging, we return the load address of the kernel binary as the
// ImageInfoAddress and we return the DynamicLoaderDarwinKernel as the GetDynamicLoader()
// name so the correct DynamicLoader plugin is chosen.
addr_t
ProcessGDBRemote::GetImageInfoAddress()
{
    if (m_kernel_load_addr != LLDB_INVALID_ADDRESS)
        return m_kernel_load_addr;
    else
        return m_gdb_comm.GetShlibInfoAddr();
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------
size_t
ProcessGDBRemote::DoReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    if (size > m_max_memory_size)
    {
        // Keep memory read sizes down to a sane limit. This function will be
        // called multiple times in order to complete the task by 
        // lldb_private::Process so it is ok to do this.
        size = m_max_memory_size;
    }

    char packet[64];
    const int packet_len = ::snprintf (packet, sizeof(packet), "m%llx,%llx", (uint64_t)addr, (uint64_t)size);
    assert (packet_len + 1 < sizeof(packet));
    StringExtractorGDBRemote response;
    if (m_gdb_comm.SendPacketAndWaitForResponse(packet, packet_len, response, true))
    {
        if (response.IsNormalResponse())
        {
            error.Clear();
            return response.GetHexBytes(buf, size, '\xdd');
        }
        else if (response.IsErrorResponse())
            error.SetErrorString("memory read failed");
        else if (response.IsUnsupportedResponse())
            error.SetErrorStringWithFormat("GDB server does not support reading memory");
        else
            error.SetErrorStringWithFormat("unexpected response to GDB server memory read packet '%s': '%s'", packet, response.GetStringRef().c_str());
    }
    else
    {
        error.SetErrorStringWithFormat("failed to sent packet: '%s'", packet);
    }
    return 0;
}

size_t
ProcessGDBRemote::DoWriteMemory (addr_t addr, const void *buf, size_t size, Error &error)
{
    if (size > m_max_memory_size)
    {
        // Keep memory read sizes down to a sane limit. This function will be
        // called multiple times in order to complete the task by 
        // lldb_private::Process so it is ok to do this.
        size = m_max_memory_size;
    }

    StreamString packet;
    packet.Printf("M%llx,%llx:", addr, (uint64_t)size);
    packet.PutBytesAsRawHex8(buf, size, lldb::endian::InlHostByteOrder(), lldb::endian::InlHostByteOrder());
    StringExtractorGDBRemote response;
    if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetData(), packet.GetSize(), response, true))
    {
        if (response.IsOKResponse())
        {
            error.Clear();
            return size;
        }
        else if (response.IsErrorResponse())
            error.SetErrorString("memory write failed");
        else if (response.IsUnsupportedResponse())
            error.SetErrorStringWithFormat("GDB server does not support writing memory");
        else
            error.SetErrorStringWithFormat("unexpected response to GDB server memory write packet '%s': '%s'", packet.GetString().c_str(), response.GetStringRef().c_str());
    }
    else
    {
        error.SetErrorStringWithFormat("failed to sent packet: '%s'", packet.GetString().c_str());
    }
    return 0;
}

lldb::addr_t
ProcessGDBRemote::DoAllocateMemory (size_t size, uint32_t permissions, Error &error)
{
    addr_t allocated_addr = LLDB_INVALID_ADDRESS;
    
    LazyBool supported = m_gdb_comm.SupportsAllocDeallocMemory();
    switch (supported)
    {
        case eLazyBoolCalculate:
        case eLazyBoolYes:
            allocated_addr = m_gdb_comm.AllocateMemory (size, permissions);
            if (allocated_addr != LLDB_INVALID_ADDRESS || supported == eLazyBoolYes)
                return allocated_addr;

        case eLazyBoolNo:
            // Call mmap() to create memory in the inferior..
            unsigned prot = 0;
            if (permissions & lldb::ePermissionsReadable)
                prot |= eMmapProtRead;
            if (permissions & lldb::ePermissionsWritable)
                prot |= eMmapProtWrite;
            if (permissions & lldb::ePermissionsExecutable)
                prot |= eMmapProtExec;

            if (InferiorCallMmap(this, allocated_addr, 0, size, prot,
                                 eMmapFlagsAnon | eMmapFlagsPrivate, -1, 0))
                m_addr_to_mmap_size[allocated_addr] = size;
            else
                allocated_addr = LLDB_INVALID_ADDRESS;
            break;
    }
    
    if (allocated_addr == LLDB_INVALID_ADDRESS)
        error.SetErrorStringWithFormat("unable to allocate %llu bytes of memory with permissions %s", (uint64_t)size, GetPermissionsAsCString (permissions));
    else
        error.Clear();
    return allocated_addr;
}

Error
ProcessGDBRemote::GetMemoryRegionInfo (addr_t load_addr, 
                                       MemoryRegionInfo &region_info)
{
    
    Error error (m_gdb_comm.GetMemoryRegionInfo (load_addr, region_info));
    return error;
}

Error
ProcessGDBRemote::GetWatchpointSupportInfo (uint32_t &num)
{
    
    Error error (m_gdb_comm.GetWatchpointSupportInfo (num));
    return error;
}

Error
ProcessGDBRemote::GetWatchpointSupportInfo (uint32_t &num, bool& after)
{
    Error error (m_gdb_comm.GetWatchpointSupportInfo (num, after));
    return error;
}

Error
ProcessGDBRemote::DoDeallocateMemory (lldb::addr_t addr)
{
    Error error; 
    LazyBool supported = m_gdb_comm.SupportsAllocDeallocMemory();

    switch (supported)
    {
        case eLazyBoolCalculate:
            // We should never be deallocating memory without allocating memory 
            // first so we should never get eLazyBoolCalculate
            error.SetErrorString ("tried to deallocate memory without ever allocating memory");
            break;

        case eLazyBoolYes:
            if (!m_gdb_comm.DeallocateMemory (addr))
                error.SetErrorStringWithFormat("unable to deallocate memory at 0x%llx", addr);
            break;
            
        case eLazyBoolNo:
            // Call munmap() to deallocate memory in the inferior..
            {
                MMapMap::iterator pos = m_addr_to_mmap_size.find(addr);
                if (pos != m_addr_to_mmap_size.end() &&
                    InferiorCallMunmap(this, addr, pos->second))
                    m_addr_to_mmap_size.erase (pos);
                else
                    error.SetErrorStringWithFormat("unable to deallocate memory at 0x%llx", addr);
            }
            break;
    }

    return error;
}


//------------------------------------------------------------------
// Process STDIO
//------------------------------------------------------------------
size_t
ProcessGDBRemote::PutSTDIN (const char *src, size_t src_len, Error &error)
{
    if (m_stdio_communication.IsConnected())
    {
        ConnectionStatus status;
        m_stdio_communication.Write(src, src_len, status, NULL);
    }
    return 0;
}

Error
ProcessGDBRemote::EnableBreakpoint (BreakpointSite *bp_site)
{
    Error error;
    assert (bp_site != NULL);

    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_BREAKPOINTS));
    user_id_t site_id = bp_site->GetID();
    const addr_t addr = bp_site->GetLoadAddress();
    if (log)
        log->Printf ("ProcessGDBRemote::EnableBreakpoint (size_id = %llu) address = 0x%llx", site_id, (uint64_t)addr);

    if (bp_site->IsEnabled())
    {
        if (log)
            log->Printf ("ProcessGDBRemote::EnableBreakpoint (size_id = %llu) address = 0x%llx -- SUCCESS (already enabled)", site_id, (uint64_t)addr);
        return error;
    }
    else
    {
        const size_t bp_op_size = GetSoftwareBreakpointTrapOpcode (bp_site);

        if (bp_site->HardwarePreferred())
        {
            // Try and set hardware breakpoint, and if that fails, fall through
            // and set a software breakpoint?
            if (m_gdb_comm.SupportsGDBStoppointPacket (eBreakpointHardware))
            {
                if (m_gdb_comm.SendGDBStoppointTypePacket(eBreakpointHardware, true, addr, bp_op_size) == 0)
                {
                    bp_site->SetEnabled(true);
                    bp_site->SetType (BreakpointSite::eHardware);
                    return error;
                }
            }
        }

        if (m_gdb_comm.SupportsGDBStoppointPacket (eBreakpointSoftware))
        {
            if (m_gdb_comm.SendGDBStoppointTypePacket(eBreakpointSoftware, true, addr, bp_op_size) == 0)
            {
                bp_site->SetEnabled(true);
                bp_site->SetType (BreakpointSite::eExternal);
                return error;
            }
        }

        return EnableSoftwareBreakpoint (bp_site);
    }

    if (log)
    {
        const char *err_string = error.AsCString();
        log->Printf ("ProcessGDBRemote::EnableBreakpoint() error for breakpoint at 0x%8.8llx: %s",
                     bp_site->GetLoadAddress(),
                     err_string ? err_string : "NULL");
    }
    // We shouldn't reach here on a successful breakpoint enable...
    if (error.Success())
        error.SetErrorToGenericError();
    return error;
}

Error
ProcessGDBRemote::DisableBreakpoint (BreakpointSite *bp_site)
{
    Error error;
    assert (bp_site != NULL);
    addr_t addr = bp_site->GetLoadAddress();
    user_id_t site_id = bp_site->GetID();
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("ProcessGDBRemote::DisableBreakpoint (site_id = %llu) addr = 0x%8.8llx", site_id, (uint64_t)addr);

    if (bp_site->IsEnabled())
    {
        const size_t bp_op_size = GetSoftwareBreakpointTrapOpcode (bp_site);

        BreakpointSite::Type bp_type = bp_site->GetType();
        switch (bp_type)
        {
        case BreakpointSite::eSoftware:
            error = DisableSoftwareBreakpoint (bp_site);
            break;

        case BreakpointSite::eHardware:
            if (m_gdb_comm.SendGDBStoppointTypePacket(eBreakpointSoftware, false, addr, bp_op_size))
                error.SetErrorToGenericError();
            break;

        case BreakpointSite::eExternal:
            if (m_gdb_comm.SendGDBStoppointTypePacket(eBreakpointSoftware, false, addr, bp_op_size))
                error.SetErrorToGenericError();
            break;
        }
        if (error.Success())
            bp_site->SetEnabled(false);
    }
    else
    {
        if (log)
            log->Printf ("ProcessGDBRemote::DisableBreakpoint (site_id = %llu) addr = 0x%8.8llx -- SUCCESS (already disabled)", site_id, (uint64_t)addr);
        return error;
    }

    if (error.Success())
        error.SetErrorToGenericError();
    return error;
}

// Pre-requisite: wp != NULL.
static GDBStoppointType
GetGDBStoppointType (Watchpoint *wp)
{
    assert(wp);
    bool watch_read = wp->WatchpointRead();
    bool watch_write = wp->WatchpointWrite();

    // watch_read and watch_write cannot both be false.
    assert(watch_read || watch_write);
    if (watch_read && watch_write)
        return eWatchpointReadWrite;
    else if (watch_read)
        return eWatchpointRead;
    else // Must be watch_write, then.
        return eWatchpointWrite;
}

Error
ProcessGDBRemote::EnableWatchpoint (Watchpoint *wp)
{
    Error error;
    if (wp)
    {
        user_id_t watchID = wp->GetID();
        addr_t addr = wp->GetLoadAddress();
        LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_WATCHPOINTS));
        if (log)
            log->Printf ("ProcessGDBRemote::EnableWatchpoint(watchID = %llu)", watchID);
        if (wp->IsEnabled())
        {
            if (log)
                log->Printf("ProcessGDBRemote::EnableWatchpoint(watchID = %llu) addr = 0x%8.8llx: watchpoint already enabled.", watchID, (uint64_t)addr);
            return error;
        }

        GDBStoppointType type = GetGDBStoppointType(wp);
        // Pass down an appropriate z/Z packet...
        if (m_gdb_comm.SupportsGDBStoppointPacket (type))
        {
            if (m_gdb_comm.SendGDBStoppointTypePacket(type, true, addr, wp->GetByteSize()) == 0)
            {
                wp->SetEnabled(true);
                return error;
            }
            else
                error.SetErrorString("sending gdb watchpoint packet failed");
        }
        else
            error.SetErrorString("watchpoints not supported");
    }
    else
    {
        error.SetErrorString("Watchpoint argument was NULL.");
    }
    if (error.Success())
        error.SetErrorToGenericError();
    return error;
}

Error
ProcessGDBRemote::DisableWatchpoint (Watchpoint *wp)
{
    Error error;
    if (wp)
    {
        user_id_t watchID = wp->GetID();

        LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_WATCHPOINTS));

        addr_t addr = wp->GetLoadAddress();
        if (log)
            log->Printf ("ProcessGDBRemote::DisableWatchpoint (watchID = %llu) addr = 0x%8.8llx", watchID, (uint64_t)addr);

        if (!wp->IsEnabled())
        {
            if (log)
                log->Printf ("ProcessGDBRemote::DisableWatchpoint (watchID = %llu) addr = 0x%8.8llx -- SUCCESS (already disabled)", watchID, (uint64_t)addr);
            // See also 'class WatchpointSentry' within StopInfo.cpp.
            // This disabling attempt might come from the user-supplied actions, we'll route it in order for
            // the watchpoint object to intelligently process this action.
            wp->SetEnabled(false);
            return error;
        }
        
        if (wp->IsHardware())
        {
            GDBStoppointType type = GetGDBStoppointType(wp);
            // Pass down an appropriate z/Z packet...
            if (m_gdb_comm.SendGDBStoppointTypePacket(type, false, addr, wp->GetByteSize()) == 0)
            {
                wp->SetEnabled(false);
                return error;
            }
            else
                error.SetErrorString("sending gdb watchpoint packet failed"); 
        }
        // TODO: clear software watchpoints if we implement them
    }
    else
    {
        error.SetErrorString("Watchpoint argument was NULL.");
    }
    if (error.Success())
        error.SetErrorToGenericError();
    return error;
}

void
ProcessGDBRemote::Clear()
{
    m_flags = 0;
    m_thread_list.Clear();
}

Error
ProcessGDBRemote::DoSignal (int signo)
{
    Error error;
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::DoSignal (signal = %d)", signo);

    if (!m_gdb_comm.SendAsyncSignal (signo))
        error.SetErrorStringWithFormat("failed to send signal %i", signo);
    return error;
}

Error
ProcessGDBRemote::StartDebugserverProcess (const char *debugserver_url)
{
    ProcessLaunchInfo launch_info;
    return StartDebugserverProcess(debugserver_url, launch_info);
}

Error
ProcessGDBRemote::StartDebugserverProcess (const char *debugserver_url, const ProcessInfo &process_info)    // The connection string to use in the spawned debugserver ("localhost:1234" or "/dev/tty...")
{
    Error error;
    if (m_debugserver_pid == LLDB_INVALID_PROCESS_ID)
    {
        // If we locate debugserver, keep that located version around
        static FileSpec g_debugserver_file_spec;

        ProcessLaunchInfo debugserver_launch_info;
        char debugserver_path[PATH_MAX];
        FileSpec &debugserver_file_spec = debugserver_launch_info.GetExecutableFile();

        // Always check to see if we have an environment override for the path
        // to the debugserver to use and use it if we do.
        const char *env_debugserver_path = getenv("LLDB_DEBUGSERVER_PATH");
        if (env_debugserver_path)
            debugserver_file_spec.SetFile (env_debugserver_path, false);
        else
            debugserver_file_spec = g_debugserver_file_spec;
        bool debugserver_exists = debugserver_file_spec.Exists();
        if (!debugserver_exists)
        {
            // The debugserver binary is in the LLDB.framework/Resources
            // directory. 
            if (Host::GetLLDBPath (ePathTypeSupportExecutableDir, debugserver_file_spec))
            {
                debugserver_file_spec.GetFilename().SetCString(DEBUGSERVER_BASENAME);
                debugserver_exists = debugserver_file_spec.Exists();
                if (debugserver_exists)
                {
                    g_debugserver_file_spec = debugserver_file_spec;
                }
                else
                {
                    g_debugserver_file_spec.Clear();
                    debugserver_file_spec.Clear();
                }
            }
        }

        if (debugserver_exists)
        {
            debugserver_file_spec.GetPath (debugserver_path, sizeof(debugserver_path));

            m_stdio_communication.Clear();

            LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));

            Args &debugserver_args = debugserver_launch_info.GetArguments();
            char arg_cstr[PATH_MAX];

            // Start args with "debugserver /file/path -r --"
            debugserver_args.AppendArgument(debugserver_path);
            debugserver_args.AppendArgument(debugserver_url);
            // use native registers, not the GDB registers
            debugserver_args.AppendArgument("--native-regs");   
            // make debugserver run in its own session so signals generated by 
            // special terminal key sequences (^C) don't affect debugserver
            debugserver_args.AppendArgument("--setsid");

            const char *env_debugserver_log_file = getenv("LLDB_DEBUGSERVER_LOG_FILE");
            if (env_debugserver_log_file)
            {
                ::snprintf (arg_cstr, sizeof(arg_cstr), "--log-file=%s", env_debugserver_log_file);
                debugserver_args.AppendArgument(arg_cstr);
            }

            const char *env_debugserver_log_flags = getenv("LLDB_DEBUGSERVER_LOG_FLAGS");
            if (env_debugserver_log_flags)
            {
                ::snprintf (arg_cstr, sizeof(arg_cstr), "--log-flags=%s", env_debugserver_log_flags);
                debugserver_args.AppendArgument(arg_cstr);
            }
//            debugserver_args.AppendArgument("--log-file=/tmp/debugserver.txt");
//            debugserver_args.AppendArgument("--log-flags=0x802e0e");

            // We currently send down all arguments, attach pids, or attach 
            // process names in dedicated GDB server packets, so we don't need
            // to pass them as arguments. This is currently because of all the
            // things we need to setup prior to launching: the environment,
            // current working dir, file actions, etc.
#if 0
            // Now append the program arguments
            if (inferior_argv)
            {
                // Terminate the debugserver args so we can now append the inferior args
                debugserver_args.AppendArgument("--");

                for (int i = 0; inferior_argv[i] != NULL; ++i)
                    debugserver_args.AppendArgument (inferior_argv[i]);
            }
            else if (attach_pid != LLDB_INVALID_PROCESS_ID)
            {
                ::snprintf (arg_cstr, sizeof(arg_cstr), "--attach=%u", attach_pid);
                debugserver_args.AppendArgument (arg_cstr);
            }
            else if (attach_name && attach_name[0])
            {
                if (wait_for_launch)
                    debugserver_args.AppendArgument ("--waitfor");
                else
                    debugserver_args.AppendArgument ("--attach");
                debugserver_args.AppendArgument (attach_name);
            }
#endif
            
            ProcessLaunchInfo::FileAction file_action;
            
            // Close STDIN, STDOUT and STDERR. We might need to redirect them
            // to "/dev/null" if we run into any problems.
            file_action.Close (STDIN_FILENO);
            debugserver_launch_info.AppendFileAction (file_action);
            file_action.Close (STDOUT_FILENO);
            debugserver_launch_info.AppendFileAction (file_action);
            file_action.Close (STDERR_FILENO);
            debugserver_launch_info.AppendFileAction (file_action);

            if (log)
            {
                StreamString strm;
                debugserver_args.Dump (&strm);
                log->Printf("%s arguments:\n%s", debugserver_args.GetArgumentAtIndex(0), strm.GetData());
            }

            debugserver_launch_info.SetMonitorProcessCallback (MonitorDebugserverProcess, this, false);
            debugserver_launch_info.SetUserID(process_info.GetUserID());

            error = Host::LaunchProcess(debugserver_launch_info);

            if (error.Success ())
                m_debugserver_pid = debugserver_launch_info.GetProcessID();
            else
                m_debugserver_pid = LLDB_INVALID_PROCESS_ID;

            if (error.Fail() || log)
                error.PutToLog(log.get(), "Host::LaunchProcess (launch_info) => pid=%llu, path='%s'", m_debugserver_pid, debugserver_path);
        }
        else
        {
            error.SetErrorStringWithFormat ("unable to locate " DEBUGSERVER_BASENAME);
        }

        if (m_debugserver_pid != LLDB_INVALID_PROCESS_ID)
            StartAsyncThread ();
    }
    return error;
}

bool
ProcessGDBRemote::MonitorDebugserverProcess
(
    void *callback_baton,
    lldb::pid_t debugserver_pid,
    bool exited,        // True if the process did exit
    int signo,          // Zero for no signal
    int exit_status     // Exit value of process if signal is zero
)
{
    // The baton is a "ProcessGDBRemote *". Now this class might be gone
    // and might not exist anymore, so we need to carefully try to get the
    // target for this process first since we have a race condition when
    // we are done running between getting the notice that the inferior 
    // process has died and the debugserver that was debugging this process.
    // In our test suite, we are also continually running process after
    // process, so we must be very careful to make sure:
    // 1 - process object hasn't been deleted already
    // 2 - that a new process object hasn't been recreated in its place

    // "debugserver_pid" argument passed in is the process ID for
    // debugserver that we are tracking...
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));

    ProcessGDBRemote *process = (ProcessGDBRemote *)callback_baton;

    // Get a shared pointer to the target that has a matching process pointer.
    // This target could be gone, or the target could already have a new process
    // object inside of it
    TargetSP target_sp (Debugger::FindTargetWithProcess(process));

    if (log)
        log->Printf ("ProcessGDBRemote::MonitorDebugserverProcess (baton=%p, pid=%llu, signo=%i (0x%x), exit_status=%i)", callback_baton, debugserver_pid, signo, signo, exit_status);

    if (target_sp)
    {
        // We found a process in a target that matches, but another thread
        // might be in the process of launching a new process that will
        // soon replace it, so get a shared pointer to the process so we
        // can keep it alive.
        ProcessSP process_sp (target_sp->GetProcessSP());
        // Now we have a shared pointer to the process that can't go away on us
        // so we now make sure it was the same as the one passed in, and also make
        // sure that our previous "process *" didn't get deleted and have a new 
        // "process *" created in its place with the same pointer. To verify this
        // we make sure the process has our debugserver process ID. If we pass all
        // of these tests, then we are sure that this process is the one we were
        // looking for.
        if (process_sp && process == process_sp.get() && process->m_debugserver_pid == debugserver_pid)
        {
            // Sleep for a half a second to make sure our inferior process has
            // time to set its exit status before we set it incorrectly when
            // both the debugserver and the inferior process shut down.
            usleep (500000);
            // If our process hasn't yet exited, debugserver might have died.
            // If the process did exit, the we are reaping it.
            const StateType state = process->GetState();
            
            if (process->m_debugserver_pid != LLDB_INVALID_PROCESS_ID &&
                state != eStateInvalid &&
                state != eStateUnloaded &&
                state != eStateExited &&
                state != eStateDetached)
            {
                char error_str[1024];
                if (signo)
                {
                    const char *signal_cstr = process->GetUnixSignals().GetSignalAsCString (signo);
                    if (signal_cstr)
                        ::snprintf (error_str, sizeof (error_str), DEBUGSERVER_BASENAME " died with signal %s", signal_cstr);
                    else
                        ::snprintf (error_str, sizeof (error_str), DEBUGSERVER_BASENAME " died with signal %i", signo);
                }
                else
                {
                    ::snprintf (error_str, sizeof (error_str), DEBUGSERVER_BASENAME " died with an exit status of 0x%8.8x", exit_status);
                }

                process->SetExitStatus (-1, error_str);
            }
            // Debugserver has exited we need to let our ProcessGDBRemote
            // know that it no longer has a debugserver instance
            process->m_debugserver_pid = LLDB_INVALID_PROCESS_ID;
        }
    }
    return true;
}

void
ProcessGDBRemote::KillDebugserverProcess ()
{
    if (m_debugserver_pid != LLDB_INVALID_PROCESS_ID)
    {
        ::kill (m_debugserver_pid, SIGINT);
        m_debugserver_pid = LLDB_INVALID_PROCESS_ID;
    }
}

void
ProcessGDBRemote::Initialize()
{
    static bool g_initialized = false;

    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);

        Log::Callbacks log_callbacks = {
            ProcessGDBRemoteLog::DisableLog,
            ProcessGDBRemoteLog::EnableLog,
            ProcessGDBRemoteLog::ListLogCategories
        };

        Log::RegisterLogChannel (ProcessGDBRemote::GetPluginNameStatic(), log_callbacks);
    }
}

bool
ProcessGDBRemote::StartAsyncThread ()
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));

    if (log)
        log->Printf ("ProcessGDBRemote::%s ()", __FUNCTION__);
    
    Mutex::Locker start_locker(m_async_thread_state_mutex);
    if (m_async_thread_state == eAsyncThreadNotStarted)
    {
        // Create a thread that watches our internal state and controls which
        // events make it to clients (into the DCProcess event queue).
        m_async_thread = Host::ThreadCreate ("<lldb.process.gdb-remote.async>", ProcessGDBRemote::AsyncThread, this, NULL);
        if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
        {
            m_async_thread_state = eAsyncThreadRunning;
            return true;
        }
        else
            return false;
    }
    else
    {
        // Somebody tried to start the async thread while it was either being started or stopped.  If the former, and
        // it started up successfully, then say all's well.  Otherwise it is an error, since we aren't going to restart it.
        if (log)
            log->Printf ("ProcessGDBRemote::%s () - Called when Async thread was in state: %d.", __FUNCTION__, m_async_thread_state);
        if (m_async_thread_state == eAsyncThreadRunning)
            return true;
        else
            return false;
    }
}

void
ProcessGDBRemote::StopAsyncThread ()
{
    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));

    if (log)
        log->Printf ("ProcessGDBRemote::%s ()", __FUNCTION__);

    Mutex::Locker start_locker(m_async_thread_state_mutex);
    if (m_async_thread_state == eAsyncThreadRunning)
    {
        m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncThreadShouldExit);
        
        //  This will shut down the async thread.
        m_gdb_comm.Disconnect();    // Disconnect from the debug server.

        // Stop the stdio thread
        if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
        {
            Host::ThreadJoin (m_async_thread, NULL, NULL);
        }
        m_async_thread_state = eAsyncThreadDone;
    }
    else
    {
        if (log)
            log->Printf ("ProcessGDBRemote::%s () - Called when Async thread was in state: %d.", __FUNCTION__, m_async_thread_state);
    }
}


void *
ProcessGDBRemote::AsyncThread (void *arg)
{
    ProcessGDBRemote *process = (ProcessGDBRemote*) arg;

    LogSP log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) thread starting...", __FUNCTION__, arg, process->GetID());

    Listener listener ("ProcessGDBRemote::AsyncThread");
    EventSP event_sp;
    const uint32_t desired_event_mask = eBroadcastBitAsyncContinue |
                                        eBroadcastBitAsyncThreadShouldExit;

    if (listener.StartListeningForEvents (&process->m_async_broadcaster, desired_event_mask) == desired_event_mask)
    {
        listener.StartListeningForEvents (&process->m_gdb_comm, Communication::eBroadcastBitReadThreadDidExit);
    
        bool done = false;
        while (!done)
        {
            if (log)
                log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) listener.WaitForEvent (NULL, event_sp)...", __FUNCTION__, arg, process->GetID());
            if (listener.WaitForEvent (NULL, event_sp))
            {
                const uint32_t event_type = event_sp->GetType();
                if (event_sp->BroadcasterIs (&process->m_async_broadcaster))
                {
                    if (log)
                        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) Got an event of type: %d...", __FUNCTION__, arg, process->GetID(), event_type);

                    switch (event_type)
                    {
                        case eBroadcastBitAsyncContinue:
                            {
                                const EventDataBytes *continue_packet = EventDataBytes::GetEventDataFromEvent(event_sp.get());

                                if (continue_packet)
                                {
                                    const char *continue_cstr = (const char *)continue_packet->GetBytes ();
                                    const size_t continue_cstr_len = continue_packet->GetByteSize ();
                                    if (log)
                                        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) got eBroadcastBitAsyncContinue: %s", __FUNCTION__, arg, process->GetID(), continue_cstr);

                                    if (::strstr (continue_cstr, "vAttach") == NULL)
                                        process->SetPrivateState(eStateRunning);
                                    StringExtractorGDBRemote response;
                                    StateType stop_state = process->GetGDBRemote().SendContinuePacketAndWaitForResponse (process, continue_cstr, continue_cstr_len, response);

                                    // We need to immediately clear the thread ID list so we are sure to get a valid list of threads.
                                    // The thread ID list might be contained within the "response", or the stop reply packet that
                                    // caused the stop. So clear it now before we give the stop reply packet to the process
                                    // using the process->SetLastStopPacket()...
                                    process->ClearThreadIDList ();

                                    switch (stop_state)
                                    {
                                    case eStateStopped:
                                    case eStateCrashed:
                                    case eStateSuspended:
                                        process->SetLastStopPacket (response);
                                        process->SetPrivateState (stop_state);
                                        break;

                                    case eStateExited:
                                        process->SetLastStopPacket (response);
                                        process->ClearThreadIDList();
                                        response.SetFilePos(1);
                                        process->SetExitStatus(response.GetHexU8(), NULL);
                                        done = true;
                                        break;

                                    case eStateInvalid:
                                        process->SetExitStatus(-1, "lost connection");
                                        break;

                                    default:
                                        process->SetPrivateState (stop_state);
                                        break;
                                    }
                                }
                            }
                            break;

                        case eBroadcastBitAsyncThreadShouldExit:
                            if (log)
                                log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) got eBroadcastBitAsyncThreadShouldExit...", __FUNCTION__, arg, process->GetID());
                            done = true;
                            break;

                        default:
                            if (log)
                                log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) got unknown event 0x%8.8x", __FUNCTION__, arg, process->GetID(), event_type);
                            done = true;
                            break;
                    }
                }
                else if (event_sp->BroadcasterIs (&process->m_gdb_comm))
                {
                    if (event_type & Communication::eBroadcastBitReadThreadDidExit)
                    {
                        process->SetExitStatus (-1, "lost connection");
                        done = true;
                    }
                }
            }
            else
            {
                if (log)
                    log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) listener.WaitForEvent (NULL, event_sp) => false", __FUNCTION__, arg, process->GetID());
                done = true;
            }
        }
    }

    if (log)
        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %llu) thread exiting...", __FUNCTION__, arg, process->GetID());

    process->m_async_thread = LLDB_INVALID_HOST_THREAD;
    return NULL;
}

const char *
ProcessGDBRemote::GetDispatchQueueNameForThread
(
    addr_t thread_dispatch_qaddr,
    std::string &dispatch_queue_name
)
{
    dispatch_queue_name.clear();
    if (thread_dispatch_qaddr != 0 && thread_dispatch_qaddr != LLDB_INVALID_ADDRESS)
    {
        // Cache the dispatch_queue_offsets_addr value so we don't always have
        // to look it up
        if (m_dispatch_queue_offsets_addr == LLDB_INVALID_ADDRESS)
        {
            static ConstString g_dispatch_queue_offsets_symbol_name ("dispatch_queue_offsets");
            const Symbol *dispatch_queue_offsets_symbol = NULL;
            ModuleSpec libSystem_module_spec (FileSpec("libSystem.B.dylib", false));
            ModuleSP module_sp(GetTarget().GetImages().FindFirstModule (libSystem_module_spec));
            if (module_sp)
                dispatch_queue_offsets_symbol = module_sp->FindFirstSymbolWithNameAndType (g_dispatch_queue_offsets_symbol_name, eSymbolTypeData);
            
            if (dispatch_queue_offsets_symbol == NULL)
            {
                ModuleSpec libdispatch_module_spec (FileSpec("libdispatch.dylib", false));
                module_sp = GetTarget().GetImages().FindFirstModule (libdispatch_module_spec);
                if (module_sp)
                    dispatch_queue_offsets_symbol = module_sp->FindFirstSymbolWithNameAndType (g_dispatch_queue_offsets_symbol_name, eSymbolTypeData);
            }
            if (dispatch_queue_offsets_symbol)
                m_dispatch_queue_offsets_addr = dispatch_queue_offsets_symbol->GetAddress().GetLoadAddress(&m_target);

            if (m_dispatch_queue_offsets_addr == LLDB_INVALID_ADDRESS)
                return NULL;
        }

        uint8_t memory_buffer[8];
        DataExtractor data (memory_buffer, 
                            sizeof(memory_buffer), 
                            m_target.GetArchitecture().GetByteOrder(), 
                            m_target.GetArchitecture().GetAddressByteSize());

        // Excerpt from src/queue_private.h
        struct dispatch_queue_offsets_s
        {
            uint16_t dqo_version;
            uint16_t dqo_label;      // in version 1-3, offset to string; in version 4+, offset to a pointer to a string
            uint16_t dqo_label_size; // in version 1-3, length of string; in version 4+, size of a (void*) in this process
        } dispatch_queue_offsets;


        Error error;
        if (ReadMemory (m_dispatch_queue_offsets_addr, memory_buffer, sizeof(dispatch_queue_offsets), error) == sizeof(dispatch_queue_offsets))
        {
            uint32_t data_offset = 0;
            if (data.GetU16(&data_offset, &dispatch_queue_offsets.dqo_version, sizeof(dispatch_queue_offsets)/sizeof(uint16_t)))
            {
                if (ReadMemory (thread_dispatch_qaddr, &memory_buffer, data.GetAddressByteSize(), error) == data.GetAddressByteSize())
                {
                    data_offset = 0;
                    lldb::addr_t queue_addr = data.GetAddress(&data_offset);
                    if (dispatch_queue_offsets.dqo_version >= 4)
                    {
                        // libdispatch versions 4+, pointer to dispatch name is in the 
                        // queue structure.
                        lldb::addr_t pointer_to_label_address = queue_addr + dispatch_queue_offsets.dqo_label;
                        if (ReadMemory (pointer_to_label_address, &memory_buffer, data.GetAddressByteSize(), error) == data.GetAddressByteSize())
                        {
                            data_offset = 0;
                            lldb::addr_t label_addr = data.GetAddress(&data_offset);
                            ReadCStringFromMemory (label_addr, dispatch_queue_name, error);
                        }
                    }
                    else
                    {
                        // libdispatch versions 1-3, dispatch name is a fixed width char array
                        // in the queue structure.
                        lldb::addr_t label_addr = queue_addr + dispatch_queue_offsets.dqo_label;
                        dispatch_queue_name.resize(dispatch_queue_offsets.dqo_label_size, '\0');
                        size_t bytes_read = ReadMemory (label_addr, &dispatch_queue_name[0], dispatch_queue_offsets.dqo_label_size, error);
                        if (bytes_read < dispatch_queue_offsets.dqo_label_size)
                            dispatch_queue_name.erase (bytes_read);
                    }
                }
            }
        }
    }
    if (dispatch_queue_name.empty())
        return NULL;
    return dispatch_queue_name.c_str();
}

//uint32_t
//ProcessGDBRemote::ListProcessesMatchingName (const char *name, StringList &matches, std::vector<lldb::pid_t> &pids)
//{
//    // If we are planning to launch the debugserver remotely, then we need to fire up a debugserver
//    // process and ask it for the list of processes. But if we are local, we can let the Host do it.
//    if (m_local_debugserver)
//    {
//        return Host::ListProcessesMatchingName (name, matches, pids);
//    }
//    else 
//    {
//        // FIXME: Implement talking to the remote debugserver.
//        return 0;
//    }
//
//}
//
bool
ProcessGDBRemote::NewThreadNotifyBreakpointHit (void *baton,
                             lldb_private::StoppointCallbackContext *context,
                             lldb::user_id_t break_id,
                             lldb::user_id_t break_loc_id)
{
    // I don't think I have to do anything here, just make sure I notice the new thread when it starts to 
    // run so I can stop it if that's what I want to do.
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf("Hit New Thread Notification breakpoint.");
    return false;
}


bool
ProcessGDBRemote::StartNoticingNewThreads()
{
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (m_thread_create_bp_sp)
    {
        if (log && log->GetVerbose())
            log->Printf("Enabled noticing new thread breakpoint.");
        m_thread_create_bp_sp->SetEnabled(true);
    }
    else
    {
        PlatformSP platform_sp (m_target.GetPlatform());
        if (platform_sp)
        {
            m_thread_create_bp_sp = platform_sp->SetThreadCreationBreakpoint(m_target);
            if (m_thread_create_bp_sp)
            {
                if (log && log->GetVerbose())
                    log->Printf("Successfully created new thread notification breakpoint %i", m_thread_create_bp_sp->GetID());
                m_thread_create_bp_sp->SetCallback (ProcessGDBRemote::NewThreadNotifyBreakpointHit, this, true);
            }
            else
            {
                if (log)
                    log->Printf("Failed to create new thread notification breakpoint.");
            }
        }
    }
    return m_thread_create_bp_sp.get() != NULL;
}

bool
ProcessGDBRemote::StopNoticingNewThreads()
{   
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log && log->GetVerbose())
        log->Printf ("Disabling new thread notification breakpoint.");

    if (m_thread_create_bp_sp)
        m_thread_create_bp_sp->SetEnabled(false);

    return true;
}
    
lldb_private::DynamicLoader *
ProcessGDBRemote::GetDynamicLoader ()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, m_dyld_plugin_name.empty() ? NULL : m_dyld_plugin_name.c_str()));
    return m_dyld_ap.get();
}


class CommandObjectProcessGDBRemotePacketHistory : public CommandObjectParsed
{
private:
    
public:
    CommandObjectProcessGDBRemotePacketHistory(CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter,
                         "process plugin packet history",
                         "Dumps the packet history buffer. ",
                         NULL)
    {
    }
    
    ~CommandObjectProcessGDBRemotePacketHistory ()
    {
    }
    
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        if (argc == 0)
        {
            ProcessGDBRemote *process = (ProcessGDBRemote *)m_interpreter.GetExecutionContext().GetProcessPtr();
            if (process)
            {
                process->GetGDBRemote().DumpHistory(result.GetOutputStream());
                result.SetStatus (eReturnStatusSuccessFinishResult);
                return true;
            }
        }
        else
        {
            result.AppendErrorWithFormat ("'%s' takes no arguments", m_cmd_name.c_str());
        }
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
};

class CommandObjectProcessGDBRemotePacketSend : public CommandObjectParsed
{
private:
    
public:
    CommandObjectProcessGDBRemotePacketSend(CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "process plugin packet send",
                             "Send a custom packet through the GDB remote protocol and print the answer. "
                             "The packet header and footer will automatically be added to the packet prior to sending and stripped from the result.",
                             NULL)
    {
    }
    
    ~CommandObjectProcessGDBRemotePacketSend ()
    {
    }
    
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        if (argc == 0)
        {
            result.AppendErrorWithFormat ("'%s' takes a one or more packet content arguments", m_cmd_name.c_str());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        ProcessGDBRemote *process = (ProcessGDBRemote *)m_interpreter.GetExecutionContext().GetProcessPtr();
        if (process)
        {
            const StateType state = process->GetState();

            if (StateIsStoppedState (state, true))
            {
                for (size_t i=0; i<argc; ++ i)
                {
                    const char *packet_cstr = command.GetArgumentAtIndex(0);
                    bool send_async = false;
                    StringExtractorGDBRemote response;
                    process->GetGDBRemote().SendPacketAndWaitForResponse(packet_cstr, response, send_async);
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                    Stream &output_strm = result.GetOutputStream();
                    output_strm.Printf ("  packet: %s\n", packet_cstr);
                    const std::string &response_str = response.GetStringRef();
                    if (response_str.empty())
                        output_strm.PutCString ("response: \nerror: UNIMPLEMENTED\n");
                    else
                        output_strm.Printf ("response: %s\n", response.GetStringRef().c_str());
                }
            }
            else
            {
                result.AppendErrorWithFormat ("process must be stopped in order to send GDB remote packets, state is %s", StateAsCString (state));
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

        }
        return true;
    }
};

class CommandObjectProcessGDBRemotePacket : public CommandObjectMultiword
{
private:
    
public:
    CommandObjectProcessGDBRemotePacket(CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "process plugin packet",
                                "Commands that deal with GDB remote packets.",
                                NULL)
    {
        LoadSubCommand ("history", CommandObjectSP (new CommandObjectProcessGDBRemotePacketHistory (interpreter)));
        LoadSubCommand ("send", CommandObjectSP (new CommandObjectProcessGDBRemotePacketSend (interpreter)));
    }
    
    ~CommandObjectProcessGDBRemotePacket ()
    {
    }    
};

class CommandObjectMultiwordProcessGDBRemote : public CommandObjectMultiword
{
public:
    CommandObjectMultiwordProcessGDBRemote (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "process plugin",
                                "A set of commands for operating on a ProcessGDBRemote process.",
                                "process plugin <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("packet", CommandObjectSP (new CommandObjectProcessGDBRemotePacket    (interpreter)));
    }

    ~CommandObjectMultiwordProcessGDBRemote ()
    {
    }
};

CommandObject *
ProcessGDBRemote::GetPluginCommandObject()
{
    if (!m_command_sp)
        m_command_sp.reset (new CommandObjectMultiwordProcessGDBRemote (GetTarget().GetDebugger().GetCommandInterpreter()));
    return m_command_sp.get();
}
