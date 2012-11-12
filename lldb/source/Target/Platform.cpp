//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Platform.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
    
// Use a singleton function for g_local_platform_sp to avoid init
// constructors since LLDB is often part of a shared library
static PlatformSP&
GetDefaultPlatformSP ()
{
    static PlatformSP g_default_platform_sp;
    return g_default_platform_sp;
}

static Mutex &
GetConnectedPlatformListMutex ()
{
    static Mutex g_remote_connected_platforms_mutex (Mutex::eMutexTypeRecursive);
    return g_remote_connected_platforms_mutex;
}
static std::vector<PlatformSP> &
GetConnectedPlatformList ()
{
    static std::vector<PlatformSP> g_remote_connected_platforms;
    return g_remote_connected_platforms;
}


const char *
Platform::GetHostPlatformName ()
{
    return "host";
}

//------------------------------------------------------------------
/// Get the native host platform plug-in. 
///
/// There should only be one of these for each host that LLDB runs
/// upon that should be statically compiled in and registered using
/// preprocessor macros or other similar build mechanisms.
///
/// This platform will be used as the default platform when launching
/// or attaching to processes unless another platform is specified.
//------------------------------------------------------------------
PlatformSP
Platform::GetDefaultPlatform ()
{
    return GetDefaultPlatformSP ();
}

void
Platform::SetDefaultPlatform (const lldb::PlatformSP &platform_sp)
{
    // The native platform should use its static void Platform::Initialize()
    // function to register itself as the native platform.
    GetDefaultPlatformSP () = platform_sp;
}

Error
Platform::GetFile (const FileSpec &platform_file, 
                   const UUID *uuid_ptr,
                   FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}

FileSpec
Platform::LocateExecutableScriptingResource (const ModuleSpec &module_spec)
{
    return FileSpec();
}

Error
Platform::GetSharedModule (const ModuleSpec &module_spec,
                           ModuleSP &module_sp,
                           const FileSpecList *module_search_paths_ptr,
                           ModuleSP *old_module_sp_ptr,
                           bool *did_create_ptr)
{
    // Don't do any path remapping for the default implementation
    // of the platform GetSharedModule function, just call through
    // to our static ModuleList function. Platform subclasses that
    // implement remote debugging, might have a developer kits
    // installed that have cached versions of the files for the
    // remote target, or might implement a download and cache 
    // locally implementation.
    const bool always_create = false;
    return ModuleList::GetSharedModule (module_spec, 
                                        module_sp,
                                        module_search_paths_ptr,
                                        old_module_sp_ptr,
                                        did_create_ptr,
                                        always_create);
}

PlatformSP
Platform::Create (const char *platform_name, Error &error)
{
    PlatformCreateInstance create_callback = NULL;
    lldb::PlatformSP platform_sp;
    if (platform_name && platform_name[0])
    {
        create_callback = PluginManager::GetPlatformCreateCallbackForPluginName (platform_name);
        if (create_callback)
            platform_sp.reset(create_callback(true, NULL));
        else
            error.SetErrorStringWithFormat ("unable to find a plug-in for the platform named \"%s\"", platform_name);
    }
    else
        error.SetErrorString ("invalid platform name");
    return platform_sp;
}


PlatformSP
Platform::Create (const ArchSpec &arch, ArchSpec *platform_arch_ptr, Error &error)
{
    lldb::PlatformSP platform_sp;
    if (arch.IsValid())
    {
        uint32_t idx;
        PlatformCreateInstance create_callback;
        for (idx = 0; (create_callback = PluginManager::GetPlatformCreateCallbackAtIndex (idx)); ++idx)
        {
            if (create_callback)
                platform_sp.reset(create_callback(false, &arch));
            if (platform_sp && platform_sp->IsCompatibleArchitecture(arch, platform_arch_ptr))
                return platform_sp;
        }
    }
    else
        error.SetErrorString ("invalid platform name");
    if (platform_arch_ptr)
        platform_arch_ptr->Clear();
    platform_sp.reset();
    return platform_sp;
}

uint32_t
Platform::GetNumConnectedRemotePlatforms ()
{
    Mutex::Locker locker (GetConnectedPlatformListMutex ());
    return GetConnectedPlatformList().size();
}

PlatformSP
Platform::GetConnectedRemotePlatformAtIndex (uint32_t idx)
{
    PlatformSP platform_sp;
    {
        Mutex::Locker locker (GetConnectedPlatformListMutex ());
        if (idx < GetConnectedPlatformList().size())
            platform_sp = GetConnectedPlatformList ()[idx];
    }
    return platform_sp;
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
Platform::Platform (bool is_host) :
    m_is_host (is_host),
    m_os_version_set_while_connected (false),
    m_system_arch_set_while_connected (false),
    m_sdk_sysroot (),
    m_sdk_build (),
    m_remote_url (),
    m_name (),
    m_major_os_version (UINT32_MAX),
    m_minor_os_version (UINT32_MAX),
    m_update_os_version (UINT32_MAX),
    m_system_arch(),
    m_uid_map_mutex (Mutex::eMutexTypeNormal),
    m_gid_map_mutex (Mutex::eMutexTypeNormal),
    m_uid_map(),
    m_gid_map(),
    m_max_uid_name_len (0),
    m_max_gid_name_len (0)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Platform::Platform()", this);
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
Platform::~Platform()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Platform::~Platform()", this);
}

void
Platform::GetStatus (Stream &strm)
{
    uint32_t major = UINT32_MAX;
    uint32_t minor = UINT32_MAX;
    uint32_t update = UINT32_MAX;
    std::string s;
    strm.Printf ("  Platform: %s\n", GetShortPluginName());

    ArchSpec arch (GetSystemArchitecture());
    if (arch.IsValid())
    {
        if (!arch.GetTriple().str().empty())
        strm.Printf("    Triple: %s\n", arch.GetTriple().str().c_str());        
    }

    if (GetOSVersion(major, minor, update))
    {
        strm.Printf("OS Version: %u", major);
        if (minor != UINT32_MAX)
            strm.Printf(".%u", minor);
        if (update != UINT32_MAX)
            strm.Printf(".%u", update);

        if (GetOSBuildString (s))
            strm.Printf(" (%s)", s.c_str());

        strm.EOL();
    }

    if (GetOSKernelDescription (s))
        strm.Printf("    Kernel: %s\n", s.c_str());

    if (IsHost())
    {
        strm.Printf("  Hostname: %s\n", GetHostname());
    }
    else
    {
        const bool is_connected = IsConnected();
        if (is_connected)
            strm.Printf("  Hostname: %s\n", GetHostname());
        strm.Printf(" Connected: %s\n", is_connected ? "yes" : "no");
    }
}


bool
Platform::GetOSVersion (uint32_t &major, 
                        uint32_t &minor, 
                        uint32_t &update)
{
    bool success = m_major_os_version != UINT32_MAX;
    if (IsHost())
    {
        if (!success)
        {
            // We have a local host platform
            success = Host::GetOSVersion (m_major_os_version, 
                                          m_minor_os_version, 
                                          m_update_os_version);
            m_os_version_set_while_connected = success;
        }
    }
    else 
    {
        // We have a remote platform. We can only fetch the remote
        // OS version if we are connected, and we don't want to do it
        // more than once.
        
        const bool is_connected = IsConnected();

        bool fetch = false;
        if (success)
        {
            // We have valid OS version info, check to make sure it wasn't
            // manually set prior to connecting. If it was manually set prior
            // to connecting, then lets fetch the actual OS version info
            // if we are now connected.
            if (is_connected && !m_os_version_set_while_connected)
                fetch = true;
        }
        else
        {
            // We don't have valid OS version info, fetch it if we are connected
            fetch = is_connected;
        }

        if (fetch)
        {
            success = GetRemoteOSVersion ();
            m_os_version_set_while_connected = success;
        }
    }

    if (success)
    {
        major = m_major_os_version;
        minor = m_minor_os_version;
        update = m_update_os_version;
    }
    return success;
}

bool
Platform::GetOSBuildString (std::string &s)
{
    if (IsHost())
        return Host::GetOSBuildString (s);
    else
        return GetRemoteOSBuildString (s);
}

bool
Platform::GetOSKernelDescription (std::string &s)
{
    if (IsHost())
        return Host::GetOSKernelDescription (s);
    else
        return GetRemoteOSKernelDescription (s);
}

const char *
Platform::GetName ()
{
    const char *name = GetHostname();
    if (name == NULL || name[0] == '\0')
        name = GetShortPluginName();
    return name;
}

const char *
Platform::GetHostname ()
{
    if (IsHost())
        return "localhost";

    if (m_name.empty())        
        return NULL;
    return m_name.c_str();
}

const char *
Platform::GetUserName (uint32_t uid)
{
    const char *user_name = GetCachedUserName(uid);
    if (user_name)
        return user_name;
    if (IsHost())
    {
        std::string name;
        if (Host::GetUserName(uid, name))
            return SetCachedUserName (uid, name.c_str(), name.size());
    }
    return NULL;
}

const char *
Platform::GetGroupName (uint32_t gid)
{
    const char *group_name = GetCachedGroupName(gid);
    if (group_name)
        return group_name;
    if (IsHost())
    {
        std::string name;
        if (Host::GetGroupName(gid, name))
            return SetCachedGroupName (gid, name.c_str(), name.size());
    }
    return NULL;
}

bool
Platform::SetOSVersion (uint32_t major, 
                        uint32_t minor, 
                        uint32_t update)
{
    if (IsHost())
    {
        // We don't need anyone setting the OS version for the host platform, 
        // we should be able to figure it out by calling Host::GetOSVersion(...).
        return false; 
    }
    else
    {
        // We have a remote platform, allow setting the target OS version if
        // we aren't connected, since if we are connected, we should be able to
        // request the remote OS version from the connected platform.
        if (IsConnected())
            return false;
        else
        {
            // We aren't connected and we might want to set the OS version
            // ahead of time before we connect so we can peruse files and
            // use a local SDK or PDK cache of support files to disassemble
            // or do other things.
            m_major_os_version = major;
            m_minor_os_version = minor;
            m_update_os_version = update;
            return true;
        }
    }
    return false;
}


Error
Platform::ResolveExecutable (const FileSpec &exe_file,
                             const ArchSpec &exe_arch,
                             lldb::ModuleSP &exe_module_sp,
                             const FileSpecList *module_search_paths_ptr)
{
    Error error;
    if (exe_file.Exists())
    {
        ModuleSpec module_spec (exe_file, exe_arch);
        if (module_spec.GetArchitecture().IsValid())
        {
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 module_search_paths_ptr,
                                                 NULL, 
                                                 NULL);
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, module_spec.GetArchitecture()); ++idx)
            {
                error = ModuleList::GetSharedModule (module_spec, 
                                                     exe_module_sp, 
                                                     module_search_paths_ptr,
                                                     NULL, 
                                                     NULL);
                // Did we find an executable using one of the 
                if (error.Success() && exe_module_sp)
                    break;
            }
        }
    }
    else
    {
        error.SetErrorStringWithFormat ("'%s%s%s' does not exist",
                                        exe_file.GetDirectory().AsCString(""),
                                        exe_file.GetDirectory() ? "/" : "",
                                        exe_file.GetFilename().AsCString(""));
    }
    return error;
}

Error
Platform::ResolveSymbolFile (Target &target,
                             const ModuleSpec &sym_spec,
                             FileSpec &sym_file)
{
    Error error;
    if (sym_spec.GetSymbolFileSpec().Exists())
        sym_file = sym_spec.GetSymbolFileSpec();
    else
        error.SetErrorString("unable to resolve symbol file");
    return error;
    
}



bool
Platform::ResolveRemotePath (const FileSpec &platform_path,
                             FileSpec &resolved_platform_path)
{
    resolved_platform_path = platform_path;
    return resolved_platform_path.ResolvePath();
}


const ArchSpec &
Platform::GetSystemArchitecture()
{
    if (IsHost())
    {
        if (!m_system_arch.IsValid())
        {
            // We have a local host platform
            m_system_arch = Host::GetArchitecture();
            m_system_arch_set_while_connected = m_system_arch.IsValid();
        }
    }
    else 
    {
        // We have a remote platform. We can only fetch the remote
        // system architecture if we are connected, and we don't want to do it
        // more than once.
        
        const bool is_connected = IsConnected();

        bool fetch = false;
        if (m_system_arch.IsValid())
        {
            // We have valid OS version info, check to make sure it wasn't
            // manually set prior to connecting. If it was manually set prior
            // to connecting, then lets fetch the actual OS version info
            // if we are now connected.
            if (is_connected && !m_system_arch_set_while_connected)
                fetch = true;
        }
        else
        {
            // We don't have valid OS version info, fetch it if we are connected
            fetch = is_connected;
        }

        if (fetch)
        {
            m_system_arch = GetRemoteSystemArchitecture ();
            m_system_arch_set_while_connected = m_system_arch.IsValid();
        }
    }
    return m_system_arch;
}


Error
Platform::ConnectRemote (Args& args)
{
    Error error;
    if (IsHost())
        error.SetErrorStringWithFormat ("The currently selected platform (%s) is the host platform and is always connected.", GetShortPluginName());
    else
        error.SetErrorStringWithFormat ("Platform::ConnectRemote() is not supported by %s", GetShortPluginName());
    return error;
}

Error
Platform::DisconnectRemote ()
{
    Error error;
    if (IsHost())
        error.SetErrorStringWithFormat ("The currently selected platform (%s) is the host platform and is always connected.", GetShortPluginName());
    else
        error.SetErrorStringWithFormat ("Platform::DisconnectRemote() is not supported by %s", GetShortPluginName());
    return error;
}

bool
Platform::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    // Take care of the host case so that each subclass can just 
    // call this function to get the host functionality.
    if (IsHost())
        return Host::GetProcessInfo (pid, process_info);
    return false;
}

uint32_t
Platform::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &process_infos)
{
    // Take care of the host case so that each subclass can just 
    // call this function to get the host functionality.
    uint32_t match_count = 0;
    if (IsHost())
        match_count = Host::FindProcesses (match_info, process_infos);
    return match_count;    
}


Error
Platform::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    // Take care of the host case so that each subclass can just 
    // call this function to get the host functionality.
    if (IsHost())
    {
        if (::getenv ("LLDB_LAUNCH_FLAG_LAUNCH_IN_TTY"))
            launch_info.GetFlags().Set (eLaunchFlagLaunchInTTY);
        
        if (launch_info.GetFlags().Test (eLaunchFlagLaunchInShell))
        {
            const bool is_localhost = true;
            const bool will_debug = launch_info.GetFlags().Test(eLaunchFlagDebug);
            const bool first_arg_is_full_shell_command = false;
            if (!launch_info.ConvertArgumentsForLaunchingInShell (error,
                                                                  is_localhost,
                                                                  will_debug,
                                                                  first_arg_is_full_shell_command))
                return error;
        }

        error = Host::LaunchProcess (launch_info);
    }
    else
        error.SetErrorString ("base lldb_private::Platform class can't launch remote processes");
    return error;
}

lldb::ProcessSP
Platform::DebugProcess (ProcessLaunchInfo &launch_info, 
                        Debugger &debugger,
                        Target *target,       // Can be NULL, if NULL create a new target, else use existing one
                        Listener &listener,
                        Error &error)
{
    ProcessSP process_sp;
    // Make sure we stop at the entry point
    launch_info.GetFlags ().Set (eLaunchFlagDebug);
    // We always launch the process we are going to debug in a separate process
    // group, since then we can handle ^C interrupts ourselves w/o having to worry
    // about the target getting them as well.
    launch_info.SetLaunchInSeparateProcessGroup(true);
    
    error = LaunchProcess (launch_info);
    if (error.Success())
    {
        if (launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID)
        {
            ProcessAttachInfo attach_info (launch_info);
            process_sp = Attach (attach_info, debugger, target, listener, error);
            if (process_sp)
            {
                // Since we attached to the process, it will think it needs to detach
                // if the process object just goes away without an explicit call to
                // Process::Kill() or Process::Detach(), so let it know to kill the 
                // process if this happens.
                process_sp->SetShouldDetach (false);
                
                // If we didn't have any file actions, the pseudo terminal might
                // have been used where the slave side was given as the file to
                // open for stdin/out/err after we have already opened the master
                // so we can read/write stdin/out/err.
                int pty_fd = launch_info.GetPTY().ReleaseMasterFileDescriptor();
                if (pty_fd != lldb_utility::PseudoTerminal::invalid_fd)
                {
                    process_sp->SetSTDIOFileDescriptor(pty_fd);
                }
            }
        }
    }
    return process_sp;
}


lldb::PlatformSP
Platform::GetPlatformForArchitecture (const ArchSpec &arch, ArchSpec *platform_arch_ptr)
{
    lldb::PlatformSP platform_sp;
    Error error;
    if (arch.IsValid())
        platform_sp = Platform::Create (arch, platform_arch_ptr, error);
    return platform_sp;
}


//------------------------------------------------------------------
/// Lets a platform answer if it is compatible with a given
/// architecture and the target triple contained within.
//------------------------------------------------------------------
bool
Platform::IsCompatibleArchitecture (const ArchSpec &arch, ArchSpec *compatible_arch_ptr)
{
    // If the architecture is invalid, we must answer true...
    if (arch.IsValid())
    {
        ArchSpec platform_arch;
        for (uint32_t arch_idx=0; GetSupportedArchitectureAtIndex (arch_idx, platform_arch); ++arch_idx)
        {
            if (arch == platform_arch)
            {
                if (compatible_arch_ptr)
                    *compatible_arch_ptr = platform_arch;
                return true;
            }
        }
    }
    if (compatible_arch_ptr)
        compatible_arch_ptr->Clear();
    return false;
    
}

lldb::BreakpointSP
Platform::SetThreadCreationBreakpoint (lldb_private::Target &target)
{
    return lldb::BreakpointSP();
}

size_t
Platform::GetEnvironment (StringList &environment)
{
    environment.Clear();
    return false;
}

