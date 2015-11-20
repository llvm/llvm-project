//===-- PlatformNetBSD.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformNetBSD.h"
#include "lldb/Host/Config.h"

// C Includes
#include <stdio.h>
#ifndef LLDB_DISABLE_POSIX
#include <sys/utsname.h>
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointSite.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_netbsd;

PlatformSP
PlatformNetBSD::CreateInstance(bool force, const ArchSpec *arch)
{
    // The only time we create an instance is when we are creating a remote
    // netbsd platform
    const bool is_host = false;

    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getOS())
        {
            case llvm::Triple::NetBSD:
                create = true;
                break;

            default:
                break;
        }
    }
    if (create)
        return PlatformSP(new PlatformNetBSD (is_host));
    return PlatformSP();

}

ConstString
PlatformNetBSD::GetPluginNameStatic(bool is_host)
{
    if (is_host)
    {
        static ConstString g_host_name(Platform::GetHostPlatformName ());
        return g_host_name;
    }
    else
    {
        static ConstString g_remote_name("remote-netbsd");
        return g_remote_name;
    }
}

const char *
PlatformNetBSD::GetDescriptionStatic (bool is_host)
{
    if (is_host)
        return "Local NetBSD user platform plug-in.";
    else
        return "Remote NetBSD user platform plug-in.";
}

static uint32_t g_initialize_count = 0;

void
PlatformNetBSD::Initialize ()
{
    Platform::Initialize ();

    if (g_initialize_count++ == 0)
    {
#if defined(__NetBSD__)
        // Force a host flag to true for the default platform object.
        PlatformSP default_platform_sp (new PlatformNetBSD(true));
        default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
        Platform::SetHostPlatform (default_platform_sp);
#endif
        PluginManager::RegisterPlugin(PlatformNetBSD::GetPluginNameStatic(false),
                                      PlatformNetBSD::GetDescriptionStatic(false),
                                      PlatformNetBSD::CreateInstance);
    }
}

void
PlatformNetBSD::Terminate ()
{
    if (g_initialize_count > 0 && --g_initialize_count == 0)
        PluginManager::UnregisterPlugin (PlatformNetBSD::CreateInstance);

    Platform::Terminate ();
}

bool
PlatformNetBSD::GetModuleSpec (const FileSpec& module_file_spec,
                               const ArchSpec& arch,
                               ModuleSpec &module_spec)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetModuleSpec (module_file_spec, arch, module_spec);

    return Platform::GetModuleSpec (module_file_spec, arch, module_spec);
}

Error
PlatformNetBSD::RunShellCommand(const char *command,
                                const FileSpec &working_dir,
                                int *status_ptr,
                                int *signo_ptr,
                                std::string *command_output,
                                uint32_t timeout_sec)
{
    if (IsHost())
        return Host::RunShellCommand(command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
    else
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->RunShellCommand(command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
        else
            return Error("unable to run a remote command without a platform");
    }
}

Error
PlatformNetBSD::ResolveExecutable (const ModuleSpec &module_spec,
                                   lldb::ModuleSP &exe_module_sp,
                                   const FileSpecList *module_search_paths_ptr)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    char exe_path[PATH_MAX];
    ModuleSpec resolved_module_spec(module_spec);

    if (IsHost())
    {
        // If we have "ls" as the module_spec's file, resolve the executable location based on
        // the current path variables
        if (!resolved_module_spec.GetFileSpec().Exists())
        {
            module_spec.GetFileSpec().GetPath(exe_path, sizeof(exe_path));
            resolved_module_spec.GetFileSpec().SetFile(exe_path, true);
        }

        if (!resolved_module_spec.GetFileSpec().Exists())
            resolved_module_spec.GetFileSpec().ResolveExecutableLocation ();

        if (resolved_module_spec.GetFileSpec().Exists())
            error.Clear();
        else
        {
            error.SetErrorStringWithFormat("unable to find executable for '%s'", resolved_module_spec.GetFileSpec().GetPath().c_str());
        }
    }
    else
    {
        if (m_remote_platform_sp)
        {
            error = GetCachedExecutable (resolved_module_spec, exe_module_sp, module_search_paths_ptr, *m_remote_platform_sp);
        }
        else
        {
            // We may connect to a process and use the provided executable (Don't use local $PATH).

            // Resolve any executable within a bundle on MacOSX
            Host::ResolveExecutableInBundle (resolved_module_spec.GetFileSpec());

            if (resolved_module_spec.GetFileSpec().Exists())
            {
                error.Clear();
            }
            else
            {
                error.SetErrorStringWithFormat("the platform is not currently connected, and '%s' doesn't exist in the system root.", resolved_module_spec.GetFileSpec().GetPath().c_str());
            }
        }
    }

    if (error.Success())
    {
        if (resolved_module_spec.GetArchitecture().IsValid())
        {
            error = ModuleList::GetSharedModule (resolved_module_spec,
                                                 exe_module_sp,
                                                 module_search_paths_ptr,
                                                 NULL,
                                                 NULL);

            if (!exe_module_sp || exe_module_sp->GetObjectFile() == NULL)
            {
                exe_module_sp.reset();
                error.SetErrorStringWithFormat ("'%s' doesn't contain the architecture %s",
                                                resolved_module_spec.GetFileSpec().GetPath().c_str(),
                                                resolved_module_spec.GetArchitecture().GetArchitectureName());
            }
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            StreamString arch_names;
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, resolved_module_spec.GetArchitecture()); ++idx)
            {
                error = ModuleList::GetSharedModule (resolved_module_spec,
                                                     exe_module_sp,
                                                     module_search_paths_ptr,
                                                     NULL,
                                                     NULL);
                // Did we find an executable using one of the
                if (error.Success())
                {
                    if (exe_module_sp && exe_module_sp->GetObjectFile())
                        break;
                    else
                        error.SetErrorToGenericError();
                }

                if (idx > 0)
                    arch_names.PutCString (", ");
                arch_names.PutCString (resolved_module_spec.GetArchitecture().GetArchitectureName());
            }

            if (error.Fail() || !exe_module_sp)
            {
                if (resolved_module_spec.GetFileSpec().Readable())
                {
                    error.SetErrorStringWithFormat ("'%s' doesn't contain any '%s' platform architectures: %s",
                                                    resolved_module_spec.GetFileSpec().GetPath().c_str(),
                                                    GetPluginName().GetCString(),
                                                    arch_names.GetString().c_str());
                }
                else
                {
                    error.SetErrorStringWithFormat("'%s' is not readable", resolved_module_spec.GetFileSpec().GetPath().c_str());
                }
            }
        }
    }

    return error;
}

// From PlatformMacOSX only
Error
PlatformNetBSD::GetFileWithUUID (const FileSpec &platform_file,
                                 const UUID *uuid_ptr,
                                 FileSpec &local_file)
{
    if (IsRemote())
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetFileWithUUID (platform_file, uuid_ptr, local_file);
    }

    // Default to the local case
    local_file = platform_file;
    return Error();
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformNetBSD::PlatformNetBSD (bool is_host) :
    Platform(is_host),
    m_remote_platform_sp()
{
}

bool
PlatformNetBSD::GetRemoteOSVersion ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetOSVersion (m_major_os_version,
                                                   m_minor_os_version,
                                                   m_update_os_version);
    return false;
}

bool
PlatformNetBSD::GetRemoteOSBuildString (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSBuildString (s);
    s.clear();
    return false;
}

bool
PlatformNetBSD::GetRemoteOSKernelDescription (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSKernelDescription (s);
    s.clear();
    return false;
}

// Remote Platform subclasses need to override this function
ArchSpec
PlatformNetBSD::GetRemoteSystemArchitecture ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteSystemArchitecture ();
    return ArchSpec();
}


const char *
PlatformNetBSD::GetHostname ()
{
    if (IsHost())
        return Platform::GetHostname();

    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetHostname ();
    return NULL;
}

bool
PlatformNetBSD::IsConnected () const
{
    if (IsHost())
        return true;
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->IsConnected();
    return false;
}

Error
PlatformNetBSD::ConnectRemote (Args& args)
{
    Error error;
    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't connect to the host platform '%s', always connected", GetPluginName().GetCString());
    }
    else
    {
        if (!m_remote_platform_sp)
            m_remote_platform_sp = Platform::Create (ConstString("remote-gdb-server"), error);

        if (m_remote_platform_sp)
        {
            if (error.Success())
            {
                if (m_remote_platform_sp)
                {
                    error = m_remote_platform_sp->ConnectRemote (args);
                }
                else
                {
                    error.SetErrorString ("\"platform connect\" takes a single argument: <connect-url>");
                }
            }
        }
        else
            error.SetErrorString ("failed to create a 'remote-gdb-server' platform");

        if (error.Fail())
            m_remote_platform_sp.reset();
    }

    return error;
}

Error
PlatformNetBSD::DisconnectRemote ()
{
    Error error;

    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't disconnect from the host platform '%s', always connected", GetPluginName().GetCString());
    }
    else
    {
        if (m_remote_platform_sp)
            error = m_remote_platform_sp->DisconnectRemote ();
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

bool
PlatformNetBSD::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    bool success = false;
    if (IsHost())
    {
        success = Platform::GetProcessInfo (pid, process_info);
    }
    else if (m_remote_platform_sp)
    {
        success = m_remote_platform_sp->GetProcessInfo (pid, process_info);
    }
    return success;
}

uint32_t
PlatformNetBSD::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                               ProcessInstanceInfoList &process_infos)
{
    uint32_t match_count = 0;
    if (IsHost())
    {
        // Let the base class figure out the host details
        match_count = Platform::FindProcesses (match_info, process_infos);
    }
    else
    {
        // If we are remote, we can only return results if we are connected
        if (m_remote_platform_sp)
            match_count = m_remote_platform_sp->FindProcesses (match_info, process_infos);
    }
    return match_count;
}

const char *
PlatformNetBSD::GetUserName (uint32_t uid)
{
    // Check the cache in Platform in case we have already looked this uid up
    const char *user_name = Platform::GetUserName(uid);
    if (user_name)
        return user_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetUserName(uid);
    return NULL;
}

const char *
PlatformNetBSD::GetGroupName (uint32_t gid)
{
    const char *group_name = Platform::GetGroupName(gid);
    if (group_name)
        return group_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetGroupName(gid);
    return NULL;
}


Error
PlatformNetBSD::GetSharedModule (const ModuleSpec &module_spec,
                                 Process* process,
                                 ModuleSP &module_sp,
                                 const FileSpecList *module_search_paths_ptr,
                                 ModuleSP *old_module_sp_ptr,
                                 bool *did_create_ptr)
{
    Error error;
    module_sp.reset();

    if (IsRemote())
    {
        // If we have a remote platform always, let it try and locate
        // the shared module first.
        if (m_remote_platform_sp)
        {
            error = m_remote_platform_sp->GetSharedModule (module_spec,
                                                           process,
                                                           module_sp,
                                                           module_search_paths_ptr,
                                                           old_module_sp_ptr,
                                                           did_create_ptr);
        }
    }

    if (!module_sp)
    {
        // Fall back to the local platform and find the file locally
        error = Platform::GetSharedModule (module_spec,
                                           process,
                                           module_sp,
                                           module_search_paths_ptr,
                                           old_module_sp_ptr,
                                           did_create_ptr);
    }
    if (module_sp)
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
    return error;
}


bool
PlatformNetBSD::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (IsHost())
    {
        ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
        if (hostArch.GetTriple().isOSNetBSD())
        {
            if (idx == 0)
            {
                arch = hostArch;
                return arch.IsValid();
            }
            else if (idx == 1)
            {
                // If the default host architecture is 64-bit, look for a 32-bit variant
                if (hostArch.IsValid() && hostArch.GetTriple().isArch64Bit())
                {
                    arch = HostInfo::GetArchitecture(HostInfo::eArchKind32);
                    return arch.IsValid();
                }
            }
        }
    }
    else
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetSupportedArchitectureAtIndex(idx, arch);

        llvm::Triple triple;
        // Set the OS to NetBSD
        triple.setOS(llvm::Triple::NetBSD);
        // Set the architecture
        switch (idx)
        {
            case 0: triple.setArchName("x86_64"); break;
            case 1: triple.setArchName("i386"); break;
            default: return false;
        }
        // Leave the vendor as "llvm::Triple:UnknownVendor" and don't specify the vendor by
        // calling triple.SetVendorName("unknown") so that it is a "unspecified unknown".
        // This means when someone calls triple.GetVendorName() it will return an empty string
        // which indicates that the vendor can be set when two architectures are merged

        // Now set the triple into "arch" and return true
        arch.SetTriple(triple);
        return true;
    }
    return false;
}

void
PlatformNetBSD::GetStatus (Stream &strm)
{
#ifndef LLDB_DISABLE_POSIX
    struct ::utsname un;

    strm << "      Host: ";

    ::memset(&un, 0, sizeof(utsname));
    if (::uname(&un) == -1) {
        strm << "NetBSD" << '\n';
    } else {
        strm << un.sysname << ' ' << un.release;
        if (un.nodename[0] != '\0')
            strm << " (" << un.nodename << ')';
        strm << '\n';

        // Dump a common information about the platform status.
        strm << "Host: " << un.sysname << ' ' << un.release << ' ' << un.version << '\n';
    }
#endif

    Platform::GetStatus(strm);
}

size_t
PlatformNetBSD::GetSoftwareBreakpointTrapOpcode (Target &target, BreakpointSite *bp_site)
{
    ArchSpec arch = target.GetArchitecture();
    const uint8_t *trap_opcode = NULL;
    size_t trap_opcode_size = 0;

    switch (arch.GetMachine())
    {
    default:
        assert(false && "Unhandled architecture in PlatformNetBSD::GetSoftwareBreakpointTrapOpcode()");
        break;
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        {
            static const uint8_t g_i386_opcode[] = { 0xCC };
            trap_opcode = g_i386_opcode;
            trap_opcode_size = sizeof(g_i386_opcode);
        }
        break;
    }

    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
        return trap_opcode_size;
    return 0;
}


void
PlatformNetBSD::CalculateTrapHandlerSymbolNames ()
{
    m_trap_handlers.push_back (ConstString ("_sigtramp"));
}

Error
PlatformNetBSD::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    if (IsHost())
    {
        error = Platform::LaunchProcess (launch_info);
    }
    else
    {
        if (m_remote_platform_sp)
            error = m_remote_platform_sp->LaunchProcess (launch_info);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

lldb::ProcessSP
PlatformNetBSD::Attach(ProcessAttachInfo &attach_info,
                       Debugger &debugger,
                       Target *target,
                       Error &error)
{
    lldb::ProcessSP process_sp;
    if (IsHost())
    {
        if (target == NULL)
        {
            TargetSP new_target_sp;
            ArchSpec emptyArchSpec;

            error = debugger.GetTargetList().CreateTarget (debugger,
                                                           NULL,
                                                           emptyArchSpec,
                                                           false,
                                                           m_remote_platform_sp,
                                                           new_target_sp);
            target = new_target_sp.get();
        }
        else
            error.Clear();

        if (target && error.Success())
        {
            debugger.GetTargetList().SetSelectedTarget(target);
            // The netbsd always currently uses the GDB remote debugger plug-in
            // so even when debugging locally we are debugging remotely!
            // Just like the darwin plugin.
            process_sp = target->CreateProcess (attach_info.GetListenerForProcess(debugger), "gdb-remote", NULL);

            if (process_sp)
                error = process_sp->Attach (attach_info);
        }
    }
    else
    {
        if (m_remote_platform_sp)
            process_sp = m_remote_platform_sp->Attach (attach_info, debugger, target, error);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return process_sp;
}
