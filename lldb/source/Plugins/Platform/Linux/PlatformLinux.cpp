//===-- PlatformLinux.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformLinux.h"

// C Includes
#include <stdio.h>
#include <sys/utsname.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

Platform *
PlatformLinux::CreateInstance (bool force, const ArchSpec *arch)
{
    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getVendor())
        {
            case llvm::Triple::PC:
                create = true;
                break;
                
#if defined(__linux__)
            // Only accept "unknown" for the vendor if the host is linux and
            // it "unknown" wasn't specified (it was just returned becasue it
            // was NOT specified_
            case llvm::Triple::UnknownArch:
                create = !arch->TripleVendorWasSpecified();
                break;
#endif
            default:
                break;
        }
        
        if (create)
        {
            switch (triple.getOS())
            {
                case llvm::Triple::Linux:
                    break;
                    
#if defined(__linux__)
                // Only accept "unknown" for the OS if the host is linux and
                // it "unknown" wasn't specified (it was just returned becasue it
                // was NOT specified)
                case llvm::Triple::UnknownOS:
                    create = !arch->TripleOSWasSpecified();
                    break;
#endif
                default:
                    create = false;
                    break;
            }
        }
    }
    if (create)
        return new PlatformLinux(true);
    return NULL;
}

const char *
PlatformLinux::GetPluginNameStatic()
{
    return "plugin.platform.linux";
}

const char *
PlatformLinux::GetShortPluginNameStatic (bool is_host)
{
    if (is_host)
        return Platform::GetHostPlatformName ();
    else
        return "remote-linux";
}

const char *
PlatformLinux::GetPluginDescriptionStatic (bool is_host)
{
    if (is_host)
        return "Local Linux user platform plug-in.";
    else
        return "Remote Linux user platform plug-in.";
}

void
PlatformLinux::Initialize ()
{
    if (g_initialize_count++ == 0)
    {
#if defined(__linux__)
        PlatformSP default_platform_sp (new PlatformLinux(true));
        default_platform_sp->SetSystemArchitecture (Host::GetArchitecture());
        Platform::SetDefaultPlatform (default_platform_sp);
#endif
        PluginManager::RegisterPlugin(PlatformLinux::GetShortPluginNameStatic(false),
                                      PlatformLinux::GetPluginDescriptionStatic(false),
                                      PlatformLinux::CreateInstance);
    }
}

void
PlatformLinux::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformLinux::CreateInstance);
        }
    }
}

Error
PlatformLinux::ResolveExecutable (const FileSpec &exe_file,
                                  const ArchSpec &exe_arch,
                                  lldb::ModuleSP &exe_module_sp,
                                  const FileSpecList *module_search_paths_ptr)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    char exe_path[PATH_MAX];
    FileSpec resolved_exe_file (exe_file);
    
    if (IsHost())
    {
        // If we have "ls" as the exe_file, resolve the executable location based on
        // the current path variables
        if (!resolved_exe_file.Exists())
        {
            exe_file.GetPath(exe_path, sizeof(exe_path));
            resolved_exe_file.SetFile(exe_path, true);
        }

        if (!resolved_exe_file.Exists())
            resolved_exe_file.ResolveExecutableLocation ();

        if (resolved_exe_file.Exists())
            error.Clear();
        else
        {
            exe_file.GetPath(exe_path, sizeof(exe_path));
            error.SetErrorStringWithFormat("unable to find executable for '%s'", exe_path);
        }
    }
    else
    {
        if (m_remote_platform_sp)
        {
            error = m_remote_platform_sp->ResolveExecutable (exe_file,
                                                             exe_arch,
                                                             exe_module_sp,
                                                             NULL);
        }
        else
        {
            // We may connect to a process and use the provided executable (Don't use local $PATH).
            
            if (resolved_exe_file.Exists())
                error.Clear();
            else
                error.SetErrorStringWithFormat("the platform is not currently connected, and '%s' doesn't exist in the system root.", exe_path);
        }
    }

    if (error.Success())
    {
        ModuleSpec module_spec (resolved_exe_file, exe_arch);
        if (exe_arch.IsValid())
        {
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 NULL, 
                                                 NULL,
                                                 NULL);
        
            // TODO find out why exe_module_sp might be NULL            
            if (!exe_module_sp || exe_module_sp->GetObjectFile() == NULL)
            {
                exe_module_sp.reset();
                error.SetErrorStringWithFormat ("'%s%s%s' doesn't contain the architecture %s",
                                                exe_file.GetDirectory().AsCString(""),
                                                exe_file.GetDirectory() ? "/" : "",
                                                exe_file.GetFilename().AsCString(""),
                                                exe_arch.GetArchitectureName());
            }
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            StreamString arch_names;
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, module_spec.GetArchitecture()); ++idx)
            {
                error = ModuleList::GetSharedModule (module_spec, 
                                                     exe_module_sp, 
                                                     NULL, 
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
                arch_names.PutCString (module_spec.GetArchitecture().GetArchitectureName());
            }
            
            if (error.Fail() || !exe_module_sp)
            {
                error.SetErrorStringWithFormat ("'%s%s%s' doesn't contain any '%s' platform architectures: %s",
                                                exe_file.GetDirectory().AsCString(""),
                                                exe_file.GetDirectory() ? "/" : "",
                                                exe_file.GetFilename().AsCString(""),
                                                GetShortPluginName(),
                                                arch_names.GetString().c_str());
            }
        }
    }

    return error;
}

Error
PlatformLinux::GetFile (const FileSpec &platform_file, 
                        const UUID *uuid_ptr, FileSpec &local_file)
{
    if (IsRemote())
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetFile (platform_file, uuid_ptr, local_file);
    }

    // Default to the local case
    local_file = platform_file;
    return Error();
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformLinux::PlatformLinux (bool is_host) :
    Platform(is_host),  // This is the local host platform
    m_remote_platform_sp ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformLinux::~PlatformLinux()
{
}

bool
PlatformLinux::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    bool success = false;
    if (IsHost())
    {
        success = Platform::GetProcessInfo (pid, process_info);
    }
    else
    {
        if (m_remote_platform_sp) 
            success = m_remote_platform_sp->GetProcessInfo (pid, process_info);
    }
    return success;
}

bool
PlatformLinux::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (idx == 0)
    {
        arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        return arch.IsValid();
    }
    else if (idx == 1)
    {
        // If the default host architecture is 64-bit, look for a 32-bit variant
        ArchSpec hostArch
                      = Host::GetArchitecture(Host::eSystemDefaultArchitecture);
        if (hostArch.IsValid() && hostArch.GetTriple().isArch64Bit())
        {
            arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
            return arch.IsValid();
        }
    }
    return false;
}

void
PlatformLinux::GetStatus (Stream &strm)
{
    struct utsname un;

    if (uname(&un)) {
        strm << "Linux";
        return;
    }

    strm << un.sysname << ' ' << un.release << ' ' << un.version << '\n';
}

size_t
PlatformLinux::GetSoftwareBreakpointTrapOpcode (Target &target, 
                                                BreakpointSite *bp_site)
{
    ArchSpec arch = target.GetArchitecture();
    const uint8_t *trap_opcode = NULL;
    size_t trap_opcode_size = 0;

    switch (arch.GetCore())
    {
    default:
        assert(false && "CPU type not supported!");
        break;

    case ArchSpec::eCore_x86_32_i386:
    case ArchSpec::eCore_x86_64_x86_64:
        {
            static const uint8_t g_i386_breakpoint_opcode[] = { 0xCC };
            trap_opcode = g_i386_breakpoint_opcode;
            trap_opcode_size = sizeof(g_i386_breakpoint_opcode);
        }
        break;
    }

    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
        return trap_opcode_size;
    return 0;
}

Error
PlatformLinux::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    
    if (IsHost())
    {
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
        error = Platform::LaunchProcess (launch_info);
    }
    else
    {
        error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

lldb::ProcessSP
PlatformLinux::Attach(ProcessAttachInfo &attach_info,
                      Debugger &debugger,
                      Target *target,
                      Listener &listener,
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

            process_sp = target->CreateProcess (listener,
                                                attach_info.GetProcessPluginName(),
                                                NULL);

            if (process_sp)
                error = process_sp->Attach (attach_info);
        }
    }
    else
    {
        if (m_remote_platform_sp)
            process_sp = m_remote_platform_sp->Attach (attach_info, debugger, target, listener, error);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return process_sp;
}
