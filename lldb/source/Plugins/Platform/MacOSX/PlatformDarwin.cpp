//===-- PlatformDarwin.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwin.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
    

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformDarwin::PlatformDarwin (bool is_host) :
    Platform(is_host),  // This is the local host platform
    m_remote_platform_sp (),
    m_developer_directory ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformDarwin::~PlatformDarwin()
{
}

FileSpec
PlatformDarwin::LocateExecutableScriptingResource (const ModuleSpec &module_spec)
{
    const FileSpec *exec_fspec = module_spec.GetFileSpecPtr();
    const ArchSpec *arch = module_spec.GetArchitecturePtr();
    const UUID *uuid = module_spec.GetUUIDPtr();
    
    const char* module_directory = exec_fspec->GetDirectory().GetCString();
    const char* module_basename = exec_fspec->GetFileNameStrippingExtension().GetCString();
    
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "LocateExecutableScriptingResource (file = %s, arch = %s, uuid = %p)",
                        exec_fspec ? exec_fspec->GetFilename().AsCString ("<NULL>") : "<NULL>",
                        arch ? arch->GetArchitectureName() : "<NULL>",
                        uuid);
    
    
    FileSpec symbol_fspec (Symbols::LocateExecutableSymbolFile(module_spec));
    
    FileSpec script_fspec;
    
    StreamString path_string;
    
    if (symbol_fspec && symbol_fspec.Exists())
    {
        // for OSX we are going to be in .dSYM/Contents/Resources/DWARF/<basename>
        // let us go to .dSYM/Contents/Resources/Python/<basename>.py and see if the file exists
        path_string.Printf("%s/../Python/%s.py",symbol_fspec.GetDirectory().GetCString(),module_basename);
        script_fspec.SetFile(path_string.GetData(), true);
        if (!script_fspec.Exists())
            script_fspec.Clear();
    }
    
    // no symbols or symbols did not have a scripting resource
    if (!symbol_fspec || !script_fspec)
    {
        path_string.Clear();
        path_string.Printf("%s.framework",module_basename);
        if (module_directory && strstr(module_directory, path_string.GetData()))
        {
            // we are going to be in foo.framework/Versions/X/foo
            path_string.Clear();
            // let's go to foo.framework/Versions/X/Resources/Python/foo.py
            path_string.Printf("%s/Resources/Python/%s.py",module_directory,module_basename);
            script_fspec.SetFile(path_string.GetData(), true);
            if (!script_fspec.Exists())
                script_fspec.Clear();
        }
    }
    
    return script_fspec;
}

Error
PlatformDarwin::ResolveExecutable (const FileSpec &exe_file,
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
        // If we have "ls" as the exe_file, resolve the executable loation based on
        // the current path variables
        if (!resolved_exe_file.Exists())
        {
            exe_file.GetPath (exe_path, sizeof(exe_path));
            resolved_exe_file.SetFile(exe_path, true);
        }

        if (!resolved_exe_file.Exists())
            resolved_exe_file.ResolveExecutableLocation ();

        // Resolve any executable within a bundle on MacOSX
        Host::ResolveExecutableInBundle (resolved_exe_file);
        
        if (resolved_exe_file.Exists())
            error.Clear();
        else
        {
            exe_file.GetPath (exe_path, sizeof(exe_path));
            error.SetErrorStringWithFormat ("unable to find executable for '%s'", exe_path);
        }
    }
    else
    {
        if (m_remote_platform_sp)
        {
            error = m_remote_platform_sp->ResolveExecutable (exe_file, 
                                                             exe_arch,
                                                             exe_module_sp,
                                                             module_search_paths_ptr);
        }
        else
        {
            // We may connect to a process and use the provided executable (Don't use local $PATH).

            // Resolve any executable within a bundle on MacOSX
            Host::ResolveExecutableInBundle (resolved_exe_file);

            if (resolved_exe_file.Exists())
                error.Clear();
            else
                error.SetErrorStringWithFormat("the platform is not currently connected, and '%s' doesn't exist in the system root.", resolved_exe_file.GetFilename().AsCString(""));
        }
    }
    

    if (error.Success())
    {
        ModuleSpec module_spec (resolved_exe_file, exe_arch);
        if (module_spec.GetArchitecture().IsValid())
        {
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 module_search_paths_ptr,
                                                 NULL, 
                                                 NULL);
        
            if (error.Fail() || exe_module_sp.get() == NULL || exe_module_sp->GetObjectFile() == NULL)
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
PlatformDarwin::ResolveSymbolFile (Target &target,
                                   const ModuleSpec &sym_spec,
                                   FileSpec &sym_file)
{
    Error error;
    sym_file = sym_spec.GetSymbolFileSpec();
    if (sym_file.Exists())
    {
        if (sym_file.GetFileType() == FileSpec::eFileTypeDirectory)
        {
            sym_file = Symbols::FindSymbolFileInBundle (sym_file,
                                                        sym_spec.GetUUIDPtr(),
                                                        sym_spec.GetArchitecturePtr());
        }
    }
    else
    {
        if (sym_spec.GetUUID().IsValid())
        {
            
        }
    }
    return error;
    
}



Error
PlatformDarwin::GetSharedModule (const ModuleSpec &module_spec,
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
                                           module_sp,
                                           module_search_paths_ptr,
                                           old_module_sp_ptr,
                                           did_create_ptr);
        
        const FileSpec &platform_file = module_spec.GetFileSpec();
        if (!module_sp && module_search_paths_ptr && platform_file)
        {
            // We can try to pull off part of the file path up to the bundle
            // directory level and try any module search paths...
            FileSpec bundle_directory;
            if (Host::GetBundleDirectory (platform_file, bundle_directory))
            {
                char platform_path[PATH_MAX];
                char bundle_dir[PATH_MAX];
                platform_file.GetPath (platform_path, sizeof(platform_path));
                const size_t bundle_directory_len = bundle_directory.GetPath (bundle_dir, sizeof(bundle_dir));
                char new_path[PATH_MAX];
                size_t num_module_search_paths = module_search_paths_ptr->GetSize();
                for (size_t i=0; i<num_module_search_paths; ++i)
                {
                    const size_t search_path_len = module_search_paths_ptr->GetFileSpecAtIndex(i).GetPath(new_path, sizeof(new_path));
                    if (search_path_len < sizeof(new_path))
                    {
                        snprintf (new_path + search_path_len, sizeof(new_path) - search_path_len, "/%s", platform_path + bundle_directory_len);
                        FileSpec new_file_spec (new_path, false);
                        if (new_file_spec.Exists())
                        {
                            ModuleSpec new_module_spec (module_spec);
                            new_module_spec.GetFileSpec() = new_file_spec;
                            Error new_error (Platform::GetSharedModule (new_module_spec,
                                                                        module_sp,
                                                                        NULL,
                                                                        old_module_sp_ptr,
                                                                        did_create_ptr));
                            
                            if (module_sp)
                            {
                                module_sp->SetPlatformFileSpec(new_file_spec);
                                return new_error;
                            }
                        }
                    }
                }
            }
        }
    }
    if (module_sp)
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
    return error;
}

size_t
PlatformDarwin::GetSoftwareBreakpointTrapOpcode (Target &target, BreakpointSite *bp_site)
{
    const uint8_t *trap_opcode = NULL;
    uint32_t trap_opcode_size = 0;
    bool bp_is_thumb = false;
        
    llvm::Triple::ArchType machine = target.GetArchitecture().GetMachine();
    switch (machine)
    {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        {
            static const uint8_t g_i386_breakpoint_opcode[] = { 0xCC };
            trap_opcode = g_i386_breakpoint_opcode;
            trap_opcode_size = sizeof(g_i386_breakpoint_opcode);
        }
        break;

    case llvm::Triple::thumb:
        bp_is_thumb = true; // Fall through...
    case llvm::Triple::arm:
        {
            static const uint8_t g_arm_breakpoint_opcode[] = { 0xFE, 0xDE, 0xFF, 0xE7 };
            static const uint8_t g_thumb_breakpooint_opcode[] = { 0xFE, 0xDE };

            // Auto detect arm/thumb if it wasn't explicitly specified
            if (!bp_is_thumb)
            {
                lldb::BreakpointLocationSP bp_loc_sp (bp_site->GetOwnerAtIndex (0));
                if (bp_loc_sp)
                    bp_is_thumb = bp_loc_sp->GetAddress().GetAddressClass () == eAddressClassCodeAlternateISA;
            }
            if (bp_is_thumb)
            {
                trap_opcode = g_thumb_breakpooint_opcode;
                trap_opcode_size = sizeof(g_thumb_breakpooint_opcode);
                break;
            }
            trap_opcode = g_arm_breakpoint_opcode;
            trap_opcode_size = sizeof(g_arm_breakpoint_opcode);
        }
        break;
        
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
        {
            static const uint8_t g_ppc_breakpoint_opcode[] = { 0x7F, 0xC0, 0x00, 0x08 };
            trap_opcode = g_ppc_breakpoint_opcode;
            trap_opcode_size = sizeof(g_ppc_breakpoint_opcode);
        }
        break;
        
    default:
        assert(!"Unhandled architecture in PlatformDarwin::GetSoftwareBreakpointTrapOpcode()");
        break;
    }
    
    if (trap_opcode && trap_opcode_size)
    {
        if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
            return trap_opcode_size;
    }
    return 0;

}

bool
PlatformDarwin::GetRemoteOSVersion ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetOSVersion (m_major_os_version, 
                                                   m_minor_os_version, 
                                                   m_update_os_version);
    return false;
}

bool
PlatformDarwin::GetRemoteOSBuildString (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSBuildString (s);
    s.clear();
    return false;
}

bool
PlatformDarwin::GetRemoteOSKernelDescription (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSKernelDescription (s);
    s.clear();
    return false;
}

// Remote Platform subclasses need to override this function
ArchSpec
PlatformDarwin::GetRemoteSystemArchitecture ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteSystemArchitecture ();
    return ArchSpec();
}


const char *
PlatformDarwin::GetHostname ()
{
    if (IsHost())
        return Platform::GetHostname();

    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetHostname ();
    return NULL;
}

bool
PlatformDarwin::IsConnected () const
{
    if (IsHost())
        return true;
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->IsConnected();
    return false;
}

Error
PlatformDarwin::ConnectRemote (Args& args)
{
    Error error;
    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't connect to the host platform '%s', always connected", GetShortPluginName());
    }
    else
    {
        if (!m_remote_platform_sp)
            m_remote_platform_sp = Platform::Create ("remote-gdb-server", error);

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
PlatformDarwin::DisconnectRemote ()
{
    Error error;
    
    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't disconnect from the host platform '%s', always connected", GetShortPluginName());
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
PlatformDarwin::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    bool sucess = false;
    if (IsHost())
    {
        sucess = Platform::GetProcessInfo (pid, process_info);
    }
    else
    {
        if (m_remote_platform_sp)
            sucess = m_remote_platform_sp->GetProcessInfo (pid, process_info);
    }
    return sucess;
}



uint32_t
PlatformDarwin::FindProcesses (const ProcessInstanceInfoMatch &match_info,
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

Error
PlatformDarwin::LaunchProcess (ProcessLaunchInfo &launch_info)
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
PlatformDarwin::Attach (ProcessAttachInfo &attach_info,
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
            
            error = debugger.GetTargetList().CreateTarget (debugger,
                                                           NULL,
                                                           NULL, 
                                                           false,
                                                           NULL,
                                                           new_target_sp);
            target = new_target_sp.get();
        }
        else
            error.Clear();
    
        if (target && error.Success())
        {
            debugger.GetTargetList().SetSelectedTarget(target);

            process_sp = target->CreateProcess (listener, attach_info.GetProcessPluginName(), NULL);
            
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

const char *
PlatformDarwin::GetUserName (uint32_t uid)
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
PlatformDarwin::GetGroupName (uint32_t gid)
{
    const char *group_name = Platform::GetGroupName(gid);
    if (group_name)
        return group_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetGroupName(gid);
    return NULL;
}

bool
PlatformDarwin::ModuleIsExcludedForNonModuleSpecificSearches (lldb_private::Target &target, const lldb::ModuleSP &module_sp)
{
    if (!module_sp)
        return false;
        
    ObjectFile *obj_file = module_sp->GetObjectFile();
    if (!obj_file)
        return false;
    
    ObjectFile::Type obj_type = obj_file->GetType();
    if (obj_type == ObjectFile::eTypeDynamicLinker)
        return true;
    else
        return false;
}


bool
PlatformDarwin::x86GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (idx == 0)
    {
        arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        return arch.IsValid();
    }
    else if (idx == 1)
    {
        ArchSpec platform_arch (Host::GetArchitecture (Host::eSystemDefaultArchitecture));
        ArchSpec platform_arch64 (Host::GetArchitecture (Host::eSystemDefaultArchitecture64));
        if (platform_arch == platform_arch64)
        {
            // This macosx platform supports both 32 and 64 bit. Since we already
            // returned the 64 bit arch for idx == 0, return the 32 bit arch 
            // for idx == 1
            arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
            return arch.IsValid();
        }
    }
    return false;
}

// The architecture selection rules for arm processors
// These cpu subtypes have distinct names (e.g. armv7f) but armv7 binaries run fine on an armv7f processor.

bool
PlatformDarwin::ARMGetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    ArchSpec system_arch (GetSystemArchitecture());
    const ArchSpec::Core system_core = system_arch.GetCore();
    switch (system_core)
    {
    default:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7-apple-ios");    return true;
            case  1: arch.SetTriple ("armv7f-apple-ios");   return true;
            case  2: arch.SetTriple ("armv7k-apple-ios");   return true;
            case  3: arch.SetTriple ("armv7s-apple-ios");   return true;
            case  4: arch.SetTriple ("armv6-apple-ios");    return true;
            case  5: arch.SetTriple ("armv5-apple-ios");    return true;
            case  6: arch.SetTriple ("armv4-apple-ios");    return true;
            case  7: arch.SetTriple ("arm-apple-ios");      return true;
            case  8: arch.SetTriple ("thumbv7-apple-ios");  return true;
            case  9: arch.SetTriple ("thumbv7f-apple-ios"); return true;
            case 10: arch.SetTriple ("thumbv7k-apple-ios"); return true;
            case 11: arch.SetTriple ("thumbv7s-apple-ios"); return true;
            case 12: arch.SetTriple ("thumbv6-apple-ios");  return true;
            case 13: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 14: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 15: arch.SetTriple ("thumb-apple-ios");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7f:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7f-apple-ios");   return true;
            case  1: arch.SetTriple ("armv7-apple-ios");    return true;
            case  2: arch.SetTriple ("armv6-apple-ios");    return true;
            case  3: arch.SetTriple ("armv5-apple-ios");    return true;
            case  4: arch.SetTriple ("armv4-apple-ios");    return true;
            case  5: arch.SetTriple ("arm-apple-ios");      return true;
            case  6: arch.SetTriple ("thumbv7f-apple-ios"); return true;
            case  7: arch.SetTriple ("thumbv7-apple-ios");  return true;
            case  8: arch.SetTriple ("thumbv6-apple-ios");  return true;
            case  9: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 10: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 11: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7k:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7k-apple-ios");   return true;
            case  1: arch.SetTriple ("armv7-apple-ios");    return true;
            case  2: arch.SetTriple ("armv6-apple-ios");    return true;
            case  3: arch.SetTriple ("armv5-apple-ios");    return true;
            case  4: arch.SetTriple ("armv4-apple-ios");    return true;
            case  5: arch.SetTriple ("arm-apple-ios");      return true;
            case  6: arch.SetTriple ("thumbv7k-apple-ios"); return true;
            case  7: arch.SetTriple ("thumbv7-apple-ios");  return true;
            case  8: arch.SetTriple ("thumbv6-apple-ios");  return true;
            case  9: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 10: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 11: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7s:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7s-apple-ios");   return true;
            case  1: arch.SetTriple ("armv7-apple-ios");    return true;
            case  2: arch.SetTriple ("armv6-apple-ios");    return true;
            case  3: arch.SetTriple ("armv5-apple-ios");    return true;
            case  4: arch.SetTriple ("armv4-apple-ios");    return true;
            case  5: arch.SetTriple ("arm-apple-ios");      return true;
            case  6: arch.SetTriple ("thumbv7s-apple-ios"); return true;
            case  7: arch.SetTriple ("thumbv7-apple-ios");  return true;
            case  8: arch.SetTriple ("thumbv6-apple-ios");  return true;
            case  9: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 10: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 11: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7:
        switch (idx)
        {
            case 0: arch.SetTriple ("armv7-apple-ios");    return true;
            case 1: arch.SetTriple ("armv6-apple-ios");    return true;
            case 2: arch.SetTriple ("armv5-apple-ios");    return true;
            case 3: arch.SetTriple ("armv4-apple-ios");    return true;
            case 4: arch.SetTriple ("arm-apple-ios");      return true;
            case 5: arch.SetTriple ("thumbv7-apple-ios");  return true;
            case 6: arch.SetTriple ("thumbv6-apple-ios");  return true;
            case 7: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 8: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 9: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv6:
        switch (idx)
        {
            case 0: arch.SetTriple ("armv6-apple-ios");    return true;
            case 1: arch.SetTriple ("armv5-apple-ios");    return true;
            case 2: arch.SetTriple ("armv4-apple-ios");    return true;
            case 3: arch.SetTriple ("arm-apple-ios");      return true;
            case 4: arch.SetTriple ("thumbv6-apple-ios");  return true;
            case 5: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 6: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 7: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv5:
        switch (idx)
        {
            case 0: arch.SetTriple ("armv5-apple-ios");    return true;
            case 1: arch.SetTriple ("armv4-apple-ios");    return true;
            case 2: arch.SetTriple ("arm-apple-ios");      return true;
            case 3: arch.SetTriple ("thumbv5-apple-ios");  return true;
            case 4: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 5: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv4:
        switch (idx)
        {
            case 0: arch.SetTriple ("armv4-apple-ios");    return true;
            case 1: arch.SetTriple ("arm-apple-ios");      return true;
            case 2: arch.SetTriple ("thumbv4t-apple-ios"); return true;
            case 3: arch.SetTriple ("thumb-apple-ios");    return true;
            default: break;
        }
        break;
    }
    arch.Clear();
    return false;
}


const char *
PlatformDarwin::GetDeveloperDirectory()
{
    if (m_developer_directory.empty())
    {
        bool developer_dir_path_valid = false;
        char developer_dir_path[PATH_MAX];
        FileSpec temp_file_spec;
        if (Host::GetLLDBPath (ePathTypeLLDBShlibDir, temp_file_spec))
        {
            if (temp_file_spec.GetPath (developer_dir_path, sizeof(developer_dir_path)))
            {
                char *shared_frameworks = strstr (developer_dir_path, "/SharedFrameworks/LLDB.framework");
                if (shared_frameworks)
                {
                    ::snprintf (shared_frameworks, 
                                sizeof(developer_dir_path) - (shared_frameworks - developer_dir_path),
                                "/Developer");
                    developer_dir_path_valid = true;
                }
                else
                {
                    char *lib_priv_frameworks = strstr (developer_dir_path, "/Library/PrivateFrameworks/LLDB.framework");
                    if (lib_priv_frameworks)
                    {
                        *lib_priv_frameworks = '\0';
                        developer_dir_path_valid = true;
                    }
                }
            }
        }
        
        if (!developer_dir_path_valid)
        {
            std::string xcode_dir_path;
            const char *xcode_select_prefix_dir = getenv ("XCODE_SELECT_PREFIX_DIR");
            if (xcode_select_prefix_dir)
                xcode_dir_path.append (xcode_select_prefix_dir);
            xcode_dir_path.append ("/usr/share/xcode-select/xcode_dir_path");
            temp_file_spec.SetFile(xcode_dir_path.c_str(), false);
            size_t bytes_read = temp_file_spec.ReadFileContents(0, developer_dir_path, sizeof(developer_dir_path), NULL);
            if (bytes_read > 0)
            {
                developer_dir_path[bytes_read] = '\0';
                while (developer_dir_path[bytes_read-1] == '\r' ||
                       developer_dir_path[bytes_read-1] == '\n')
                    developer_dir_path[--bytes_read] = '\0';
                developer_dir_path_valid = true;
            }
        }
        
        if (developer_dir_path_valid)
        {
            temp_file_spec.SetFile (developer_dir_path, false);
            if (temp_file_spec.Exists())
            {
                m_developer_directory.assign (developer_dir_path);
                return m_developer_directory.c_str();
            }
        }
        // Assign a single NULL character so we know we tried to find the device
        // support directory and we don't keep trying to find it over and over.
        m_developer_directory.assign (1, '\0');
    }
    
    // We should have put a single NULL character into m_developer_directory
    // or it should have a valid path if the code gets here
    assert (m_developer_directory.empty() == false);
    if (m_developer_directory[0])
        return m_developer_directory.c_str();
    return NULL;
}


BreakpointSP
PlatformDarwin::SetThreadCreationBreakpoint (Target &target)
{
    BreakpointSP bp_sp;
    static const char *g_bp_names[] =
    {
        "start_wqthread",
        "_pthread_wqthread",
        "_pthread_start",
    };

    static const char *g_bp_modules[] =
    {
        "libsystem_c.dylib",
        "libSystem.B.dylib"
    };

    FileSpecList bp_modules;
    for (int i = 0; i < sizeof(g_bp_modules)/sizeof(const char *); i++)
    {
        const char *bp_module = g_bp_modules[i];
        bp_modules.Append(FileSpec(bp_module, false));
    }

    bool internal = true;
    LazyBool skip_prologue = eLazyBoolNo;
    bp_sp = target.CreateBreakpoint (&bp_modules,
                                     NULL,
                                     g_bp_names,
                                     sizeof(g_bp_names)/sizeof(const char *),
                                     eFunctionNameTypeFull,
                                     skip_prologue,
                                     internal);

    return bp_sp;
}

size_t
PlatformDarwin::GetEnvironment (StringList &env)
{
    if (IsRemote())
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetEnvironment(env);
        return 0;
    }
    return Host::GetEnvironment(env);
}
