//===-- PlatformiOSSimulator.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformiOSSimulator.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------
// Static Variables
//------------------------------------------------------------------
static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
void
PlatformiOSSimulator::Initialize ()
{
    if (g_initialize_count++ == 0)
    {
        PluginManager::RegisterPlugin (PlatformiOSSimulator::GetShortPluginNameStatic(),
                                       PlatformiOSSimulator::GetDescriptionStatic(),
                                       PlatformiOSSimulator::CreateInstance);
    }
}

void
PlatformiOSSimulator::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformiOSSimulator::CreateInstance);
        }
    }
}

Platform* 
PlatformiOSSimulator::CreateInstance (bool force, const ArchSpec *arch)
{
    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        switch (arch->GetMachine())
        {
        // Currently simulator is i386 only...
        case llvm::Triple::x86:
            {
                const llvm::Triple &triple = arch->GetTriple();
                switch (triple.getVendor())
                {
                    case llvm::Triple::Apple:
                        create = true;
                        break;
                        
#if defined(__APPLE__)
                    // Only accept "unknown" for the vendor if the host is Apple and
                    // it "unknown" wasn't specified (it was just returned becasue it
                    // was NOT specified)
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
                        case llvm::Triple::Darwin:  // Deprecated, but still support Darwin for historical reasons
                        case llvm::Triple::MacOSX:
                        case llvm::Triple::IOS:     // IOS is not used for simulator triples, but accept it just in case
                            break;
                            
#if defined(__APPLE__)
                        // Only accept "unknown" for the OS if the host is Apple and
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
            break;
        default:
            break;
        }
    }
    if (create)
        return new PlatformiOSSimulator ();
    return NULL;
}


const char *
PlatformiOSSimulator::GetPluginNameStatic ()
{
    return "PlatformiOSSimulator";
}

const char *
PlatformiOSSimulator::GetShortPluginNameStatic()
{
    return "ios-simulator";
}

const char *
PlatformiOSSimulator::GetDescriptionStatic()
{
    return "iOS simulator platform plug-in.";
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformiOSSimulator::PlatformiOSSimulator () :
    PlatformDarwin (true),
    m_sdk_directory ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformiOSSimulator::~PlatformiOSSimulator()
{
}


void
PlatformiOSSimulator::GetStatus (Stream &strm)
{
    Platform::GetStatus (strm);
    const char *sdk_directory = GetSDKDirectory();
    if (sdk_directory)
        strm.Printf ("  SDK Path: \"%s\"\n", sdk_directory);
    else
        strm.PutCString ("  SDK Path: error: unable to locate SDK\n");
}


Error
PlatformiOSSimulator::ResolveExecutable (const FileSpec &exe_file,
                                         const ArchSpec &exe_arch,
                                         lldb::ModuleSP &exe_module_sp,
                                         const FileSpecList *module_search_paths_ptr)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    FileSpec resolved_exe_file (exe_file);
    
    // If we have "ls" as the exe_file, resolve the executable loation based on
    // the current path variables
    // TODO: resolve bare executables in the Platform SDK
//    if (!resolved_exe_file.Exists())
//        resolved_exe_file.ResolveExecutableLocation ();

    // Resolve any executable within a bundle on MacOSX
    // TODO: verify that this handles shallow bundles, if not then implement one ourselves
    Host::ResolveExecutableInBundle (resolved_exe_file);

    if (resolved_exe_file.Exists())
    {
        ModuleSpec module_spec(resolved_exe_file, exe_arch);
        if (exe_arch.IsValid())
        {
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 NULL,
                                                 NULL, 
                                                 NULL);
        
            if (exe_module_sp && exe_module_sp->GetObjectFile())
                return error;
            exe_module_sp.reset();
        }
        // No valid architecture was specified or the exact ARM slice wasn't
        // found so ask the platform for the architectures that we should be
        // using (in the correct order) and see if we can find a match that way
        StreamString arch_names;
        ArchSpec platform_arch;
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
            arch_names.PutCString (platform_arch.GetArchitectureName());
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
    else
    {
        error.SetErrorStringWithFormat ("'%s%s%s' does not exist",
                                        exe_file.GetDirectory().AsCString(""),
                                        exe_file.GetDirectory() ? "/" : "",
                                        exe_file.GetFilename().AsCString(""));
    }

    return error;
}

static FileSpec::EnumerateDirectoryResult 
EnumerateDirectoryCallback (void *baton, FileSpec::FileType file_type, const FileSpec &file_spec)
{
    if (file_type == FileSpec::eFileTypeDirectory)
    {
        const char *filename = file_spec.GetFilename().GetCString();
        if (filename && strncmp(filename, "iPhoneSimulator", strlen ("iPhoneSimulator")) == 0)
        {
            ::snprintf ((char *)baton, PATH_MAX, "%s", filename);
            return FileSpec::eEnumerateDirectoryResultQuit;
        }
    }
    return FileSpec::eEnumerateDirectoryResultNext;
}



const char *
PlatformiOSSimulator::GetSDKDirectory()
{
    if (m_sdk_directory.empty())
    {
        const char *developer_dir = GetDeveloperDirectory();
        if (developer_dir)
        {
            char sdks_directory[PATH_MAX];
            char sdk_dirname[PATH_MAX];
            sdk_dirname[0] = '\0';
            snprintf (sdks_directory, 
                      sizeof(sdks_directory), 
                      "%s/Platforms/iPhoneSimulator.platform/Developer/SDKs",
                      developer_dir);
            FileSpec simulator_sdk_spec;
            bool find_directories = true;
            bool find_files = false;
            bool find_other = false;
            FileSpec::EnumerateDirectory (sdks_directory,
                                          find_directories,
                                          find_files,
                                          find_other,
                                          EnumerateDirectoryCallback,
                                          sdk_dirname);
            
            if (sdk_dirname[0])
            {
                m_sdk_directory = sdks_directory;
                m_sdk_directory.append (1, '/');
                m_sdk_directory.append (sdk_dirname);
                return m_sdk_directory.c_str();
            }
        }
        // Assign a single NULL character so we know we tried to find the device
        // support directory and we don't keep trying to find it over and over.
        m_sdk_directory.assign (1, '\0');
    }

    // We should have put a single NULL character into m_sdk_directory
    // or it should have a valid path if the code gets here
    assert (m_sdk_directory.empty() == false);
    if (m_sdk_directory[0])
        return m_sdk_directory.c_str();
    return NULL;
}

Error
PlatformiOSSimulator::GetFile (const FileSpec &platform_file, 
                               const UUID *uuid_ptr,
                               FileSpec &local_file)
{
    Error error;
    char platform_file_path[PATH_MAX];
    if (platform_file.GetPath(platform_file_path, sizeof(platform_file_path)))
    {
        char resolved_path[PATH_MAX];
    
        const char * sdk_dir = GetSDKDirectory();
        if (sdk_dir)
        {
            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/%s", 
                        sdk_dir, 
                        platform_file_path);
            
            // First try in the SDK and see if the file is in there
            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;

            // Else fall back to the actual path itself
            local_file.SetFile(platform_file_path, true);
            if (local_file.Exists())
                return error;

        }
        error.SetErrorStringWithFormat ("unable to locate a platform file for '%s' in platform '%s'", 
                                        platform_file_path,
                                        GetPluginName());
    }
    else
    {
        error.SetErrorString ("invalid platform file argument");
    }
    return error;
}

Error
PlatformiOSSimulator::GetSharedModule (const ModuleSpec &module_spec,
                                       ModuleSP &module_sp,
                                       const FileSpecList *module_search_paths_ptr,
                                       ModuleSP *old_module_sp_ptr,
                                       bool *did_create_ptr)
{
    // For iOS, the SDK files are all cached locally on the host
    // system. So first we ask for the file in the cached SDK,
    // then we attempt to get a shared module for the right architecture
    // with the right UUID.
    Error error;
    FileSpec local_file;
    const FileSpec &platform_file = module_spec.GetFileSpec();
    error = GetFile (platform_file, module_spec.GetUUIDPtr(), local_file);
    if (error.Success())
    {
        error = ResolveExecutable (local_file, module_spec.GetArchitecture(), module_sp, module_search_paths_ptr);
    }
    else
    {
        const bool always_create = false;
        error = ModuleList::GetSharedModule (module_spec, 
                                             module_sp,
                                             module_search_paths_ptr,
                                             old_module_sp_ptr,
                                             did_create_ptr,
                                             always_create);

    }
    if (module_sp)
        module_sp->SetPlatformFileSpec(platform_file);

    return error;
}


uint32_t
PlatformiOSSimulator::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                                     ProcessInstanceInfoList &process_infos)
{
    // TODO: if connected, send a packet to get the remote process infos by name
    process_infos.Clear();
    return 0;
}

bool
PlatformiOSSimulator::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (idx == 0)
    {
        // All iOS simulator binaries are currently i386
        arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
        return arch.IsValid();
    }
    return false;
}
