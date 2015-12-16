//===-- PlatformRemoteAppleWatch.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "PlatformRemoteAppleWatch.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
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
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteAppleWatch::PlatformRemoteAppleWatch () :
    PlatformDarwin (false),    // This is a remote platform
    m_sdk_directory_infos(),
    m_device_support_directory(),
    m_device_support_directory_for_os_version (),
    m_build_update(),
    m_last_module_sdk_idx (UINT32_MAX),
    m_connected_module_sdk_idx (UINT32_MAX)
{
}

PlatformRemoteAppleWatch::SDKDirectoryInfo::SDKDirectoryInfo (const lldb_private::FileSpec &sdk_dir) :
    directory(sdk_dir),
    build(),
    version_major(0),
    version_minor(0),
    version_update(0),
    user_cached(false)
{
    const char *dirname_cstr = sdk_dir.GetFilename().GetCString();
    const char *pos = Args::StringToVersion (dirname_cstr,
                                             version_major,
                                             version_minor,
                                             version_update);
    
    if (pos && pos[0] == ' ' && pos[1] == '(')
    {
        const char *build_start = pos + 2;
        const char *end_paren = strchr (build_start, ')');
        if (end_paren && build_start < end_paren)
            build.SetCStringWithLength(build_start, end_paren - build_start);
    }
}

//------------------------------------------------------------------
// Static Variables
//------------------------------------------------------------------
static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
void
PlatformRemoteAppleWatch::Initialize ()
{
    PlatformDarwin::Initialize ();

    if (g_initialize_count++ == 0)
    {
        PluginManager::RegisterPlugin (PlatformRemoteAppleWatch::GetPluginNameStatic(),
                                       PlatformRemoteAppleWatch::GetDescriptionStatic(),
                                       PlatformRemoteAppleWatch::CreateInstance);
    }
}

void
PlatformRemoteAppleWatch::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformRemoteAppleWatch::CreateInstance);
        }
    }

    PlatformDarwin::Terminate ();
}

PlatformSP
PlatformRemoteAppleWatch::CreateInstance (bool force, const ArchSpec *arch)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_PLATFORM));
    if (log)
    {
        const char *arch_name;
        if (arch && arch->GetArchitectureName ())
            arch_name = arch->GetArchitectureName ();
        else
            arch_name = "<null>";

        const char *triple_cstr = arch ? arch->GetTriple ().getTriple ().c_str() : "<null>";

        log->Printf ("PlatformRemoteAppleWatch::%s(force=%s, arch={%s,%s})", __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
    }

    bool create = force;
    if (!create && arch && arch->IsValid())
    {
        switch (arch->GetMachine())
        {
        case llvm::Triple::arm:
        case llvm::Triple::aarch64:
        case llvm::Triple::thumb:
            {
                const llvm::Triple &triple = arch->GetTriple();
                llvm::Triple::VendorType vendor = triple.getVendor();
                switch (vendor)
                {
                    case llvm::Triple::Apple:
                        create = true;
                        break;

#if defined(__APPLE__)
                    // Only accept "unknown" for the vendor if the host is Apple and
                    // it "unknown" wasn't specified (it was just returned because it
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
                        case llvm::Triple::WatchOS:     // This is the right triple value for Apple Watch debugging
                            break;

#if defined(__APPLE__)
                        // Only accept "unknown" for the OS if the host is Apple and
                        // it "unknown" wasn't specified (it was just returned because it
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

#if defined(__APPLE__) && (defined(__arm__) || defined(__arm64__) || defined(__aarch64__))
    // If lldb is running on a watch, this isn't a RemoteWatch environment; it's a local system environment.
    if (force == false)
    {
        create = false;
    }
#endif

    if (create)
    {
        if (log)
            log->Printf ("PlatformRemoteAppleWatch::%s() creating platform", __FUNCTION__);

        return lldb::PlatformSP(new PlatformRemoteAppleWatch ());
    }

    if (log)
        log->Printf ("PlatformRemoteAppleWatch::%s() aborting creation of platform", __FUNCTION__);

    return lldb::PlatformSP();
}

lldb_private::ConstString
PlatformRemoteAppleWatch::GetPluginNameStatic ()
{
    static ConstString g_name("remote-watchos");
    return g_name;
}

const char *
PlatformRemoteAppleWatch::GetDescriptionStatic()
{
    return "Remote Apple Watch platform plug-in.";
}

void
PlatformRemoteAppleWatch::GetStatus (Stream &strm)
{
    Platform::GetStatus (strm);
    const char *sdk_directory = GetDeviceSupportDirectoryForOSVersion();
    if (sdk_directory)
        strm.Printf ("  SDK Path: \"%s\"\n", sdk_directory);
    else
        strm.PutCString ("  SDK Path: error: unable to locate SDK\n");
    
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
    for (uint32_t i=0; i<num_sdk_infos; ++i)
    {
        const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
        strm.Printf (" SDK Roots: [%2u] \"%s\"\n",
                     i,
                     sdk_dir_info.directory.GetPath().c_str());
    }
}

Error
PlatformRemoteAppleWatch::ResolveExecutable (const ModuleSpec &ms,
                                          lldb::ModuleSP &exe_module_sp,
                                          const FileSpecList *module_search_paths_ptr)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    ModuleSpec resolved_module_spec(ms);

    // Resolve any executable within a bundle on MacOSX
    // TODO: verify that this handles shallow bundles, if not then implement one ourselves
    Host::ResolveExecutableInBundle (resolved_module_spec.GetFileSpec());

    if (resolved_module_spec.GetFileSpec().Exists())
    {
        if (resolved_module_spec.GetArchitecture().IsValid() || resolved_module_spec.GetUUID().IsValid())
        {
            error = ModuleList::GetSharedModule(resolved_module_spec,
                                                exe_module_sp,
                                                nullptr,
                                                nullptr,
                                                nullptr);
        
            if (exe_module_sp && exe_module_sp->GetObjectFile())
                return error;
            exe_module_sp.reset();
        }
        // No valid architecture was specified or the exact ARM slice wasn't
        // found so ask the platform for the architectures that we should be
        // using (in the correct order) and see if we can find a match that way
        StreamString arch_names;
        for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, resolved_module_spec.GetArchitecture()); ++idx)
        {
            error = ModuleList::GetSharedModule(resolved_module_spec,
                                                exe_module_sp,
                                                nullptr,
                                                nullptr,
                                                nullptr);
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
    else
    {
        error.SetErrorStringWithFormat ("'%s' does not exist",
                                        resolved_module_spec.GetFileSpec().GetPath().c_str());
    }

    return error;
}

FileSpec::EnumerateDirectoryResult 
PlatformRemoteAppleWatch::GetContainedFilesIntoVectorOfStringsCallback (void *baton,
                                                                     FileSpec::FileType file_type,
                                                                     const FileSpec &file_spec)
{
    ((PlatformRemoteAppleWatch::SDKDirectoryInfoCollection *)baton)->push_back(PlatformRemoteAppleWatch::SDKDirectoryInfo(file_spec));
    return FileSpec::eEnumerateDirectoryResultNext;
}

bool
PlatformRemoteAppleWatch::UpdateSDKDirectoryInfosIfNeeded()
{
    if (m_sdk_directory_infos.empty())
    {
        const char *device_support_dir = GetDeviceSupportDirectory();
        if (device_support_dir)
        {
            const bool find_directories = true;
            const bool find_files = false;
            const bool find_other = false;

            SDKDirectoryInfoCollection builtin_sdk_directory_infos;
            FileSpec::EnumerateDirectory (m_device_support_directory.c_str(),
                                          find_directories,
                                          find_files,
                                          find_other,
                                          GetContainedFilesIntoVectorOfStringsCallback,
                                          &builtin_sdk_directory_infos);

            // Only add SDK directories that have symbols in them, some SDKs only contain
            // developer disk images and no symbols, so they aren't useful to us.
            FileSpec sdk_symbols_symlink_fspec;
            for (const auto &sdk_directory_info : builtin_sdk_directory_infos)
            {
                sdk_symbols_symlink_fspec = sdk_directory_info.directory;
                sdk_symbols_symlink_fspec.AppendPathComponent("Symbols.Internal");
                if (sdk_symbols_symlink_fspec.Exists())
                {
                    m_sdk_directory_infos.push_back(sdk_directory_info);
                }
                else
                {
                    sdk_symbols_symlink_fspec.GetFilename().SetCString("Symbols");
                    if (sdk_symbols_symlink_fspec.Exists())
                        m_sdk_directory_infos.push_back(sdk_directory_info);
                }
            }

            const uint32_t num_installed = m_sdk_directory_infos.size();
            FileSpec local_sdk_cache("~/Library/Developer/Xcode/watchOS DeviceSupport", true);
            if (!local_sdk_cache.Exists())
            {
                local_sdk_cache = FileSpec("~/Library/Developer/Xcode/watch OS DeviceSupport", true);
            }
            if (!local_sdk_cache.Exists())
            {
                local_sdk_cache = FileSpec("~/Library/Developer/Xcode/WatchOS DeviceSupport", true);
            }
            if (!local_sdk_cache.Exists())
            {
                local_sdk_cache = FileSpec("~/Library/Developer/Xcode/Watch OS DeviceSupport", true);
            }
            if (local_sdk_cache.Exists())
            {
                char path[PATH_MAX];
                if (local_sdk_cache.GetPath(path, sizeof(path)))
                {
                    FileSpec::EnumerateDirectory (path,
                                                  find_directories,
                                                  find_files,
                                                  find_other,
                                                  GetContainedFilesIntoVectorOfStringsCallback,
                                                  &m_sdk_directory_infos);
                    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
                    // First try for an exact match of major, minor and update
                    for (uint32_t i=num_installed; i<num_sdk_infos; ++i)
                    {
                        m_sdk_directory_infos[i].user_cached = true;
                    }
                }
            }
        }
    }
    return !m_sdk_directory_infos.empty();
}

const PlatformRemoteAppleWatch::SDKDirectoryInfo *
PlatformRemoteAppleWatch::GetSDKDirectoryForCurrentOSVersion ()
{
    uint32_t i;
    if (UpdateSDKDirectoryInfosIfNeeded())
    {
        const uint32_t num_sdk_infos = m_sdk_directory_infos.size();

        // Check to see if the user specified a build string. If they did, then
        // be sure to match it.
        std::vector<bool> check_sdk_info(num_sdk_infos, true);
        ConstString build(m_sdk_build);
        if (build)
        {
            for (i=0; i<num_sdk_infos; ++i)
                check_sdk_info[i] = m_sdk_directory_infos[i].build == build;
        }
        
        // If we are connected we can find the version of the OS the platform
        // us running on and select the right SDK
        uint32_t major, minor, update;
        if (GetOSVersion(major, minor, update))
        {
            if (UpdateSDKDirectoryInfosIfNeeded())
            {
                // First try for an exact match of major, minor and update
                for (i=0; i<num_sdk_infos; ++i)
                {
                    if (check_sdk_info[i])
                    {
                        if (m_sdk_directory_infos[i].version_major == major &&
                            m_sdk_directory_infos[i].version_minor == minor &&
                            m_sdk_directory_infos[i].version_update == update)
                        {
                            return &m_sdk_directory_infos[i];
                        }
                    }
                }
                // First try for an exact match of major and minor
                for (i=0; i<num_sdk_infos; ++i)
                {
                    if (check_sdk_info[i])
                    {
                        if (m_sdk_directory_infos[i].version_major == major &&
                            m_sdk_directory_infos[i].version_minor == minor)
                        {
                            return &m_sdk_directory_infos[i];
                        }
                    }
                }
                // Lastly try to match of major version only..
                for (i=0; i<num_sdk_infos; ++i)
                {
                    if (check_sdk_info[i])
                    {
                        if (m_sdk_directory_infos[i].version_major == major)
                        {
                            return &m_sdk_directory_infos[i];
                        }
                    }
                }
            }
        }
        else if (build)
        {
            // No version, just a build number, search for the first one that matches
            for (i=0; i<num_sdk_infos; ++i)
                if (check_sdk_info[i])
                    return &m_sdk_directory_infos[i];
        }
    }
    return nullptr;
}

const PlatformRemoteAppleWatch::SDKDirectoryInfo *
PlatformRemoteAppleWatch::GetSDKDirectoryForLatestOSVersion ()
{
    const PlatformRemoteAppleWatch::SDKDirectoryInfo *result = nullptr;
    if (UpdateSDKDirectoryInfosIfNeeded())
    {
        const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
        // First try for an exact match of major, minor and update
        for (uint32_t i=0; i<num_sdk_infos; ++i)
        {
            const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
            if (sdk_dir_info.version_major != UINT32_MAX)
            {
                if (result == nullptr || sdk_dir_info.version_major > result->version_major)
                {
                    result = &sdk_dir_info;
                }
                else if (sdk_dir_info.version_major == result->version_major)
                {
                    if (sdk_dir_info.version_minor > result->version_minor)
                    {
                        result = &sdk_dir_info;
                    }
                    else if (sdk_dir_info.version_minor == result->version_minor)
                    {
                        if (sdk_dir_info.version_update > result->version_update)
                        {
                            result = &sdk_dir_info;
                        }
                    }
                }
            }
        }
    }
    return result;
}

const char *
PlatformRemoteAppleWatch::GetDeviceSupportDirectory()
{
    if (m_device_support_directory.empty())
    {
        const char *device_support_dir = GetDeveloperDirectory();
        if (device_support_dir)
        {
            m_device_support_directory.assign (device_support_dir);
            m_device_support_directory.append ("/Platforms/watchOS.platform/DeviceSupport");
            FileSpec platform_device_support_dir (m_device_support_directory.c_str(), true);
            if (!platform_device_support_dir.Exists())
            {
                std::string alt_platform_dirname = device_support_dir;
                alt_platform_dirname.append ("/Platforms/WatchOS.platform/DeviceSupport");
                FileSpec alt_platform_device_support_dir (m_device_support_directory.c_str(), true);
                if (alt_platform_device_support_dir.Exists())
                {
                    m_device_support_directory = alt_platform_dirname;
                }
            }
        }
        else
        {
            // Assign a single NULL character so we know we tried to find the device
            // support directory and we don't keep trying to find it over and over.
            m_device_support_directory.assign (1, '\0');
        }
    }
    // We should have put a single NULL character into m_device_support_directory
    // or it should have a valid path if the code gets here
    assert (m_device_support_directory.empty() == false);
    if (m_device_support_directory[0])
        return m_device_support_directory.c_str();
    return nullptr;
}

const char *
PlatformRemoteAppleWatch::GetDeviceSupportDirectoryForOSVersion()
{
    if (m_sdk_sysroot)
        return m_sdk_sysroot.GetCString();

    if (m_device_support_directory_for_os_version.empty())
    {
        const PlatformRemoteAppleWatch::SDKDirectoryInfo *sdk_dir_info = GetSDKDirectoryForCurrentOSVersion ();
        if (sdk_dir_info == nullptr)
            sdk_dir_info = GetSDKDirectoryForLatestOSVersion ();
        if (sdk_dir_info)
        {
            char path[PATH_MAX];
            if (sdk_dir_info->directory.GetPath(path, sizeof(path)))
            {
                m_device_support_directory_for_os_version = path;
                return m_device_support_directory_for_os_version.c_str();
            }
        }
        else
        {
            // Assign a single NULL character so we know we tried to find the device
            // support directory and we don't keep trying to find it over and over.
            m_device_support_directory_for_os_version.assign (1, '\0');
        }
    }
    // We should have put a single NULL character into m_device_support_directory_for_os_version
    // or it should have a valid path if the code gets here
    assert (m_device_support_directory_for_os_version.empty() == false);
    if (m_device_support_directory_for_os_version[0])
        return m_device_support_directory_for_os_version.c_str();
    return nullptr;
}

uint32_t
PlatformRemoteAppleWatch::FindFileInAllSDKs (const char *platform_file_path,
                                      FileSpecList &file_list)
{
    if (platform_file_path && platform_file_path[0] && UpdateSDKDirectoryInfosIfNeeded())
    {
        const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
        lldb_private::FileSpec local_file;
        // First try for an exact match of major, minor and update
        for (uint32_t sdk_idx=0; sdk_idx<num_sdk_infos; ++sdk_idx)
        {
            if (GetFileInSDK (platform_file_path,
                              sdk_idx,
                              local_file))
            {
                file_list.Append(local_file);
            }
        }
    }
    return file_list.GetSize();
}

bool
PlatformRemoteAppleWatch::GetFileInSDK (const char *platform_file_path,
                                 uint32_t sdk_idx,
                                 lldb_private::FileSpec &local_file)
{
    if (sdk_idx < m_sdk_directory_infos.size())
    {
        char sdkroot_path[PATH_MAX];
        const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[sdk_idx];
        if (sdk_dir_info.directory.GetPath(sdkroot_path, sizeof(sdkroot_path)))
        {
            const bool symbols_dirs_only = true;

            return GetFileInSDKRoot (platform_file_path,
                                     sdkroot_path,
                                     symbols_dirs_only,
                                     local_file);
        }
    }
    return false;
}

bool
PlatformRemoteAppleWatch::GetFileInSDKRoot (const char *platform_file_path,
                                     const char *sdkroot_path,
                                     bool symbols_dirs_only,
                                     lldb_private::FileSpec &local_file)
{
    if (sdkroot_path && sdkroot_path[0] && platform_file_path && platform_file_path[0])
    {
        char resolved_path[PATH_MAX];
        
        if (!symbols_dirs_only)
        {
            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s%s",
                        sdkroot_path,
                        platform_file_path);
            
            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return true;
        }
            
        ::snprintf (resolved_path,
                    sizeof(resolved_path), 
                    "%s/Symbols.Internal%s",
                    sdkroot_path,
                    platform_file_path);
        
        local_file.SetFile(resolved_path, true);
        if (local_file.Exists())
            return true;
        ::snprintf (resolved_path,
                    sizeof(resolved_path), 
                    "%s/Symbols%s", 
                    sdkroot_path, 
                    platform_file_path);
        
        local_file.SetFile(resolved_path, true);
        if (local_file.Exists())
            return true;                
    }
    return false;
}

Error
PlatformRemoteAppleWatch::GetSymbolFile (const FileSpec &platform_file,
                                  const UUID *uuid_ptr,
                                  FileSpec &local_file)
{
    Error error;
    char platform_file_path[PATH_MAX];
    if (platform_file.GetPath(platform_file_path, sizeof(platform_file_path)))
    {
        char resolved_path[PATH_MAX];
    
        const char * os_version_dir = GetDeviceSupportDirectoryForOSVersion();
        if (os_version_dir)
        {
            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/%s", 
                        os_version_dir, 
                        platform_file_path);
            
            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;

            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/Symbols.Internal/%s", 
                        os_version_dir, 
                        platform_file_path);

            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;
            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/Symbols/%s", 
                        os_version_dir, 
                        platform_file_path);

            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;

        }
        local_file = platform_file;
        if (local_file.Exists())
            return error;

        error.SetErrorStringWithFormat ("unable to locate a platform file for '%s' in platform '%s'", 
                                        platform_file_path,
                                        GetPluginName().GetCString());
    }
    else
    {
        error.SetErrorString ("invalid platform file argument");
    }
    return error;
}

Error
PlatformRemoteAppleWatch::GetSharedModule (const ModuleSpec &module_spec,
                                           lldb_private::Process* process,
                                           ModuleSP &module_sp,
                                           const FileSpecList *module_search_paths_ptr,
                                           ModuleSP *old_module_sp_ptr,
                                           bool *did_create_ptr)
{
    // For Apple Watch, the SDK files are all cached locally on the host
    // system. So first we ask for the file in the cached SDK,
    // then we attempt to get a shared module for the right architecture
    // with the right UUID.
    const FileSpec &platform_file = module_spec.GetFileSpec();

    Error error;
    char platform_file_path[PATH_MAX];
    
    if (platform_file.GetPath(platform_file_path, sizeof(platform_file_path)))
    {
        ModuleSpec platform_module_spec(module_spec);

        UpdateSDKDirectoryInfosIfNeeded();

        const uint32_t num_sdk_infos = m_sdk_directory_infos.size();

        // If we are connected we migth be able to correctly deduce the SDK directory
        // using the OS build.
        const uint32_t connected_sdk_idx = GetConnectedSDKIndex ();
        if (connected_sdk_idx < num_sdk_infos)
        {
            if (GetFileInSDK (platform_file_path, connected_sdk_idx, platform_module_spec.GetFileSpec()))
            {
                module_sp.reset();
                error = ResolveExecutable(platform_module_spec,
                                          module_sp,
                                          nullptr);
                if (module_sp)
                {
                    m_last_module_sdk_idx = connected_sdk_idx;
                    error.Clear();
                    return error;
                }
            }
        }

        // Try the last SDK index if it is set as most files from an SDK
        // will tend to be valid in that same SDK.
        if (m_last_module_sdk_idx < num_sdk_infos)
        {
            if (GetFileInSDK (platform_file_path, m_last_module_sdk_idx, platform_module_spec.GetFileSpec()))
            {
                module_sp.reset();
                error = ResolveExecutable(platform_module_spec,
                                          module_sp,
                                          nullptr);
                if (module_sp)
                {
                    error.Clear();
                    return error;
                }
            }
        }
        
        // First try for an exact match of major, minor and update
        for (uint32_t sdk_idx=0; sdk_idx<num_sdk_infos; ++sdk_idx)
        {
            if (m_last_module_sdk_idx == sdk_idx)
            {
                // Skip the last module SDK index if we already searched
                // it above
                continue;
            }
            if (GetFileInSDK (platform_file_path, sdk_idx, platform_module_spec.GetFileSpec()))
            {
                //printf ("sdk[%u]: '%s'\n", sdk_idx, local_file.GetPath().c_str());
                
                error = ResolveExecutable(platform_module_spec, module_sp, nullptr);
                if (module_sp)
                {
                    // Remember the index of the last SDK that we found a file
                    // in in case the wrong SDK was selected.
                    m_last_module_sdk_idx = sdk_idx;
                    error.Clear();
                    return error;
                }
            }
        }
    }
    // Not the module we are looking for... Nothing to see here...
    module_sp.reset();

    // This may not be an SDK-related module.  Try whether we can bring in the thing to our local cache.
    error = GetSharedModuleWithLocalCache(module_spec, module_sp, module_search_paths_ptr, old_module_sp_ptr, did_create_ptr);
    if (error.Success())
        return error;

    const bool always_create = false;
    error = ModuleList::GetSharedModule (module_spec,
                                         module_sp,
                                         module_search_paths_ptr,
                                         old_module_sp_ptr,
                                         did_create_ptr,
                                         always_create);

    if (module_sp)
        module_sp->SetPlatformFileSpec(platform_file);

    return error;
}

bool
PlatformRemoteAppleWatch::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    ArchSpec system_arch (GetSystemArchitecture());

    const ArchSpec::Core system_core = system_arch.GetCore();
    switch (system_core)
    {
    default:
        switch (idx)
        {
            case  0: arch.SetTriple ("arm64-apple-watchos");    return true;
            case  1: arch.SetTriple ("armv7k-apple-watchos");   return true;
            case  2: arch.SetTriple ("armv7s-apple-watchos");   return true;
            case  3: arch.SetTriple ("armv7-apple-watchos");    return true;
            case  4: arch.SetTriple ("thumbv7k-apple-watchos");   return true;
            case  5: arch.SetTriple ("thumbv7-apple-watchos");    return true;
            case  6: arch.SetTriple ("thumbv7s-apple-watchos");   return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_arm64:
        switch (idx)
        {
            case  0: arch.SetTriple ("arm64-apple-watchos");    return true;
            case  1: arch.SetTriple ("armv7k-apple-watchos");   return true;
            case  2: arch.SetTriple ("armv7s-apple-watchos");   return true;
            case  3: arch.SetTriple ("armv7-apple-watchos");    return true;
            case  4: arch.SetTriple ("thumbv7k-apple-watchos");   return true;
            case  5: arch.SetTriple ("thumbv7-apple-watchos");    return true;
            case  6: arch.SetTriple ("thumbv7s-apple-watchos");   return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7k:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7k-apple-watchos");   return true;
            case  1: arch.SetTriple ("armv7s-apple-watchos");   return true;
            case  2: arch.SetTriple ("armv7-apple-watchos");    return true;
            case  3: arch.SetTriple ("thumbv7k-apple-watchos");   return true;
            case  4: arch.SetTriple ("thumbv7-apple-watchos");    return true;
            case  5: arch.SetTriple ("thumbv7s-apple-watchos");   return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7s:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7s-apple-watchos");   return true;
            case  1: arch.SetTriple ("armv7k-apple-watchos");   return true;
            case  2: arch.SetTriple ("armv7-apple-watchos");    return true;
            case  3: arch.SetTriple ("thumbv7k-apple-watchos");   return true;
            case  4: arch.SetTriple ("thumbv7-apple-watchos");    return true;
            case  5: arch.SetTriple ("thumbv7s-apple-watchos");   return true;
            default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7:
        switch (idx)
        {
            case  0: arch.SetTriple ("armv7-apple-watchos");    return true;
            case  1: arch.SetTriple ("armv7k-apple-watchos");   return true;
            case  2: arch.SetTriple ("thumbv7k-apple-watchos");   return true;
            case  3: arch.SetTriple ("thumbv7-apple-watchos");    return true;
            default: break;
        }
        break;
    }
    arch.Clear();
    return false;
}

uint32_t
PlatformRemoteAppleWatch::GetConnectedSDKIndex ()
{
    if (IsConnected())
    {
        if (m_connected_module_sdk_idx == UINT32_MAX)
        {
            std::string build;
            if (GetRemoteOSBuildString(build))
            {
                const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
                for (uint32_t i=0; i<num_sdk_infos; ++i)
                {
                    const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
                    if (strstr(sdk_dir_info.directory.GetFilename().AsCString(""), build.c_str()))
                    {
                        m_connected_module_sdk_idx = i;
                    }
                }
            }
        }
    }
    else
    {
        m_connected_module_sdk_idx = UINT32_MAX;
    }
    return m_connected_module_sdk_idx;
}
