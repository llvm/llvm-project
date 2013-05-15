//===-- PlatformDarwinKernel.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwinKernel.h"

#if defined (__APPLE__)  // This Plugin uses the Mac-specific source/Host/macosx/cfcpp utilities


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
#include "lldb/Interpreter/OptionValueFileSpecList.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include <CoreFoundation/CoreFoundation.h>

#include "Host/macosx/cfcpp/CFCBundle.h"

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
PlatformDarwinKernel::Initialize ()
{
    if (g_initialize_count++ == 0)
    {
        PluginManager::RegisterPlugin (PlatformDarwinKernel::GetPluginNameStatic(),
                                       PlatformDarwinKernel::GetDescriptionStatic(),
                                       PlatformDarwinKernel::CreateInstance,
                                       PlatformDarwinKernel::DebuggerInitialize);
    }
}

void
PlatformDarwinKernel::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformDarwinKernel::CreateInstance);
        }
    }
}

Platform*
PlatformDarwinKernel::CreateInstance (bool force, const ArchSpec *arch)
{
    // This is a special plugin that we don't want to activate just based on an ArchSpec for normal
    // userlnad debugging.  It is only useful in kernel debug sessions and the DynamicLoaderDarwinPlugin
    // (or a user doing 'platform select') will force the creation of this Platform plugin.
    if (force == false)
        return NULL;

    bool create = force;
    LazyBool is_ios_debug_session = eLazyBoolCalculate;

    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getVendor())
        {
            case llvm::Triple::Apple:
                create = true;
                break;
                
            // Only accept "unknown" for vendor if the host is Apple and
            // it "unknown" wasn't specified (it was just returned becasue it
            // was NOT specified)
            case llvm::Triple::UnknownArch:
                create = !arch->TripleVendorWasSpecified();
                break;
            default:
                break;
        }
        
        if (create)
        {
            switch (triple.getOS())
            {
                case llvm::Triple::Darwin:  // Deprecated, but still support Darwin for historical reasons
                case llvm::Triple::MacOSX:
                    break;
                // Only accept "vendor" for vendor if the host is Apple and
                // it "unknown" wasn't specified (it was just returned becasue it
                // was NOT specified)
                case llvm::Triple::UnknownOS:
                    create = !arch->TripleOSWasSpecified();
                    break;
                default:
                    create = false;
                    break;
            }
        }
    }
    if (arch && arch->IsValid())
    {
        switch (arch->GetMachine())
        {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
        case llvm::Triple::ppc:
        case llvm::Triple::ppc64:
            is_ios_debug_session = eLazyBoolNo;
            break;
        case llvm::Triple::arm:
        case llvm::Triple::thumb:
            is_ios_debug_session = eLazyBoolYes;
            break;
        default:
            is_ios_debug_session = eLazyBoolCalculate;
            break;
        }
    }
    if (create)
        return new PlatformDarwinKernel (is_ios_debug_session);
    return NULL;
}


lldb_private::ConstString
PlatformDarwinKernel::GetPluginNameStatic ()
{
    static ConstString g_name("darwin-kernel");
    return g_name;
}

const char *
PlatformDarwinKernel::GetDescriptionStatic()
{
    return "Darwin Kernel platform plug-in.";
}

//------------------------------------------------------------------
/// Code to handle the PlatformDarwinKernel settings
//------------------------------------------------------------------

static PropertyDefinition
g_properties[] =
{
    { "search-locally-for-kexts" , OptionValue::eTypeBoolean,      true, true, NULL, NULL, "Automatically search for kexts on the local system when doing kernel debugging." },
    { "kext-directories",          OptionValue::eTypeFileSpecList, false, 0,   NULL, NULL, "Directories/KDKs to search for kexts in when starting a kernel debug session." },
    {  NULL        , OptionValue::eTypeInvalid, false, 0  , NULL, NULL, NULL  }
};

enum {
    ePropertySearchForKexts = 0,
    ePropertyKextDirectories
};



class PlatformDarwinKernelProperties : public Properties
{
public:
    
    static ConstString &
    GetSettingName ()
    {
        static ConstString g_setting_name("darwin-kernel");
        return g_setting_name;
    }

    PlatformDarwinKernelProperties() :
        Properties ()
    {
        m_collection_sp.reset (new OptionValueProperties(GetSettingName()));
        m_collection_sp->Initialize(g_properties);
    }

    virtual
    ~PlatformDarwinKernelProperties()
    {
    }

    bool
    GetSearchForKexts() const
    {
        const uint32_t idx = ePropertySearchForKexts;
        return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
    }

    FileSpecList &
    GetKextDirectories() const
    {
        const uint32_t idx = ePropertyKextDirectories;
        OptionValueFileSpecList *option_value = m_collection_sp->GetPropertyAtIndexAsOptionValueFileSpecList (NULL, false, idx);
        assert(option_value);
        return option_value->GetCurrentValue();
    }
};

typedef std::shared_ptr<PlatformDarwinKernelProperties> PlatformDarwinKernelPropertiesSP;

static const PlatformDarwinKernelPropertiesSP &
GetGlobalProperties()
{
    static PlatformDarwinKernelPropertiesSP g_settings_sp;
    if (!g_settings_sp)
        g_settings_sp.reset (new PlatformDarwinKernelProperties ());
    return g_settings_sp;
}

void
PlatformDarwinKernel::DebuggerInitialize (lldb_private::Debugger &debugger)
{
    if (!PluginManager::GetSettingForPlatformPlugin (debugger, PlatformDarwinKernelProperties::GetSettingName()))
    {
        const bool is_global_setting = true;
        PluginManager::CreateSettingForPlatformPlugin (debugger,
                                                            GetGlobalProperties()->GetValueProperties(),
                                                            ConstString ("Properties for the PlatformDarwinKernel plug-in."),
                                                            is_global_setting);
    }
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformDarwinKernel::PlatformDarwinKernel (lldb_private::LazyBool is_ios_debug_session) :
    PlatformDarwin (false),    // This is a remote platform
    m_name_to_kext_path_map(),
    m_directories_searched(),
    m_ios_debug_session(is_ios_debug_session)

{
    if (GetGlobalProperties()->GetSearchForKexts())
    {
        SearchForKexts ();
    }
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformDarwinKernel::~PlatformDarwinKernel()
{
}


void
PlatformDarwinKernel::GetStatus (Stream &strm)
{
    Platform::GetStatus (strm);
    strm.Printf (" Debug session type: ");
    if (m_ios_debug_session == eLazyBoolYes)
        strm.Printf ("iOS kernel debugging\n");
    else if (m_ios_debug_session == eLazyBoolNo)
        strm.Printf ("Mac OS X kernel debugging\n");
    else
            strm.Printf ("unknown kernel debugging\n");
    const uint32_t num_kext_dirs = m_directories_searched.size();
    for (uint32_t i=0; i<num_kext_dirs; ++i)
    {
        const FileSpec &kext_dir = m_directories_searched[i];
        strm.Printf (" Kext directories: [%2u] \"%s\"\n", i, kext_dir.GetPath().c_str());
    }
    strm.Printf (" Total number of kexts indexed: %d\n", (int) m_name_to_kext_path_map.size());
}

void
PlatformDarwinKernel::SearchForKexts ()
{
    // Differentiate between "ios debug session" and "mac debug session" so we don't index
    // kext bundles that won't be used in this debug session.  If this is an ios kext debug
    // session, looking in /System/Library/Extensions is a waste of stat()s, for example.

    // Build up a list of all SDKs we'll be searching for directories of kexts
    // e.g. /Applications/Xcode.app//Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.Internal.sdk
    std::vector<FileSpec> sdk_dirs;
    if (m_ios_debug_session != eLazyBoolNo)
        GetiOSSDKDirectoriesToSearch (sdk_dirs);
    if (m_ios_debug_session != eLazyBoolYes)
        GetMacSDKDirectoriesToSearch (sdk_dirs);

    GetGenericSDKDirectoriesToSearch (sdk_dirs);

    // Build up a list of directories that hold kext bundles on the system
    // e.g. /Applications/Xcode.app//Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.Internal.sdk/System/Library/Extensions
    std::vector<FileSpec> kext_dirs;
    SearchSDKsForKextDirectories (sdk_dirs, kext_dirs);

    if (m_ios_debug_session != eLazyBoolNo)
        GetiOSDirectoriesToSearch (kext_dirs);
    if (m_ios_debug_session != eLazyBoolYes)
        GetMacDirectoriesToSearch (kext_dirs);

    GetGenericDirectoriesToSearch (kext_dirs);

    GetUserSpecifiedDirectoriesToSearch (kext_dirs);

    // We now have a complete list of directories that we will search for kext bundles
    m_directories_searched = kext_dirs;

    IndexKextsInDirectories (kext_dirs);
}

void
PlatformDarwinKernel::GetiOSSDKDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
    // DeveloperDirectory is something like "/Applications/Xcode.app/Contents/Developer"
    const char *developer_dir = GetDeveloperDirectory();
    if (developer_dir == NULL)
        developer_dir = "/Applications/Xcode.app/Contents/Developer";

    char pathbuf[PATH_MAX];
    ::snprintf (pathbuf, sizeof (pathbuf), "%s/Platforms/iPhoneOS.platform/Developer/SDKs", developer_dir);
    FileSpec ios_sdk(pathbuf, true);
    if (ios_sdk.Exists() && ios_sdk.IsDirectory())
    {
        directories.push_back (ios_sdk);
    }
}

void
PlatformDarwinKernel::GetMacSDKDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
    // DeveloperDirectory is something like "/Applications/Xcode.app/Contents/Developer"
    const char *developer_dir = GetDeveloperDirectory();
    if (developer_dir == NULL)
        developer_dir = "/Applications/Xcode.app/Contents/Developer";

    char pathbuf[PATH_MAX];
    ::snprintf (pathbuf, sizeof (pathbuf), "%s/Platforms/MacOSX.platform/Developer/SDKs", developer_dir);
    FileSpec mac_sdk(pathbuf, true);
    if (mac_sdk.Exists() && mac_sdk.IsDirectory())
    {
        directories.push_back (mac_sdk);
    }
}

void
PlatformDarwinKernel::GetGenericSDKDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
    FileSpec generic_sdk("/AppleInternal/Developer/KDKs", true);
    if (generic_sdk.Exists() && generic_sdk.IsDirectory())
    {
        directories.push_back (generic_sdk);
    }
}

void
PlatformDarwinKernel::GetiOSDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
}

void
PlatformDarwinKernel::GetMacDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
    FileSpec sle("/System/Library/Extensions", true);
    if (sle.Exists() && sle.IsDirectory())
    {
        directories.push_back(sle);
    }

    FileSpec le("/Library/Extensions", true);
    if (le.Exists() && le.IsDirectory())
    {
        directories.push_back(le);
    }

    FileSpec kdk("/Volumes/KernelDebugKit", true);
    if (kdk.Exists() && kdk.IsDirectory())
    {
        directories.push_back(kdk);
    }
}

void
PlatformDarwinKernel::GetGenericDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
    // DeveloperDirectory is something like "/Applications/Xcode.app/Contents/Developer"
    const char *developer_dir = GetDeveloperDirectory();
    if (developer_dir == NULL)
        developer_dir = "/Applications/Xcode.app/Contents/Developer";

    char pathbuf[PATH_MAX];
    ::snprintf (pathbuf, sizeof (pathbuf), "%s/../Symbols", developer_dir);
    FileSpec symbols_dir (pathbuf, true);
    if (symbols_dir.Exists() && symbols_dir.IsDirectory())
    {
        directories.push_back (symbols_dir);
    }
}

void
PlatformDarwinKernel::GetUserSpecifiedDirectoriesToSearch (std::vector<lldb_private::FileSpec> &directories)
{
    FileSpecList user_dirs(GetGlobalProperties()->GetKextDirectories());
    std::vector<FileSpec> possible_sdk_dirs;

    const uint32_t user_dirs_count = user_dirs.GetSize();
    for (uint32_t i = 0; i < user_dirs_count; i++)
    {
        FileSpec dir = user_dirs.GetFileSpecAtIndex (i);
        dir.ResolvePath();
        if (dir.Exists() && dir.IsDirectory())
        {
            directories.push_back (dir);
            possible_sdk_dirs.push_back (dir);  // does this directory have a *.sdk or *.kdk that we should look in?

            // Is there a "System/Library/Extensions" subdir of this directory?
            std::string dir_sle_path = dir.GetPath();
            dir_sle_path.append ("/System/Library/Extensions");
            FileSpec dir_sle(dir_sle_path.c_str(), true);
            if (dir_sle.Exists() && dir_sle.IsDirectory())
            {
                directories.push_back (dir_sle);
            }
        }
    }

    SearchSDKsForKextDirectories (possible_sdk_dirs, directories);
}

// Scan through the SDK directories, looking for directories where kexts are likely.
// Add those directories to kext_dirs.
void
PlatformDarwinKernel::SearchSDKsForKextDirectories (std::vector<lldb_private::FileSpec> sdk_dirs, std::vector<lldb_private::FileSpec> &kext_dirs)
{
    const uint32_t num_sdks = sdk_dirs.size();
    for (uint32_t i = 0; i < num_sdks; i++)
    {
        const FileSpec &sdk_dir = sdk_dirs[i];
        std::string sdk_dir_path = sdk_dir.GetPath();
        if (!sdk_dir_path.empty())
        {
            const bool find_directories = true;
            const bool find_files = false;
            const bool find_other = false;
            FileSpec::EnumerateDirectory (sdk_dir_path.c_str(),
                                          find_directories,
                                          find_files,
                                          find_other,
                                          GetKextDirectoriesInSDK,
                                          &kext_dirs);
        }
    }
}

// Callback for FileSpec::EnumerateDirectory().  
// Step through the entries in a directory like
//    /Applications/Xcode.app//Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
// looking for any subdirectories of the form MacOSX10.8.Internal.sdk/System/Library/Extensions
// Adds these to the vector of FileSpec's.

FileSpec::EnumerateDirectoryResult
PlatformDarwinKernel::GetKextDirectoriesInSDK (void *baton,
                                               FileSpec::FileType file_type,
                                               const FileSpec &file_spec)
{
    if (file_type == FileSpec::eFileTypeDirectory 
        && (file_spec.GetFileNameExtension() == ConstString("sdk")
            || file_spec.GetFileNameExtension() == ConstString("kdk")))
    {
        std::string kext_directory_path = file_spec.GetPath();
        kext_directory_path.append ("/System/Library/Extensions");
        FileSpec kext_directory (kext_directory_path.c_str(), true);
        if (kext_directory.Exists() && kext_directory.IsDirectory())
        {
            ((std::vector<lldb_private::FileSpec> *)baton)->push_back(kext_directory);
        }
    }
    return FileSpec::eEnumerateDirectoryResultNext;
}

void
PlatformDarwinKernel::IndexKextsInDirectories (std::vector<lldb_private::FileSpec> kext_dirs)
{
    std::vector<FileSpec> kext_bundles;

    const uint32_t num_dirs = kext_dirs.size();
    for (uint32_t i = 0; i < num_dirs; i++)
    {
        const FileSpec &dir = kext_dirs[i];
        const bool find_directories = true;
        const bool find_files = false;
        const bool find_other = false;
        FileSpec::EnumerateDirectory (dir.GetPath().c_str(),
                                      find_directories,
                                      find_files,
                                      find_other,
                                      GetKextsInDirectory,
                                      &kext_bundles);
    }

    const uint32_t num_kexts = kext_bundles.size();
    for (uint32_t i = 0; i < num_kexts; i++)
    {
        const FileSpec &kext = kext_bundles[i];
        CFCBundle bundle (kext.GetPath().c_str());
        CFStringRef bundle_id (bundle.GetIdentifier());
        if (bundle_id && CFGetTypeID (bundle_id) == CFStringGetTypeID ())
        {
            char bundle_id_buf[PATH_MAX];
            if (CFStringGetCString (bundle_id, bundle_id_buf, sizeof (bundle_id_buf), kCFStringEncodingUTF8))
            {
                ConstString bundle_conststr(bundle_id_buf);
                m_name_to_kext_path_map.insert(std::pair<ConstString, FileSpec>(bundle_conststr, kext));
            }
        }
    }
}

// Callback for FileSpec::EnumerateDirectory().
// Step through the entries in a directory like /System/Library/Extensions, find .kext bundles, add them
// to the vector of FileSpecs.
// If a .kext bundle has a Contents/PlugIns or PlugIns subdir, search for kexts in there too.

FileSpec::EnumerateDirectoryResult
PlatformDarwinKernel::GetKextsInDirectory (void *baton,
                                           FileSpec::FileType file_type,
                                           const FileSpec &file_spec)
{
    if (file_type == FileSpec::eFileTypeDirectory && file_spec.GetFileNameExtension() == ConstString("kext"))
    {
        ((std::vector<lldb_private::FileSpec> *)baton)->push_back(file_spec);
        std::string kext_bundle_path = file_spec.GetPath();
        std::string search_here_too;
        std::string contents_plugins_path = kext_bundle_path + "/Contents/PlugIns";
        FileSpec contents_plugins (contents_plugins_path.c_str(), false);
        if (contents_plugins.Exists() && contents_plugins.IsDirectory())
        {
            search_here_too = contents_plugins_path;
        }
        else
        {
            std::string plugins_path = kext_bundle_path + "/PlugIns";
            FileSpec plugins (plugins_path.c_str(), false);
            if (plugins.Exists() && plugins.IsDirectory())
            {
                search_here_too = plugins_path;
            }
        }

        if (!search_here_too.empty())
        {
            const bool find_directories = true;
            const bool find_files = false;
            const bool find_other = false;
            FileSpec::EnumerateDirectory (search_here_too.c_str(),
                                          find_directories,
                                          find_files,
                                          find_other,
                                          GetKextsInDirectory,
                                          baton);
        }
    }
    return FileSpec::eEnumerateDirectoryResultNext;
}

Error
PlatformDarwinKernel::GetSharedModule (const ModuleSpec &module_spec,
                                       ModuleSP &module_sp,
                                       const FileSpecList *module_search_paths_ptr,
                                       ModuleSP *old_module_sp_ptr,
                                       bool *did_create_ptr)
{
    Error error;
    module_sp.reset();
    const FileSpec &platform_file = module_spec.GetFileSpec();
    std::string kext_bundle_id = platform_file.GetPath();
    if (!kext_bundle_id.empty())
    {
        ConstString kext_bundle_cs(kext_bundle_id.c_str());
        if (m_name_to_kext_path_map.count(kext_bundle_cs) > 0)
        {
            for (BundleIDToKextIterator it = m_name_to_kext_path_map.begin (); it != m_name_to_kext_path_map.end (); ++it)
            {
                if (it->first == kext_bundle_cs)
                {
                    error = ExamineKextForMatchingUUID (it->second, module_spec.GetUUID(), module_spec.GetArchitecture(), module_sp);
                    if (module_sp.get())
                    {
                        return error;
                    }
                }
            }
        }
    }

    return error;
}

Error
PlatformDarwinKernel::ExamineKextForMatchingUUID (const FileSpec &kext_bundle_path, const lldb_private::UUID &uuid, const ArchSpec &arch, ModuleSP &exe_module_sp)
{
    Error error;
    FileSpec exe_file = kext_bundle_path;
    Host::ResolveExecutableInBundle (exe_file);
    if (exe_file.Exists())
    {
        ModuleSpec exe_spec (exe_file);
        exe_spec.GetUUID() = uuid;
        exe_spec.GetArchitecture() = arch;

        // First try to create a ModuleSP with the file / arch and see if the UUID matches.
        // If that fails (this exec file doesn't have the correct uuid), don't call GetSharedModule
        // (which may call in to the DebugSymbols framework and therefore can be slow.)
        ModuleSP module_sp (new Module (exe_file, arch));
        if (module_sp && module_sp->GetObjectFile() && module_sp->MatchesModuleSpec (exe_spec))
        {
            error = ModuleList::GetSharedModule (exe_spec, exe_module_sp, NULL, NULL, NULL);
            if (exe_module_sp && exe_module_sp->GetObjectFile())
            {
                return error;
            }
        }
        exe_module_sp.reset();
    }
    return error;
}

bool
PlatformDarwinKernel::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
#if defined (__arm__)
    return ARMGetSupportedArchitectureAtIndex (idx, arch);
#else
    return x86GetSupportedArchitectureAtIndex (idx, arch);
#endif
}

#else  // __APPLE__

// Since DynamicLoaderDarwinKernel is compiled in for all systems, and relies on
// PlatformDarwinKernel for the plug-in name, we compile just the plug-in name in
// here to avoid issues. We are tracking an internal bug to resolve this issue by
// either not compiling in DynamicLoaderDarwinKernel for non-apple builds, or to make
// PlatformDarwinKernel build on all systems. PlatformDarwinKernel is currently not
// compiled on other platforms due to the use of the Mac-specific
// source/Host/macosx/cfcpp utilities.

lldb_private::ConstString
PlatformDarwinKernel::GetPluginNameStatic ()
{
    static lldb_private::ConstString g_name("darwin-kernel");
    return g_name;
}


#endif // __APPLE__
