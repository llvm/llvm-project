//===-- PlatformDarwinKernel.cpp -----------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwinKernel.h"

#if defined(__APPLE__) // This Plugin uses the Mac-specific
                       // source/Host/macosx/cfcpp utilities

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/OptionValueFileSpecList.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/FileSystem.h"

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
void PlatformDarwinKernel::Initialize() {
  PlatformDarwin::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(PlatformDarwinKernel::GetPluginNameStatic(),
                                  PlatformDarwinKernel::GetDescriptionStatic(),
                                  PlatformDarwinKernel::CreateInstance,
                                  PlatformDarwinKernel::DebuggerInitialize);
  }
}

void PlatformDarwinKernel::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformDarwinKernel::CreateInstance);
    }
  }

  PlatformDarwin::Terminate();
}

PlatformSP PlatformDarwinKernel::CreateInstance(bool force,
                                                const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    log->Printf("PlatformDarwinKernel::%s(force=%s, arch={%s,%s})",
                __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
  }

  // This is a special plugin that we don't want to activate just based on an
  // ArchSpec for normal
  // userland debugging.  It is only useful in kernel debug sessions and the
  // DynamicLoaderDarwinPlugin
  // (or a user doing 'platform select') will force the creation of this
  // Platform plugin.
  if (force == false) {
    if (log)
      log->Printf("PlatformDarwinKernel::%s() aborting creation of platform "
                  "because force == false",
                  __FUNCTION__);
    return PlatformSP();
  }

  bool create = force;
  LazyBool is_ios_debug_session = eLazyBoolCalculate;

  if (create == false && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::Apple:
      create = true;
      break;

    // Only accept "unknown" for vendor if the host is Apple and
    // it "unknown" wasn't specified (it was just returned because it
    // was NOT specified)
    case llvm::Triple::UnknownArch:
      create = !arch->TripleVendorWasSpecified();
      break;
    default:
      break;
    }

    if (create) {
      switch (triple.getOS()) {
      case llvm::Triple::Darwin:
      case llvm::Triple::MacOSX:
      case llvm::Triple::IOS:
      case llvm::Triple::WatchOS:
      case llvm::Triple::TvOS:
        break;
      // Only accept "vendor" for vendor if the host is Apple and
      // it "unknown" wasn't specified (it was just returned because it
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
  if (arch && arch->IsValid()) {
    switch (arch->GetMachine()) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
      is_ios_debug_session = eLazyBoolNo;
      break;
    case llvm::Triple::arm:
    case llvm::Triple::aarch64:
    case llvm::Triple::thumb:
      is_ios_debug_session = eLazyBoolYes;
      break;
    default:
      is_ios_debug_session = eLazyBoolCalculate;
      break;
    }
  }
  if (create) {
    if (log)
      log->Printf("PlatformDarwinKernel::%s() creating platform", __FUNCTION__);

    return PlatformSP(new PlatformDarwinKernel(is_ios_debug_session));
  }

  if (log)
    log->Printf("PlatformDarwinKernel::%s() aborting creation of platform",
                __FUNCTION__);

  return PlatformSP();
}

lldb_private::ConstString PlatformDarwinKernel::GetPluginNameStatic() {
  static ConstString g_name("darwin-kernel");
  return g_name;
}

const char *PlatformDarwinKernel::GetDescriptionStatic() {
  return "Darwin Kernel platform plug-in.";
}

//------------------------------------------------------------------
/// Code to handle the PlatformDarwinKernel settings
//------------------------------------------------------------------

static PropertyDefinition g_properties[] = {
    {"search-locally-for-kexts", OptionValue::eTypeBoolean, true, true, NULL,
     NULL, "Automatically search for kexts on the local system when doing "
           "kernel debugging."},
    {"kext-directories", OptionValue::eTypeFileSpecList, false, 0, NULL, NULL,
     "Directories/KDKs to search for kexts in when starting a kernel debug "
     "session."},
    {NULL, OptionValue::eTypeInvalid, false, 0, NULL, NULL, NULL}};

enum { ePropertySearchForKexts = 0, ePropertyKextDirectories };

class PlatformDarwinKernelProperties : public Properties {
public:
  static ConstString &GetSettingName() {
    static ConstString g_setting_name("darwin-kernel");
    return g_setting_name;
  }

  PlatformDarwinKernelProperties() : Properties() {
    m_collection_sp.reset(new OptionValueProperties(GetSettingName()));
    m_collection_sp->Initialize(g_properties);
  }

  virtual ~PlatformDarwinKernelProperties() {}

  bool GetSearchForKexts() const {
    const uint32_t idx = ePropertySearchForKexts;
    return m_collection_sp->GetPropertyAtIndexAsBoolean(
        NULL, idx, g_properties[idx].default_uint_value != 0);
  }

  FileSpecList &GetKextDirectories() const {
    const uint32_t idx = ePropertyKextDirectories;
    OptionValueFileSpecList *option_value =
        m_collection_sp->GetPropertyAtIndexAsOptionValueFileSpecList(
            NULL, false, idx);
    assert(option_value);
    return option_value->GetCurrentValue();
  }
};

typedef std::shared_ptr<PlatformDarwinKernelProperties>
    PlatformDarwinKernelPropertiesSP;

static const PlatformDarwinKernelPropertiesSP &GetGlobalProperties() {
  static PlatformDarwinKernelPropertiesSP g_settings_sp;
  if (!g_settings_sp)
    g_settings_sp.reset(new PlatformDarwinKernelProperties());
  return g_settings_sp;
}

void PlatformDarwinKernel::DebuggerInitialize(
    lldb_private::Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(
          debugger, PlatformDarwinKernelProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties()->GetValueProperties(),
        ConstString("Properties for the PlatformDarwinKernel plug-in."),
        is_global_setting);
  }
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformDarwinKernel::PlatformDarwinKernel(
    lldb_private::LazyBool is_ios_debug_session)
    : PlatformDarwin(false), // This is a remote platform
      m_name_to_kext_path_map_with_dsyms(),
      m_name_to_kext_path_map_without_dsyms(), m_search_directories(),
      m_search_directories_no_recursing(), m_kernel_binaries_with_dsyms(),
      m_kernel_binaries_without_dsyms(),
      m_ios_debug_session(is_ios_debug_session)

{
  if (GetGlobalProperties()->GetSearchForKexts()) {
    CollectKextAndKernelDirectories();
    SearchForKextsAndKernelsRecursively();
  }
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformDarwinKernel::~PlatformDarwinKernel() {}

void PlatformDarwinKernel::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);
  strm.Printf(" Debug session type: ");
  if (m_ios_debug_session == eLazyBoolYes)
    strm.Printf("iOS kernel debugging\n");
  else if (m_ios_debug_session == eLazyBoolNo)
    strm.Printf("Mac OS X kernel debugging\n");
  else
    strm.Printf("unknown kernel debugging\n");

  strm.Printf("Directories searched recursively:\n");
  const uint32_t num_kext_dirs = m_search_directories.size();
  for (uint32_t i = 0; i < num_kext_dirs; ++i) {
    strm.Printf("[%d] %s\n", i, m_search_directories[i].GetPath().c_str());
  }

  strm.Printf("Directories not searched recursively:\n");
  const uint32_t num_kext_dirs_no_recursion =
      m_search_directories_no_recursing.size();
  for (uint32_t i = 0; i < num_kext_dirs_no_recursion; i++) {
    strm.Printf("[%d] %s\n", i,
                m_search_directories_no_recursing[i].GetPath().c_str());
  }

  strm.Printf(" Number of kexts with dSYMs indexed: %d\n",
              (int)m_name_to_kext_path_map_with_dsyms.size());
  strm.Printf(" Number of kexts without dSYMs indexed: %d\n",
              (int)m_name_to_kext_path_map_without_dsyms.size());
  strm.Printf(" Number of Kernel binaries with dSYMs indexed: %d\n",
              (int)m_kernel_binaries_with_dsyms.size());
  strm.Printf(" Number of Kernel binaries without dSYMs indexed: %d\n",
              (int)m_kernel_binaries_without_dsyms.size());

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    log->Printf("\nkexts with dSYMs");
    for (auto pos : m_name_to_kext_path_map_with_dsyms) {
      log->Printf("%s", pos.second.GetPath().c_str());
    }
    log->Printf("\nkexts without dSYMs");

    for (auto pos : m_name_to_kext_path_map_without_dsyms) {
      log->Printf("%s", pos.second.GetPath().c_str());
    }
    log->Printf("\nkernels with dSYMS");
    for (auto fs : m_kernel_binaries_with_dsyms) {
      log->Printf("%s", fs.GetPath().c_str());
    }
    log->Printf("\nkernels without dSYMS");
    for (auto fs : m_kernel_binaries_without_dsyms) {
      log->Printf("%s", fs.GetPath().c_str());
    }
    log->Printf("\n");
  }
}

// Populate the m_search_directories vector with directories we should search
// for kernel & kext binaries.

void PlatformDarwinKernel::CollectKextAndKernelDirectories() {
  // Differentiate between "ios debug session" and "mac debug session" so we
  // don't index
  // kext bundles that won't be used in this debug session.  If this is an ios
  // kext debug
  // session, looking in /System/Library/Extensions is a waste of stat()s, for
  // example.

  // DeveloperDirectory is something like
  // "/Applications/Xcode.app/Contents/Developer"
  std::string developer_dir = GetDeveloperDirectory();
  if (developer_dir.empty())
    developer_dir = "/Applications/Xcode.app/Contents/Developer";

  if (m_ios_debug_session != eLazyBoolNo) {
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/iPhoneOS.platform/Developer/SDKs");
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/AppleTVOS.platform/Developer/SDKs");
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/WatchOS.platform/Developer/SDKs");
  }
  if (m_ios_debug_session != eLazyBoolYes) {
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/MacOSX.platform/Developer/SDKs");
  }

  AddSDKSubdirsToSearchPaths("/Volumes/KernelDebugKit");
  AddSDKSubdirsToSearchPaths("/AppleInternal/Developer/KDKs");
  // The KDKs distributed from Apple installed on external
  // developer systems may be in directories like
  // /Library/Developer/KDKs/KDK_10.10_14A298i.kdk
  AddSDKSubdirsToSearchPaths("/Library/Developer/KDKs");

  if (m_ios_debug_session != eLazyBoolNo) {
  }
  if (m_ios_debug_session != eLazyBoolYes) {
    AddRootSubdirsToSearchPaths(this, "/");
  }

  GetUserSpecifiedDirectoriesToSearch();

  // Add simple directory /Applications/Xcode.app/Contents/Developer/../Symbols
  FileSpec possible_dir(developer_dir + "/../Symbols", true);
  if (llvm::sys::fs::is_directory(possible_dir.GetPath()))
    m_search_directories.push_back(possible_dir);

  // Add simple directory of the current working directory
  m_search_directories_no_recursing.push_back(FileSpec(".", true));
}

void PlatformDarwinKernel::GetUserSpecifiedDirectoriesToSearch() {
  FileSpecList user_dirs(GetGlobalProperties()->GetKextDirectories());
  std::vector<FileSpec> possible_sdk_dirs;

  const uint32_t user_dirs_count = user_dirs.GetSize();
  for (uint32_t i = 0; i < user_dirs_count; i++) {
    FileSpec dir = user_dirs.GetFileSpecAtIndex(i);
    dir.ResolvePath();
    if (llvm::sys::fs::is_directory(dir.GetPath())) {
      m_search_directories.push_back(dir);
    }
  }
}

void PlatformDarwinKernel::AddRootSubdirsToSearchPaths(
    PlatformDarwinKernel *thisp, const std::string &dir) {
  const char *subdirs[] = {
      "/System/Library/Extensions", "/Library/Extensions",
      "/System/Library/Kernels",
      "/System/Library/Extensions/KDK", // this one probably only exist in
                                        // /AppleInternal/Developer/KDKs/*.kdk/...
      nullptr};
  for (int i = 0; subdirs[i] != nullptr; i++) {
    FileSpec testdir(dir + subdirs[i], true);
    if (llvm::sys::fs::is_directory(testdir.GetPath()))
      thisp->m_search_directories.push_back(testdir);
  }

  // Look for kernel binaries in the top level directory, without any recursion
  thisp->m_search_directories_no_recursing.push_back(
      FileSpec(dir + "/", false));
}

// Given a directory path dir, look for any subdirs named *.kdk and *.sdk
void PlatformDarwinKernel::AddSDKSubdirsToSearchPaths(const std::string &dir) {
  // Look for *.kdk and *.sdk in dir
  const bool find_directories = true;
  const bool find_files = false;
  const bool find_other = false;
  FileSpec::EnumerateDirectory(dir.c_str(), find_directories, find_files,
                               find_other, FindKDKandSDKDirectoriesInDirectory,
                               this);
}

// Helper function to find *.sdk and *.kdk directories in a given directory.
FileSpec::EnumerateDirectoryResult
PlatformDarwinKernel::FindKDKandSDKDirectoriesInDirectory(
    void *baton, llvm::sys::fs::file_type ft, const FileSpec &file_spec) {
  static ConstString g_sdk_suffix = ConstString("sdk");
  static ConstString g_kdk_suffix = ConstString("kdk");

  PlatformDarwinKernel *thisp = (PlatformDarwinKernel *)baton;
  if (ft == llvm::sys::fs::file_type::directory_file &&
      (file_spec.GetFileNameExtension() == g_sdk_suffix ||
       file_spec.GetFileNameExtension() == g_kdk_suffix)) {
    AddRootSubdirsToSearchPaths(thisp, file_spec.GetPath());
  }
  return FileSpec::eEnumerateDirectoryResultNext;
}

// Recursively search trough m_search_directories looking for
// kext and kernel binaries, adding files found to the appropriate
// lists.
void PlatformDarwinKernel::SearchForKextsAndKernelsRecursively() {
  const uint32_t num_dirs = m_search_directories.size();
  for (uint32_t i = 0; i < num_dirs; i++) {
    const FileSpec &dir = m_search_directories[i];
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true; // I think eFileTypeSymbolicLink are "other"s.
    FileSpec::EnumerateDirectory(
        dir.GetPath().c_str(), find_directories, find_files, find_other,
        GetKernelsAndKextsInDirectoryWithRecursion, this);
  }
  const uint32_t num_dirs_no_recurse = m_search_directories_no_recursing.size();
  for (uint32_t i = 0; i < num_dirs_no_recurse; i++) {
    const FileSpec &dir = m_search_directories_no_recursing[i];
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true; // I think eFileTypeSymbolicLink are "other"s.
    FileSpec::EnumerateDirectory(
        dir.GetPath().c_str(), find_directories, find_files, find_other,
        GetKernelsAndKextsInDirectoryNoRecursion, this);
  }
}

// We're only doing a filename match here.  We won't try opening the file to see
// if it's really
// a kernel or not until we need to find a kernel of a given UUID.  There's no
// cheap way to find
// the UUID of a file (or if it's a Mach-O binary at all) without creating a
// whole Module for
// the file and throwing it away if it's not wanted.
//
// Recurse into any subdirectories found.

FileSpec::EnumerateDirectoryResult
PlatformDarwinKernel::GetKernelsAndKextsInDirectoryWithRecursion(
    void *baton, llvm::sys::fs::file_type ft, const FileSpec &file_spec) {
  return GetKernelsAndKextsInDirectoryHelper(baton, ft, file_spec, true);
}

FileSpec::EnumerateDirectoryResult
PlatformDarwinKernel::GetKernelsAndKextsInDirectoryNoRecursion(
    void *baton, llvm::sys::fs::file_type ft, const FileSpec &file_spec) {
  return GetKernelsAndKextsInDirectoryHelper(baton, ft, file_spec, false);
}

FileSpec::EnumerateDirectoryResult
PlatformDarwinKernel::GetKernelsAndKextsInDirectoryHelper(
    void *baton, llvm::sys::fs::file_type ft, const FileSpec &file_spec,
    bool recurse) {
  static ConstString g_kext_suffix = ConstString("kext");
  static ConstString g_dsym_suffix = ConstString("dSYM");
  static ConstString g_bundle_suffix = ConstString("Bundle");
  ConstString file_spec_extension = file_spec.GetFileNameExtension();

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  Log *log_verbose(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM | LLDB_LOG_OPTION_VERBOSE));

  if (log_verbose)
      log_verbose->Printf ("PlatformDarwinKernel examining '%s'", file_spec.GetPath().c_str());

  PlatformDarwinKernel *thisp = (PlatformDarwinKernel *)baton;
  if (ft == llvm::sys::fs::file_type::regular_file ||
      ft == llvm::sys::fs::file_type::symlink_file) {
    ConstString filename = file_spec.GetFilename();
    if ((strncmp(filename.GetCString(), "kernel", 6) == 0 ||
         strncmp(filename.GetCString(), "mach", 4) == 0) &&
        file_spec_extension != g_dsym_suffix) {
      if (KernelHasdSYMSibling(file_spec))
      {
        if (log)
        {
            log->Printf ("PlatformDarwinKernel registering kernel binary '%s' with dSYM sibling", file_spec.GetPath().c_str());
        }
        thisp->m_kernel_binaries_with_dsyms.push_back(file_spec);
      }
      else
      {
        if (log)
        {
            log->Printf ("PlatformDarwinKernel registering kernel binary '%s', no dSYM", file_spec.GetPath().c_str());
        }
        thisp->m_kernel_binaries_without_dsyms.push_back(file_spec);
      }
      return FileSpec::eEnumerateDirectoryResultNext;
    }
  } else if (ft == llvm::sys::fs::file_type::directory_file &&
             file_spec_extension == g_kext_suffix) {
    AddKextToMap(thisp, file_spec);
    // Look to see if there is a PlugIns subdir with more kexts
    FileSpec contents_plugins(file_spec.GetPath() + "/Contents/PlugIns", false);
    std::string search_here_too;
    if (llvm::sys::fs::is_directory(contents_plugins.GetPath())) {
      search_here_too = contents_plugins.GetPath();
    } else {
      FileSpec plugins(file_spec.GetPath() + "/PlugIns", false);
      if (llvm::sys::fs::is_directory(plugins.GetPath())) {
        search_here_too = plugins.GetPath();
      }
    }

    if (!search_here_too.empty()) {
      const bool find_directories = true;
      const bool find_files = false;
      const bool find_other = false;
      FileSpec::EnumerateDirectory(
          search_here_too.c_str(), find_directories, find_files, find_other,
          recurse ? GetKernelsAndKextsInDirectoryWithRecursion
                  : GetKernelsAndKextsInDirectoryNoRecursion,
          baton);
    }
    return FileSpec::eEnumerateDirectoryResultNext;
  }
  // Don't recurse into dSYM/kext/bundle directories
  if (recurse && file_spec_extension != g_dsym_suffix &&
      file_spec_extension != g_kext_suffix &&
      file_spec_extension != g_bundle_suffix) {
    if (log_verbose)
        log_verbose->Printf ("PlatformDarwinKernel descending into directory '%s'", file_spec.GetPath().c_str());
    return FileSpec::eEnumerateDirectoryResultEnter;
  } else {
    return FileSpec::eEnumerateDirectoryResultNext;
  }
}

void PlatformDarwinKernel::AddKextToMap(PlatformDarwinKernel *thisp,
                                        const FileSpec &file_spec) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  CFCBundle bundle(file_spec.GetPath().c_str());
  CFStringRef bundle_id(bundle.GetIdentifier());
  if (bundle_id && CFGetTypeID(bundle_id) == CFStringGetTypeID()) {
    char bundle_id_buf[PATH_MAX];
    if (CFStringGetCString(bundle_id, bundle_id_buf, sizeof(bundle_id_buf),
                           kCFStringEncodingUTF8)) {
      ConstString bundle_conststr(bundle_id_buf);
      if (KextHasdSYMSibling(file_spec))
      {
        if (log)
        {
            log->Printf ("PlatformDarwinKernel registering kext binary '%s' with dSYM sibling", file_spec.GetPath().c_str());
        }
        thisp->m_name_to_kext_path_map_with_dsyms.insert(
            std::pair<ConstString, FileSpec>(bundle_conststr, file_spec));
      }
      else
      {
        if (log)
        {
            log->Printf ("PlatformDarwinKernel registering kext binary '%s', no dSYM", file_spec.GetPath().c_str());
        }
        thisp->m_name_to_kext_path_map_without_dsyms.insert(
            std::pair<ConstString, FileSpec>(bundle_conststr, file_spec));
      }
    }
  }
}

// Given a FileSpec of /dir/dir/foo.kext
// Return true if any of these exist:
//    /dir/dir/foo.kext.dSYM
//    /dir/dir/foo.kext/Contents/MacOS/foo.dSYM
//    /dir/dir/foo.kext/foo.dSYM
bool PlatformDarwinKernel::KextHasdSYMSibling(
    const FileSpec &kext_bundle_filepath) {
  FileSpec dsym_fspec = kext_bundle_filepath;
  std::string filename = dsym_fspec.GetFilename().AsCString();
  filename += ".dSYM";
  dsym_fspec.GetFilename() = ConstString(filename);
  if (llvm::sys::fs::is_directory(dsym_fspec.GetPath())) {
    return true;
  }
  // Should probably get the CFBundleExecutable here or call
  // CFBundleCopyExecutableURL

  // Look for a deep bundle foramt
  ConstString executable_name =
      kext_bundle_filepath.GetFileNameStrippingExtension();
  std::string deep_bundle_str =
      kext_bundle_filepath.GetPath() + "/Contents/MacOS/";
  deep_bundle_str += executable_name.AsCString();
  deep_bundle_str += ".dSYM";
  dsym_fspec.SetFile(deep_bundle_str, true);
  if (llvm::sys::fs::is_directory(dsym_fspec.GetPath())) {
    return true;
  }

  // look for a shallow bundle format
  //
  std::string shallow_bundle_str = kext_bundle_filepath.GetPath() + "/";
  shallow_bundle_str += executable_name.AsCString();
  shallow_bundle_str += ".dSYM";
  dsym_fspec.SetFile(shallow_bundle_str, true);
  if (llvm::sys::fs::is_directory(dsym_fspec.GetPath())) {
    return true;
  }
  return false;
}

// Given a FileSpec of /dir/dir/mach.development.t7004
// Return true if a dSYM exists next to it:
//    /dir/dir/mach.development.t7004.dSYM
bool PlatformDarwinKernel::KernelHasdSYMSibling(const FileSpec &kernel_binary) {
  FileSpec kernel_dsym = kernel_binary;
  std::string filename = kernel_binary.GetFilename().AsCString();
  filename += ".dSYM";
  kernel_dsym.GetFilename() = ConstString(filename);
  if (llvm::sys::fs::is_directory(kernel_dsym.GetPath())) {
    return true;
  }
  return false;
}

Error PlatformDarwinKernel::GetSharedModule(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr, ModuleSP *old_module_sp_ptr,
    bool *did_create_ptr) {
  Error error;
  module_sp.reset();
  const FileSpec &platform_file = module_spec.GetFileSpec();

  // Treat the file's path as a kext bundle ID (e.g.
  // "com.apple.driver.AppleIRController") and search our kext index.
  std::string kext_bundle_id = platform_file.GetPath();
  if (!kext_bundle_id.empty()) {
    ConstString kext_bundle_cs(kext_bundle_id.c_str());

    // First look through the kext bundles that had a dsym next to them
    if (m_name_to_kext_path_map_with_dsyms.count(kext_bundle_cs) > 0) {
      for (BundleIDToKextIterator it =
               m_name_to_kext_path_map_with_dsyms.begin();
           it != m_name_to_kext_path_map_with_dsyms.end(); ++it) {
        if (it->first == kext_bundle_cs) {
          error = ExamineKextForMatchingUUID(it->second, module_spec.GetUUID(),
                                             module_spec.GetArchitecture(),
                                             module_sp);
          if (module_sp.get()) {
            return error;
          }
        }
      }
    }

    // Second look through the kext binarys without dSYMs
    if (m_name_to_kext_path_map_without_dsyms.count(kext_bundle_cs) > 0) {
      for (BundleIDToKextIterator it =
               m_name_to_kext_path_map_without_dsyms.begin();
           it != m_name_to_kext_path_map_without_dsyms.end(); ++it) {
        if (it->first == kext_bundle_cs) {
          error = ExamineKextForMatchingUUID(it->second, module_spec.GetUUID(),
                                             module_spec.GetArchitecture(),
                                             module_sp);
          if (module_sp.get()) {
            return error;
          }
        }
      }
    }
  }

  if (kext_bundle_id.compare("mach_kernel") == 0 &&
      module_spec.GetUUID().IsValid()) {
    // First try all kernel binaries that have a dSYM next to them
    for (auto possible_kernel : m_kernel_binaries_with_dsyms) {
      if (possible_kernel.Exists()) {
        ModuleSpec kern_spec(possible_kernel);
        kern_spec.GetUUID() = module_spec.GetUUID();
        ModuleSP module_sp(new Module(kern_spec));
        if (module_sp && module_sp->GetObjectFile() &&
            module_sp->MatchesModuleSpec(kern_spec)) {
          // module_sp is an actual kernel binary we want to add.
          if (process) {
            process->GetTarget().GetImages().AppendIfNeeded(module_sp);
            error.Clear();
            return error;
          } else {
            error = ModuleList::GetSharedModule(kern_spec, module_sp, NULL,
                                                NULL, NULL);
            if (module_sp && module_sp->GetObjectFile() &&
                module_sp->GetObjectFile()->GetType() !=
                    ObjectFile::Type::eTypeCoreFile) {
              return error;
            }
            module_sp.reset();
          }
        }
      }
    }
    // Second try all kernel binaries that don't have a dSYM
    for (auto possible_kernel : m_kernel_binaries_without_dsyms) {
      if (possible_kernel.Exists()) {
        ModuleSpec kern_spec(possible_kernel);
        kern_spec.GetUUID() = module_spec.GetUUID();
        ModuleSP module_sp(new Module(kern_spec));
        if (module_sp && module_sp->GetObjectFile() &&
            module_sp->MatchesModuleSpec(kern_spec)) {
          // module_sp is an actual kernel binary we want to add.
          if (process) {
            process->GetTarget().GetImages().AppendIfNeeded(module_sp);
            error.Clear();
            return error;
          } else {
            error = ModuleList::GetSharedModule(kern_spec, module_sp, NULL,
                                                NULL, NULL);
            if (module_sp && module_sp->GetObjectFile() &&
                module_sp->GetObjectFile()->GetType() !=
                    ObjectFile::Type::eTypeCoreFile) {
              return error;
            }
            module_sp.reset();
          }
        }
      }
    }
  }

  // Else fall back to treating the file's path as an actual file path - defer
  // to PlatformDarwin's GetSharedModule.
  return PlatformDarwin::GetSharedModule(module_spec, process, module_sp,
                                         module_search_paths_ptr,
                                         old_module_sp_ptr, did_create_ptr);
}

Error PlatformDarwinKernel::ExamineKextForMatchingUUID(
    const FileSpec &kext_bundle_path, const lldb_private::UUID &uuid,
    const ArchSpec &arch, ModuleSP &exe_module_sp) {
  Error error;
  FileSpec exe_file = kext_bundle_path;
  Host::ResolveExecutableInBundle(exe_file);
  if (exe_file.Exists()) {
    ModuleSpec exe_spec(exe_file);
    exe_spec.GetUUID() = uuid;
    if (!uuid.IsValid()) {
      exe_spec.GetArchitecture() = arch;
    }

    // First try to create a ModuleSP with the file / arch and see if the UUID
    // matches.
    // If that fails (this exec file doesn't have the correct uuid), don't call
    // GetSharedModule
    // (which may call in to the DebugSymbols framework and therefore can be
    // slow.)
    ModuleSP module_sp(new Module(exe_spec));
    if (module_sp && module_sp->GetObjectFile() &&
        module_sp->MatchesModuleSpec(exe_spec)) {
      error = ModuleList::GetSharedModule(exe_spec, exe_module_sp, NULL, NULL,
                                          NULL);
      if (exe_module_sp && exe_module_sp->GetObjectFile()) {
        return error;
      }
    }
    exe_module_sp.reset();
  }
  return error;
}

bool PlatformDarwinKernel::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                           ArchSpec &arch) {
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
  return ARMGetSupportedArchitectureAtIndex(idx, arch);
#else
  return x86GetSupportedArchitectureAtIndex(idx, arch);
#endif
}

void PlatformDarwinKernel::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("trap_from_kernel"));
  m_trap_handlers.push_back(ConstString("hndl_machine_check"));
  m_trap_handlers.push_back(ConstString("hndl_double_fault"));
  m_trap_handlers.push_back(ConstString("hndl_allintrs"));
  m_trap_handlers.push_back(ConstString("hndl_alltraps"));
  m_trap_handlers.push_back(ConstString("interrupt"));
  m_trap_handlers.push_back(ConstString("fleh_prefabt"));
  m_trap_handlers.push_back(ConstString("ExceptionVectorsBase"));
  m_trap_handlers.push_back(ConstString("ExceptionVectorsTable"));
  m_trap_handlers.push_back(ConstString("fleh_undef"));
  m_trap_handlers.push_back(ConstString("fleh_dataabt"));
  m_trap_handlers.push_back(ConstString("fleh_irq"));
  m_trap_handlers.push_back(ConstString("fleh_decirq"));
  m_trap_handlers.push_back(ConstString("fleh_fiq_generic"));
  m_trap_handlers.push_back(ConstString("fleh_dec"));
}

#else // __APPLE__

// Since DynamicLoaderDarwinKernel is compiled in for all systems, and relies on
// PlatformDarwinKernel for the plug-in name, we compile just the plug-in name
// in
// here to avoid issues. We are tracking an internal bug to resolve this issue
// by
// either not compiling in DynamicLoaderDarwinKernel for non-apple builds, or to
// make
// PlatformDarwinKernel build on all systems. PlatformDarwinKernel is currently
// not
// compiled on other platforms due to the use of the Mac-specific
// source/Host/macosx/cfcpp utilities.

lldb_private::ConstString PlatformDarwinKernel::GetPluginNameStatic() {
  static lldb_private::ConstString g_name("darwin-kernel");
  return g_name;
}

#endif // __APPLE__
