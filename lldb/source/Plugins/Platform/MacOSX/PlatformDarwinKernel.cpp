//===-- PlatformDarwinKernel.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwinKernel.h"

#if defined(__APPLE__) // This Plugin uses the Mac-specific
                       // source/Host/macosx/cfcpp utilities

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/OptionValueFileSpecList.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/FileSystem.h"

#include <CoreFoundation/CoreFoundation.h>

#include <memory>

#include "Host/macosx/cfcpp/CFCBundle.h"

using namespace lldb;
using namespace lldb_private;

// Static Variables
static uint32_t g_initialize_count = 0;

// Static Functions
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

    LLDB_LOGF(log, "PlatformDarwinKernel::%s(force=%s, arch={%s,%s})",
              __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
  }

  // This is a special plugin that we don't want to activate just based on an
  // ArchSpec for normal userland debugging.  It is only useful in kernel debug
  // sessions and the DynamicLoaderDarwinPlugin (or a user doing 'platform
  // select') will force the creation of this Platform plugin.
  if (!force) {
    LLDB_LOGF(log,
              "PlatformDarwinKernel::%s() aborting creation of platform "
              "because force == false",
              __FUNCTION__);
    return PlatformSP();
  }

  bool create = force;
  LazyBool is_ios_debug_session = eLazyBoolCalculate;

  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::Apple:
      create = true;
      break;

    // Only accept "unknown" for vendor if the host is Apple and it "unknown"
    // wasn't specified (it was just returned because it was NOT specified)
    case llvm::Triple::UnknownVendor:
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
      // NEED_BRIDGEOS_TRIPLE case llvm::Triple::BridgeOS:
        break;
      // Only accept "vendor" for vendor if the host is Apple and it "unknown"
      // wasn't specified (it was just returned because it was NOT specified)
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
    LLDB_LOGF(log, "PlatformDarwinKernel::%s() creating platform",
              __FUNCTION__);

    return PlatformSP(new PlatformDarwinKernel(is_ios_debug_session));
  }

  LLDB_LOGF(log, "PlatformDarwinKernel::%s() aborting creation of platform",
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

/// Code to handle the PlatformDarwinKernel settings

#define LLDB_PROPERTIES_platformdarwinkernel
#include "PlatformMacOSXProperties.inc"

enum {
#define LLDB_PROPERTIES_platformdarwinkernel
#include "PlatformMacOSXPropertiesEnum.inc"
};

class PlatformDarwinKernelProperties : public Properties {
public:
  static ConstString &GetSettingName() {
    static ConstString g_setting_name("darwin-kernel");
    return g_setting_name;
  }

  PlatformDarwinKernelProperties() : Properties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_platformdarwinkernel_properties);
  }

  virtual ~PlatformDarwinKernelProperties() {}

  FileSpecList GetKextDirectories() const {
    const uint32_t idx = ePropertyKextDirectories;
    const OptionValueFileSpecList *option_value =
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
    g_settings_sp = std::make_shared<PlatformDarwinKernelProperties>();
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

/// Default Constructor
PlatformDarwinKernel::PlatformDarwinKernel(
    lldb_private::LazyBool is_ios_debug_session)
    : PlatformDarwin(false), // This is a remote platform
      m_name_to_kext_path_map_with_dsyms(),
      m_name_to_kext_path_map_without_dsyms(), m_search_directories(),
      m_search_directories_no_recursing(), m_kernel_binaries_with_dsyms(),
      m_kernel_binaries_without_dsyms(), m_kernel_dsyms_no_binaries(),
      m_kernel_dsyms_yaas(), m_ios_debug_session(is_ios_debug_session)

{
  CollectKextAndKernelDirectories();
  SearchForKextsAndKernelsRecursively();
}

/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
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
  strm.Printf(" Number of Kernel dSYMs with no binaries indexed: %d\n",
              (int)m_kernel_dsyms_no_binaries.size());
  strm.Printf(" Number of Kernel dSYM.yaa's indexed: %d\n",
              (int)m_kernel_dsyms_yaas.size());

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    LLDB_LOGF(log, "\nkexts with dSYMs");
    for (auto pos : m_name_to_kext_path_map_with_dsyms) {
      LLDB_LOGF(log, "%s", pos.second.GetPath().c_str());
    }
    LLDB_LOGF(log, "\nkexts without dSYMs");

    for (auto pos : m_name_to_kext_path_map_without_dsyms) {
      LLDB_LOGF(log, "%s", pos.second.GetPath().c_str());
    }
    LLDB_LOGF(log, "\nkernel binaries with dSYMS");
    for (auto fs : m_kernel_binaries_with_dsyms) {
      LLDB_LOGF(log, "%s", fs.GetPath().c_str());
    }
    LLDB_LOGF(log, "\nkernel binaries without dSYMS");
    for (auto fs : m_kernel_binaries_without_dsyms) {
      LLDB_LOGF(log, "%s", fs.GetPath().c_str());
    }
    LLDB_LOGF(log, "\nkernel dSYMS with no binaries");
    for (auto fs : m_kernel_dsyms_no_binaries) {
      LLDB_LOGF(log, "%s", fs.GetPath().c_str());
    }
    LLDB_LOGF(log, "\nkernels .dSYM.yaa's");
    for (auto fs : m_kernel_dsyms_yaas) {
      LLDB_LOGF(log, "%s", fs.GetPath().c_str());
    }
    LLDB_LOGF(log, "\n");
  }
}

// Populate the m_search_directories vector with directories we should search
// for kernel & kext binaries.

void PlatformDarwinKernel::CollectKextAndKernelDirectories() {
  // Differentiate between "ios debug session" and "mac debug session" so we
  // don't index kext bundles that won't be used in this debug session.  If
  // this is an ios kext debug session, looking in /System/Library/Extensions
  // is a waste of stat()s, for example.

  // DeveloperDirectory is something like
  // "/Applications/Xcode.app/Contents/Developer"
  std::string developer_dir = HostInfo::GetXcodeDeveloperDirectory().GetPath();
  if (developer_dir.empty())
    developer_dir = "/Applications/Xcode.app/Contents/Developer";

  if (m_ios_debug_session != eLazyBoolNo) {
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/iPhoneOS.platform/Developer/SDKs");
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/AppleTVOS.platform/Developer/SDKs");
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/WatchOS.platform/Developer/SDKs");
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/BridgeOS.platform/Developer/SDKs");
  }
  if (m_ios_debug_session != eLazyBoolYes) {
    AddSDKSubdirsToSearchPaths(developer_dir +
                               "/Platforms/MacOSX.platform/Developer/SDKs");
  }

  AddSDKSubdirsToSearchPaths("/Volumes/KernelDebugKit");
  AddSDKSubdirsToSearchPaths("/AppleInternal/Developer/KDKs");
  // The KDKs distributed from Apple installed on external developer systems
  // may be in directories like /Library/Developer/KDKs/KDK_10.10_14A298i.kdk
  AddSDKSubdirsToSearchPaths("/Library/Developer/KDKs");

  if (m_ios_debug_session != eLazyBoolNo) {
  }
  if (m_ios_debug_session != eLazyBoolYes) {
    AddRootSubdirsToSearchPaths(this, "/");
  }

  GetUserSpecifiedDirectoriesToSearch();

  // Add simple directory /Applications/Xcode.app/Contents/Developer/../Symbols
  FileSpec possible_dir(developer_dir + "/../Symbols");
  FileSystem::Instance().Resolve(possible_dir);
  if (FileSystem::Instance().IsDirectory(possible_dir))
    m_search_directories.push_back(possible_dir);

  // Add simple directory of the current working directory
  FileSpec cwd(".");
  FileSystem::Instance().Resolve(cwd);
  m_search_directories_no_recursing.push_back(cwd);
}

void PlatformDarwinKernel::GetUserSpecifiedDirectoriesToSearch() {
  FileSpecList user_dirs(GetGlobalProperties()->GetKextDirectories());
  std::vector<FileSpec> possible_sdk_dirs;

  const uint32_t user_dirs_count = user_dirs.GetSize();
  for (uint32_t i = 0; i < user_dirs_count; i++) {
    FileSpec dir = user_dirs.GetFileSpecAtIndex(i);
    FileSystem::Instance().Resolve(dir);
    if (FileSystem::Instance().IsDirectory(dir)) {
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
    FileSpec testdir(dir + subdirs[i]);
    FileSystem::Instance().Resolve(testdir);
    if (FileSystem::Instance().IsDirectory(testdir))
      thisp->m_search_directories.push_back(testdir);
  }

  // Look for kernel binaries in the top level directory, without any recursion
  thisp->m_search_directories_no_recursing.push_back(FileSpec(dir + "/"));
}

// Given a directory path dir, look for any subdirs named *.kdk and *.sdk
void PlatformDarwinKernel::AddSDKSubdirsToSearchPaths(const std::string &dir) {
  // Look for *.kdk and *.sdk in dir
  const bool find_directories = true;
  const bool find_files = false;
  const bool find_other = false;
  FileSystem::Instance().EnumerateDirectory(
      dir.c_str(), find_directories, find_files, find_other,
      FindKDKandSDKDirectoriesInDirectory, this);
}

// Helper function to find *.sdk and *.kdk directories in a given directory.
FileSystem::EnumerateDirectoryResult
PlatformDarwinKernel::FindKDKandSDKDirectoriesInDirectory(
    void *baton, llvm::sys::fs::file_type ft, llvm::StringRef path) {
  static ConstString g_sdk_suffix = ConstString(".sdk");
  static ConstString g_kdk_suffix = ConstString(".kdk");

  PlatformDarwinKernel *thisp = (PlatformDarwinKernel *)baton;
  FileSpec file_spec(path);
  if (ft == llvm::sys::fs::file_type::directory_file &&
      (file_spec.GetFileNameExtension() == g_sdk_suffix ||
       file_spec.GetFileNameExtension() == g_kdk_suffix)) {
    AddRootSubdirsToSearchPaths(thisp, file_spec.GetPath());
  }
  return FileSystem::eEnumerateDirectoryResultNext;
}

// Recursively search trough m_search_directories looking for kext and kernel
// binaries, adding files found to the appropriate lists.
void PlatformDarwinKernel::SearchForKextsAndKernelsRecursively() {
  const uint32_t num_dirs = m_search_directories.size();
  for (uint32_t i = 0; i < num_dirs; i++) {
    const FileSpec &dir = m_search_directories[i];
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true; // I think eFileTypeSymbolicLink are "other"s.
    FileSystem::Instance().EnumerateDirectory(
        dir.GetPath().c_str(), find_directories, find_files, find_other,
        GetKernelsAndKextsInDirectoryWithRecursion, this);
  }
  const uint32_t num_dirs_no_recurse = m_search_directories_no_recursing.size();
  for (uint32_t i = 0; i < num_dirs_no_recurse; i++) {
    const FileSpec &dir = m_search_directories_no_recursing[i];
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true; // I think eFileTypeSymbolicLink are "other"s.
    FileSystem::Instance().EnumerateDirectory(
        dir.GetPath().c_str(), find_directories, find_files, find_other,
        GetKernelsAndKextsInDirectoryNoRecursion, this);
  }
}

// We're only doing a filename match here.  We won't try opening the file to
// see if it's really a kernel or not until we need to find a kernel of a given
// UUID.  There's no cheap way to find the UUID of a file (or if it's a Mach-O
// binary at all) without creating a whole Module for the file and throwing it
// away if it's not wanted.
//
// Recurse into any subdirectories found.

FileSystem::EnumerateDirectoryResult
PlatformDarwinKernel::GetKernelsAndKextsInDirectoryWithRecursion(
    void *baton, llvm::sys::fs::file_type ft, llvm::StringRef path) {
  return GetKernelsAndKextsInDirectoryHelper(baton, ft, path, true);
}

FileSystem::EnumerateDirectoryResult
PlatformDarwinKernel::GetKernelsAndKextsInDirectoryNoRecursion(
    void *baton, llvm::sys::fs::file_type ft, llvm::StringRef path) {
  return GetKernelsAndKextsInDirectoryHelper(baton, ft, path, false);
}

FileSystem::EnumerateDirectoryResult
PlatformDarwinKernel::GetKernelsAndKextsInDirectoryHelper(
    void *baton, llvm::sys::fs::file_type ft, llvm::StringRef path,
    bool recurse) {
  static ConstString g_kext_suffix = ConstString(".kext");
  static ConstString g_dsym_suffix = ConstString(".dSYM");
  static ConstString g_bundle_suffix = ConstString("Bundle");

  FileSpec file_spec(path);
  ConstString file_spec_extension = file_spec.GetFileNameExtension();

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  Log *log_verbose(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM | LLDB_LOG_OPTION_VERBOSE));

  LLDB_LOGF(log_verbose, "PlatformDarwinKernel examining '%s'",
            file_spec.GetPath().c_str());

  PlatformDarwinKernel *thisp = (PlatformDarwinKernel *)baton;

  llvm::StringRef filename = file_spec.GetFilename().GetStringRef();
  bool is_kernel_filename =
      filename.startswith("kernel") || filename.startswith("mach");
  bool is_dsym_yaa = filename.endswith(".dSYM.yaa");

  if (ft == llvm::sys::fs::file_type::regular_file ||
      ft == llvm::sys::fs::file_type::symlink_file) {
    if (is_kernel_filename) {
      if (file_spec_extension != g_dsym_suffix && !is_dsym_yaa) {
        if (KernelHasdSYMSibling(file_spec)) {
          LLDB_LOGF(log,
                    "PlatformDarwinKernel registering kernel binary '%s' with "
                    "dSYM sibling",
                    file_spec.GetPath().c_str());
          thisp->m_kernel_binaries_with_dsyms.push_back(file_spec);
        } else {
          LLDB_LOGF(
              log,
              "PlatformDarwinKernel registering kernel binary '%s', no dSYM",
              file_spec.GetPath().c_str());
          thisp->m_kernel_binaries_without_dsyms.push_back(file_spec);
        }
      }
      if (is_dsym_yaa) {
        LLDB_LOGF(log, "PlatformDarwinKernel registering kernel .dSYM.yaa '%s'",
                  file_spec.GetPath().c_str());
        thisp->m_kernel_dsyms_yaas.push_back(file_spec);
      }
      return FileSystem::eEnumerateDirectoryResultNext;
    }
  } else {
    if (ft == llvm::sys::fs::file_type::directory_file) {
      if (file_spec_extension == g_kext_suffix) {
        AddKextToMap(thisp, file_spec);
        // Look to see if there is a PlugIns subdir with more kexts
        FileSpec contents_plugins(file_spec.GetPath() + "/Contents/PlugIns");
        std::string search_here_too;
        if (FileSystem::Instance().IsDirectory(contents_plugins)) {
          search_here_too = contents_plugins.GetPath();
        } else {
          FileSpec plugins(file_spec.GetPath() + "/PlugIns");
          if (FileSystem::Instance().IsDirectory(plugins)) {
            search_here_too = plugins.GetPath();
          }
        }

        if (!search_here_too.empty()) {
          const bool find_directories = true;
          const bool find_files = false;
          const bool find_other = false;
          FileSystem::Instance().EnumerateDirectory(
              search_here_too.c_str(), find_directories, find_files, find_other,
              recurse ? GetKernelsAndKextsInDirectoryWithRecursion
                      : GetKernelsAndKextsInDirectoryNoRecursion,
              baton);
        }
        return FileSystem::eEnumerateDirectoryResultNext;
      }
      // Do we have a kernel dSYM with no kernel binary?
      if (is_kernel_filename && file_spec_extension == g_dsym_suffix) {
        if (KerneldSYMHasNoSiblingBinary(file_spec)) {
          LLDB_LOGF(log,
                    "PlatformDarwinKernel registering kernel dSYM '%s' with "
                    "no binary sibling",
                    file_spec.GetPath().c_str());
          thisp->m_kernel_dsyms_no_binaries.push_back(file_spec);
          return FileSystem::eEnumerateDirectoryResultNext;
        }
      }
    }
  }

  // Don't recurse into dSYM/kext/bundle directories
  if (recurse && file_spec_extension != g_dsym_suffix &&
      file_spec_extension != g_kext_suffix &&
      file_spec_extension != g_bundle_suffix) {
    LLDB_LOGF(log_verbose,
              "PlatformDarwinKernel descending into directory '%s'",
              file_spec.GetPath().c_str());
    return FileSystem::eEnumerateDirectoryResultEnter;
  } else {
    return FileSystem::eEnumerateDirectoryResultNext;
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
        LLDB_LOGF(log,
                  "PlatformDarwinKernel registering kext binary '%s' with dSYM "
                  "sibling",
                  file_spec.GetPath().c_str());
        thisp->m_name_to_kext_path_map_with_dsyms.insert(
            std::pair<ConstString, FileSpec>(bundle_conststr, file_spec));
      }
      else
      {
        LLDB_LOGF(log,
                  "PlatformDarwinKernel registering kext binary '%s', no dSYM",
                  file_spec.GetPath().c_str());
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
  if (FileSystem::Instance().IsDirectory(dsym_fspec)) {
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
  dsym_fspec.SetFile(deep_bundle_str, FileSpec::Style::native);
  FileSystem::Instance().Resolve(dsym_fspec);
  if (FileSystem::Instance().IsDirectory(dsym_fspec)) {
    return true;
  }

  // look for a shallow bundle format
  //
  std::string shallow_bundle_str = kext_bundle_filepath.GetPath() + "/";
  shallow_bundle_str += executable_name.AsCString();
  shallow_bundle_str += ".dSYM";
  dsym_fspec.SetFile(shallow_bundle_str, FileSpec::Style::native);
  FileSystem::Instance().Resolve(dsym_fspec);
  return FileSystem::Instance().IsDirectory(dsym_fspec);
}

// Given a FileSpec of /dir/dir/mach.development.t7004 Return true if a dSYM
// exists next to it:
//    /dir/dir/mach.development.t7004.dSYM
bool PlatformDarwinKernel::KernelHasdSYMSibling(const FileSpec &kernel_binary) {
  FileSpec kernel_dsym = kernel_binary;
  std::string filename = kernel_binary.GetFilename().AsCString();
  filename += ".dSYM";
  kernel_dsym.GetFilename() = ConstString(filename);
  return FileSystem::Instance().IsDirectory(kernel_dsym);
}

// Given a FileSpec of /dir/dir/mach.development.t7004.dSYM
// Return true if only the dSYM exists, no binary next to it.
//    /dir/dir/mach.development.t7004.dSYM
//    but no
//    /dir/dir/mach.development.t7004
bool PlatformDarwinKernel::KerneldSYMHasNoSiblingBinary(
    const FileSpec &kernel_dsym) {
  static ConstString g_dsym_suffix = ConstString(".dSYM");
  std::string possible_path = kernel_dsym.GetPath();
  if (kernel_dsym.GetFileNameExtension() != g_dsym_suffix)
    return false;

  FileSpec binary_filespec = kernel_dsym;
  // Chop off the '.dSYM' extension on the filename
  binary_filespec.GetFilename() =
      binary_filespec.GetFileNameStrippingExtension();

  // Is there a binary next to this this?  Then return false.
  if (FileSystem::Instance().Exists(binary_filespec))
    return false;

  // If we have at least one binary in the DWARF subdir, then
  // this is a properly formed dSYM and it has no binary next
  // to it.
  if (GetDWARFBinaryInDSYMBundle(kernel_dsym).size() > 0)
    return true;

  return false;
}

// TODO: This method returns a vector of FileSpec's because a
// dSYM bundle may contain multiple DWARF binaries, but it
// only implements returning the base name binary for now;
// it should iterate over every binary in the DWARF subdir
// and return them all.
std::vector<FileSpec>
PlatformDarwinKernel::GetDWARFBinaryInDSYMBundle(FileSpec dsym_bundle) {
  std::vector<FileSpec> results;
  static ConstString g_dsym_suffix = ConstString(".dSYM");
  if (dsym_bundle.GetFileNameExtension() != g_dsym_suffix) {
    return results;
  }
  // Drop the '.dSYM' from the filename
  std::string filename =
      dsym_bundle.GetFileNameStrippingExtension().GetCString();
  std::string dirname = dsym_bundle.GetDirectory().GetCString();

  std::string binary_filepath = dsym_bundle.GetPath();
  binary_filepath += "/Contents/Resources/DWARF/";
  binary_filepath += filename;

  FileSpec binary_fspec(binary_filepath);
  if (FileSystem::Instance().Exists(binary_fspec))
    results.push_back(binary_fspec);
  return results;
}

Status PlatformDarwinKernel::GetSharedModule(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr,
    llvm::SmallVectorImpl<ModuleSP> *old_modules, bool *did_create_ptr) {
  Status error;
  module_sp.reset();
  const FileSpec &platform_file = module_spec.GetFileSpec();

  // Treat the file's path as a kext bundle ID (e.g.
  // "com.apple.driver.AppleIRController") and search our kext index.
  std::string kext_bundle_id = platform_file.GetPath();

  if (!kext_bundle_id.empty() && module_spec.GetUUID().IsValid()) {
    if (kext_bundle_id == "mach_kernel") {
      return GetSharedModuleKernel(module_spec, process, module_sp,
                                   module_search_paths_ptr, old_modules,
                                   did_create_ptr);
    } else {
      return GetSharedModuleKext(module_spec, process, module_sp,
                                 module_search_paths_ptr, old_modules,
                                 did_create_ptr);
    }
  } else {
    // Give the generic methods, including possibly calling into  DebugSymbols
    // framework on macOS systems, a chance.
    return PlatformDarwin::GetSharedModule(module_spec, process, module_sp,
                                           module_search_paths_ptr, old_modules,
                                           did_create_ptr);
  }
}

Status PlatformDarwinKernel::GetSharedModuleKext(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr,
    llvm::SmallVectorImpl<ModuleSP> *old_modules, bool *did_create_ptr) {
  Status error;
  module_sp.reset();
  const FileSpec &platform_file = module_spec.GetFileSpec();

  // Treat the file's path as a kext bundle ID (e.g.
  // "com.apple.driver.AppleIRController") and search our kext index.
  ConstString kext_bundle(platform_file.GetPath().c_str());
  // First look through the kext bundles that had a dsym next to them
  if (m_name_to_kext_path_map_with_dsyms.count(kext_bundle) > 0) {
    for (BundleIDToKextIterator it = m_name_to_kext_path_map_with_dsyms.begin();
         it != m_name_to_kext_path_map_with_dsyms.end(); ++it) {
      if (it->first == kext_bundle) {
        error = ExamineKextForMatchingUUID(it->second, module_spec.GetUUID(),
                                           module_spec.GetArchitecture(),
                                           module_sp);
        if (module_sp.get()) {
          return error;
        }
      }
    }
  }

  // Give the generic methods, including possibly calling into  DebugSymbols
  // framework on macOS systems, a chance.
  error = PlatformDarwin::GetSharedModule(module_spec, process, module_sp,
                                          module_search_paths_ptr, old_modules,
                                          did_create_ptr);
  if (error.Success() && module_sp.get()) {
    return error;
  }

  // Lastly, look through the kext binarys without dSYMs
  if (m_name_to_kext_path_map_without_dsyms.count(kext_bundle) > 0) {
    for (BundleIDToKextIterator it =
             m_name_to_kext_path_map_without_dsyms.begin();
         it != m_name_to_kext_path_map_without_dsyms.end(); ++it) {
      if (it->first == kext_bundle) {
        error = ExamineKextForMatchingUUID(it->second, module_spec.GetUUID(),
                                           module_spec.GetArchitecture(),
                                           module_sp);
        if (module_sp.get()) {
          return error;
        }
      }
    }
  }
  return error;
}

Status PlatformDarwinKernel::GetSharedModuleKernel(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr,
    llvm::SmallVectorImpl<ModuleSP> *old_modules, bool *did_create_ptr) {
  Status error;
  module_sp.reset();

  // First try all kernel binaries that have a dSYM next to them
  for (auto possible_kernel : m_kernel_binaries_with_dsyms) {
    if (FileSystem::Instance().Exists(possible_kernel)) {
      ModuleSpec kern_spec(possible_kernel);
      kern_spec.GetUUID() = module_spec.GetUUID();
      module_sp.reset(new Module(kern_spec));
      if (module_sp && module_sp->GetObjectFile() &&
          module_sp->MatchesModuleSpec(kern_spec)) {
        // module_sp is an actual kernel binary we want to add.
        if (process) {
          process->GetTarget().GetImages().AppendIfNeeded(module_sp);
          error.Clear();
          return error;
        } else {
          error = ModuleList::GetSharedModule(kern_spec, module_sp, nullptr,
                                              nullptr, nullptr);
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

  // Next try all dSYMs that have no kernel binary next to them (load
  // the kernel DWARF stub as the main binary)
  for (auto possible_kernel_dsym : m_kernel_dsyms_no_binaries) {
    std::vector<FileSpec> objfile_names =
        GetDWARFBinaryInDSYMBundle(possible_kernel_dsym);
    for (FileSpec objfile : objfile_names) {
      ModuleSpec kern_spec(objfile);
      kern_spec.GetUUID() = module_spec.GetUUID();
      kern_spec.GetSymbolFileSpec() = possible_kernel_dsym;

      module_sp.reset(new Module(kern_spec));
      if (module_sp && module_sp->GetObjectFile() &&
          module_sp->MatchesModuleSpec(kern_spec)) {
        // module_sp is an actual kernel binary we want to add.
        if (process) {
          process->GetTarget().GetImages().AppendIfNeeded(module_sp);
          error.Clear();
          return error;
        } else {
          error = ModuleList::GetSharedModule(kern_spec, module_sp, nullptr,
                                              nullptr, nullptr);
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

  // Give the generic methods, including possibly calling into  DebugSymbols
  // framework on macOS systems, a chance.
  error = PlatformDarwin::GetSharedModule(module_spec, process, module_sp,
                                          module_search_paths_ptr, old_modules,
                                          did_create_ptr);
  if (error.Success() && module_sp.get()) {
    return error;
  }

  // Lastly, try all kernel binaries that don't have a dSYM
  for (auto possible_kernel : m_kernel_binaries_without_dsyms) {
    if (FileSystem::Instance().Exists(possible_kernel)) {
      ModuleSpec kern_spec(possible_kernel);
      kern_spec.GetUUID() = module_spec.GetUUID();
      module_sp.reset(new Module(kern_spec));
      if (module_sp && module_sp->GetObjectFile() &&
          module_sp->MatchesModuleSpec(kern_spec)) {
        // module_sp is an actual kernel binary we want to add.
        if (process) {
          process->GetTarget().GetImages().AppendIfNeeded(module_sp);
          error.Clear();
          return error;
        } else {
          error = ModuleList::GetSharedModule(kern_spec, module_sp, nullptr,
                                              nullptr, nullptr);
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

  return error;
}

std::vector<lldb_private::FileSpec>
PlatformDarwinKernel::SearchForExecutablesRecursively(const std::string &dir) {
  std::vector<FileSpec> executables;
  std::error_code EC;
  for (llvm::sys::fs::recursive_directory_iterator it(dir.c_str(), EC),
       end;
       it != end && !EC; it.increment(EC)) {
    auto status = it->status();
    if (!status)
      break;
    if (llvm::sys::fs::is_regular_file(*status) &&
        llvm::sys::fs::can_execute(it->path()))
      executables.emplace_back(it->path());
  }
  return executables;
}

Status PlatformDarwinKernel::ExamineKextForMatchingUUID(
    const FileSpec &kext_bundle_path, const lldb_private::UUID &uuid,
    const ArchSpec &arch, ModuleSP &exe_module_sp) {
  for (const auto &exe_file :
       SearchForExecutablesRecursively(kext_bundle_path.GetPath())) {
    if (FileSystem::Instance().Exists(exe_file)) {
      ModuleSpec exe_spec(exe_file);
      exe_spec.GetUUID() = uuid;
      if (!uuid.IsValid()) {
        exe_spec.GetArchitecture() = arch;
      }

      // First try to create a ModuleSP with the file / arch and see if the UUID
      // matches. If that fails (this exec file doesn't have the correct uuid),
      // don't call GetSharedModule (which may call in to the DebugSymbols
      // framework and therefore can be slow.)
      ModuleSP module_sp(new Module(exe_spec));
      if (module_sp && module_sp->GetObjectFile() &&
          module_sp->MatchesModuleSpec(exe_spec)) {
        Status error = ModuleList::GetSharedModule(exe_spec, exe_module_sp,
                                                   NULL, NULL, NULL);
        if (exe_module_sp && exe_module_sp->GetObjectFile()) {
          return error;
        }
      }
      exe_module_sp.reset();
    }
  }

  return {};
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

// Since DynamicLoaderDarwinKernel is compiled in for all systems, and relies
// on PlatformDarwinKernel for the plug-in name, we compile just the plug-in
// name in here to avoid issues. We are tracking an internal bug to resolve
// this issue by either not compiling in DynamicLoaderDarwinKernel for non-
// apple builds, or to make PlatformDarwinKernel build on all systems.
// PlatformDarwinKernel is currently not compiled on other platforms due to the
// use of the Mac-specific source/Host/macosx/cfcpp utilities.

lldb_private::ConstString PlatformDarwinKernel::GetPluginNameStatic() {
  static lldb_private::ConstString g_name("darwin-kernel");
  return g_name;
}

#endif // __APPLE__
