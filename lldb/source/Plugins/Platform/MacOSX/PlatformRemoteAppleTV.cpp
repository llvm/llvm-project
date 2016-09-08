//===-- PlatformRemoteAppleTV.cpp -------------------------------*- C++ -*-===//
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
#include "PlatformRemoteAppleTV.h"

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
PlatformRemoteAppleTV::PlatformRemoteAppleTV()
    : PlatformDarwin(false), // This is a remote platform
      m_sdk_directory_infos(), m_device_support_directory(),
      m_device_support_directory_for_os_version(), m_build_update(),
      m_last_module_sdk_idx(UINT32_MAX),
      m_connected_module_sdk_idx(UINT32_MAX) {}

PlatformRemoteAppleTV::SDKDirectoryInfo::SDKDirectoryInfo(
    const lldb_private::FileSpec &sdk_dir)
    : directory(sdk_dir), build(), version_major(0), version_minor(0),
      version_update(0), user_cached(false) {
  const char *dirname_cstr = sdk_dir.GetFilename().GetCString();
  const char *pos = Args::StringToVersion(dirname_cstr, version_major,
                                          version_minor, version_update);

  if (pos && pos[0] == ' ' && pos[1] == '(') {
    const char *build_start = pos + 2;
    const char *end_paren = strchr(build_start, ')');
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
void PlatformRemoteAppleTV::Initialize() {
  PlatformDarwin::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(PlatformRemoteAppleTV::GetPluginNameStatic(),
                                  PlatformRemoteAppleTV::GetDescriptionStatic(),
                                  PlatformRemoteAppleTV::CreateInstance);
  }
}

void PlatformRemoteAppleTV::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformRemoteAppleTV::CreateInstance);
    }
  }

  PlatformDarwin::Terminate();
}

PlatformSP PlatformRemoteAppleTV::CreateInstance(bool force,
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

    log->Printf("PlatformRemoteAppleTV::%s(force=%s, arch={%s,%s})",
                __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
  }

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    switch (arch->GetMachine()) {
    case llvm::Triple::arm:
    case llvm::Triple::aarch64:
    case llvm::Triple::thumb: {
      const llvm::Triple &triple = arch->GetTriple();
      llvm::Triple::VendorType vendor = triple.getVendor();
      switch (vendor) {
      case llvm::Triple::Apple:
        create = true;
        break;

      default:
        break;
      }
      if (create) {
        switch (triple.getOS()) {
        case llvm::Triple::TvOS: // This is the right triple value for Apple TV
                                 // debugging
          break;

        default:
          create = false;
          break;
        }
      }
    } break;
    default:
      break;
    }
  }

  if (create) {
    if (log)
      log->Printf("PlatformRemoteAppleTV::%s() creating platform",
                  __FUNCTION__);

    return lldb::PlatformSP(new PlatformRemoteAppleTV());
  }

  if (log)
    log->Printf("PlatformRemoteAppleTV::%s() aborting creation of platform",
                __FUNCTION__);

  return lldb::PlatformSP();
}

lldb_private::ConstString PlatformRemoteAppleTV::GetPluginNameStatic() {
  static ConstString g_name("remote-tvos");
  return g_name;
}

const char *PlatformRemoteAppleTV::GetDescriptionStatic() {
  return "Remote Apple TV platform plug-in.";
}

void PlatformRemoteAppleTV::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);
  const char *sdk_directory = GetDeviceSupportDirectoryForOSVersion();
  if (sdk_directory)
    strm.Printf("  SDK Path: \"%s\"\n", sdk_directory);
  else
    strm.PutCString("  SDK Path: error: unable to locate SDK\n");

  const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
  for (uint32_t i = 0; i < num_sdk_infos; ++i) {
    const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
    strm.Printf(" SDK Roots: [%2u] \"%s\"\n", i,
                sdk_dir_info.directory.GetPath().c_str());
  }
}

Error PlatformRemoteAppleTV::ResolveExecutable(
    const ModuleSpec &ms, lldb::ModuleSP &exe_module_sp,
    const FileSpecList *module_search_paths_ptr) {
  Error error;
  // Nothing special to do here, just use the actual file and architecture

  ModuleSpec resolved_module_spec(ms);

  // Resolve any executable within a bundle on MacOSX
  // TODO: verify that this handles shallow bundles, if not then implement one
  // ourselves
  Host::ResolveExecutableInBundle(resolved_module_spec.GetFileSpec());

  if (resolved_module_spec.GetFileSpec().Exists()) {
    if (resolved_module_spec.GetArchitecture().IsValid() ||
        resolved_module_spec.GetUUID().IsValid()) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          nullptr, nullptr, nullptr);

      if (exe_module_sp && exe_module_sp->GetObjectFile())
        return error;
      exe_module_sp.reset();
    }
    // No valid architecture was specified or the exact ARM slice wasn't
    // found so ask the platform for the architectures that we should be
    // using (in the correct order) and see if we can find a match that way
    StreamString arch_names;
    for (uint32_t idx = 0; GetSupportedArchitectureAtIndex(
             idx, resolved_module_spec.GetArchitecture());
         ++idx) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          nullptr, nullptr, nullptr);
      // Did we find an executable using one of the
      if (error.Success()) {
        if (exe_module_sp && exe_module_sp->GetObjectFile())
          break;
        else
          error.SetErrorToGenericError();
      }

      if (idx > 0)
        arch_names.PutCString(", ");
      arch_names.PutCString(
          resolved_module_spec.GetArchitecture().GetArchitectureName());
    }

    if (error.Fail() || !exe_module_sp) {
      if (resolved_module_spec.GetFileSpec().Readable()) {
        error.SetErrorStringWithFormat(
            "'%s' doesn't contain any '%s' platform architectures: %s",
            resolved_module_spec.GetFileSpec().GetPath().c_str(),
            GetPluginName().GetCString(), arch_names.GetString().c_str());
      } else {
        error.SetErrorStringWithFormat(
            "'%s' is not readable",
            resolved_module_spec.GetFileSpec().GetPath().c_str());
      }
    }
  } else {
    error.SetErrorStringWithFormat(
        "'%s' does not exist",
        resolved_module_spec.GetFileSpec().GetPath().c_str());
  }

  return error;
}

FileSpec::EnumerateDirectoryResult
PlatformRemoteAppleTV::GetContainedFilesIntoVectorOfStringsCallback(
    void *baton, FileSpec::FileType file_type, const FileSpec &file_spec) {
  ((PlatformRemoteAppleTV::SDKDirectoryInfoCollection *)baton)
      ->push_back(PlatformRemoteAppleTV::SDKDirectoryInfo(file_spec));
  return FileSpec::eEnumerateDirectoryResultNext;
}

bool PlatformRemoteAppleTV::UpdateSDKDirectoryInfosIfNeeded() {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  if (m_sdk_directory_infos.empty()) {
    const char *device_support_dir = GetDeviceSupportDirectory();
    if (log) {
      log->Printf("PlatformRemoteAppleTV::UpdateSDKDirectoryInfosIfNeeded Got "
                  "DeviceSupport directory %s",
                  device_support_dir);
    }
    if (device_support_dir) {
      const bool find_directories = true;
      const bool find_files = false;
      const bool find_other = false;

      SDKDirectoryInfoCollection builtin_sdk_directory_infos;
      FileSpec::EnumerateDirectory(m_device_support_directory.c_str(),
                                   find_directories, find_files, find_other,
                                   GetContainedFilesIntoVectorOfStringsCallback,
                                   &builtin_sdk_directory_infos);

      // Only add SDK directories that have symbols in them, some SDKs only
      // contain
      // developer disk images and no symbols, so they aren't useful to us.
      FileSpec sdk_symbols_symlink_fspec;
      for (const auto &sdk_directory_info : builtin_sdk_directory_infos) {
        sdk_symbols_symlink_fspec = sdk_directory_info.directory;
        sdk_symbols_symlink_fspec.AppendPathComponent("Symbols.Internal");
        if (sdk_symbols_symlink_fspec.Exists()) {
          m_sdk_directory_infos.push_back(sdk_directory_info);
          if (log) {
            log->Printf("PlatformRemoteAppleTV::"
                        "UpdateSDKDirectoryInfosIfNeeded added builtin SDK "
                        "directory %s",
                        sdk_symbols_symlink_fspec.GetPath().c_str());
          }
        } else {
          sdk_symbols_symlink_fspec.GetFilename().SetCString("Symbols");
          if (sdk_symbols_symlink_fspec.Exists())
            m_sdk_directory_infos.push_back(sdk_directory_info);
          if (log) {
            log->Printf("PlatformRemoteAppleTV::"
                        "UpdateSDKDirectoryInfosIfNeeded added builtin SDK "
                        "directory %s",
                        sdk_symbols_symlink_fspec.GetPath().c_str());
          }
        }
      }

      const uint32_t num_installed = m_sdk_directory_infos.size();
      FileSpec local_sdk_cache("~/Library/Developer/Xcode/tvOS DeviceSupport",
                               true);
      if (!local_sdk_cache.Exists()) {
        // Try looking for another possible name
        local_sdk_cache = FileSpec(
            "~/Library/Developer/Xcode/Apple TVOS DeviceSupport", true);
      }
      if (!local_sdk_cache.Exists()) {
        // Try looking for another possible name
        local_sdk_cache =
            FileSpec("~/Library/Developer/Xcode/AppleTVOS DeviceSupport", true);
      }
      if (!local_sdk_cache.Exists()) {
        // Try looking for another possible name
        local_sdk_cache = FileSpec(
            "~/Library/Developer/Xcode/AppleTV OS DeviceSupport", true);
      }
      if (!local_sdk_cache.Exists()) {
        // Try looking for another possible name
        local_sdk_cache = FileSpec(
            "~/Library/Developer/Xcode/Apple TV OS DeviceSupport", true);
      }
      if (local_sdk_cache.Exists()) {
        if (log) {
          log->Printf("PlatformRemoteAppleTV::UpdateSDKDirectoryInfosIfNeeded "
                      "searching %s for additional SDKs",
                      local_sdk_cache.GetPath().c_str());
        }
        char path[PATH_MAX];
        if (local_sdk_cache.GetPath(path, sizeof(path))) {
          FileSpec::EnumerateDirectory(
              path, find_directories, find_files, find_other,
              GetContainedFilesIntoVectorOfStringsCallback,
              &m_sdk_directory_infos);
          const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
          // First try for an exact match of major, minor and update
          for (uint32_t i = num_installed; i < num_sdk_infos; ++i) {
            m_sdk_directory_infos[i].user_cached = true;
            if (log) {
              log->Printf("PlatformRemoteAppleTV::"
                          "UpdateSDKDirectoryInfosIfNeeded user SDK directory "
                          "%s",
                          m_sdk_directory_infos[i].directory.GetPath().c_str());
            }
          }
        }
      }
    }
  }
  return !m_sdk_directory_infos.empty();
}

const PlatformRemoteAppleTV::SDKDirectoryInfo *
PlatformRemoteAppleTV::GetSDKDirectoryForCurrentOSVersion() {
  uint32_t i;
  if (UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();

    // Check to see if the user specified a build string. If they did, then
    // be sure to match it.
    std::vector<bool> check_sdk_info(num_sdk_infos, true);
    ConstString build(m_sdk_build);
    if (build) {
      for (i = 0; i < num_sdk_infos; ++i)
        check_sdk_info[i] = m_sdk_directory_infos[i].build == build;
    }

    // If we are connected we can find the version of the OS the platform
    // us running on and select the right SDK
    uint32_t major, minor, update;
    if (GetOSVersion(major, minor, update)) {
      if (UpdateSDKDirectoryInfosIfNeeded()) {
        // First try for an exact match of major, minor and update
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version_major == major &&
                m_sdk_directory_infos[i].version_minor == minor &&
                m_sdk_directory_infos[i].version_update == update) {
              return &m_sdk_directory_infos[i];
            }
          }
        }
        // First try for an exact match of major and minor
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version_major == major &&
                m_sdk_directory_infos[i].version_minor == minor) {
              return &m_sdk_directory_infos[i];
            }
          }
        }
        // Lastly try to match of major version only..
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version_major == major) {
              return &m_sdk_directory_infos[i];
            }
          }
        }
      }
    } else if (build) {
      // No version, just a build number, search for the first one that matches
      for (i = 0; i < num_sdk_infos; ++i)
        if (check_sdk_info[i])
          return &m_sdk_directory_infos[i];
    }
  }
  return nullptr;
}

const PlatformRemoteAppleTV::SDKDirectoryInfo *
PlatformRemoteAppleTV::GetSDKDirectoryForLatestOSVersion() {
  const PlatformRemoteAppleTV::SDKDirectoryInfo *result = nullptr;
  if (UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
    // First try for an exact match of major, minor and update
    for (uint32_t i = 0; i < num_sdk_infos; ++i) {
      const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
      if (sdk_dir_info.version_major != UINT32_MAX) {
        if (result == nullptr ||
            sdk_dir_info.version_major > result->version_major) {
          result = &sdk_dir_info;
        } else if (sdk_dir_info.version_major == result->version_major) {
          if (sdk_dir_info.version_minor > result->version_minor) {
            result = &sdk_dir_info;
          } else if (sdk_dir_info.version_minor == result->version_minor) {
            if (sdk_dir_info.version_update > result->version_update) {
              result = &sdk_dir_info;
            }
          }
        }
      }
    }
  }
  return result;
}

const char *PlatformRemoteAppleTV::GetDeviceSupportDirectory() {
  if (m_device_support_directory.empty()) {
    const char *device_support_dir = GetDeveloperDirectory();
    if (device_support_dir) {
      m_device_support_directory.assign(device_support_dir);
      m_device_support_directory.append(
          "/Platforms/AppleTVOS.platform/DeviceSupport");
    } else {
      // Assign a single NULL character so we know we tried to find the device
      // support directory and we don't keep trying to find it over and over.
      m_device_support_directory.assign(1, '\0');
    }
  }
  // We should have put a single NULL character into m_device_support_directory
  // or it should have a valid path if the code gets here
  assert(m_device_support_directory.empty() == false);
  if (m_device_support_directory[0])
    return m_device_support_directory.c_str();
  return nullptr;
}

const char *PlatformRemoteAppleTV::GetDeviceSupportDirectoryForOSVersion() {
  if (m_sdk_sysroot)
    return m_sdk_sysroot.GetCString();

  if (m_device_support_directory_for_os_version.empty()) {
    const PlatformRemoteAppleTV::SDKDirectoryInfo *sdk_dir_info =
        GetSDKDirectoryForCurrentOSVersion();
    if (sdk_dir_info == nullptr)
      sdk_dir_info = GetSDKDirectoryForLatestOSVersion();
    if (sdk_dir_info) {
      char path[PATH_MAX];
      if (sdk_dir_info->directory.GetPath(path, sizeof(path))) {
        m_device_support_directory_for_os_version = path;
        return m_device_support_directory_for_os_version.c_str();
      }
    } else {
      // Assign a single NULL character so we know we tried to find the device
      // support directory and we don't keep trying to find it over and over.
      m_device_support_directory_for_os_version.assign(1, '\0');
    }
  }
  // We should have put a single NULL character into
  // m_device_support_directory_for_os_version
  // or it should have a valid path if the code gets here
  assert(m_device_support_directory_for_os_version.empty() == false);
  if (m_device_support_directory_for_os_version[0])
    return m_device_support_directory_for_os_version.c_str();
  return nullptr;
}

uint32_t
PlatformRemoteAppleTV::FindFileInAllSDKs(const char *platform_file_path,
                                         FileSpecList &file_list) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST |
                                                    LIBLLDB_LOG_VERBOSE);
  if (platform_file_path && platform_file_path[0] &&
      UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
    lldb_private::FileSpec local_file;
    // First try for an exact match of major, minor and update
    for (uint32_t sdk_idx = 0; sdk_idx < num_sdk_infos; ++sdk_idx) {
      if (log) {
        log->Printf("Searching for %s in sdk path %s", platform_file_path,
                    m_sdk_directory_infos[sdk_idx].directory.GetPath().c_str());
      }
      if (GetFileInSDK(platform_file_path, sdk_idx, local_file)) {
        file_list.Append(local_file);
      }
    }
  }
  return file_list.GetSize();
}

bool PlatformRemoteAppleTV::GetFileInSDK(const char *platform_file_path,
                                         uint32_t sdk_idx,
                                         lldb_private::FileSpec &local_file) {
  if (sdk_idx < m_sdk_directory_infos.size()) {
    char sdkroot_path[PATH_MAX];
    const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[sdk_idx];
    if (sdk_dir_info.directory.GetPath(sdkroot_path, sizeof(sdkroot_path))) {
      const bool symbols_dirs_only = true;

      return GetFileInSDKRoot(platform_file_path, sdkroot_path,
                              symbols_dirs_only, local_file);
    }
  }
  return false;
}

bool PlatformRemoteAppleTV::GetFileInSDKRoot(
    const char *platform_file_path, const char *sdkroot_path,
    bool symbols_dirs_only, lldb_private::FileSpec &local_file) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  if (sdkroot_path && sdkroot_path[0] && platform_file_path &&
      platform_file_path[0]) {
    char resolved_path[PATH_MAX];

    if (!symbols_dirs_only) {
      ::snprintf(resolved_path, sizeof(resolved_path), "%s%s", sdkroot_path,
                 platform_file_path);

      local_file.SetFile(resolved_path, true);
      if (local_file.Exists()) {
        if (log) {
          log->Printf("Found a copy of %s in the SDK dir %s",
                      platform_file_path, sdkroot_path);
        }
        return true;
      }
    }

    ::snprintf(resolved_path, sizeof(resolved_path), "%s/Symbols.Internal%s",
               sdkroot_path, platform_file_path);

    local_file.SetFile(resolved_path, true);
    if (local_file.Exists()) {
      if (log) {
        log->Printf("Found a copy of %s in the SDK dir %s/Symbols.Internal",
                    platform_file_path, sdkroot_path);
      }
      return true;
    }
    ::snprintf(resolved_path, sizeof(resolved_path), "%s/Symbols%s",
               sdkroot_path, platform_file_path);

    local_file.SetFile(resolved_path, true);
    if (local_file.Exists()) {
      if (log) {
        log->Printf("Found a copy of %s in the SDK dir %s/Symbols",
                    platform_file_path, sdkroot_path);
      }
      return true;
    }
  }
  return false;
}

Error PlatformRemoteAppleTV::GetSymbolFile(const FileSpec &platform_file,
                                           const UUID *uuid_ptr,
                                           FileSpec &local_file) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  Error error;
  char platform_file_path[PATH_MAX];
  if (platform_file.GetPath(platform_file_path, sizeof(platform_file_path))) {
    char resolved_path[PATH_MAX];

    const char *os_version_dir = GetDeviceSupportDirectoryForOSVersion();
    if (os_version_dir) {
      ::snprintf(resolved_path, sizeof(resolved_path), "%s/%s", os_version_dir,
                 platform_file_path);

      local_file.SetFile(resolved_path, true);
      if (local_file.Exists()) {
        if (log) {
          log->Printf("Found a copy of %s in the DeviceSupport dir %s",
                      platform_file_path, os_version_dir);
        }
        return error;
      }

      ::snprintf(resolved_path, sizeof(resolved_path), "%s/Symbols.Internal/%s",
                 os_version_dir, platform_file_path);

      local_file.SetFile(resolved_path, true);
      if (local_file.Exists()) {
        if (log) {
          log->Printf(
              "Found a copy of %s in the DeviceSupport dir %s/Symbols.Internal",
              platform_file_path, os_version_dir);
        }
        return error;
      }
      ::snprintf(resolved_path, sizeof(resolved_path), "%s/Symbols/%s",
                 os_version_dir, platform_file_path);

      local_file.SetFile(resolved_path, true);
      if (local_file.Exists()) {
        if (log) {
          log->Printf("Found a copy of %s in the DeviceSupport dir %s/Symbols",
                      platform_file_path, os_version_dir);
        }
        return error;
      }
    }
    local_file = platform_file;
    if (local_file.Exists())
      return error;

    error.SetErrorStringWithFormat(
        "unable to locate a platform file for '%s' in platform '%s'",
        platform_file_path, GetPluginName().GetCString());
  } else {
    error.SetErrorString("invalid platform file argument");
  }
  return error;
}

Error PlatformRemoteAppleTV::GetSharedModule(
    const ModuleSpec &module_spec, lldb_private::Process *process,
    ModuleSP &module_sp, const FileSpecList *module_search_paths_ptr,
    ModuleSP *old_module_sp_ptr, bool *did_create_ptr) {
  // For Apple TV, the SDK files are all cached locally on the host
  // system. So first we ask for the file in the cached SDK,
  // then we attempt to get a shared module for the right architecture
  // with the right UUID.
  const FileSpec &platform_file = module_spec.GetFileSpec();
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST |
                                                    LIBLLDB_LOG_VERBOSE);

  Error error;
  char platform_file_path[PATH_MAX];

  if (platform_file.GetPath(platform_file_path, sizeof(platform_file_path))) {
    ModuleSpec platform_module_spec(module_spec);

    UpdateSDKDirectoryInfosIfNeeded();

    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();

    // If we are connected we migth be able to correctly deduce the SDK
    // directory
    // using the OS build.
    const uint32_t connected_sdk_idx = GetConnectedSDKIndex();
    if (connected_sdk_idx < num_sdk_infos) {
      if (log) {
        log->Printf("Searching for %s in sdk path %s", platform_file_path,
                    m_sdk_directory_infos[connected_sdk_idx]
                        .directory.GetPath()
                        .c_str());
      }
      if (GetFileInSDK(platform_file_path, connected_sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        module_sp.reset();
        error = ResolveExecutable(platform_module_spec, module_sp, nullptr);
        if (module_sp) {
          m_last_module_sdk_idx = connected_sdk_idx;
          error.Clear();
          return error;
        }
      }
    }

    // Try the last SDK index if it is set as most files from an SDK
    // will tend to be valid in that same SDK.
    if (m_last_module_sdk_idx < num_sdk_infos) {
      if (log) {
        log->Printf("Searching for %s in sdk path %s", platform_file_path,
                    m_sdk_directory_infos[m_last_module_sdk_idx]
                        .directory.GetPath()
                        .c_str());
      }
      if (GetFileInSDK(platform_file_path, m_last_module_sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        module_sp.reset();
        error = ResolveExecutable(platform_module_spec, module_sp, nullptr);
        if (module_sp) {
          error.Clear();
          return error;
        }
      }
    }

    // First try for an exact match of major, minor and update
    for (uint32_t sdk_idx = 0; sdk_idx < num_sdk_infos; ++sdk_idx) {
      if (m_last_module_sdk_idx == sdk_idx) {
        // Skip the last module SDK index if we already searched
        // it above
        continue;
      }
      if (log) {
        log->Printf("Searching for %s in sdk path %s", platform_file_path,
                    m_sdk_directory_infos[sdk_idx].directory.GetPath().c_str());
      }
      if (GetFileInSDK(platform_file_path, sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        // printf ("sdk[%u]: '%s'\n", sdk_idx, local_file.GetPath().c_str());

        error = ResolveExecutable(platform_module_spec, module_sp, nullptr);
        if (module_sp) {
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

  // This may not be an SDK-related module.  Try whether we can bring in the
  // thing to our local cache.
  error = GetSharedModuleWithLocalCache(module_spec, module_sp,
                                        module_search_paths_ptr,
                                        old_module_sp_ptr, did_create_ptr);
  if (error.Success())
    return error;

  // See if the file is present in any of the module_search_paths_ptr
  // directories.
  if (!module_sp && module_search_paths_ptr && platform_file) {
    // create a vector of all the file / directory names in platform_file
    // e.g. this might be
    // /System/Library/PrivateFrameworks/UIFoundation.framework/UIFoundation
    //
    // We'll need to look in the module_search_paths_ptr directories for
    // both "UIFoundation" and "UIFoundation.framework" -- most likely the
    // latter will be the one we find there.

    FileSpec platform_pull_apart(platform_file);
    std::vector<std::string> path_parts;
    ConstString unix_root_dir("/");
    while (true) {
      ConstString part = platform_pull_apart.GetLastPathComponent();
      platform_pull_apart.RemoveLastPathComponent();
      if (part.IsEmpty() || part == unix_root_dir)
        break;
      path_parts.push_back(part.AsCString());
    }
    const size_t path_parts_size = path_parts.size();

    size_t num_module_search_paths = module_search_paths_ptr->GetSize();
    for (size_t i = 0; i < num_module_search_paths; ++i) {
      // Create a new FileSpec with this module_search_paths_ptr
      // plus just the filename ("UIFoundation"), then the parent
      // dir plus filename ("UIFoundation.framework/UIFoundation")
      // etc - up to four names (to handle "Foo.framework/Contents/MacOS/Foo")

      for (size_t j = 0; j < 4 && j < path_parts_size - 1; ++j) {
        FileSpec path_to_try(module_search_paths_ptr->GetFileSpecAtIndex(i));

        // Add the components backwards.  For
        // .../PrivateFrameworks/UIFoundation.framework/UIFoundation
        // path_parts is
        //   [0] UIFoundation
        //   [1] UIFoundation.framework
        //   [2] PrivateFrameworks
        //
        // and if 'j' is 2, we want to append path_parts[1] and then
        // path_parts[0], aka
        // 'UIFoundation.framework/UIFoundation', to the module_search_paths_ptr
        // path.

        for (int k = j; k >= 0; --k) {
          path_to_try.AppendPathComponent(path_parts[k]);
        }

        if (path_to_try.Exists()) {
          ModuleSpec new_module_spec(module_spec);
          new_module_spec.GetFileSpec() = path_to_try;
          Error new_error(Platform::GetSharedModule(
              new_module_spec, process, module_sp, NULL, old_module_sp_ptr,
              did_create_ptr));

          if (module_sp) {
            module_sp->SetPlatformFileSpec(path_to_try);
            return new_error;
          }
        }
      }
    }
  }

  const bool always_create = false;
  error = ModuleList::GetSharedModule(
      module_spec, module_sp, module_search_paths_ptr, old_module_sp_ptr,
      did_create_ptr, always_create);

  if (module_sp)
    module_sp->SetPlatformFileSpec(platform_file);

  return error;
}

bool PlatformRemoteAppleTV::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                            ArchSpec &arch) {
  ArchSpec system_arch(GetSystemArchitecture());

  const ArchSpec::Core system_core = system_arch.GetCore();
  switch (system_core) {
  default:
    switch (idx) {
    case 0:
      arch.SetTriple("arm64-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("armv7s-apple-tvos");
      return true;
    case 2:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 3:
      arch.SetTriple("thumbv7s-apple-tvos");
      return true;
    case 4:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_arm64:
    switch (idx) {
    case 0:
      arch.SetTriple("arm64-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("armv7s-apple-tvos");
      return true;
    case 2:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 3:
      arch.SetTriple("thumbv7s-apple-tvos");
      return true;
    case 4:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7s:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7s-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 2:
      arch.SetTriple("thumbv7s-apple-tvos");
      return true;
    case 3:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;
  }
  arch.Clear();
  return false;
}

uint32_t PlatformRemoteAppleTV::GetConnectedSDKIndex() {
  if (IsConnected()) {
    if (m_connected_module_sdk_idx == UINT32_MAX) {
      std::string build;
      if (GetRemoteOSBuildString(build)) {
        const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
        for (uint32_t i = 0; i < num_sdk_infos; ++i) {
          const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
          if (strstr(sdk_dir_info.directory.GetFilename().AsCString(""),
                     build.c_str())) {
            m_connected_module_sdk_idx = i;
          }
        }
      }
    }
  } else {
    m_connected_module_sdk_idx = UINT32_MAX;
  }
  return m_connected_module_sdk_idx;
}
