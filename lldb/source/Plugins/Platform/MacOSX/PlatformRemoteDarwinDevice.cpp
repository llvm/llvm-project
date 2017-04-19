//===-- PlatformRemoteDarwinDevice.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformRemoteDarwinDevice.h"

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
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

PlatformRemoteDarwinDevice::SDKDirectoryInfo::SDKDirectoryInfo(
    const lldb_private::FileSpec &sdk_dir)
    : directory(sdk_dir), build(), version_major(0), version_minor(0),
      version_update(0), user_cached(false) {
  llvm::StringRef dirname_str = sdk_dir.GetFilename().GetStringRef();
  llvm::StringRef build_str;
  std::tie(version_major, version_minor, version_update, build_str) =
      ParseVersionBuildDir(dirname_str);
  build.SetString(build_str);
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteDarwinDevice::PlatformRemoteDarwinDevice()
    : PlatformDarwin(false), // This is a remote platform
      m_sdk_directory_infos(), m_device_support_directory(),
      m_device_support_directory_for_os_version(), m_build_update(),
      m_last_module_sdk_idx(UINT32_MAX),
      m_connected_module_sdk_idx(UINT32_MAX) {}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformRemoteDarwinDevice::~PlatformRemoteDarwinDevice() {}

void PlatformRemoteDarwinDevice::GetStatus(Stream &strm) {
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

Error PlatformRemoteDarwinDevice::ResolveExecutable(
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
                                          NULL, NULL, NULL);

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
                                          NULL, NULL, NULL);
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
            GetPluginName().GetCString(), arch_names.GetData());
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
PlatformRemoteDarwinDevice::GetContainedFilesIntoVectorOfStringsCallback(
    void *baton, llvm::sys::fs::file_type ft, const FileSpec &file_spec) {
  ((PlatformRemoteDarwinDevice::SDKDirectoryInfoCollection *)baton)
      ->push_back(PlatformRemoteDarwinDevice::SDKDirectoryInfo(file_spec));
  return FileSpec::eEnumerateDirectoryResultNext;
}

bool PlatformRemoteDarwinDevice::UpdateSDKDirectoryInfosIfNeeded() {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  std::lock_guard<std::mutex> guard(m_sdk_dir_mutex);
  if (m_sdk_directory_infos.empty()) {
    // A --sysroot option was supplied - add it to our list of SDKs to check
    if (m_sdk_sysroot) {
      FileSpec sdk_sysroot_fspec(m_sdk_sysroot.GetCString(), true);
      const SDKDirectoryInfo sdk_sysroot_directory_info(sdk_sysroot_fspec);
      m_sdk_directory_infos.push_back(sdk_sysroot_directory_info);
      if (log) {
        log->Printf("PlatformRemoteDarwinDevice::UpdateSDKDirectoryInfosIfNeeded added "
                    "--sysroot SDK directory %s",
                    m_sdk_sysroot.GetCString());
      }
      return true;
    }
    const char *device_support_dir = GetDeviceSupportDirectory();
    if (log) {
      log->Printf("PlatformRemoteDarwinDevice::UpdateSDKDirectoryInfosIfNeeded Got "
                  "DeviceSupport directory %s",
                  device_support_dir);
    }
    if (device_support_dir) {
      const bool find_directories = true;
      const bool find_files = false;
      const bool find_other = false;

      SDKDirectoryInfoCollection builtin_sdk_directory_infos;
      FileSpec::EnumerateDirectory(m_device_support_directory, find_directories,
                                   find_files, find_other,
                                   GetContainedFilesIntoVectorOfStringsCallback,
                                   &builtin_sdk_directory_infos);

      // Only add SDK directories that have symbols in them, some SDKs only
      // contain
      // developer disk images and no symbols, so they aren't useful to us.
      FileSpec sdk_symbols_symlink_fspec;
      for (const auto &sdk_directory_info : builtin_sdk_directory_infos) {
        sdk_symbols_symlink_fspec = sdk_directory_info.directory;
        sdk_symbols_symlink_fspec.AppendPathComponent("Symbols");
        if (sdk_symbols_symlink_fspec.Exists()) {
          m_sdk_directory_infos.push_back(sdk_directory_info);
          if (log) {
            log->Printf("PlatformRemoteDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                        "added builtin SDK directory %s",
                        sdk_symbols_symlink_fspec.GetPath().c_str());
          }
        }
      }

      std::vector<std::string>  device_support_dirnames;
      GetDeviceSupportDirectoryNames (device_support_dirnames);

      for (std::string &dirname : device_support_dirnames)
      {
        const uint32_t num_installed = m_sdk_directory_infos.size();
        std::string local_sdk_cache_str = "~/Library/Developer/Xcode/";
        local_sdk_cache_str += dirname;
        FileSpec local_sdk_cache(local_sdk_cache_str.c_str(), true);
        if (local_sdk_cache.Exists()) {
            if (log) {
            log->Printf("PlatformRemoteDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
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
                log->Printf("PlatformRemoteDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                            "user SDK directory %s",
                            m_sdk_directory_infos[i].directory.GetPath().c_str());
                }
            }
          }
        }
      }
    }
  }
  return !m_sdk_directory_infos.empty();
}

const PlatformRemoteDarwinDevice::SDKDirectoryInfo *
PlatformRemoteDarwinDevice::GetSDKDirectoryForCurrentOSVersion() {
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
  return NULL;
}

const PlatformRemoteDarwinDevice::SDKDirectoryInfo *
PlatformRemoteDarwinDevice::GetSDKDirectoryForLatestOSVersion() {
  const PlatformRemoteDarwinDevice::SDKDirectoryInfo *result = NULL;
  if (UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
    // First try for an exact match of major, minor and update
    for (uint32_t i = 0; i < num_sdk_infos; ++i) {
      const SDKDirectoryInfo &sdk_dir_info = m_sdk_directory_infos[i];
      if (sdk_dir_info.version_major != UINT32_MAX) {
        if (result == NULL ||
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

const char *PlatformRemoteDarwinDevice::GetDeviceSupportDirectory() {
  std::string platform_dir = "/Platforms/" + GetPlatformName() + "/DeviceSupport";
  if (m_device_support_directory.empty()) {
    const char *device_support_dir = GetDeveloperDirectory();
    if (device_support_dir) {
      m_device_support_directory.assign(device_support_dir);
      m_device_support_directory.append(platform_dir.c_str());
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
  return NULL;
}

const char *PlatformRemoteDarwinDevice::GetDeviceSupportDirectoryForOSVersion() {
  if (m_sdk_sysroot)
    return m_sdk_sysroot.GetCString();

  if (m_device_support_directory_for_os_version.empty()) {
    const PlatformRemoteDarwinDevice::SDKDirectoryInfo *sdk_dir_info =
        GetSDKDirectoryForCurrentOSVersion();
    if (sdk_dir_info == NULL)
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
  return NULL;
}

uint32_t PlatformRemoteDarwinDevice::FindFileInAllSDKs(const char *platform_file_path,
                                              FileSpecList &file_list) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  if (platform_file_path && platform_file_path[0] &&
      UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
    lldb_private::FileSpec local_file;
    // First try for an exact match of major, minor and update
    for (uint32_t sdk_idx = 0; sdk_idx < num_sdk_infos; ++sdk_idx) {
      LLDB_LOGV(log, "Searching for {0} in sdk path {1}", platform_file_path,
                m_sdk_directory_infos[sdk_idx].directory);
      if (GetFileInSDK(platform_file_path, sdk_idx, local_file)) {
        file_list.Append(local_file);
      }
    }
  }
  return file_list.GetSize();
}

bool PlatformRemoteDarwinDevice::GetFileInSDK(const char *platform_file_path,
                                     uint32_t sdk_idx,
                                     lldb_private::FileSpec &local_file) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  if (sdk_idx < m_sdk_directory_infos.size()) {
    std::string sdkroot_path =
        m_sdk_directory_infos[sdk_idx].directory.GetPath();
    local_file.Clear();

    if (!sdkroot_path.empty() && platform_file_path && platform_file_path[0]) {
      // We may need to interpose "/Symbols/" or "/Symbols.Internal/" between
      // the
      // SDK root directory and the file path.

      const char *paths_to_try[] = {"Symbols", "", "Symbols.Internal", nullptr};
      for (size_t i = 0; paths_to_try[i] != nullptr; i++) {
        local_file.SetFile(sdkroot_path, false);
        if (paths_to_try[i][0] != '\0')
          local_file.AppendPathComponent(paths_to_try[i]);
        local_file.AppendPathComponent(platform_file_path);
        local_file.ResolvePath();
        if (local_file.Exists()) {
          if (log)
            log->Printf("Found a copy of %s in the SDK dir %s/%s",
                        platform_file_path, sdkroot_path.c_str(),
                        paths_to_try[i]);
          return true;
        }
        local_file.Clear();
      }
    }
  }
  return false;
}

Error PlatformRemoteDarwinDevice::GetSymbolFile(const FileSpec &platform_file,
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

Error PlatformRemoteDarwinDevice::GetSharedModule(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr, ModuleSP *old_module_sp_ptr,
    bool *did_create_ptr) {
  // For iOS, the SDK files are all cached locally on the host
  // system. So first we ask for the file in the cached SDK,
  // then we attempt to get a shared module for the right architecture
  // with the right UUID.
  const FileSpec &platform_file = module_spec.GetFileSpec();
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);

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
      LLDB_LOGV(log, "Searching for {0} in sdk path {1}", platform_file,
                m_sdk_directory_infos[connected_sdk_idx].directory);
      if (GetFileInSDK(platform_file_path, connected_sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        module_sp.reset();
        error = ResolveExecutable(platform_module_spec, module_sp, NULL);
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
      LLDB_LOGV(log, "Searching for {0} in sdk path {1}", platform_file,
                m_sdk_directory_infos[m_last_module_sdk_idx].directory);
      if (GetFileInSDK(platform_file_path, m_last_module_sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        module_sp.reset();
        error = ResolveExecutable(platform_module_spec, module_sp, NULL);
        if (module_sp) {
          error.Clear();
          return error;
        }
      }
    }

    // First try for an exact match of major, minor and update:
    // If a particalar SDK version was specified via --version or --build, look
    // for a match on disk.
    const SDKDirectoryInfo *current_sdk_info =
        GetSDKDirectoryForCurrentOSVersion();
    const uint32_t current_sdk_idx =
        GetSDKIndexBySDKDirectoryInfo(current_sdk_info);
    if (current_sdk_idx < num_sdk_infos &&
        current_sdk_idx != m_last_module_sdk_idx) {
      LLDB_LOGV(log, "Searching for {0} in sdk path {1}", platform_file,
                m_sdk_directory_infos[current_sdk_idx].directory);
      if (GetFileInSDK(platform_file_path, current_sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        module_sp.reset();
        error = ResolveExecutable(platform_module_spec, module_sp, NULL);
        if (module_sp) {
          m_last_module_sdk_idx = current_sdk_idx;
          error.Clear();
          return error;
        }
      }
    }

    // Second try all SDKs that were found.
    for (uint32_t sdk_idx = 0; sdk_idx < num_sdk_infos; ++sdk_idx) {
      if (m_last_module_sdk_idx == sdk_idx) {
        // Skip the last module SDK index if we already searched
        // it above
        continue;
      }
      LLDB_LOGV(log, "Searching for {0} in sdk path {1}", platform_file,
                m_sdk_directory_infos[sdk_idx].directory);
      if (GetFileInSDK(platform_file_path, sdk_idx,
                       platform_module_spec.GetFileSpec())) {
        // printf ("sdk[%u]: '%s'\n", sdk_idx, local_file.GetPath().c_str());

        error = ResolveExecutable(platform_module_spec, module_sp, NULL);
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
      LLDB_LOGV(log, "searching for binary in search-path {0}",
                module_search_paths_ptr->GetFileSpecAtIndex(i));
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

uint32_t PlatformRemoteDarwinDevice::GetConnectedSDKIndex() {
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

uint32_t PlatformRemoteDarwinDevice::GetSDKIndexBySDKDirectoryInfo(
    const SDKDirectoryInfo *sdk_info) {
  if (sdk_info == NULL) {
    return UINT32_MAX;
  }

  return sdk_info - &m_sdk_directory_infos[0];
}
