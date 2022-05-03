//===-- PlatformDarwinDevice.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwinDevice.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

PlatformDarwinDevice::~PlatformDarwinDevice() = default;

FileSystem::EnumerateDirectoryResult
PlatformDarwinDevice::GetContainedFilesIntoVectorOfStringsCallback(
    void *baton, llvm::sys::fs::file_type ft, llvm::StringRef path) {
  ((PlatformDarwinDevice::SDKDirectoryInfoCollection *)baton)
      ->push_back(PlatformDarwinDevice::SDKDirectoryInfo(FileSpec(path)));
  return FileSystem::eEnumerateDirectoryResultNext;
}

bool PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded() {
  Log *log = GetLog(LLDBLog::Host);
  std::lock_guard<std::mutex> guard(m_sdk_dir_mutex);
  if (m_sdk_directory_infos.empty()) {
    // A --sysroot option was supplied - add it to our list of SDKs to check
    if (m_sdk_sysroot) {
      FileSpec sdk_sysroot_fspec(m_sdk_sysroot.GetCString());
      FileSystem::Instance().Resolve(sdk_sysroot_fspec);
      const SDKDirectoryInfo sdk_sysroot_directory_info(sdk_sysroot_fspec);
      m_sdk_directory_infos.push_back(sdk_sysroot_directory_info);
      if (log) {
        LLDB_LOGF(log,
                  "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded added "
                  "--sysroot SDK directory %s",
                  m_sdk_sysroot.GetCString());
      }
      return true;
    }
    const char *device_support_dir = GetDeviceSupportDirectory();
    if (log) {
      LLDB_LOGF(log,
                "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded Got "
                "DeviceSupport directory %s",
                device_support_dir);
    }
    if (device_support_dir) {
      const bool find_directories = true;
      const bool find_files = false;
      const bool find_other = false;

      SDKDirectoryInfoCollection builtin_sdk_directory_infos;
      FileSystem::Instance().EnumerateDirectory(
          m_device_support_directory, find_directories, find_files, find_other,
          GetContainedFilesIntoVectorOfStringsCallback,
          &builtin_sdk_directory_infos);

      // Only add SDK directories that have symbols in them, some SDKs only
      // contain developer disk images and no symbols, so they aren't useful to
      // us.
      FileSpec sdk_symbols_symlink_fspec;
      for (const auto &sdk_directory_info : builtin_sdk_directory_infos) {
        sdk_symbols_symlink_fspec = sdk_directory_info.directory;
        sdk_symbols_symlink_fspec.AppendPathComponent("Symbols");
        if (FileSystem::Instance().Exists(sdk_symbols_symlink_fspec)) {
          m_sdk_directory_infos.push_back(sdk_directory_info);
          if (log) {
            LLDB_LOGF(log,
                      "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                      "added builtin SDK directory %s",
                      sdk_symbols_symlink_fspec.GetPath().c_str());
          }
        }
      }

      const uint32_t num_installed = m_sdk_directory_infos.size();
      llvm::StringRef dirname = GetDeviceSupportDirectoryName();
      std::string local_sdk_cache_str = "~/Library/Developer/Xcode/";
      local_sdk_cache_str += std::string(dirname);
      FileSpec local_sdk_cache(local_sdk_cache_str.c_str());
      FileSystem::Instance().Resolve(local_sdk_cache);
      if (FileSystem::Instance().Exists(local_sdk_cache)) {
        if (log) {
          LLDB_LOGF(log,
                    "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                    "searching %s for additional SDKs",
                    local_sdk_cache.GetPath().c_str());
        }
        char path[PATH_MAX];
        if (local_sdk_cache.GetPath(path, sizeof(path))) {
          FileSystem::Instance().EnumerateDirectory(
              path, find_directories, find_files, find_other,
              GetContainedFilesIntoVectorOfStringsCallback,
              &m_sdk_directory_infos);
          const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
          // First try for an exact match of major, minor and update
          for (uint32_t i = num_installed; i < num_sdk_infos; ++i) {
            m_sdk_directory_infos[i].user_cached = true;
            if (log) {
              LLDB_LOGF(log,
                        "PlatformDarwinDevice::"
                        "UpdateSDKDirectoryInfosIfNeeded "
                        "user SDK directory %s",
                        m_sdk_directory_infos[i].directory.GetPath().c_str());
            }
          }
        }
      }

      const char *addtional_platform_dirs = getenv("PLATFORM_SDK_DIRECTORY");
      if (addtional_platform_dirs) {
        SDKDirectoryInfoCollection env_var_sdk_directory_infos;
        FileSystem::Instance().EnumerateDirectory(
            addtional_platform_dirs, find_directories, find_files, find_other,
            GetContainedFilesIntoVectorOfStringsCallback,
            &env_var_sdk_directory_infos);
        FileSpec sdk_symbols_symlink_fspec;
        for (const auto &sdk_directory_info : env_var_sdk_directory_infos) {
          sdk_symbols_symlink_fspec = sdk_directory_info.directory;
          sdk_symbols_symlink_fspec.AppendPathComponent("Symbols");
          if (FileSystem::Instance().Exists(sdk_symbols_symlink_fspec)) {
            m_sdk_directory_infos.push_back(sdk_directory_info);
            if (log) {
              LLDB_LOGF(log,
                        "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                        "added env var SDK directory %s",
                        sdk_symbols_symlink_fspec.GetPath().c_str());
            }
          }
        }
      }
    }
  }
  return !m_sdk_directory_infos.empty();
}

const PlatformDarwinDevice::SDKDirectoryInfo *
PlatformDarwinDevice::GetSDKDirectoryForCurrentOSVersion() {
  uint32_t i;
  if (UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();

    // Check to see if the user specified a build string. If they did, then be
    // sure to match it.
    std::vector<bool> check_sdk_info(num_sdk_infos, true);
    ConstString build(m_sdk_build);
    if (build) {
      for (i = 0; i < num_sdk_infos; ++i)
        check_sdk_info[i] = m_sdk_directory_infos[i].build == build;
    }

    // If we are connected we can find the version of the OS the platform us
    // running on and select the right SDK
    llvm::VersionTuple version = GetOSVersion();
    if (!version.empty()) {
      if (UpdateSDKDirectoryInfosIfNeeded()) {
        // First try for an exact match of major, minor and update
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version == version)
              return &m_sdk_directory_infos[i];
          }
        }
        // First try for an exact match of major and minor
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version.getMajor() ==
                    version.getMajor() &&
                m_sdk_directory_infos[i].version.getMinor() ==
                    version.getMinor()) {
              return &m_sdk_directory_infos[i];
            }
          }
        }
        // Lastly try to match of major version only..
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version.getMajor() ==
                version.getMajor()) {
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

const PlatformDarwinDevice::SDKDirectoryInfo *
PlatformDarwinDevice::GetSDKDirectoryForLatestOSVersion() {
  const PlatformDarwinDevice::SDKDirectoryInfo *result = nullptr;
  if (UpdateSDKDirectoryInfosIfNeeded()) {
    auto max = std::max_element(
        m_sdk_directory_infos.begin(), m_sdk_directory_infos.end(),
        [](const SDKDirectoryInfo &a, const SDKDirectoryInfo &b) {
          return a.version < b.version;
        });
    if (max != m_sdk_directory_infos.end())
      result = &*max;
  }
  return result;
}

const char *PlatformDarwinDevice::GetDeviceSupportDirectory() {
  std::string platform_dir =
      ("/Platforms/" + GetPlatformName() + "/DeviceSupport").str();
  if (m_device_support_directory.empty()) {
    if (FileSpec fspec = HostInfo::GetXcodeDeveloperDirectory()) {
      m_device_support_directory = fspec.GetPath();
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
  return nullptr;
}

const char *PlatformDarwinDevice::GetDeviceSupportDirectoryForOSVersion() {
  if (m_sdk_sysroot)
    return m_sdk_sysroot.GetCString();

  if (m_device_support_directory_for_os_version.empty()) {
    const PlatformDarwinDevice::SDKDirectoryInfo *sdk_dir_info =
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
  // m_device_support_directory_for_os_version or it should have a valid path
  // if the code gets here
  assert(m_device_support_directory_for_os_version.empty() == false);
  if (m_device_support_directory_for_os_version[0])
    return m_device_support_directory_for_os_version.c_str();
  return nullptr;
}
