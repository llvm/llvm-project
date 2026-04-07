//===-- PlatformDarwinDevice.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwinDevice.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;

PlatformDarwinDevice::~PlatformDarwinDevice() = default;

FileSystem::EnumerateDirectoryResult
PlatformDarwinDevice::GetContainedFilesIntoVectorOfFileSpecsCallback(
    void *baton, llvm::sys::fs::file_type ft, llvm::StringRef path) {
  ((std::vector<FileSpec> *)baton)->push_back(FileSpec(path));
  return FileSystem::eEnumerateDirectoryResultNext;
}

void PlatformDarwinDevice::AddSharedCacheDirectory(
    llvm::StringRef dir, llvm::StringRef log_msg_descriptor) {
  Log *log = GetLog(LLDBLog::Host);
  const bool find_directories = true;
  const bool find_files = false;
  const bool find_other = false;
  std::vector<FileSpec> shared_cache_expanded_directories;
  FileSystem::Instance().EnumerateDirectory(
      dir, find_directories, find_files, find_other,
      GetContainedFilesIntoVectorOfFileSpecsCallback,
      &shared_cache_expanded_directories);

  /// shared_cache_expanded_directories will have the directories under \a dir.
  /// Those that have a /Symbols/ subdir are shared cache dirs.
  /// Those that have /<arch>/Symbols/ subdirs are shared cache dirs.

  for (const FileSpec &sc_directory : shared_cache_expanded_directories) {
    FileSpec sc_directory_symbols = sc_directory;
    sc_directory_symbols.AppendPathComponent("Symbols");
    if (FileSystem::Instance().Exists(sc_directory_symbols)) {
      SDKDirectoryInfo thisdir(sc_directory,
                               sc_directory.GetFilename().GetStringRef());
      m_sdk_directory_infos.push_back(thisdir);
      LLDB_LOGF(log,
                "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                "added %s %s",
                log_msg_descriptor.str().c_str(),
                sc_directory.GetPath().c_str());
    }

    // See if we have arch subdirs under sc_directory, and if there is
    // a Symbols subdir under those.
    std::vector<FileSpec> subdirs;
    FileSystem::Instance().EnumerateDirectory(
        sc_directory.GetPath().c_str(), find_directories, find_files,
        find_other, GetContainedFilesIntoVectorOfFileSpecsCallback, &subdirs);
    for (const FileSpec &subdir : subdirs) {
      FileSpec subdir_directory_symbols = subdir;
      subdir_directory_symbols.AppendPathComponent("Symbols");
      if (FileSystem::Instance().Exists(subdir_directory_symbols)) {
        SDKDirectoryInfo thisdir(subdir,
                                 sc_directory.GetFilename().GetStringRef());
        m_sdk_directory_infos.push_back(thisdir);
        LLDB_LOGF(log,
                  "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                  "added %s %s",
                  log_msg_descriptor.str().c_str(), subdir.GetPath().c_str());
      }
    }
  }
}

bool PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded() {
  Log *log = GetLog(LLDBLog::Host);
  std::lock_guard<std::mutex> guard(m_sdk_dir_mutex);
  if (m_sdk_directory_infos.empty()) {

    // A --sysroot option was supplied - add it to our list of SDKs to check
    if (!m_sdk_sysroot.empty())
      AddSharedCacheDirectory(m_sdk_sysroot.c_str(), "--sysroot SDK directory");

    const char *device_support_dir = GetDeviceSupportDirectory();
    LLDB_LOGF(log,
              "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded Got "
              "DeviceSupport directory %s",
              device_support_dir);
    if (device_support_dir) {
      AddSharedCacheDirectory(device_support_dir, "builtin SDK directory");

      // "macOS DeviceSupport", "iOS DeviceSupport", etc.
      llvm::StringRef dirname = GetDeviceSupportDirectoryName();
      std::string local_sdk_cache_str = "~/Library/Developer/Xcode/";
      local_sdk_cache_str += std::string(dirname);
      FileSpec local_sdk_cache(local_sdk_cache_str.c_str());
      FileSystem::Instance().Resolve(local_sdk_cache);
      if (FileSystem::Instance().Exists(local_sdk_cache)) {
        LLDB_LOGF(log,
                  "PlatformDarwinDevice::UpdateSDKDirectoryInfosIfNeeded "
                  "searching %s for additional SDKs",
                  local_sdk_cache.GetPath().c_str());
        AddSharedCacheDirectory(local_sdk_cache.GetPath().c_str(),
                                "system developer dir directory");
      }

      const char *addtional_platform_dirs = getenv("PLATFORM_SDK_DIRECTORY");
      if (addtional_platform_dirs)
        AddSharedCacheDirectory(addtional_platform_dirs,
                                "env var SDK directory");
    }
  }
  return !m_sdk_directory_infos.empty();
}

const PlatformDarwinDevice::SDKDirectoryInfo *
PlatformDarwinDevice::GetSDKDirectoryForCurrentOSVersion() {
  uint32_t i;
  if (UpdateSDKDirectoryInfosIfNeeded()) {
    const uint32_t num_sdk_infos = m_sdk_directory_infos.size();
    std::vector<bool> check_sdk_info(num_sdk_infos, true);

    // Prefer the user SDK build string.
    std::string build = GetSDKBuild();

    // Fall back to the platform's build string.
    if (build.empty()) {
      if (std::optional<std::string> os_build_str = GetOSBuildString())
        build.assign(*os_build_str);
    }

    // If we have a build string, only check platforms for which the build
    // string matches.
    if (!build.empty()) {
      for (i = 0; i < num_sdk_infos; ++i)
        check_sdk_info[i] = m_sdk_directory_infos[i].build.GetStringRef() ==
                            llvm::StringRef(build);
    }

    // If we are connected we can find the version of the OS the platform us
    // running on and select the right SDK
    llvm::VersionTuple version = GetOSVersion();
    if (!version.empty()) {
      if (UpdateSDKDirectoryInfosIfNeeded()) {
        // First try for an exact match of major, minor and update.
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version == version)
              return &m_sdk_directory_infos[i];
          }
        }
        // Try for an exact match of major and minor.
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
        // Lastly try to match of major version only.
        for (i = 0; i < num_sdk_infos; ++i) {
          if (check_sdk_info[i]) {
            if (m_sdk_directory_infos[i].version.getMajor() ==
                version.getMajor()) {
              return &m_sdk_directory_infos[i];
            }
          }
        }
      }
    } else if (!build.empty()) {
      // No version, just a build number, return the first one that matches.
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
  if (!m_sdk_sysroot.empty())
    return m_sdk_sysroot.c_str();

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

static lldb_private::Status
MakeCacheFolderForFile(const FileSpec &module_cache_spec) {
  FileSpec module_cache_folder =
      module_cache_spec.CopyByRemovingLastPathComponent();
  return llvm::sys::fs::create_directory(module_cache_folder.GetPath());
}

static lldb_private::Status
BringInRemoteFile(Platform *platform,
                  const lldb_private::ModuleSpec &module_spec,
                  const FileSpec &module_cache_spec) {
  MakeCacheFolderForFile(module_cache_spec);
  Status err = platform->GetFile(module_spec.GetFileSpec(), module_cache_spec);
  return err;
}

lldb_private::Status PlatformDarwinDevice::GetSharedModuleWithLocalCache(
    const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
    llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules, bool *did_create_ptr,
    Process *process) {

  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log,
            "[%s] Trying to find module %s/%s - platform path %s/%s symbol "
            "path %s/%s",
            (IsHost() ? "host" : "remote"),
            module_spec.GetFileSpec().GetDirectory().AsCString(),
            module_spec.GetFileSpec().GetFilename().AsCString(),
            module_spec.GetPlatformFileSpec().GetDirectory().AsCString(),
            module_spec.GetPlatformFileSpec().GetFilename().AsCString(),
            module_spec.GetSymbolFileSpec().GetDirectory().AsCString(),
            module_spec.GetSymbolFileSpec().GetFilename().AsCString());

  Status err;

  if (CheckLocalSharedCache()) {
    err = GetModuleFromSharedCaches(module_spec, process, module_sp,
                                    old_modules, did_create_ptr);
    if (module_sp)
      return err;
  }

    // We failed to find the module in our shared cache. Let's see if we have a
    // copy in our device support directory.
    FileSpec device_support_spec(GetDeviceSupportDirectoryForOSVersion());
    device_support_spec.AppendPathComponent("Symbols");
    device_support_spec.AppendPathComponent(
        module_spec.GetFileSpec().GetPath());
    FileSystem::Instance().Resolve(device_support_spec);
    if (FileSystem::Instance().Exists(device_support_spec)) {
      ModuleSpec local_spec(device_support_spec, module_spec.GetUUID());
      err = ModuleList::GetSharedModule(local_spec, module_sp, old_modules,
                                        did_create_ptr);
      if (module_sp) {
        LLDB_LOGF(log,
                  "[%s] module %s was found in Device Support "
                  "directory: %s",
                  (IsHost() ? "host" : "remote"),
                  module_spec.GetFileSpec().GetPath().c_str(),
                  local_spec.GetFileSpec().GetPath().c_str());
        return err;
      }
  }

  err = ModuleList::GetSharedModule(module_spec, module_sp, old_modules,
                                    did_create_ptr);
  if (module_sp)
    return err;

  if (!IsHost()) {
    std::string cache_path(GetLocalCacheDirectory());
    // Only search for a locally cached file if we have a valid cache path
    if (!cache_path.empty()) {
      std::string module_path(module_spec.GetFileSpec().GetPath());
      cache_path.append(module_path);
      FileSpec module_cache_spec(cache_path);

      // if rsync is supported, always bring in the file - rsync will be very
      // efficient when files are the same on the local and remote end of the
      // connection
      if (this->GetSupportsRSync()) {
        err = BringInRemoteFile(this, module_spec, module_cache_spec);
        if (err.Fail())
          return err;
        if (FileSystem::Instance().Exists(module_cache_spec)) {
          Log *log = GetLog(LLDBLog::Platform);
          LLDB_LOGF(log, "[%s] module %s/%s was rsynced and is now there",
                    (IsHost() ? "host" : "remote"),
                    module_spec.GetFileSpec().GetDirectory().AsCString(),
                    module_spec.GetFileSpec().GetFilename().AsCString());
          ModuleSpec local_spec(module_cache_spec,
                                module_spec.GetArchitecture());
          module_sp = std::make_shared<Module>(local_spec);
          module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
          return Status();
        }
      }

      // try to find the module in the cache
      if (FileSystem::Instance().Exists(module_cache_spec)) {
        // get the local and remote MD5 and compare
        if (m_remote_platform_sp) {
          // when going over the *slow* GDB remote transfer mechanism we first
          // check the hashes of the files - and only do the actual transfer if
          // they differ
          auto MD5 = llvm::sys::fs::md5_contents(module_cache_spec.GetPath());
          if (!MD5)
            return Status(MD5.getError());

          Log *log = GetLog(LLDBLog::Platform);
          bool requires_transfer = true;
          llvm::ErrorOr<llvm::MD5::MD5Result> remote_md5 =
              m_remote_platform_sp->CalculateMD5(module_spec.GetFileSpec());
          if (std::error_code ec = remote_md5.getError())
            LLDB_LOG(log, "couldn't get md5 sum from remote: {0}",
                     ec.message());
          else
            requires_transfer = *MD5 != *remote_md5;
          if (requires_transfer) {
            // bring in the remote file
            LLDB_LOGF(log,
                      "[%s] module %s/%s needs to be replaced from remote copy",
                      (IsHost() ? "host" : "remote"),
                      module_spec.GetFileSpec().GetDirectory().AsCString(),
                      module_spec.GetFileSpec().GetFilename().AsCString());
            Status err =
                BringInRemoteFile(this, module_spec, module_cache_spec);
            if (err.Fail())
              return err;
          }
        }

        ModuleSpec local_spec(module_cache_spec, module_spec.GetArchitecture());
        module_sp = std::make_shared<Module>(local_spec);
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
        Log *log = GetLog(LLDBLog::Platform);
        LLDB_LOGF(log, "[%s] module %s/%s was found in the cache",
                  (IsHost() ? "host" : "remote"),
                  module_spec.GetFileSpec().GetDirectory().AsCString(),
                  module_spec.GetFileSpec().GetFilename().AsCString());
        return Status();
      }

      // bring in the remote module file
      LLDB_LOGF(log, "[%s] module %s/%s needs to come in remotely",
                (IsHost() ? "host" : "remote"),
                module_spec.GetFileSpec().GetDirectory().AsCString(),
                module_spec.GetFileSpec().GetFilename().AsCString());
      Status err = BringInRemoteFile(this, module_spec, module_cache_spec);
      if (err.Fail())
        return err;
      if (FileSystem::Instance().Exists(module_cache_spec)) {
        Log *log = GetLog(LLDBLog::Platform);
        LLDB_LOGF(log, "[%s] module %s/%s is now cached and fine",
                  (IsHost() ? "host" : "remote"),
                  module_spec.GetFileSpec().GetDirectory().AsCString(),
                  module_spec.GetFileSpec().GetFilename().AsCString());
        ModuleSpec local_spec(module_cache_spec, module_spec.GetArchitecture());
        module_sp = std::make_shared<Module>(local_spec);
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
        return Status();
      } else
        return Status::FromErrorString("unable to obtain valid module file");
    } else
      return Status::FromErrorString("no cache path");
  } else
    return Status::FromErrorString("unable to resolve module");
}
