//===-- PlatformRemoteDarwinDevice.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEDARWINDEVICE_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEDARWINDEVICE_H

#include <string>

#include "PlatformDarwin.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/Support/FileSystem.h"

class PlatformRemoteDarwinDevice : public PlatformDarwin {
public:
  PlatformRemoteDarwinDevice();

  ~PlatformRemoteDarwinDevice() override;

  // lldb_private::Platform functions
  lldb_private::Status ResolveExecutable(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr) override;

  void GetStatus(lldb_private::Stream &strm) override;

  virtual lldb_private::Status
  GetSymbolFile(const lldb_private::FileSpec &platform_file,
                const lldb_private::UUID *uuid_ptr,
                lldb_private::FileSpec &local_file);

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules,
                  bool *did_create_ptr) override;

  void
  AddClangModuleCompilationOptions(lldb_private::Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, lldb_private::XcodeSDK::Type::iPhoneOS);
  }

protected:
  struct SDKDirectoryInfo {
    SDKDirectoryInfo(const lldb_private::FileSpec &sdk_dir_spec);
    lldb_private::FileSpec directory;
    lldb_private::ConstString build;
    llvm::VersionTuple version;
    bool user_cached;
  };

  typedef std::vector<SDKDirectoryInfo> SDKDirectoryInfoCollection;

  std::mutex m_sdk_dir_mutex;
  SDKDirectoryInfoCollection m_sdk_directory_infos;
  std::string m_device_support_directory;
  std::string m_device_support_directory_for_os_version;
  std::string m_build_update;
  uint32_t m_last_module_sdk_idx = UINT32_MAX;
  uint32_t m_connected_module_sdk_idx = UINT32_MAX;

  bool UpdateSDKDirectoryInfosIfNeeded();

  const char *GetDeviceSupportDirectory();

  const char *GetDeviceSupportDirectoryForOSVersion();

  const SDKDirectoryInfo *GetSDKDirectoryForLatestOSVersion();

  const SDKDirectoryInfo *GetSDKDirectoryForCurrentOSVersion();

  static lldb_private::FileSystem::EnumerateDirectoryResult
  GetContainedFilesIntoVectorOfStringsCallback(void *baton,
                                               llvm::sys::fs::file_type ft,
                                               llvm::StringRef path);

  uint32_t FindFileInAllSDKs(const char *platform_file_path,
                             lldb_private::FileSpecList &file_list);

  bool GetFileInSDK(const char *platform_file_path, uint32_t sdk_idx,
                    lldb_private::FileSpec &local_file);

  uint32_t FindFileInAllSDKs(const lldb_private::FileSpec &platform_file,
                             lldb_private::FileSpecList &file_list);

  uint32_t GetConnectedSDKIndex();

  // Get index of SDK in SDKDirectoryInfoCollection by its pointer and return
  // UINT32_MAX if that SDK not found.
  uint32_t GetSDKIndexBySDKDirectoryInfo(const SDKDirectoryInfo *sdk_info);

  virtual llvm::StringRef GetDeviceSupportDirectoryName() = 0;
  virtual llvm::StringRef GetPlatformName() = 0;

private:
  PlatformRemoteDarwinDevice(const PlatformRemoteDarwinDevice &) = delete;
  const PlatformRemoteDarwinDevice &
  operator=(const PlatformRemoteDarwinDevice &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEDARWINDEVICE_H
