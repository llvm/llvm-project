//===-- PlatformDarwinDevice.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWINDEVICE_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWINDEVICE_H

#include "PlatformDarwin.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace lldb_private {

/// Abstract Darwin platform with a potential device support directory.
class PlatformDarwinDevice : public PlatformDarwin {
public:
  using PlatformDarwin::PlatformDarwin;
  ~PlatformDarwinDevice() override;

protected:
  virtual Status GetSharedModuleWithLocalCache(
      const ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules, bool *did_create_ptr,
      lldb_private::Process *process);

  struct SDKDirectoryInfo {
    SDKDirectoryInfo(const FileSpec &sdk_dir_spec, llvm::StringRef dirname_str);
    FileSpec directory;
    ConstString build;
    llvm::VersionTuple version;
  };

  typedef std::vector<SDKDirectoryInfo> SDKDirectoryInfoCollection;

  /// Look for expanded shared cache directories under the given dir.
  /// Expanded shared cache directories found under the given dir will
  /// be added to \a m_sdk_directory_infos.
  ///
  /// \param[in] dir
  ///     Directory to search under.
  ///
  /// \param[in] log_msg_descriptor
  ///     Text to describe the origin of this directory, in logging.
  void AddSharedCacheDirectory(llvm::StringRef dir,
                               llvm::StringRef log_msg_descriptor);

  bool UpdateSDKDirectoryInfosIfNeeded();

  const SDKDirectoryInfo *GetSDKDirectoryForLatestOSVersion();
  const SDKDirectoryInfo *GetSDKDirectoryForCurrentOSVersion();

  static FileSystem::EnumerateDirectoryResult
  GetContainedFilesIntoVectorOfFileSpecsCallback(void *baton,
                                                 llvm::sys::fs::file_type ft,
                                                 llvm::StringRef path);

  const char *GetDeviceSupportDirectory();
  const char *GetDeviceSupportDirectoryForOSVersion();

  virtual llvm::StringRef GetPlatformName() = 0;
  virtual llvm::StringRef GetDeviceSupportDirectoryName() = 0;

  std::mutex m_sdk_dir_mutex;
  SDKDirectoryInfoCollection m_sdk_directory_infos;

private:
  std::string m_device_support_directory;
  std::string m_device_support_directory_for_os_version;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMDARWINDEVICE_H
