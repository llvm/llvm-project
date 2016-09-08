//===-- PlatformAndroid.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformAndroid_h_
#define liblldb_PlatformAndroid_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "Plugins/Platform/Linux/PlatformLinux.h"

namespace lldb_private {
namespace platform_android {

class PlatformAndroid : public platform_linux::PlatformLinux {
public:
  PlatformAndroid(bool is_host);

  ~PlatformAndroid() override;

  static void Initialize();

  static void Terminate();

  //------------------------------------------------------------
  // lldb_private::PluginInterface functions
  //------------------------------------------------------------
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static ConstString GetPluginNameStatic(bool is_host);

  static const char *GetPluginDescriptionStatic(bool is_host);

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override { return 1; }

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------

  Error ConnectRemote(Args &args) override;

  Error GetFile(const FileSpec &source, const FileSpec &destination) override;

  Error PutFile(const FileSpec &source, const FileSpec &destination,
                uint32_t uid = UINT32_MAX, uint32_t gid = UINT32_MAX) override;

  uint32_t GetSdkVersion();

  bool GetRemoteOSVersion() override;

  Error DisconnectRemote() override;

  uint32_t GetDefaultMemoryCacheLineSize() override;

protected:
  const char *GetCacheHostname() override;

  Error DownloadModuleSlice(const FileSpec &src_file_spec,
                            const uint64_t src_offset, const uint64_t src_size,
                            const FileSpec &dst_file_spec) override;

  Error DownloadSymbolFile(const lldb::ModuleSP &module_sp,
                           const FileSpec &dst_file_spec) override;

  const char *GetLibdlFunctionDeclarations() const override;

private:
  std::string m_device_id;
  uint32_t m_sdk_version;

  DISALLOW_COPY_AND_ASSIGN(PlatformAndroid);
};

} // namespace platofor_android
} // namespace lldb_private

#endif // liblldb_PlatformAndroid_h_
