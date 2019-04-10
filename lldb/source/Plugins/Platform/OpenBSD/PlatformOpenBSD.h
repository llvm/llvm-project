//===-- PlatformOpenBSD.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformOpenBSD_h_
#define liblldb_PlatformOpenBSD_h_

#include "Plugins/Platform/POSIX/PlatformPOSIX.h"

namespace lldb_private {
namespace platform_openbsd {

class PlatformOpenBSD : public PlatformPOSIX {
public:
  PlatformOpenBSD(bool is_host);

  ~PlatformOpenBSD() override;

  static void Initialize();

  static void Terminate();

  // lldb_private::PluginInterface functions
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static ConstString GetPluginNameStatic(bool is_host);

  static const char *GetPluginDescriptionStatic(bool is_host);

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override { return 1; }

  // lldb_private::Platform functions
  const char *GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  void GetStatus(Stream &strm) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx, ArchSpec &arch) override;

  bool CanDebugProcess() override;

  void CalculateTrapHandlerSymbolNames() override;

  MmapArgList GetMmapArgumentList(const ArchSpec &arch, lldb::addr_t addr,
                                  lldb::addr_t length, unsigned prot,
                                  unsigned flags, lldb::addr_t fd,
                                  lldb::addr_t offset) override;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformOpenBSD);
};

} // namespace platform_openbsd
} // namespace lldb_private

#endif // liblldb_PlatformOpenBSD_h_
