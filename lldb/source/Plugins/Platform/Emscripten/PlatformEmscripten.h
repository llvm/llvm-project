//===-- PlatformEmscripten.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_EMSCRIPTEN_PLATFORMEMSCRIPTEN_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_EMSCRIPTEN_PLATFORMEMSCRIPTEN_H

#include "Plugins/Platform/POSIX/PlatformPOSIX.h"

namespace lldb_private::platform_emscripten {

class PlatformEmscripten : public PlatformPOSIX {
public:
  explicit PlatformEmscripten(bool is_host);

  static void Initialize();
  static void Terminate();

  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static llvm::StringRef GetPluginNameStatic(bool is_host) {
    return is_host ? Platform::GetHostPlatformName() : "remote-emscripten";
  }

  static llvm::StringRef GetPluginDescriptionStatic(bool is_host);

  llvm::StringRef GetPluginName() override {
    return GetPluginNameStatic(IsHost());
  }

  llvm::StringRef GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override;

  bool CanDebugProcess() override;

private:
  std::vector<ArchSpec> m_supported_architectures;
};

} // namespace lldb_private::platform_emscripten

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_EMSCRIPTEN_PLATFORMEMSCRIPTEN_H
