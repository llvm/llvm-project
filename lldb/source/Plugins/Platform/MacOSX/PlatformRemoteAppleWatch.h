//===-- PlatformRemoteAppleWatch.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEAPPLEWATCH_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEAPPLEWATCH_H

#include <string>
#include <vector>

#include "lldb/Utility/FileSpec.h"

#include "PlatformRemoteDarwinDevice.h"

#include "llvm/Support/FileSystem.h"

class PlatformRemoteAppleWatch : public PlatformRemoteDarwinDevice {
public:
  PlatformRemoteAppleWatch();

  // Class Functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "remote-watchos"; }

  static llvm::StringRef GetDescriptionStatic();

  // lldb_private::Platform functions

  llvm::StringRef GetDescription() override { return GetDescriptionStatic(); }

  // lldb_private::PluginInterface functions
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // lldb_private::Platform functions

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

protected:
  llvm::StringRef GetDeviceSupportDirectoryName() override;
  llvm::StringRef GetPlatformName() override;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEAPPLEWATCH_H
