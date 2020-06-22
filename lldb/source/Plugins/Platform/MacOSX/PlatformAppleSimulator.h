//===-- PlatformAppleSimulator.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H

#include <mutex>

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/Platform/MacOSX/objcxx/PlatformiOSSimulatorCoreSimulatorSupport.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/ADT/Optional.h"

class PlatformAppleSimulator : public PlatformDarwin {
public:
  // Class Functions
  static void Initialize();

  static void Terminate();

  // Class Methods
  PlatformAppleSimulator(
      CoreSimulatorSupport::DeviceType::ProductFamilyID kind);

  virtual ~PlatformAppleSimulator();

  lldb_private::Status
  LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) override;

  void GetStatus(lldb_private::Stream &strm) override;

  lldb_private::Status ConnectRemote(lldb_private::Args &args) override;

  lldb_private::Status DisconnectRemote() override;

  lldb::ProcessSP DebugProcess(lldb_private::ProcessLaunchInfo &launch_info,
                               lldb_private::Debugger &debugger,
                               lldb_private::Target *target,
                               lldb_private::Status &error) override;

protected:
  std::mutex m_core_sim_path_mutex;
  llvm::Optional<lldb_private::FileSpec> m_core_simulator_framework_path;
  llvm::Optional<CoreSimulatorSupport::Device> m_device;
  CoreSimulatorSupport::DeviceType::ProductFamilyID m_kind;

  lldb_private::FileSpec GetCoreSimulatorPath();

  void LoadCoreSimulator();

#if defined(__APPLE__)
  CoreSimulatorSupport::Device GetSimulatorDevice();
#endif

private:
  PlatformAppleSimulator(const PlatformAppleSimulator &) = delete;
  const PlatformAppleSimulator &
  operator=(const PlatformAppleSimulator &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMAPPLESIMULATOR_H
