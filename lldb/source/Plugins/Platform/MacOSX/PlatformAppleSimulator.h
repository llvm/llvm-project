//===-- PlatformAppleSimulator.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformAppleSimulator_h_
#define liblldb_PlatformAppleSimulator_h_

#include <mutex>

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/Platform/MacOSX/objcxx/PlatformiOSSimulatorCoreSimulatorSupport.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/ADT/Optional.h"

class PlatformAppleSimulator : public PlatformDarwin {
public:
  //------------------------------------------------------------
  // Class Functions
  //------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  //------------------------------------------------------------
  // Class Methods
  //------------------------------------------------------------
  PlatformAppleSimulator();

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

  lldb_private::FileSpec GetCoreSimulatorPath();

  void LoadCoreSimulator();

#if defined(__APPLE__)
  CoreSimulatorSupport::Device GetSimulatorDevice();
#endif

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformAppleSimulator);
};

#endif // liblldb_PlatformAppleSimulator_h_
