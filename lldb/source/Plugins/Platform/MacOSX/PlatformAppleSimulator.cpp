//===-- PlatformAppleSimulator.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformAppleSimulator.h"

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <mutex>
#include <thread>
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Support/Threading.h"

using namespace lldb;
using namespace lldb_private;

#if !defined(__APPLE__)
#define UNSUPPORTED_ERROR ("Apple simulators aren't supported on this platform")
#endif

// Static Functions
void PlatformAppleSimulator::Initialize() { PlatformDarwin::Initialize(); }

void PlatformAppleSimulator::Terminate() { PlatformDarwin::Terminate(); }

/// Default Constructor
PlatformAppleSimulator::PlatformAppleSimulator()
    : PlatformDarwin(true), m_core_sim_path_mutex(),
      m_core_simulator_framework_path(), m_device() {}

/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
PlatformAppleSimulator::~PlatformAppleSimulator() {}

lldb_private::Status PlatformAppleSimulator::LaunchProcess(
    lldb_private::ProcessLaunchInfo &launch_info) {
#if defined(__APPLE__)
  LoadCoreSimulator();
  CoreSimulatorSupport::Device device(GetSimulatorDevice());

  if (device.GetState() != CoreSimulatorSupport::Device::State::Booted) {
    Status boot_err;
    device.Boot(boot_err);
    if (boot_err.Fail())
      return boot_err;
  }

  auto spawned = device.Spawn(launch_info);

  if (spawned) {
    launch_info.SetProcessID(spawned.GetPID());
    return Status();
  } else
    return spawned.GetError();
#else
  Status err;
  err.SetErrorString(UNSUPPORTED_ERROR);
  return err;
#endif
}

void PlatformAppleSimulator::GetStatus(Stream &strm) {
#if defined(__APPLE__)
  // This will get called by subclasses, so just output status on the current
  // simulator
  PlatformAppleSimulator::LoadCoreSimulator();

  std::string developer_dir = HostInfo::GetXcodeDeveloperDirectory().GetPath();
  CoreSimulatorSupport::DeviceSet devices =
      CoreSimulatorSupport::DeviceSet::GetAvailableDevices(
          developer_dir.c_str());
  const size_t num_devices = devices.GetNumDevices();
  if (num_devices) {
    strm.Printf("Available devices:\n");
    for (size_t i = 0; i < num_devices; ++i) {
      CoreSimulatorSupport::Device device = devices.GetDeviceAtIndex(i);
      strm.Printf("   %s: %s\n", device.GetUDID().c_str(),
                  device.GetName().c_str());
    }

    if (m_device.hasValue() && m_device->operator bool()) {
      strm.Printf("Current device: %s: %s", m_device->GetUDID().c_str(),
                  m_device->GetName().c_str());
      if (m_device->GetState() == CoreSimulatorSupport::Device::State::Booted) {
        strm.Printf(" state = booted");
      }
      strm.Printf("\nType \"platform connect <ARG>\" where <ARG> is a device "
                  "UDID or a device name to disconnect and connect to a "
                  "different device.\n");

    } else {
      strm.Printf("No current device is selected, \"platform connect <ARG>\" "
                  "where <ARG> is a device UDID or a device name to connect to "
                  "a specific device.\n");
    }

  } else {
    strm.Printf("No devices are available.\n");
  }
#else
  strm.Printf(UNSUPPORTED_ERROR);
#endif
}

Status PlatformAppleSimulator::ConnectRemote(Args &args) {
#if defined(__APPLE__)
  Status error;
  if (args.GetArgumentCount() == 1) {
    if (m_device)
      DisconnectRemote();
    PlatformAppleSimulator::LoadCoreSimulator();
    const char *arg_cstr = args.GetArgumentAtIndex(0);
    if (arg_cstr) {
      std::string arg_str(arg_cstr);
      std::string developer_dir = HostInfo::GetXcodeDeveloperDirectory().GetPath();
      CoreSimulatorSupport::DeviceSet devices =
          CoreSimulatorSupport::DeviceSet::GetAvailableDevices(
              developer_dir.c_str());
      devices.ForEach(
          [this, &arg_str](const CoreSimulatorSupport::Device &device) -> bool {
            if (arg_str == device.GetUDID() || arg_str == device.GetName()) {
              m_device = device;
              return false; // Stop iterating
            } else {
              return true; // Keep iterating
            }
          });
      if (!m_device)
        error.SetErrorStringWithFormat(
            "no device with UDID or name '%s' was found", arg_cstr);
    }
  } else {
    error.SetErrorString("this command take a single UDID argument of the "
                         "device you want to connect to.");
  }
  return error;
#else
  Status err;
  err.SetErrorString(UNSUPPORTED_ERROR);
  return err;
#endif
}

Status PlatformAppleSimulator::DisconnectRemote() {
#if defined(__APPLE__)
  m_device.reset();
  return Status();
#else
  Status err;
  err.SetErrorString(UNSUPPORTED_ERROR);
  return err;
#endif
}

lldb::ProcessSP PlatformAppleSimulator::DebugProcess(
    ProcessLaunchInfo &launch_info, Debugger &debugger,
    Target *target, // Can be NULL, if NULL create a new target, else use
                    // existing one
    Status &error) {
#if defined(__APPLE__)
  ProcessSP process_sp;
  // Make sure we stop at the entry point
  launch_info.GetFlags().Set(eLaunchFlagDebug);
  // We always launch the process we are going to debug in a separate process
  // group, since then we can handle ^C interrupts ourselves w/o having to
  // worry about the target getting them as well.
  launch_info.SetLaunchInSeparateProcessGroup(true);

  error = LaunchProcess(launch_info);
  if (error.Success()) {
    if (launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID) {
      ProcessAttachInfo attach_info(launch_info);
      process_sp = Attach(attach_info, debugger, target, error);
      if (process_sp) {
        launch_info.SetHijackListener(attach_info.GetHijackListener());

        // Since we attached to the process, it will think it needs to detach
        // if the process object just goes away without an explicit call to
        // Process::Kill() or Process::Detach(), so let it know to kill the
        // process if this happens.
        process_sp->SetShouldDetach(false);

        // If we didn't have any file actions, the pseudo terminal might have
        // been used where the slave side was given as the file to open for
        // stdin/out/err after we have already opened the master so we can
        // read/write stdin/out/err.
        int pty_fd = launch_info.GetPTY().ReleaseMasterFileDescriptor();
        if (pty_fd != PseudoTerminal::invalid_fd) {
          process_sp->SetSTDIOFileDescriptor(pty_fd);
        }
      }
    }
  }

  return process_sp;
#else
  return ProcessSP();
#endif
}

FileSpec PlatformAppleSimulator::GetCoreSimulatorPath() {
#if defined(__APPLE__)
  std::lock_guard<std::mutex> guard(m_core_sim_path_mutex);
  if (!m_core_simulator_framework_path.hasValue()) {
    if (FileSpec fspec = HostInfo::GetXcodeDeveloperDirectory()) {
      std::string developer_dir = fspec.GetPath();
      StreamString cs_path;
      cs_path.Printf(
          "%s/Library/PrivateFrameworks/CoreSimulator.framework/CoreSimulator",
          developer_dir.c_str());
      m_core_simulator_framework_path = FileSpec(cs_path.GetData());
      FileSystem::Instance().Resolve(*m_core_simulator_framework_path);
    }
  }

  return m_core_simulator_framework_path.getValue();
#else
  return FileSpec();
#endif
}

void PlatformAppleSimulator::LoadCoreSimulator() {
#if defined(__APPLE__)
  static llvm::once_flag g_load_core_sim_flag;
  llvm::call_once(g_load_core_sim_flag, [this] {
    const std::string core_sim_path(GetCoreSimulatorPath().GetPath());
    if (core_sim_path.size())
      dlopen(core_sim_path.c_str(), RTLD_LAZY);
  });
#endif
}

#if defined(__APPLE__)
CoreSimulatorSupport::Device PlatformAppleSimulator::GetSimulatorDevice() {
  if (!m_device.hasValue()) {
    const CoreSimulatorSupport::DeviceType::ProductFamilyID dev_id =
        CoreSimulatorSupport::DeviceType::ProductFamilyID::iPhone;
    std::string developer_dir = HostInfo::GetXcodeDeveloperDirectory().GetPath();
    m_device = CoreSimulatorSupport::DeviceSet::GetAvailableDevices(
                   developer_dir.c_str())
                   .GetFanciest(dev_id);
  }

  if (m_device.hasValue())
    return m_device.getValue();
  else
    return CoreSimulatorSupport::Device();
}
#endif
