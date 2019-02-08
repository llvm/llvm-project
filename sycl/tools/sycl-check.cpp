//==----------- sycl-check.cpp ---------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace cl;

// The base class every class that want to perform some action on
// cl::sycl::device.
class Action {
public:
  // This function will be called for each cl::sycl::device
  // Indentation should be printed as a beginning of each line the method prints
  virtual void visit(const sycl::device &Device,
                     const std::string &Indentation) = 0;
  virtual ~Action() = default;
};

// The class prints if the device is a default device of this type
class PrintIfDefaultDevice : public Action {
  // Contains default device of each type
  // Using cl_device_id as SYCL implementation always create new
  // cl::sycl::device object instead of reusing existing one.
  std::vector<cl_device_id> m_DefaultDevices;

public:
  PrintIfDefaultDevice() {
    // Fill vector of default devices for comparing in future
    try {
      sycl::cpu_selector CPUSelector;
      m_DefaultDevices.push_back(sycl::device(CPUSelector).get());
    } catch (cl::sycl::invalid_parameter_error &) {
    }

    try {
      sycl::gpu_selector GPUSelector;
      m_DefaultDevices.push_back(sycl::device(GPUSelector).get());
    } catch (cl::sycl::invalid_parameter_error &) {
    }

    try {
      sycl::accelerator_selector AcceleratorSelector;
      m_DefaultDevices.push_back(sycl::device(AcceleratorSelector).get());
    } catch (cl::sycl::invalid_parameter_error &) {
    }
  }

  void visit(const sycl::device &Device,
             const std::string &Indentation) override {
    auto DefaultIt = std::find(m_DefaultDevices.begin(), m_DefaultDevices.end(),
                               Device.get());

    if (DefaultIt != m_DefaultDevices.end())
      std::cout << Indentation
                << "NOTE! The device is a DEFAULT device of this type"
                << std::endl;
  }
};

// The class prints generic info about the device
class PrintGenericInfo : public Action {
  std::string convertDeviceType2String(sycl::info::device_type DeviceType) {
    switch (DeviceType) {
    case sycl::info::device_type::cpu:
      return std::string("CPU");
      break;
    case sycl::info::device_type::gpu:
      return std::string("GPU");
      break;
    case sycl::info::device_type::accelerator:
      return std::string("ACCELERATOR");
      break;
    case sycl::info::device_type::custom:
      return std::string("CUSTOM");
      break;
    case sycl::info::device_type::all:
    case sycl::info::device_type::host:
    case sycl::info::device_type::automatic:
    default:
      assert(!"Should be concrete OpenCL device");
      return std::string("UNKNOWN");
      break;
    };
  }

public:
  PrintGenericInfo() = default;
  void visit(const sycl::device &Device,
             const std::string &Indentation) override {

    const sycl::info::device_type DeviceType =
        Device.get_info<sycl::info::device::device_type>();

    const std::string DeviceName = Device.get_info<sycl::info::device::name>();

    const std::string DeviceVendor =
        Device.get_info<sycl::info::device::vendor>();

    const std::string DeviceDriverVersion =
        Device.get_info<sycl::info::device::driver_version>();

    std::cout << Indentation
              << "Type            : " << convertDeviceType2String(DeviceType)
              << std::endl;

    std::cout << Indentation << "Name            : " << DeviceName << std::endl;
    std::cout << Indentation << "Vendor          : " << DeviceVendor
              << std::endl;
    std::cout << Indentation << "Driver version  : " << DeviceDriverVersion
              << std::endl;
  }
};

// The class prints warning if the device is not tested or the driver version
// is too low
class PrintIfDeviceSupported : public Action {

  // Convert std::string "42.13.53" to std::vector<size_t> {42, 13, 53}
  std::vector<size_t> convertToInts(const std::string SourceString) {
    std::vector<size_t> Result;
    std::stringstream SStream(SourceString);
    size_t Value = 0;

    while (SStream >> Value) {
      Result.push_back(Value);

      if (SStream.peek() == '.')
        SStream.ignore();
    }
    return Result;
  }

  void checkDriverVersion(const std::vector<std::string> &RefVersionsStr,
                          const std::string &CurVersionStr,
                          const std::string &Indentation) {

    // Convert to vector of integers
    const std::vector<size_t> CurVersion = convertToInts(CurVersionStr);

    for (const std::string &RefVersionStr : RefVersionsStr) {

      // Convert to vector of integers
      const std::vector<size_t> RefVersion = convertToInts(RefVersionStr);

      // Check branch version
      if (CurVersion[0] != RefVersion[0]) {
        continue;
      }

      // Check sizes
      bool CheckPassed = CurVersion.size() == RefVersion.size();
      // Checking version going from major version to minor
      for (size_t I = 1; CheckPassed && I < RefVersion.size(); ++I)
        if (CurVersion[I] < RefVersion[I])
          CheckPassed = false;

      if (false == CheckPassed) {
        std::cout << Indentation << "WARNING! The device driver version it too "
                                    "low and not supported."
                  << std::endl;
        std::cout
            << Indentation
            << "NOTE! The minimum supported driver version for this device is "
            << RefVersionStr << std::endl;
      }
      return;
    }

    std::cout << Indentation
              << "WARNING! The device driver version is unrecognized"
              << std::endl;
  }

public:
  PrintIfDeviceSupported() = default;
  void visit(const sycl::device &Device,
             const std::string &Indentation) override {

    const std::string IntelName("Intel");
    const std::vector<std::string> MinIntelOCLGPUVersion = {
        MIN_INTEL_OCL_GPU_VERSION};
    const std::vector<std::string> MinIntelOCLCPUVersion = {
        MIN_INTEL_OCL_CPU_VERSION};

    const sycl::info::device_type DeviceType =
        Device.get_info<sycl::info::device::device_type>();
    const std::string DeviceName = Device.get_info<sycl::info::device::name>();
    const std::string DeviceDriverVersion =
        Device.get_info<sycl::info::device::driver_version>();

    // If Intel's device
    if (DeviceName.find(IntelName) != std::string::npos)
      switch (DeviceType) {
      case sycl::info::device_type::cpu:
        checkDriverVersion(MinIntelOCLCPUVersion, DeviceDriverVersion,
                           Indentation);
        return;
      case sycl::info::device_type::gpu:
        checkDriverVersion(MinIntelOCLGPUVersion, DeviceDriverVersion,
                           Indentation);
        return;
      case sycl::info::device_type::accelerator:
        std::cout << Indentation
                  << "WARNING! The device is not officially supported."
                  << std::endl;
        return;
      case sycl::info::device_type::custom:
      case sycl::info::device_type::all:
      case sycl::info::device_type::host:
      case sycl::info::device_type::automatic:
      default:
        assert(!"Should be concrete OpenCL device");
        return;
      }
    // Non-Intel devices were not tested
    std::cout << Indentation
              << "WARNING! The device is not officially supported."
              << std::endl;
  }
};

// The class prints if the device is a default device of this type
class CheckSPIRVSupport : public Action {
public:
  void visit(const sycl::device &Device,
             const std::string &Indentation) override {

    const std::vector<size_t> MinimumDeviceVersion = {2, 1};

    const std::string DeviceVersionStr =
        Device.get_info<sycl::info::device::version>();
    std::vector<size_t> DeviceVersion = {0, 0};

    if (!DeviceVersionStr.compare(0, 10, "OpenCL 2.2"))
      DeviceVersion = {2, 2};
    else if (!DeviceVersionStr.compare(0, 10, "OpenCL 2.1"))
      DeviceVersion = {2, 1};
    else if (!DeviceVersionStr.compare(0, 10, "OpenCL 2.0"))
      DeviceVersion = {2, 0};
    else if (!DeviceVersionStr.compare(0, 10, "OpenCL 1.2"))
      DeviceVersion = {1, 2};
    else if (!DeviceVersionStr.compare(0, 10, "OpenCL 1.0"))
      DeviceVersion = {1, 0};

    if (DeviceVersion[0] < MinimumDeviceVersion[0] ||
        DeviceVersion[1] < MinimumDeviceVersion[1]) {
      std::cout << Indentation
                << "WARNING! Device doesn't support SPIRV format." << std::endl;
      return;
    }
  }
};

int main() {
  try {
    std::vector<std::unique_ptr<Action>> Actions;
    // Fill vector of actions to be performed on each device
    // Note! Actions are performed in order they are placed in vector
    Actions.emplace_back(new PrintGenericInfo());
    Actions.emplace_back(new PrintIfDefaultDevice());
    Actions.emplace_back(new PrintIfDeviceSupported());
    Actions.emplace_back(new CheckSPIRVSupport());

    std::cout << "Available OpenCL devices:" << std::endl;
    size_t DeviceNumber = 0;
    for (const sycl::device &Device : sycl::device::get_devices()) {

      // SYCL host device is not OpenCL device, skipping...
      if (Device.get_info<sycl::info::device::device_type>() ==
          sycl::info::device_type::host)
        continue;

      std::cout << "Device [" << DeviceNumber << "]:" << std::endl;

      const std::string Tab("    ");
      for (std::unique_ptr<Action> &Act : Actions) {
        Act->visit(Device, Tab);
      }
      ++DeviceNumber;
    }
  } catch (...) {
    std::cout << "Unhandled error happened." << std::endl;
    return 1;
  }
  return 0;
}
