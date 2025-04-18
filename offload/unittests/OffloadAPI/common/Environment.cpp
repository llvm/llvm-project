//===------- Offload API tests - gtest environment ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Environment.hpp"
#include "Fixtures.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include <OffloadAPI.h>
#include <fstream>

using namespace llvm;

// Wrapper so we don't have to constantly init and shutdown Offload in every
// test, while having sensible lifetime for the platform environment
struct OffloadInitWrapper {
  OffloadInitWrapper() { olInit(); }
  ~OffloadInitWrapper() { olShutDown(); }
};
static OffloadInitWrapper Wrapper{};

static cl::opt<std::string>
    SelectedPlatform("platform", cl::desc("Only test the specified platform"),
                     cl::value_desc("platform"));

raw_ostream &operator<<(raw_ostream &Out,
                        const ol_platform_handle_t &Platform) {
  size_t Size;
  olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_NAME, &Size);
  std::vector<char> Name(Size);
  olGetPlatformInfo(Platform, OL_PLATFORM_INFO_NAME, Size, Name.data());
  Out << Name.data();
  return Out;
}

void printPlatforms() {
  SmallDenseSet<ol_platform_handle_t> Platforms;
  using DeviceVecT = SmallVector<ol_device_handle_t, 8>;
  DeviceVecT Devices{};

  olIterateDevices(
      [](ol_device_handle_t D, void *Data) {
        static_cast<DeviceVecT *>(Data)->push_back(D);
        return true;
      },
      &Devices);

  for (auto &Device : Devices) {
    ol_platform_handle_t Platform;
    olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                    &Platform);
    Platforms.insert(Platform);
  }

  for (const auto &Platform : Platforms) {
    errs() << "  * " << Platform << "\n";
  }
}

ol_device_handle_t TestEnvironment::getDevice() {
  static ol_device_handle_t Device = nullptr;

  if (!Device) {
    if (SelectedPlatform != "") {
      olIterateDevices(
          [](ol_device_handle_t D, void *Data) {
            ol_platform_handle_t Platform;
            olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                            &Platform);

            std::string PlatformName;
            raw_string_ostream S(PlatformName);
            S << Platform;

            if (PlatformName == SelectedPlatform) {
              *(static_cast<ol_device_handle_t *>(Data)) = D;
              return false;
            }

            return true;
          },
          &Device);

      if (Device == nullptr) {
        errs() << "No device found with the platform \"" << SelectedPlatform
               << "\". Choose from:"
               << "\n";
        printPlatforms();
        std::exit(1);
      }
    } else {
      olIterateDevices(
          [](ol_device_handle_t D, void *Data) {
            *(static_cast<ol_device_handle_t *>(Data)) = D;
            return false;
          },
          &Device);
    }
  }

  return Device;
}

ol_device_handle_t TestEnvironment::getHostDevice() {
  static ol_device_handle_t HostDevice = nullptr;

  if (!HostDevice) {
    olIterateDevices(
        [](ol_device_handle_t D, void *Data) {
          ol_platform_handle_t Platform;
          olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                          &Platform);
          ol_platform_backend_t Backend;
          olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                            &Backend);

          if (Backend == OL_PLATFORM_BACKEND_HOST) {
            *(static_cast<ol_device_handle_t *>(Data)) = D;
            return false;
          }

          return true;
        },
        &HostDevice);
  }

  return HostDevice;
}

// TODO: Allow overriding via cmd line arg
const std::string DeviceBinsDirectory = DEVICE_CODE_PATH;

bool TestEnvironment::loadDeviceBinary(
    const std::string &BinaryName, ol_device_handle_t Device,
    std::unique_ptr<MemoryBuffer> &BinaryOut) {

  // Get the platform type
  ol_platform_handle_t Platform;
  olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform), &Platform);
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                    &Backend);
  std::string FileExtension;
  if (Backend == OL_PLATFORM_BACKEND_AMDGPU) {
    FileExtension = ".amdgpu.bin";
  } else if (Backend == OL_PLATFORM_BACKEND_CUDA) {
    FileExtension = ".nvptx64.bin";
  } else {
    errs() << "Unsupported platform type for a device binary test.\n";
    return false;
  }

  std::string SourcePath =
      DeviceBinsDirectory + "/" + BinaryName + FileExtension;

  auto SourceFile = MemoryBuffer::getFile(SourcePath, false, false);
  if (!SourceFile) {
    errs() << "failed to read device binary file: " + SourcePath;
    return false;
  }

  BinaryOut = std::move(SourceFile.get());
  return true;
}
