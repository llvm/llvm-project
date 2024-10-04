//===------- Offload API tests - gtest environment ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "environment.hpp"
#include "fixtures.hpp"
#include "llvm/Support/CommandLine.h"
#include <offload_api.h>

using namespace llvm;

static cl::opt<std::string>
    SelectedPlatform("platform", cl::desc("Only test the specified platform"),
                     cl::value_desc("platform"));

std::ostream &operator<<(std::ostream &Out,
                         const offload_platform_handle_t &Platform) {
  size_t Size;
  offloadPlatformGetInfo(Platform, OFFLOAD_PLATFORM_INFO_NAME, 0, nullptr,
                         &Size);
  std::vector<char> Name(Size);
  offloadPlatformGetInfo(Platform, OFFLOAD_PLATFORM_INFO_NAME, Size,
                         Name.data(), nullptr);
  Out << Name.data();
  return Out;
}

std::ostream &
operator<<(std::ostream &Out,
           const std::vector<offload_platform_handle_t> &Platforms) {
  for (auto Platform : Platforms) {
    Out << "\n  * \"" << Platform << "\"";
  }
  return Out;
}

const std::vector<offload_platform_handle_t> &TestEnvironment::getPlatforms() {
  static std::vector<offload_platform_handle_t> Platforms{};

  if (Platforms.empty()) {
    uint32_t PlatformCount = 0;
    offloadPlatformGet(0, nullptr, &PlatformCount);
    if (PlatformCount > 0) {
      Platforms.resize(PlatformCount);
      offloadPlatformGet(PlatformCount, Platforms.data(), nullptr);
    }
  }

  return Platforms;
}

// Get a single platform, which may be selected by the user.
offload_platform_handle_t TestEnvironment::getPlatform() {
  static offload_platform_handle_t Platform = nullptr;
  const auto &Platforms = getPlatforms();

  if (!Platform) {
    if (SelectedPlatform != "") {
      for (const auto CandidatePlatform : Platforms) {
        std::stringstream PlatformName;
        PlatformName << CandidatePlatform;
        if (SelectedPlatform == PlatformName.str()) {
          Platform = CandidatePlatform;
          return Platform;
        }
      }
      std::cout << "No platform found with the name \"" << SelectedPlatform
                << "\". Choose from:" << Platforms << "\n";
      std::exit(1);
    } else {
      // Pick a single platform. We prefer one that has available devices, but
      // just pick the first initially in case none have any devices.
      Platform = Platforms[0];
      for (auto CandidatePlatform : Platforms) {
        uint32_t NumDevices = 0;
        if (offloadDeviceGet(CandidatePlatform, OFFLOAD_DEVICE_TYPE_ALL, 0,
                             nullptr, &NumDevices) == OFFLOAD_SUCCESS) {
          if (NumDevices > 0) {
            Platform = CandidatePlatform;
            break;
          }
        }
      }
    }
  }

  return Platform;
}
