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
#include <OffloadAPI.h>

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

std::ostream &operator<<(std::ostream &Out,
                         const ol_platform_handle_t &Platform) {
  size_t Size;
  olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_NAME, &Size);
  std::vector<char> Name(Size);
  olGetPlatformInfo(Platform, OL_PLATFORM_INFO_NAME, Size, Name.data());
  Out << Name.data();
  return Out;
}

std::ostream &operator<<(std::ostream &Out,
                         const std::vector<ol_platform_handle_t> &Platforms) {
  for (auto Platform : Platforms) {
    Out << "\n  * \"" << Platform << "\"";
  }
  return Out;
}

const std::vector<ol_platform_handle_t> &TestEnvironment::getPlatforms() {
  static std::vector<ol_platform_handle_t> Platforms{};

  if (Platforms.empty()) {
    uint32_t PlatformCount = 0;
    olGetPlatformCount(&PlatformCount);
    if (PlatformCount > 0) {
      Platforms.resize(PlatformCount);
      olGetPlatform(PlatformCount, Platforms.data());
    }
  }

  return Platforms;
}

// Get a single platform, which may be selected by the user.
ol_platform_handle_t TestEnvironment::getPlatform() {
  static ol_platform_handle_t Platform = nullptr;
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
        if (olGetDeviceCount(CandidatePlatform, &NumDevices) == OL_SUCCESS) {
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
