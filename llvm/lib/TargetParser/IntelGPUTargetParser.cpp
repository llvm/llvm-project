//===-- IntelGPUTargetParser - Parser for Intel GPU targets ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser for the Intel GPU list.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/IntelGPUTargetParser.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;
using namespace IntelGPU;

// A GPU IP version, the "GMDID", packs the architecture into the bits above the
// release, which in turn sits above the revision. The bits between the release
// and the revision are reserved.
static constexpr uint32_t GMDIDArchitectureShift = 22;
static constexpr uint32_t GMDIDReleaseShift = 14;
static constexpr uint32_t GMDIDReleaseMask = 0xff;
static constexpr uint32_t GMDIDRevisionMask = 0x3f;

// The bits that identify a device: the architecture and the release. Neither
// the revision nor the reserved bits take part in the lookup, because every
// stepping of a release is one device as far as the compiler is concerned.
static constexpr uint32_t GMDIDDeviceMask = ~0u << GMDIDReleaseShift;

// Pack an architecture and a release the way a GPU IP version does, so that a
// row of the table can be compared against a reported version as it is.
static constexpr uint32_t packDevice(uint32_t Architecture, uint32_t Release) {
  return (Architecture << GMDIDArchitectureShift) |
         (Release << GMDIDReleaseShift);
}

StringRef llvm::IntelGPU::getArchName(uint32_t GPUIPVersion) {
  const uint32_t Device = GPUIPVersion & GMDIDDeviceMask;
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_TARGET, IGCA_SUFFIX) \
  if (Device == packDevice(ARCHITECTURE, RELEASE))                             \
    return NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  return "";
}

std::string llvm::IntelGPU::getNumericArchName(uint32_t GPUIPVersion) {
  const uint32_t Architecture = GPUIPVersion >> GMDIDArchitectureShift;
  const uint32_t Release =
      (GPUIPVersion >> GMDIDReleaseShift) & GMDIDReleaseMask;
  const uint32_t Revision = GPUIPVersion & GMDIDRevisionMask;
  return ("xe_" + Twine(Architecture) + "." + Twine(Release) + "." +
          Twine(Revision))
      .str();
}
