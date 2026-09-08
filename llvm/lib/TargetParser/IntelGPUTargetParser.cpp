//===-- IntelGPUTargetParser - Parser for Intel GPU targets ----*- C++ -*-===//
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

// A GMDID packs the architecture, release and revision of the GPU IP.
static constexpr uint32_t GMDIDArchitectureShift = 22;
static constexpr uint32_t GMDIDReleaseShift = 14;
static constexpr uint32_t GMDIDReleaseMask = 0xff;
static constexpr uint32_t GMDIDRevisionMask = 0x3f;

GMDID llvm::IntelGPU::decodeGMDID(uint32_t IPVersion) {
  return {IPVersion >> GMDIDArchitectureShift,
          (IPVersion >> GMDIDReleaseShift) & GMDIDReleaseMask,
          IPVersion & GMDIDRevisionMask};
}

GPUKind llvm::IntelGPU::getKindForGMDID(GMDID ID) {
  // Only INTEL_GPU rows are expanded, so a compatibility name can never match.
  // The rows are ordered so that the first match in a group names the group.
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  if (ID.Architecture == ARCHITECTURE && ID.Release == RELEASE)                \
    return GK_##KIND;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  return GK_NONE;
}

StringRef llvm::IntelGPU::getArchName(GPUKind Kind) {
  switch (Kind) {
  case GK_NONE:
    return "";
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  case GK_##KIND:                                                              \
    return NAME;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
  case GK_##KIND:                                                              \
    return NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  }
  llvm_unreachable("invalid Intel GPU GPUKind");
}

std::string llvm::IntelGPU::getNumericArchName(GMDID ID) {
  return ("xe_" + Twine(ID.Architecture) + "." + Twine(ID.Release) + "." +
          Twine(ID.Revision))
      .str();
}
