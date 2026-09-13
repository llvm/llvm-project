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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;
using namespace IntelGPU;

// A GMDID packs the architecture (bits 31:22), the release (21:14) and the
// revision (5:0) of the GPU IP into one 32-bit value. Bits 13:6 are reserved,
// which is why the release and the revision fields do not meet.
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

// Read a numeric architecture name, e.g. "xe_12.60.7", into \p ID. The revision
// may be omitted, since it takes no part in a lookup either way. Anything else
// is not a numeric name, which is not the same as naming no device: a caller
// distinguishes the two by whether this succeeds.
static bool parseNumericArchName(StringRef Name, GMDID &ID) {
  if (!Name.consume_front("xe_"))
    return false;

  StringRef Architecture, Release, Revision;
  std::tie(Architecture, Name) = Name.split('.');
  std::tie(Release, Revision) = Name.split('.');
  if (Architecture.getAsInteger(10, ID.Architecture) ||
      Release.getAsInteger(10, ID.Release))
    return false;
  if (!Revision.empty() && Revision.getAsInteger(10, ID.Revision))
    return false;
  return true;
}

GPUKind llvm::IntelGPU::parseArch(StringRef Name) {
  GPUKind Kind = StringSwitch<GPUKind>(Name)
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  .Case(NAME, GK_##KIND)
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
  .Case(NAME, GK_##KIND)
#define INTEL_GPU_ALIAS(NAME, KIND) .Case(NAME, GK_##KIND)
#include "llvm/TargetParser/IntelGPUTargetParser.def"
                     .Default(GK_NONE);
  if (Kind != GK_NONE)
    return Kind;

  // A device with no human-friendly name is spelled numerically, so the same
  // lookup the driver does for a reported GMDID has to be reachable by name.
  GMDID ID;
  if (parseNumericArchName(Name, ID))
    return getKindForGMDID(ID);
  return GK_NONE;
}

// The suffix each IGCA_SUFFIX token contributes to a level name. Pasting a
// row's token onto this prefix turns the column straight into its spelling.
#define IGCA_SUFFIX_Core ""
#define IGCA_SUFFIX_Compute "c"
#define IGCA_SUFFIX_Render "r"
#define IGCA_SUFFIX_ComputeExact "ca"
#define IGCA_SUFFIX_RenderExact "ra"

StringRef llvm::IntelGPU::getIGCAName(GPUKind Kind) {
  // Unlike the NVPTX virtual architecture name, this is not a column of its own:
  // the level and the suffix already spell it, and a column would let the three
  // disagree. Both parts are known at compile time, so each row yields one
  // literal rather than a string built on demand.
  switch (Kind) {
  case GK_NONE:
    return "";
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  case GK_##KIND:                                                              \
    return "igca_" #IGCA_LEVEL IGCA_SUFFIX_##IGCA_SUFFIX;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
  case GK_##KIND:                                                              \
    return "igca_" #IGCA_LEVEL IGCA_SUFFIX_##IGCA_SUFFIX;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  }
  llvm_unreachable("invalid Intel GPU GPUKind");
}

#undef IGCA_SUFFIX_Core
#undef IGCA_SUFFIX_Compute
#undef IGCA_SUFFIX_Render
#undef IGCA_SUFFIX_ComputeExact
#undef IGCA_SUFFIX_RenderExact

void llvm::IntelGPU::fillValidArchList(SmallVectorImpl<StringRef> &Values) {
  // An alias is a name the user may write, so it belongs here even though it is
  // never the name reported for a device.
#define INTEL_GPU(NAME, KIND, ARCHITECTURE, RELEASE, IGCA_LEVEL, IGCA_SUFFIX)  \
  Values.push_back(NAME);
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_LEVEL, IGCA_SUFFIX)                  \
  Values.push_back(NAME);
#define INTEL_GPU_ALIAS(NAME, KIND) Values.push_back(NAME);
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}
