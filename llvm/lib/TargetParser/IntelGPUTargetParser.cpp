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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <tuple>

using namespace llvm;
using namespace IntelGPU;

// A GPU IP version packs four fields, from the most significant bit down:
//
//    31           22 21      14 13       6 5        0
//   +---------------+----------+----------+----------+
//   |     major     |   minor  | reserved | revision |
//   +---------------+----------+----------+----------+
//        10 bits      8 bits     8 bits     6 bits
//
// The reserved bits carry no information.
static constexpr uint32_t GPUIPMajorShift = 22;
static constexpr uint32_t GPUIPMinorShift = 14;
static constexpr uint32_t GPUIPMajorMask = 0x3ff;
static constexpr uint32_t GPUIPMinorMask = 0xff;
static constexpr uint32_t GPUIPRevisionMask = 0x3f;

// The bits that identify a device: the major and the minor version. Neither the
// revision nor the reserved bits take part in the lookup, because every
// revision of a device is one device as far as the compiler is concerned.
static constexpr uint32_t GPUIPDeviceMask = ~0u << GPUIPMinorShift;

// Pack a major and a minor version the way a GPU IP version does, so that a
// row of the table can be compared against a reported version as it is. A value
// too wide for its field would silently corrupt the fields above it, which
// would mean a typo in IntelGPUTargetParser.def going unnoticed.
static constexpr uint32_t packDevice(uint32_t Major, uint32_t Minor) {
  assert((Major & ~GPUIPMajorMask) == 0 && "major version too wide");
  assert((Minor & ~GPUIPMinorMask) == 0 && "minor version too wide");
  return (Major << GPUIPMajorShift) | (Minor << GPUIPMinorShift);
}

// The device that \p GPUIPVersion identifies, or GK_NONE if the table lists
// none. Only INTEL_GPU rows are expanded, so a compatibility name can never
// match.
static GPUKind getKindForVersion(uint32_t GPUIPVersion) {
  const uint32_t Device = GPUIPVersion & GPUIPDeviceMask;
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  if (Device == packDevice(MAJOR, MINOR))                                      \
    return GK_##KIND;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  return GK_NONE;
}

StringRef llvm::IntelGPU::getArchName(uint32_t GPUIPVersion) {
  return getArchName(getKindForVersion(GPUIPVersion));
}

std::string llvm::IntelGPU::getNumericArchName(uint32_t GPUIPVersion) {
  const uint32_t Major = GPUIPVersion >> GPUIPMajorShift;
  const uint32_t Minor = (GPUIPVersion >> GPUIPMinorShift) & GPUIPMinorMask;
  const uint32_t Revision = GPUIPVersion & GPUIPRevisionMask;
  return ("xe_" + Twine(Major) + "." + Twine(Minor) + "." + Twine(Revision))
      .str();
}

StringRef llvm::IntelGPU::getArchName(GPUKind Kind) {
  switch (Kind) {
  case GK_NONE:
    return "";
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  case GK_##KIND:                                                              \
    return NAME;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  case GK_##KIND:                                                              \
    return NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  }
  llvm_unreachable("invalid Intel GPU GPUKind");
}

// Read a numeric architecture name, e.g. "xe_12.60.7", into the GPU IP version
// it spells. The revision may be omitted, since it takes no part in a lookup
// either way. Anything else is not a numeric name, which is not the same as
// naming no device: a caller distinguishes the two by whether this succeeds.
static bool parseNumericArchName(StringRef Name, uint32_t &GPUIPVersion) {
  if (!Name.consume_front("xe_"))
    return false;

  StringRef MajorStr, MinorStr, RevisionStr;
  std::tie(MajorStr, Name) = Name.split('.');
  // The revision may be omitted, but a separator with no revision after it is
  // malformed rather than an omitted revision.
  const bool HasRevision = Name.contains('.');
  std::tie(MinorStr, RevisionStr) = Name.split('.');
  uint32_t Major, Minor, Revision = 0;
  if (MajorStr.getAsInteger(10, Major) || MinorStr.getAsInteger(10, Minor))
    return false;
  if (HasRevision && RevisionStr.getAsInteger(10, Revision))
    return false;

  // A field too wide for the GPU IP version it spells describes no GPU that
  // could ever report it, so this is not a numeric name rather than one naming
  // no device.
  if (Major > GPUIPMajorMask || Minor > GPUIPMinorMask ||
      Revision > GPUIPRevisionMask)
    return false;
  GPUIPVersion = packDevice(Major, Minor) | Revision;
  return true;
}

GPUKind llvm::IntelGPU::parseArch(StringRef Name) {
  GPUKind Kind = StringSwitch<GPUKind>(Name)
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  .Case(NAME, GK_##KIND)
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  .Case(NAME, GK_##KIND)
#define INTEL_GPU_ALIAS(NAME, KIND) .Case(NAME, GK_##KIND)
#include "llvm/TargetParser/IntelGPUTargetParser.def"
                     .Default(GK_NONE);
  if (Kind != GK_NONE)
    return Kind;

  // A device with no human-friendly name is spelled numerically, so the same
  // lookup the driver does for a reported GPU IP version has to be reachable by
  // name.
  uint32_t GPUIPVersion;
  if (parseNumericArchName(Name, GPUIPVersion))
    return getKindForVersion(GPUIPVersion);
  return GK_NONE;
}

// The suffix each IGCA_FEATURE_SETS token contributes to a target name.
#define IGCA_FEATURE_SETS_Core ""
#define IGCA_FEATURE_SETS_Compute "c"
#define IGCA_FEATURE_SETS_Render "r"
#define IGCA_FEATURE_SETS_ComputeExact "ca"
#define IGCA_FEATURE_SETS_RenderExact "ra"

bool llvm::IntelGPU::isNumericArchName(StringRef Name) {
  uint32_t GPUIPVersion;
  return parseNumericArchName(Name, GPUIPVersion);
}

StringRef llvm::IntelGPU::getIGCAName(GPUKind Kind) {
  // Unlike the NVPTX virtual architecture name, this is not a column of its
  // own: the target and the feature sets already spell it, and a column would
  // let the three disagree.
  switch (Kind) {
  case GK_NONE:
    return "";
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  case GK_##KIND:                                                              \
    return "igca_" #IGCA_TARGET IGCA_FEATURE_SETS_##IGCA_FEATURE_SETS;
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  case GK_##KIND:                                                              \
    return "igca_" #IGCA_TARGET IGCA_FEATURE_SETS_##IGCA_FEATURE_SETS;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  }
  llvm_unreachable("invalid Intel GPU GPUKind");
}

#undef IGCA_FEATURE_SETS_Core
#undef IGCA_FEATURE_SETS_Compute
#undef IGCA_FEATURE_SETS_Render
#undef IGCA_FEATURE_SETS_ComputeExact
#undef IGCA_FEATURE_SETS_RenderExact

void llvm::IntelGPU::fillValidArchList(SmallVectorImpl<StringRef> &Values) {
  // An alias is a name the user may write, so it belongs here even though it is
  // never the name reported for a device.
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  Values.push_back(NAME);
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  Values.push_back(NAME);
#define INTEL_GPU_ALIAS(NAME, KIND) Values.push_back(NAME);
#include "llvm/TargetParser/IntelGPUTargetParser.def"
}
