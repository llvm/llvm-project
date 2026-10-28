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
#include <cassert>

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
[[maybe_unused]] static constexpr uint32_t GPUIPMajorMask = 0x3ff;
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

StringRef llvm::IntelGPU::getArchName(uint32_t GPUIPVersion) {
  const uint32_t Device = GPUIPVersion & GPUIPDeviceMask;
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  if (Device == packDevice(MAJOR, MINOR))                                      \
    return NAME;
#include "llvm/TargetParser/IntelGPUTargetParser.def"
  return "";
}

std::string llvm::IntelGPU::getNumericArchName(uint32_t GPUIPVersion) {
  const uint32_t Major = GPUIPVersion >> GPUIPMajorShift;
  const uint32_t Minor = (GPUIPVersion >> GPUIPMinorShift) & GPUIPMinorMask;
  const uint32_t Revision = GPUIPVersion & GPUIPRevisionMask;
  return ("xe_" + Twine(Major) + "." + Twine(Minor) + "." + Twine(Revision))
      .str();
}

namespace {
// All five possible names for each device's numeric IGCA target, e.g. igca_20,
// igca_20c, igca_20ca, igca_20r and igca_20ra for xe-pvc. Which of them are
// valid is decided by the IGCATarget constructor, so the rule is in one place.
// The names are string literals, so getIGCATargetName() can return them.
struct IGCATargetNameEntry {
  const char *Name;
  uint16_t Target;
  IGCAFeatureSet FeatureSet;
};
#define IGCA_TARGET_NAMES(TARGET)                                              \
  {"igca_" #TARGET, TARGET, IGCAFeatureSet::Core},                             \
      {"igca_" #TARGET "c", TARGET, IGCAFeatureSet::Compute},                  \
      {"igca_" #TARGET "ca", TARGET, IGCAFeatureSet::ComputeExact},            \
      {"igca_" #TARGET "r", TARGET, IGCAFeatureSet::Render},                   \
      {"igca_" #TARGET "ra", TARGET, IGCAFeatureSet::RenderExact},
constexpr IGCATargetNameEntry IGCATargetNames[] = {
#define INTEL_GPU(NAME, KIND, MAJOR, MINOR, IGCA_TARGET, IGCA_FEATURE_SETS)    \
  IGCA_TARGET_NAMES(IGCA_TARGET)
#define INTEL_GPU_COMPAT(NAME, KIND, IGCA_TARGET, IGCA_FEATURE_SETS)           \
  IGCA_TARGET_NAMES(IGCA_TARGET)
#include "llvm/TargetParser/IntelGPUTargetParser.def"
};
#undef IGCA_TARGET_NAMES
} // namespace

IGCATarget llvm::IntelGPU::parseIGCATarget(StringRef MaybeTarget) {
  for (const IGCATargetNameEntry &E : IGCATargetNames)
    if (MaybeTarget == E.Name)
      return IGCATarget(E.Target, E.FeatureSet);
  return IGCATarget::invalid();
}

StringRef llvm::IntelGPU::getIGCATargetName(IGCATarget T) {
  for (const IGCATargetNameEntry &E : IGCATargetNames)
    if (E.Target == T.getTarget() && E.FeatureSet == T.getFeatureSet())
      return E.Name;
  return "";
}
