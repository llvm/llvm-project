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

IGCATarget llvm::IntelGPU::parseIGCATarget(StringRef MaybeTarget) {
  if (!MaybeTarget.consume_front("igca_"))
    return IGCATarget::invalid();

  uint16_t Target;
  if (MaybeTarget.consumeInteger(10, Target))
    return IGCATarget::invalid();
  // TODO check that Target contains a valid target level
  
  IGCAFeatureSet FS = IGCAFeatureSet::IGCA_CORE;
  if (MaybeTarget.consume_front("c"))
    FS = IGCAFeatureSet::IGCA_COMPUTE;
  else if (MaybeTarget.consume_front("r"))
    FS = IGCAFeatureSet::IGCA_RENDER;
  bool IsExactFS = MaybeTarget.consume_front("a");
  if (!MaybeTarget.empty())
    return IGCATarget::invalid();

  return { Target, FS, IsExactFS };
}

static constexpr uint32_t IGCATargetShift     = 16;
static constexpr uint32_t IGCAFeatureSetShift = 1;
static constexpr uint32_t IGCAIsExactFSShift  = 0;
static constexpr uint32_t IGCAFeatureSetMask  = 0x03;
static constexpr uint32_t IGCAIsExactFSMask   = 0x1;

// An IGCA Target is packed as follows:
//
//    31              16 15           3 2           1 0         0 
//   +------------------+--------------+-------------+-----------+
//   |      Target      |   Reserved   | Feature set | Is Exact? |
//   +------------------+--------------+-------------+-----------+
//          16 bits         13 bits        2 bits        1 bit
//
uint32_t llvm::IntelGPU::IGCATarget::pack() const {
  // TODO add debug mode asserts here
  return uint32_t(Target) << IGCATargetShift |
        (uint32_t(FeatureSet) & IGCAFeatureSetMask) << IGCAFeatureSetShift |
        (uint32_t(IsExactFeatureSet) & IGCAIsExactFSMask)
            << IGCAIsExactFSShift;
}

IGCATarget llvm::IntelGPU::IGCATarget::unpack(uint32_t V) {
  return {
    uint16_t(V >> IGCATargetShift),
    IGCAFeatureSet((V >> IGCAFeatureSetShift) & IGCAFeatureSetMask),
    bool((V >> IGCAIsExactFSShift) & IGCAIsExactFSMask)
  };
}

std::string llvm::IntelGPU::getIGCATargetName(IGCATarget T) {
  if (!T)
    return "";

  StringRef FS = T.isCompute() ? "c" : T.isRender() ? "r" : "";
  StringRef Exact = T.isExact() ? "a" : "";
  return ("igca_" + Twine(T.Target) + FS + Exact).str();
}
