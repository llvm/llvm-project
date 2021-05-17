//===- PseudoProbe.h - Pseudo Probe IR Helpers ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pseudo probe IR intrinsic and dwarf discriminator manipulation routines.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PSEUDOPROBE_H
#define LLVM_IR_PSEUDOPROBE_H

#include "llvm/ADT/Optional.h"
#include <cassert>
#include <cstdint>
#include <limits>

namespace llvm {

class Instruction;
class BasicBlock;

constexpr const char *PseudoProbeDescMetadataName = "llvm.pseudo_probe_desc";

enum class PseudoProbeType { Block = 0, IndirectCall, DirectCall };

enum class PseudoProbeAttributes {
  Reserved = 0x1, // Reserved for future use.
  Dangling = 0x2, // The probe is dangling.
};

// The saturated distrution factor representing 100% for block probes.
constexpr static uint64_t PseudoProbeFullDistributionFactor =
    std::numeric_limits<uint64_t>::max();

struct PseudoProbeDwarfDiscriminator {
public:
  // The following APIs encodes/decodes per-probe information to/from a
  // 32-bit integer which is organized as:
  //  [2:0] - 0x7, this is reserved for regular discriminator,
  //          see DWARF discriminator encoding rule
  //  [18:3] - probe id
  //  [25:19] - probe distribution factor
  //  [28:26] - probe type, see PseudoProbeType
  //  [31:29] - reserved for probe attributes
  static uint32_t packProbeData(uint32_t Index, uint32_t Type, uint32_t Flags,
                                uint32_t Factor) {
    assert(Index <= 0xFFFF && "Probe index too big to encode, exceeding 2^16");
    assert(Type <= 0x7 && "Probe type too big to encode, exceeding 7");
    assert(Flags <= 0x7);
    assert(Factor <= 100 &&
           "Probe distribution factor too big to encode, exceeding 100");
    return (Index << 3) | (Factor << 19) | (Type << 26) | 0x7;
  }

  static uint32_t extractProbeIndex(uint32_t Value) {
    return (Value >> 3) & 0xFFFF;
  }

  static uint32_t extractProbeType(uint32_t Value) {
    return (Value >> 26) & 0x7;
  }

  static uint32_t extractProbeAttributes(uint32_t Value) {
    return (Value >> 29) & 0x7;
  }

  static uint32_t extractProbeFactor(uint32_t Value) {
    return (Value >> 19) & 0x7F;
  }

  // The saturated distrution factor representing 100% for callsites.
  constexpr static uint8_t FullDistributionFactor = 100;
};

struct PseudoProbe {
  uint32_t Id;
  uint32_t Type;
  uint32_t Attr;
  // Distribution factor that estimates the portion of the real execution count.
  // A saturated distribution factor stands for 1.0 or 100%. A pesudo probe has
  // a factor with the value ranged from 0.0 to 1.0.
  float Factor;

  bool isDangling() const {
    return Attr & (uint32_t)PseudoProbeAttributes::Dangling;
  }
};

Optional<PseudoProbe> extractProbe(const Instruction &Inst);

void setProbeDistributionFactor(Instruction &Inst, float Factor);

bool moveAndDanglePseudoProbes(BasicBlock *From, Instruction *To);

bool removeRedundantPseudoProbes(BasicBlock *Block);
} // end namespace llvm

#endif // LLVM_IR_PSEUDOPROBE_H
