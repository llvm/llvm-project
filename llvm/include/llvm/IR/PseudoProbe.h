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

#include <cassert>
#include <cstdint>

namespace llvm {

enum class PseudoProbeType { Block = 0, IndirectCall, DirectCall };

struct PseudoProbeDwarfDiscriminator {
  // The following APIs encodes/decodes per-probe information to/from a
  // 32-bit integer which is organized as:
  //  [2:0] - 0x7, this is reserved for regular discriminator,
  //          see DWARF discriminator encoding rule
  //  [18:3] - probe id
  //  [25:19] - reserved
  //  [28:26] - probe type, see PseudoProbeType
  //  [31:29] - reserved for probe attributes
  static uint32_t packProbeData(uint32_t Index, uint32_t Type) {
    assert(Index <= 0xFFFF && "Probe index too big to encode, exceeding 2^16");
    assert(Type <= 0x7 && "Probe type too big to encode, exceeding 7");
    return (Index << 3) | (Type << 26) | 0x7;
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
};
} // end namespace llvm

#endif // LLVM_IR_PSEUDOPROBE_H
