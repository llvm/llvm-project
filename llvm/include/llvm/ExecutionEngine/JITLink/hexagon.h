//===---- hexagon.h - JITLink hexagon edge kinds, utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing Hexagon objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_HEXAGON_H
#define LLVM_EXECUTIONENGINE_JITLINK_HEXAGON_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Compiler.h"

namespace llvm::jitlink::hexagon {

/// Represents Hexagon fixup kinds.
enum EdgeKind_hexagon : Edge::Kind {
  /// Full 32-bit absolute pointer.
  ///   Fixup <- Target + Addend : uint32
  Pointer32 = Edge::FirstRelocation,

  /// 32-bit PC-relative.
  ///   Fixup <- Target - Fixup + Addend : int32
  PCRel32,

  /// 22-bit PC-relative branch (shifted right by 2).
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int22
  ///   Mask: 0x01ff3ffe
  B22_PCREL,

  /// 15-bit PC-relative branch (shifted right by 2).
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int15
  ///   Mask: 0x00df20fe
  B15_PCREL,

  /// 13-bit PC-relative branch (shifted right by 2).
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int13
  ///   Mask: 0x00202ffe
  B13_PCREL,

  /// 9-bit PC-relative branch (shifted right by 2).
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int9
  ///   Mask: 0x003000fe
  B9_PCREL,

  /// 7-bit PC-relative branch (shifted right by 2).
  ///   Fixup <- (Target - Fixup + Addend) >> 2 : int7
  ///   Mask: 0x00001f18
  B7_PCREL,

  /// High 16 bits of absolute address.
  ///   Fixup <- (Target + Addend) >> 16
  ///   Mask: 0x00c03fff
  HI16,

  /// Low 16 bits of absolute address.
  ///   Fixup <- (Target + Addend) & 0xffff
  ///   Mask: 0x00c03fff
  LO16,

  /// 32-bit absolute, upper 26 bits via constant extender (shifted right by 6).
  ///   Fixup <- (Target + Addend) >> 6
  ///   Mask: 0x0fff3fff
  Word32_6_X,

  /// 32-bit PC-relative, upper 26 bits via constant extender (shifted by 6).
  ///   Fixup <- (Target - Fixup + Addend) >> 6
  ///   Mask: 0x0fff3fff
  B32_PCREL_X,

  /// 22-bit PC-relative branch, lower 6 bits via extender.
  ///   Fixup <- (Target - Fixup + Addend) & 0x3f
  ///   Mask: 0x01ff3ffe
  B22_PCREL_X,

  /// 15-bit PC-relative branch, lower 6 bits via extender.
  ///   Fixup <- (Target - Fixup + Addend) & 0x3f
  ///   Mask: 0x00df20fe
  B15_PCREL_X,

  /// 13-bit PC-relative branch, lower 6 bits via extender.
  ///   Fixup <- (Target - Fixup + Addend) & 0x3f
  ///   Mask: 0x00202ffe
  B13_PCREL_X,

  /// 9-bit PC-relative branch, lower 6 bits via extender.
  ///   Fixup <- (Target - Fixup + Addend) & 0x3f
  ///   Mask: 0x003000fe
  B9_PCREL_X,

  /// 7-bit PC-relative branch, lower 6 bits via extender.
  ///   Fixup <- (Target - Fixup + Addend) & 0x3f
  ///   Mask: 0x00001f18
  B7_PCREL_X,

  /// 6-bit absolute extended.
  ///   Fixup <- (Target + Addend) & mask (instruction-dependent)
  Word6_X,

  /// 6-bit PC-relative extended.
  ///   Fixup <- (Target - Fixup + Addend) & mask (instruction-dependent)
  Word6_PCREL_X,

  /// 8-bit absolute extended.
  Word8_X,

  /// 9-bit absolute extended (6 effective bits).
  Word9_X,

  /// 10-bit absolute extended (6 effective bits).
  Word10_X,

  /// 11-bit absolute extended (6 effective bits).
  Word11_X,

  /// 12-bit absolute extended.
  Word12_X,

  /// 16-bit absolute extended (6 effective bits).
  Word16_X,
};

/// Returns a string name for the given Hexagon edge kind.
LLVM_ABI const char *getEdgeKindName(Edge::Kind K);

/// Hexagon pointer size.
constexpr uint32_t PointerSize = 4;

/// Spread data bits into instruction word according to mask.
/// For each set bit in mask (scanning low to high), the corresponding
/// bit position in the result gets the next data bit.
constexpr uint32_t applyMask(uint32_t Mask, uint32_t Data) {
  uint32_t Result = 0;
  size_t Off = 0;
  for (size_t Bit = 0; Bit != 32; ++Bit) {
    uint32_t ValBit = (Data >> Off) & 1;
    uint32_t MaskBit = (Mask >> Bit) & 1;
    if (MaskBit) {
      Result |= (ValBit << Bit);
      ++Off;
    }
  }
  return Result;
}

/// Apply fixup expression for edge to block content.
LLVM_ABI Error applyFixup(LinkGraph &G, Block &B, const Edge &E);

} // namespace llvm::jitlink::hexagon

#endif // LLVM_EXECUTIONENGINE_JITLINK_HEXAGON_H
