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

#include "llvm/ADT/StringExtras.h"
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

/// Instruction mask entry for R_HEX_6_X / R_HEX_16_X lookup.
struct InstructionMask {
  uint32_t CmpMask;
  uint32_t RelocMask;
};

/// Mask table for R_HEX_6_X relocations, indexed by instruction class.
inline constexpr InstructionMask R6Masks[] = {
    {0x38000000, 0x0000201f}, {0x39000000, 0x0000201f},
    {0x3e000000, 0x00001f80}, {0x3f000000, 0x00001f80},
    {0x40000000, 0x000020f8}, {0x41000000, 0x000007e0},
    {0x42000000, 0x000020f8}, {0x43000000, 0x000007e0},
    {0x44000000, 0x000020f8}, {0x45000000, 0x000007e0},
    {0x46000000, 0x000020f8}, {0x47000000, 0x000007e0},
    {0x6a000000, 0x00001f80}, {0x7c000000, 0x001f2000},
    {0x9a000000, 0x00000f60}, {0x9b000000, 0x00000f60},
    {0x9c000000, 0x00000f60}, {0x9d000000, 0x00000f60},
    {0x9f000000, 0x001f0100}, {0xab000000, 0x0000003f},
    {0xad000000, 0x0000003f}, {0xaf000000, 0x00030078},
    {0xd7000000, 0x006020e0}, {0xd8000000, 0x006020e0},
    {0xdb000000, 0x006020e0}, {0xdf000000, 0x006020e0}};

inline constexpr uint32_t InstParsePacketEnd = 0x0000c000;

inline bool isDuplex(uint32_t Insn) { return (InstParsePacketEnd & Insn) == 0; }

inline Expected<uint32_t> findMaskR6(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  for (auto &I : R6Masks)
    if ((0xff000000 & Insn) == I.CmpMask)
      return I.RelocMask;
  return make_error<JITLinkError>("unrecognized instruction for R_HEX_6_X "
                                  "relocation: 0x" +
                                  utohexstr(Insn));
}

inline Expected<uint32_t> findMaskR8(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  if ((0xff000000 & Insn) == 0xde000000)
    return 0x00e020e8;
  if ((0xff000000 & Insn) == 0x3c000000)
    return 0x0000207f;
  return 0x00001fe0;
}

inline Expected<uint32_t> findMaskR11(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  if ((0xff000000 & Insn) == 0xa1000000)
    return 0x060020ff;
  return 0x06003fe0;
}

inline Expected<uint32_t> findMaskR16(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  Insn = Insn & ~InstParsePacketEnd;
  if ((0xff000000 & Insn) == 0x48000000)
    return 0x061f20ff;
  if ((0xff000000 & Insn) == 0x49000000)
    return 0x061f3fe0;
  if ((0xff000000 & Insn) == 0x78000000)
    return 0x00df3fe0;
  if ((0xff000000 & Insn) == 0xb0000000)
    return 0x0fe03fe0;
  if ((0xff802000 & Insn) == 0x74000000)
    return 0x00001fe0;
  if ((0xff802000 & Insn) == 0x74002000)
    return 0x00001fe0;
  if ((0xff802000 & Insn) == 0x74800000)
    return 0x00001fe0;
  if ((0xff802000 & Insn) == 0x74802000)
    return 0x00001fe0;
  for (auto &I : R6Masks)
    if ((0xff000000 & Insn) == I.CmpMask)
      return I.RelocMask;
  return make_error<JITLinkError>("unrecognized instruction for R_HEX_16_X "
                                  "relocation: 0x" +
                                  utohexstr(Insn));
}

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E) {
  using namespace llvm::support;

  char *BlockWorkingMem = B.getAlreadyMutableContent().data();
  char *FixupPtr = BlockWorkingMem + E.getOffset();
  auto FixupAddress = B.getAddress() + E.getOffset();

  int64_t TargetAddr = E.getTarget().getAddress().getValue() + E.getAddend();
  int64_t PCRelVal = E.getTarget().getAddress().getValue() -
                     FixupAddress.getValue() + E.getAddend();

  auto or32le = [](char *P, uint32_t V) {
    endian::write32le(P, endian::read32le(P) | V);
  };

  uint32_t Insn = endian::read32le(FixupPtr);

  switch (E.getKind()) {
  case Pointer32:
    endian::write32le(FixupPtr, static_cast<uint32_t>(TargetAddr));
    break;

  case PCRel32:
    endian::write32le(FixupPtr, static_cast<uint32_t>(PCRelVal));
    break;

  case B22_PCREL:
    if (!isInt<24>(PCRelVal))
      return makeTargetOutOfRangeError(G, B, E);
    or32le(FixupPtr, applyMask(0x01ff3ffe, PCRelVal >> 2));
    break;

  case B15_PCREL:
    if (!isInt<17>(PCRelVal))
      return makeTargetOutOfRangeError(G, B, E);
    or32le(FixupPtr, applyMask(0x00df20fe, PCRelVal >> 2));
    break;

  case B13_PCREL:
    if (!isInt<15>(PCRelVal))
      return makeTargetOutOfRangeError(G, B, E);
    or32le(FixupPtr, applyMask(0x00202ffe, PCRelVal >> 2));
    break;

  case B9_PCREL:
    if (!isInt<11>(PCRelVal))
      return makeTargetOutOfRangeError(G, B, E);
    or32le(FixupPtr, applyMask(0x003000fe, PCRelVal >> 2));
    break;

  case B7_PCREL:
    if (!isInt<9>(PCRelVal))
      return makeTargetOutOfRangeError(G, B, E);
    or32le(FixupPtr, applyMask(0x00001f18, PCRelVal >> 2));
    break;

  case HI16:
    or32le(FixupPtr, applyMask(0x00c03fff, TargetAddr >> 16));
    break;

  case LO16:
    or32le(FixupPtr, applyMask(0x00c03fff, TargetAddr));
    break;

  case Word32_6_X:
    or32le(FixupPtr, applyMask(0x0fff3fff, TargetAddr >> 6));
    break;

  case B32_PCREL_X:
    or32le(FixupPtr, applyMask(0x0fff3fff, PCRelVal >> 6));
    break;

  case B22_PCREL_X:
    or32le(FixupPtr, applyMask(0x01ff3ffe, PCRelVal & 0x3f));
    break;

  case B15_PCREL_X:
    or32le(FixupPtr, applyMask(0x00df20fe, PCRelVal & 0x3f));
    break;

  case B13_PCREL_X:
    or32le(FixupPtr, applyMask(0x00202ffe, PCRelVal & 0x3f));
    break;

  case B9_PCREL_X:
    or32le(FixupPtr, applyMask(0x003000fe, PCRelVal & 0x3f));
    break;

  case B7_PCREL_X:
    or32le(FixupPtr, applyMask(0x00001f18, PCRelVal & 0x3f));
    break;

  case Word6_X: {
    auto Mask = findMaskR6(Insn);
    if (!Mask)
      return Mask.takeError();
    or32le(FixupPtr, applyMask(*Mask, TargetAddr));
    break;
  }

  case Word6_PCREL_X: {
    auto Mask = findMaskR6(Insn);
    if (!Mask)
      return Mask.takeError();
    or32le(FixupPtr, applyMask(*Mask, PCRelVal));
    break;
  }

  case Word8_X: {
    auto Mask = findMaskR8(Insn);
    if (!Mask)
      return Mask.takeError();
    or32le(FixupPtr, applyMask(*Mask, TargetAddr));
    break;
  }

  case Word9_X:
    or32le(FixupPtr, applyMask(0x00003fe0, TargetAddr & 0x3f));
    break;

  case Word10_X:
    or32le(FixupPtr, applyMask(0x00203fe0, TargetAddr & 0x3f));
    break;

  case Word11_X: {
    auto Mask = findMaskR11(Insn);
    if (!Mask)
      return Mask.takeError();
    or32le(FixupPtr, applyMask(*Mask, TargetAddr & 0x3f));
    break;
  }

  case Word12_X:
    or32le(FixupPtr, applyMask(0x000007e0, TargetAddr));
    break;

  case Word16_X: {
    auto Mask = findMaskR16(Insn);
    if (!Mask)
      return Mask.takeError();
    or32le(FixupPtr, applyMask(*Mask, TargetAddr & 0x3f));
    break;
  }

  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " unsupported Hexagon edge kind " + getEdgeKindName(E.getKind()));
  }

  return Error::success();
}

} // namespace llvm::jitlink::hexagon

#endif // LLVM_EXECUTIONENGINE_JITLINK_HEXAGON_H
