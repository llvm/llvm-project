//===-------- hexagon.cpp - JITLink hexagon support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hexagon edge kind names and fixup application.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/hexagon.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

// Instruction mask tables for R_HEX_6_X.
struct InstructionMask {
  uint32_t CmpMask;
  uint32_t RelocMask;
};

static const InstructionMask R6Masks[] = {
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

constexpr uint32_t InstParsePacketEnd = 0x0000c000;

bool isDuplex(uint32_t Insn) { return (InstParsePacketEnd & Insn) == 0; }

Expected<uint32_t> findMaskR6(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  for (auto &I : R6Masks)
    if ((0xff000000 & Insn) == I.CmpMask)
      return I.RelocMask;
  return make_error<JITLinkError>("unrecognized instruction for R_HEX_6_X "
                                  "relocation: 0x" +
                                  utohexstr(Insn));
}

Expected<uint32_t> findMaskR8(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  if ((0xff000000 & Insn) == 0xde000000)
    return 0x00e020e8;
  if ((0xff000000 & Insn) == 0x3c000000)
    return 0x0000207f;
  return 0x00001fe0;
}

Expected<uint32_t> findMaskR11(uint32_t Insn) {
  if (isDuplex(Insn))
    return 0x03f00000;
  if ((0xff000000 & Insn) == 0xa1000000)
    return 0x060020ff;
  return 0x06003fe0;
}

Expected<uint32_t> findMaskR16(uint32_t Insn) {
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

} // anonymous namespace

namespace llvm::jitlink::hexagon {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Pointer32:
    return "Pointer32";
  case PCRel32:
    return "PCRel32";
  case B22_PCREL:
    return "B22_PCREL";
  case B15_PCREL:
    return "B15_PCREL";
  case B13_PCREL:
    return "B13_PCREL";
  case B9_PCREL:
    return "B9_PCREL";
  case B7_PCREL:
    return "B7_PCREL";
  case HI16:
    return "HI16";
  case LO16:
    return "LO16";
  case Word32_6_X:
    return "Word32_6_X";
  case B32_PCREL_X:
    return "B32_PCREL_X";
  case B22_PCREL_X:
    return "B22_PCREL_X";
  case B15_PCREL_X:
    return "B15_PCREL_X";
  case B13_PCREL_X:
    return "B13_PCREL_X";
  case B9_PCREL_X:
    return "B9_PCREL_X";
  case B7_PCREL_X:
    return "B7_PCREL_X";
  case Word6_X:
    return "Word6_X";
  case Word6_PCREL_X:
    return "Word6_PCREL_X";
  case Word8_X:
    return "Word8_X";
  case Word9_X:
    return "Word9_X";
  case Word10_X:
    return "Word10_X";
  case Word11_X:
    return "Word11_X";
  case Word12_X:
    return "Word12_X";
  case Word16_X:
    return "Word16_X";
  default:
    return getGenericEdgeKindName(K);
  }
}

Error applyFixup(LinkGraph &G, Block &B, const Edge &E) {
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
