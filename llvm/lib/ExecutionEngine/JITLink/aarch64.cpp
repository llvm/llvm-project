//===---- aarch64.cpp - Generic JITLink aarch64 edge kinds, utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing aarch64 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/aarch64.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace aarch64 {

bool isLoadStoreImm12(uint32_t Instr) {
  constexpr uint32_t LoadStoreImm12Mask = 0x3b000000;
  return (Instr & LoadStoreImm12Mask) == 0x39000000;
}

unsigned getPageOffset12Shift(uint32_t Instr) {
  constexpr uint32_t Vec128Mask = 0x04800000;

  if (isLoadStoreImm12(Instr)) {
    uint32_t ImplicitShift = Instr >> 30;
    if (ImplicitShift == 0)
      if ((Instr & Vec128Mask) == Vec128Mask)
        ImplicitShift = 4;

    return ImplicitShift;
  }

  return 0;
}

Error applyFixup(LinkGraph &G, Block &B, const Edge &E) {
  using namespace support;

  char *BlockWorkingMem = B.getAlreadyMutableContent().data();
  char *FixupPtr = BlockWorkingMem + E.getOffset();
  orc::ExecutorAddr FixupAddress = B.getAddress() + E.getOffset();

  switch (E.getKind()) {
  case Branch26: {
    assert((FixupAddress.getValue() & 0x3) == 0 &&
           "Branch-inst is not 32-bit aligned");

    int64_t Value = E.getTarget().getAddress() - FixupAddress + E.getAddend();

    if (static_cast<uint64_t>(Value) & 0x3)
      return make_error<JITLinkError>("Branch26 target is not 32-bit "
                                      "aligned");

    if (Value < -(1 << 27) || Value > ((1 << 27) - 1))
      return makeTargetOutOfRangeError(G, B, E);

    uint32_t RawInstr = *(little32_t *)FixupPtr;
    assert((RawInstr & 0x7fffffff) == 0x14000000 &&
           "RawInstr isn't a B or BR immediate instruction");
    uint32_t Imm = (static_cast<uint32_t>(Value) & ((1 << 28) - 1)) >> 2;
    uint32_t FixedInstr = RawInstr | Imm;
    *(little32_t *)FixupPtr = FixedInstr;
    break;
  }
  case Pointer32: {
    uint64_t Value = E.getTarget().getAddress().getValue() + E.getAddend();
    if (Value > std::numeric_limits<uint32_t>::max())
      return makeTargetOutOfRangeError(G, B, E);
    *(ulittle32_t *)FixupPtr = Value;
    break;
  }
  case Pointer64:
  case Pointer64Anon: {
    uint64_t Value = E.getTarget().getAddress().getValue() + E.getAddend();
    *(ulittle64_t *)FixupPtr = Value;
    break;
  }
  case Page21:
  case TLVPage21:
  case GOTPage21: {
    assert((E.getKind() != GOTPage21 || E.getAddend() == 0) &&
           "GOTPAGE21 with non-zero addend");
    uint64_t TargetPage =
        (E.getTarget().getAddress().getValue() + E.getAddend()) &
        ~static_cast<uint64_t>(4096 - 1);
    uint64_t PCPage =
        FixupAddress.getValue() & ~static_cast<uint64_t>(4096 - 1);

    int64_t PageDelta = TargetPage - PCPage;
    if (PageDelta < -(1 << 30) || PageDelta > ((1 << 30) - 1))
      return makeTargetOutOfRangeError(G, B, E);

    uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
    assert((RawInstr & 0xffffffe0) == 0x90000000 &&
           "RawInstr isn't an ADRP instruction");
    uint32_t ImmLo = (static_cast<uint64_t>(PageDelta) >> 12) & 0x3;
    uint32_t ImmHi = (static_cast<uint64_t>(PageDelta) >> 14) & 0x7ffff;
    uint32_t FixedInstr = RawInstr | (ImmLo << 29) | (ImmHi << 5);
    *(ulittle32_t *)FixupPtr = FixedInstr;
    break;
  }
  case PageOffset12: {
    uint64_t TargetOffset =
        (E.getTarget().getAddress() + E.getAddend()).getValue() & 0xfff;

    uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
    unsigned ImmShift = getPageOffset12Shift(RawInstr);

    if (TargetOffset & ((1 << ImmShift) - 1))
      return make_error<JITLinkError>("PAGEOFF12 target is not aligned");

    uint32_t EncodedImm = (TargetOffset >> ImmShift) << 10;
    uint32_t FixedInstr = RawInstr | EncodedImm;
    *(ulittle32_t *)FixupPtr = FixedInstr;
    break;
  }
  case TLVPageOffset12:
  case GOTPageOffset12: {
    assert(E.getAddend() == 0 && "GOTPAGEOF12 with non-zero addend");

    uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
    assert((RawInstr & 0xfffffc00) == 0xf9400000 &&
           "RawInstr isn't a 64-bit LDR immediate");

    uint32_t TargetOffset = E.getTarget().getAddress().getValue() & 0xfff;
    assert((TargetOffset & 0x7) == 0 && "GOT entry is not 8-byte aligned");
    uint32_t EncodedImm = (TargetOffset >> 3) << 10;
    uint32_t FixedInstr = RawInstr | EncodedImm;
    *(ulittle32_t *)FixupPtr = FixedInstr;
    break;
  }
  case LDRLiteral19: {
    assert((FixupAddress.getValue() & 0x3) == 0 && "LDR is not 32-bit aligned");
    assert(E.getAddend() == 0 && "LDRLiteral19 with non-zero addend");
    uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
    assert(RawInstr == 0x58000010 && "RawInstr isn't a 64-bit LDR literal");
    int64_t Delta = E.getTarget().getAddress() - FixupAddress;
    if (Delta & 0x3)
      return make_error<JITLinkError>("LDR literal target is not 32-bit "
                                      "aligned");
    if (Delta < -(1 << 20) || Delta > ((1 << 20) - 1))
      return makeTargetOutOfRangeError(G, B, E);

    uint32_t EncodedImm = ((static_cast<uint32_t>(Delta) >> 2) & 0x7ffff) << 5;
    uint32_t FixedInstr = RawInstr | EncodedImm;
    *(ulittle32_t *)FixupPtr = FixedInstr;
    break;
  }
  case Delta32:
  case Delta64:
  case NegDelta32:
  case NegDelta64: {
    int64_t Value;
    if (E.getKind() == Delta32 || E.getKind() == Delta64)
      Value = E.getTarget().getAddress() - FixupAddress + E.getAddend();
    else
      Value = FixupAddress - E.getTarget().getAddress() + E.getAddend();

    if (E.getKind() == Delta32 || E.getKind() == NegDelta32) {
      if (Value < std::numeric_limits<int32_t>::min() ||
          Value > std::numeric_limits<int32_t>::max())
        return makeTargetOutOfRangeError(G, B, E);
      *(little32_t *)FixupPtr = Value;
    } else
      *(little64_t *)FixupPtr = Value;
    break;
  }
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        "unsupported edge kind" + getEdgeKindName(E.getKind()));
  }

  return Error::success();
}

const char *getEdgeKindName(Edge::Kind R) {
  switch (R) {
  case Branch26:
    return "Branch26";
  case Pointer64:
    return "Pointer64";
  case Pointer64Anon:
    return "Pointer64Anon";
  case Page21:
    return "Page21";
  case PageOffset12:
    return "PageOffset12";
  case GOTPage21:
    return "GOTPage21";
  case GOTPageOffset12:
    return "GOTPageOffset12";
  case TLVPage21:
    return "TLVPage21";
  case TLVPageOffset12:
    return "TLVPageOffset12";
  case PointerToGOT:
    return "PointerToGOT";
  case PairedAddend:
    return "PairedAddend";
  case LDRLiteral19:
    return "LDRLiteral19";
  case Delta32:
    return "Delta32";
  case Delta64:
    return "Delta64";
  case NegDelta32:
    return "NegDelta32";
  case NegDelta64:
    return "NegDelta64";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}

} // namespace aarch64
} // namespace jitlink
} // namespace llvm
