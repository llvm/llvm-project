//===-- ppc32.cpp - Generic JITLink PPC32 fixups -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ppc32.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

namespace llvm::jitlink::ppc32 {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Pointer32:
    return "Pointer32";
  case Pointer16:
    return "Pointer16";
  case Pointer16HI:
    return "Pointer16HI";
  case Pointer16HA:
    return "Pointer16HA";
  case Delta32:
    return "Delta32";
  case NegDelta32:
    return "NegDelta32";
  case Delta16:
    return "Delta16";
  case Delta16HI:
    return "Delta16HI";
  case Delta16HA:
    return "Delta16HA";
  case Branch24:
    return "Branch24";
  case Branch14:
    return "Branch14";
  case Branch24Absolute:
    return "Branch24Absolute";
  case Branch14Absolute:
    return "Branch14Absolute";
  case GOTDelta16:
    return "GOTDelta16";
  case GOTDelta16LO:
    return "GOTDelta16LO";
  case GOTDelta16HI:
    return "GOTDelta16HI";
  case GOTDelta16HA:
    return "GOTDelta16HA";
  case RequestGOTAndTransformToGOTDelta16:
    return "RequestGOTAndTransformToGOTDelta16";
  case RequestGOTAndTransformToGOTDelta16LO:
    return "RequestGOTAndTransformToGOTDelta16LO";
  case RequestGOTAndTransformToGOTDelta16HI:
    return "RequestGOTAndTransformToGOTDelta16HI";
  case RequestGOTAndTransformToGOTDelta16HA:
    return "RequestGOTAndTransformToGOTDelta16HA";
  default:
    return getGenericEdgeKindName(K);
  }
}

Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                 const Symbol *GOTSymbol) {
  uint64_t S = E.getTarget().getAddress().getValue();
  uint64_t P = (B.getAddress() + E.getOffset()).getValue();
  uint64_t V = S + E.getAddend();
  auto Order = G.getEndianness();
  char *Fixup = B.getAlreadyMutableContent().data() + E.getOffset();
  switch (E.getKind()) {
  case GOTDelta16:
  case GOTDelta16LO:
  case GOTDelta16HI:
  case GOTDelta16HA:
    assert(GOTSymbol && "Missing PPC32 GOT base symbol");
    V -= GOTSymbol->getAddress().getValue();
    if (E.getKind() == GOTDelta16 && !isInt<16>(static_cast<int64_t>(V)))
      return makeTargetOutOfRangeError(G, B, E);
    if (E.getKind() == GOTDelta16HI)
      V >>= 16;
    else if (E.getKind() == GOTDelta16HA)
      V = (V + 0x8000) >> 16;
    support::endian::write16(Fixup, V, Order);
    return Error::success();
  case Pointer32:
    if (!isUInt<32>(V))
      return makeTargetOutOfRangeError(G, B, E);
    support::endian::write32(Fixup, V, Order);
    return Error::success();
  case Delta32:
    if (!isInt<32>(static_cast<int64_t>(V - P)))
      return makeTargetOutOfRangeError(G, B, E);
    support::endian::write32(Fixup, V - P, Order);
    return Error::success();
  case NegDelta32:
    support::endian::write32(Fixup, P - V, Order);
    return Error::success();
  case Pointer16:
  case Pointer16HI:
  case Pointer16HA:
  case Delta16:
  case Delta16HI:
  case Delta16HA: {
    if (E.getKind() == Delta16 || E.getKind() == Delta16HI ||
        E.getKind() == Delta16HA)
      V -= P;
    if (E.getKind() == Pointer16HI || E.getKind() == Delta16HI)
      V >>= 16;
    else if (E.getKind() == Pointer16HA || E.getKind() == Delta16HA)
      V = (V + 0x8000) >> 16;
    support::endian::write16(Fixup, V, Order);
    return Error::success();
  }
  case Branch24:
  case Branch14:
  case Branch24Absolute:
  case Branch14Absolute: {
    if (E.getKind() == Branch24 || E.getKind() == Branch14)
      V -= P;
    unsigned Bits =
        E.getKind() == Branch24 || E.getKind() == Branch24Absolute ? 26 : 16;
    if (!isIntN(Bits, static_cast<int64_t>(V)))
      return makeTargetOutOfRangeError(G, B, E);
    if (V & 3)
      return makeAlignmentError(orc::ExecutorAddr(P), V, 4, E);
    uint32_t Mask = Bits == 26 ? 0x03fffffc : 0x0000fffc;
    uint32_t Insn = support::endian::read32(Fixup, Order);
    support::endian::write32(Fixup, (Insn & ~Mask) | (V & Mask), Order);
    return Error::success();
  }
  default:
    return make_error<JITLinkError>("Unsupported PPC32 edge kind");
  }
}

Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                               Symbol *InitialTarget,
                               Edge::AddendT InitialAddend) {
  static const char Empty[4] = {};
  auto &B =
      G.createContentBlock(PointerSection, Empty, orc::ExecutorAddr(), 4, 0);
  if (InitialTarget)
    B.addEdge(Pointer32, 0, *InitialTarget, InitialAddend);
  return G.addAnonymousSymbol(B, 0, 4, false, false);
}

Symbol &createAnonymousPointerJumpStub(LinkGraph &G, Section &StubSection,
                                       Symbol &PointerSymbol) {
  static constexpr uint32_t Insns[] = {
      0x3d800000, // lis r12, PointerSymbol@ha
      0x818c0000, // lwz r12, PointerSymbol@l(r12)
      0x7d8903a6, // mtctr r12
      0x4e800420  // bctr
  };
  auto Content = G.allocateBuffer(sizeof(Insns));
  for (unsigned I = 0; I != std::size(Insns); ++I)
    support::endian::write32(Content.data() + 4 * I, Insns[I],
                             G.getEndianness());
  auto &B = G.createMutableContentBlock(StubSection, Content,
                                        orc::ExecutorAddr(), 4, 0);
  unsigned ImmOffset = G.getEndianness() == endianness::big ? 2 : 0;
  B.addEdge(Pointer16HA, ImmOffset, PointerSymbol, 0);
  B.addEdge(Pointer16, 4 + ImmOffset, PointerSymbol, 0);
  return G.addAnonymousSymbol(B, 0, Content.size(), true, false);
}

} // namespace llvm::jitlink::ppc32
