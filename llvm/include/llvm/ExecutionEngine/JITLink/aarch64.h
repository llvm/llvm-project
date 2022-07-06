//=== aarch64.h - Generic JITLink aarch64 edge kinds, utilities -*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H
#define LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H

#include "TableManager.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/MemoryFlags.h"

namespace llvm {
namespace jitlink {
namespace aarch64 {

enum EdgeKind_aarch64 : Edge::Kind {
  Branch26 = Edge::FirstRelocation,
  Pointer32,
  Pointer64,
  Pointer64Anon,
  Page21,
  PageOffset12,
  MoveWide16,
  GOTPage21,
  GOTPageOffset12,
  TLVPage21,
  TLVPageOffset12,
  TLSDescPage21,
  TLSDescPageOffset12,
  PointerToGOT,
  PairedAddend,
  LDRLiteral19,
  Delta32,
  Delta64,
  NegDelta32,
  NegDelta64,
};

/// Returns a string name for the given aarch64 edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);

// Returns whether the Instr is LD/ST (imm12)
inline bool isLoadStoreImm12(uint32_t Instr) {
  constexpr uint32_t LoadStoreImm12Mask = 0x3b000000;
  return (Instr & LoadStoreImm12Mask) == 0x39000000;
}

// Returns the amount the address operand of LD/ST (imm12)
// should be shifted right by.
//
// The shift value varies by the data size of LD/ST instruction.
// For instance, LDH instructoin needs the address to be shifted
// right by 1.
inline unsigned getPageOffset12Shift(uint32_t Instr) {
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

// Returns whether the Instr is MOVK/MOVZ (imm16) with a zero immediate field
inline bool isMoveWideImm16(uint32_t Instr) {
  constexpr uint32_t MoveWideImm16Mask = 0x5f9fffe0;
  return (Instr & MoveWideImm16Mask) == 0x52800000;
}

// Returns the amount the address operand of MOVK/MOVZ (imm16)
// should be shifted right by.
//
// The shift value is specfied in the assembly as LSL #<shift>.
inline unsigned getMoveWide16Shift(uint32_t Instr) {
  if (isMoveWideImm16(Instr)) {
    uint32_t ImplicitShift = (Instr >> 21) & 0b11;
    return ImplicitShift << 4;
  }

  return 0;
}

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E) {
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
  case Page21: {
    assert((E.getKind() != GOTPage21 || E.getAddend() == 0) &&
           "GOTPAGE21 with non-zero addend");
    uint64_t TargetPage =
        (E.getTarget().getAddress().getValue() + E.getAddend()) &
        ~static_cast<uint64_t>(4096 - 1);
    uint64_t PCPage =
        FixupAddress.getValue() & ~static_cast<uint64_t>(4096 - 1);

    int64_t PageDelta = TargetPage - PCPage;
    if (!isInt<33>(PageDelta))
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
  case MoveWide16: {
    uint64_t TargetOffset =
        (E.getTarget().getAddress() + E.getAddend()).getValue();

    uint32_t RawInstr = *(ulittle32_t *)FixupPtr;
    assert(isMoveWideImm16(RawInstr) &&
           "RawInstr isn't a MOVK/MOVZ instruction");

    unsigned ImmShift = getMoveWide16Shift(RawInstr);
    uint32_t Imm = (TargetOffset >> ImmShift) & 0xffff;
    uint32_t FixedInstr = RawInstr | (Imm << 5);
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
  case TLVPage21:
  case TLVPageOffset12:
  case TLSDescPage21:
  case TLSDescPageOffset12:
  case GOTPage21:
  case GOTPageOffset12:
  case PointerToGOT: {
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        "GOT/TLV edge kinds not lowered: " + getEdgeKindName(E.getKind()));
  }
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        "unsupported edge kind" + getEdgeKindName(E.getKind()));
  }

  return Error::success();
}

/// AArch64 null pointer content.
extern const uint8_t NullGOTEntryContent[8];

/// AArch64 PLT stub content.
extern const uint8_t StubContent[8];

/// Global Offset Table Builder.
class GOTTableManager : public TableManager<GOTTableManager> {
public:
  static StringRef getSectionName() { return "$__GOT"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    Edge::Kind KindToSet = Edge::Invalid;
    const char *BlockWorkingMem = B->getContent().data();
    const char *FixupPtr = BlockWorkingMem + E.getOffset();

    switch (E.getKind()) {
    case aarch64::GOTPage21:
    case aarch64::TLVPage21: {
      KindToSet = aarch64::Page21;
      break;
    }
    case aarch64::GOTPageOffset12:
    case aarch64::TLVPageOffset12: {
      KindToSet = aarch64::PageOffset12;
      uint32_t RawInstr = *(const support::ulittle32_t *)FixupPtr;
      (void)RawInstr;
      assert(E.getAddend() == 0 &&
             "GOTPageOffset12/TLVPageOffset12 with non-zero addend");
      assert((RawInstr & 0xfffffc00) == 0xf9400000 &&
             "RawInstr isn't a 64-bit LDR immediate");
      break;
    }
    case aarch64::PointerToGOT: {
      KindToSet = aarch64::Delta64;
      break;
    }
    default:
      return false;
    }
    assert(KindToSet != Edge::Invalid &&
           "Fell through switch, but no new kind to set");
    DEBUG_WITH_TYPE("jitlink", {
      dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
             << B->getFixupAddress(E) << " (" << B->getAddress() << " + "
             << formatv("{0:x}", E.getOffset()) << ")\n";
    });
    E.setKind(KindToSet);
    E.setTarget(getEntryForTarget(G, E.getTarget()));
    return true;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    auto &GOTEntryBlock = G.createContentBlock(
        getGOTSection(G), getGOTEntryBlockContent(), orc::ExecutorAddr(), 8, 0);
    GOTEntryBlock.addEdge(aarch64::Pointer64, 0, Target, 0);
    return G.addAnonymousSymbol(GOTEntryBlock, 0, 8, false, false);
  }

private:
  Section &getGOTSection(LinkGraph &G) {
    if (!GOTSection)
      GOTSection =
          &G.createSection(getSectionName(), MemProt::Read | MemProt::Exec);
    return *GOTSection;
  }

  ArrayRef<char> getGOTEntryBlockContent() {
    return {reinterpret_cast<const char *>(NullGOTEntryContent),
            sizeof(NullGOTEntryContent)};
  }

  Section *GOTSection = nullptr;
};

/// Procedure Linkage Table Builder.
class PLTTableManager : public TableManager<PLTTableManager> {
public:
  PLTTableManager(GOTTableManager &GOT) : GOT(GOT) {}

  static StringRef getSectionName() { return "$__STUBS"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if (E.getKind() == aarch64::Branch26 && !E.getTarget().isDefined()) {
      DEBUG_WITH_TYPE("jitlink", {
        dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
               << B->getFixupAddress(E) << " (" << B->getAddress() << " + "
               << formatv("{0:x}", E.getOffset()) << ")\n";
      });
      E.setTarget(getEntryForTarget(G, E.getTarget()));
      return true;
    }
    return false;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    auto &StubContentBlock = G.createContentBlock(
        getStubsSection(G), getStubBlockContent(), orc::ExecutorAddr(), 1, 0);
    // Re-use GOT entries for stub targets.
    auto &GOTEntrySymbol = GOT.getEntryForTarget(G, Target);
    StubContentBlock.addEdge(aarch64::LDRLiteral19, 0, GOTEntrySymbol, 0);
    return G.addAnonymousSymbol(StubContentBlock, 0, 8, true, false);
  }

public:
  Section &getStubsSection(LinkGraph &G) {
    if (!StubsSection)
      StubsSection =
          &G.createSection(getSectionName(), MemProt::Read | MemProt::Exec);
    return *StubsSection;
  }

  ArrayRef<char> getStubBlockContent() {
    return {reinterpret_cast<const char *>(StubContent), sizeof(StubContent)};
  }

  GOTTableManager &GOT;
  Section *StubsSection = nullptr;
};

} // namespace aarch64
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H
