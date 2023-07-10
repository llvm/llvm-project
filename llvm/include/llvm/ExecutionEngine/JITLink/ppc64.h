//===--- ppc64.h - Generic JITLink ppc64 edge kinds, utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing 64-bit PowerPC objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_PPC64_H
#define LLVM_EXECUTIONENGINE_JITLINK_PPC64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/TableManager.h"
#include "llvm/Support/Endian.h"

namespace llvm::jitlink::ppc64 {

/// Represents ppc64 fixups and other ppc64-specific edge kinds.
enum EdgeKind_ppc64 : Edge::Kind {
  Pointer64 = Edge::FirstRelocation,
  Pointer32,
  Delta64,
  Delta32,
  NegDelta32,
  Delta16,
  Delta16HA,
  Delta16LO,
  TOCDelta16HA,
  TOCDelta16LO,
  TOCDelta16DS,
  TOCDelta16LODS,
  CallBranchDelta,
  // Need to restore r2 after the bl, suggesting the bl is followed by a nop.
  CallBranchDeltaRestoreTOC,
  // Need PLT call stub.
  RequestPLTCallStub,
  // Need PLT call stub following a save of r2.
  RequestPLTCallStubSaveTOC,
};

extern const char NullPointerContent[8];
extern const char PointerJumpStubContent_big[20];
extern const char PointerJumpStubContent_little[20];

inline Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                                      Symbol *InitialTarget = nullptr,
                                      uint64_t InitialAddend = 0) {
  assert(G.getPointerSize() == sizeof(NullPointerContent) &&
         "LinkGraph's pointer size should be consistent with size of "
         "NullPointerContent");
  Block &B = G.createContentBlock(PointerSection, NullPointerContent,
                                  orc::ExecutorAddr(), G.getPointerSize(), 0);
  if (InitialTarget)
    B.addEdge(Pointer64, 0, *InitialTarget, InitialAddend);
  return G.addAnonymousSymbol(B, 0, G.getPointerSize(), false, false);
}

template <support::endianness Endianness>
inline Block &createPointerJumpStubBlock(LinkGraph &G, Section &StubSection,
                                         Symbol &PointerSymbol, bool SaveR2) {
  constexpr bool isLE = Endianness == support::endianness::little;
  ArrayRef<char> C =
      isLE ? PointerJumpStubContent_little : PointerJumpStubContent_big;
  if (!SaveR2)
    // Skip storing r2.
    C = C.slice(4);
  Block &B = G.createContentBlock(StubSection, C, orc::ExecutorAddr(), 4, 0);
  size_t Offset = SaveR2 ? 4 : 0;
  B.addEdge(TOCDelta16HA, Offset, PointerSymbol, 0);
  B.addEdge(TOCDelta16LO, Offset + 4, PointerSymbol, 0);
  return B;
}

template <support::endianness Endianness>
inline Symbol &
createAnonymousPointerJumpStub(LinkGraph &G, Section &StubSection,
                               Symbol &PointerSymbol, bool SaveR2) {
  constexpr bool isLE = Endianness == support::endianness::little;
  constexpr ArrayRef<char> Stub =
      isLE ? PointerJumpStubContent_little : PointerJumpStubContent_big;
  return G.addAnonymousSymbol(createPointerJumpStubBlock<Endianness>(
                                  G, StubSection, PointerSymbol, SaveR2),
                              0, SaveR2 ? sizeof(Stub) : sizeof(Stub) - 4, true,
                              false);
}

template <support::endianness Endianness>
class TOCTableManager : public TableManager<TOCTableManager<Endianness>> {
public:
  // FIXME: `llvm-jitlink -check` relies this name to be $__GOT.
  static StringRef getSectionName() { return "$__GOT"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    Edge::Kind K = E.getKind();
    switch (K) {
    case TOCDelta16HA:
    case TOCDelta16LO:
    case TOCDelta16DS:
    case TOCDelta16LODS:
    case CallBranchDeltaRestoreTOC:
    case RequestPLTCallStub:
    case RequestPLTCallStubSaveTOC:
      // Create TOC section if TOC relocation, PLT or GOT is used.
      getOrCreateTOCSection(G);
      return false;
    default:
      return false;
    }
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    return createAnonymousPointer(G, getOrCreateTOCSection(G), &Target);
  }

private:
  Section &getOrCreateTOCSection(LinkGraph &G) {
    TOCSection = G.findSectionByName(getSectionName());
    if (!TOCSection)
      TOCSection = &G.createSection(getSectionName(), orc::MemProt::Read);
    return *TOCSection;
  }

  Section *TOCSection = nullptr;
};

template <support::endianness Endianness>
class PLTTableManager : public TableManager<PLTTableManager<Endianness>> {
public:
  PLTTableManager(TOCTableManager<Endianness> &TOC) : TOC(TOC) {}

  static StringRef getSectionName() { return "$__STUBS"; }

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    Edge::Kind K = E.getKind();
    if (K == ppc64::RequestPLTCallStubSaveTOC && E.getTarget().isExternal()) {
      E.setKind(ppc64::CallBranchDeltaRestoreTOC);
      this->SaveR2InStub = true;
      E.setTarget(this->getEntryForTarget(G, E.getTarget()));
      return true;
    }
    if (K == ppc64::RequestPLTCallStub && E.getTarget().isExternal()) {
      E.setKind(ppc64::CallBranchDelta);
      this->SaveR2InStub = false;
      E.setTarget(this->getEntryForTarget(G, E.getTarget()));
      return true;
    }
    return false;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    return createAnonymousPointerJumpStub<Endianness>(
        G, getOrCreateStubsSection(G), TOC.getEntryForTarget(G, Target),
        this->SaveR2InStub);
  }

private:
  Section &getOrCreateStubsSection(LinkGraph &G) {
    PLTSection = G.findSectionByName(getSectionName());
    if (!PLTSection)
      PLTSection = &G.createSection(getSectionName(),
                                    orc::MemProt::Read | orc::MemProt::Exec);
    return *PLTSection;
  }

  TOCTableManager<Endianness> &TOC;
  Section *PLTSection = nullptr;
  bool SaveR2InStub = false;
};

/// Returns a string name for the given ppc64 edge. For debugging purposes
/// only.
const char *getEdgeKindName(Edge::Kind K);

inline static uint16_t ha16(uint64_t x) { return (x + 0x8000) >> 16; }

inline static uint16_t lo16(uint64_t x) { return x & 0xffff; }

/// Apply fixup expression for edge to block content.
template <support::endianness Endianness>
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                        const Symbol *TOCSymbol) {
  char *BlockWorkingMem = B.getAlreadyMutableContent().data();
  char *FixupPtr = BlockWorkingMem + E.getOffset();
  orc::ExecutorAddr FixupAddress = B.getAddress() + E.getOffset();
  int64_t S = E.getTarget().getAddress().getValue();
  int64_t A = E.getAddend();
  int64_t P = FixupAddress.getValue();
  int64_t TOCBase = TOCSymbol ? TOCSymbol->getAddress().getValue() : 0;
  Edge::Kind K = E.getKind();

  DEBUG_WITH_TYPE("jitlink", {
    dbgs() << "    Applying fixup on " << G.getEdgeKindName(K)
           << " edge, (S, A, P, .TOC.) = (" << formatv("{0:x}", S) << ", "
           << formatv("{0:x}", A) << ", " << formatv("{0:x}", P) << ", "
           << formatv("{0:x}", TOCBase) << ")\n";
  });

  switch (K) {
  case Pointer64: {
    uint64_t Value = S + A;
    support::endian::write64<Endianness>(FixupPtr, Value);
    break;
  }
  case Delta16HA:
  case Delta16LO: {
    int64_t Value = S + A - P;
    if (LLVM_UNLIKELY(!isInt<32>(Value))) {
      return makeTargetOutOfRangeError(G, B, E);
    }
    if (K == Delta16LO)
      support::endian::write16<Endianness>(FixupPtr, lo16(Value));
    else
      support::endian::write16<Endianness>(FixupPtr, ha16(Value));
    break;
  }
  case TOCDelta16HA:
  case TOCDelta16LO: {
    int64_t Value = S + A - TOCBase;
    if (LLVM_UNLIKELY(!isInt<32>(Value))) {
      return makeTargetOutOfRangeError(G, B, E);
    }
    if (K == TOCDelta16LO)
      support::endian::write16<Endianness>(FixupPtr, lo16(Value));
    else
      support::endian::write16<Endianness>(FixupPtr, ha16(Value));
    break;
  }
  case TOCDelta16DS:
  case TOCDelta16LODS: {
    int64_t Value = S + A - TOCBase;
    if (LLVM_UNLIKELY(!isInt<32>(Value))) {
      return makeTargetOutOfRangeError(G, B, E);
    }
    if (K == TOCDelta16LODS)
      support::endian::write16<Endianness>(FixupPtr, lo16(Value) & ~3);
    else
      support::endian::write16<Endianness>(FixupPtr, Value & ~3);
    break;
  }
  case CallBranchDeltaRestoreTOC:
  case CallBranchDelta: {
    int64_t Value = S + A - P;
    if (LLVM_UNLIKELY(!isInt<26>(Value))) {
      return makeTargetOutOfRangeError(G, B, E);
    }
    uint32_t Inst = support::endian::read32<Endianness>(FixupPtr);
    support::endian::write32<Endianness>(FixupPtr, (Inst & 0xfc000003) |
                                                       (Value & 0x03fffffc));
    if (K == CallBranchDeltaRestoreTOC) {
      uint32_t NopInst = support::endian::read32<Endianness>(FixupPtr + 4);
      assert(NopInst == 0x60000000 &&
             "NOP should be placed here for restoring r2");
      // Restore r2 by instruction 0xe8410018 which is `ld r2, 24(r1)`.
      support::endian::write32<Endianness>(FixupPtr + 4, 0xe8410018);
    }
    break;
  }
  case Delta64: {
    int64_t Value = S + A - P;
    support::endian::write64<Endianness>(FixupPtr, Value);
    break;
  }
  case Delta32: {
    int64_t Value = S + A - P;
    if (LLVM_UNLIKELY(!isInt<32>(Value))) {
      return makeTargetOutOfRangeError(G, B, E);
    }
    support::endian::write32<Endianness>(FixupPtr, Value);
    break;
  }
  case NegDelta32: {
    int64_t Value = P - S + A;
    if (LLVM_UNLIKELY(!isInt<32>(Value))) {
      return makeTargetOutOfRangeError(G, B, E);
    }
    support::endian::write32<Endianness>(FixupPtr, Value);
    break;
  }
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " unsupported edge kind " + getEdgeKindName(E.getKind()));
  }
  return Error::success();
}

} // end namespace llvm::jitlink::ppc64

#endif // LLVM_EXECUTIONENGINE_JITLINK_PPC64_H
