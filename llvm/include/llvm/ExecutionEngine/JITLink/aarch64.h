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
  GOTPage21,
  GOTPageOffset12,
  TLVPage21,
  TLVPageOffset12,
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

bool isLoadStoreImm12(uint32_t Instr);

unsigned getPageOffset12Shift(uint32_t Instr);

Error applyFixup(LinkGraph &G, Block &B, const Edge &E);

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
