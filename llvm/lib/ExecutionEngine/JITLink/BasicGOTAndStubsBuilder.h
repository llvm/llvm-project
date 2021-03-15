//===--- BasicGOTAndStubsBuilder.h - Generic GOT/Stub creation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A base for simple GOT and stub creation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_JITLINK_BASICGOTANDSTUBSBUILDER_H
#define LLVM_LIB_EXECUTIONENGINE_JITLINK_BASICGOTANDSTUBSBUILDER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

template <typename BuilderImpl> class BasicGOTAndStubsBuilder {
public:
  BasicGOTAndStubsBuilder(LinkGraph &G) : G(G) {}

  void run() {
    // We're going to be adding new blocks, but we don't want to iterate over
    // the newly added ones, so just copy the existing blocks out.
    std::vector<Block *> Blocks(G.blocks().begin(), G.blocks().end());

    LLVM_DEBUG(dbgs() << "Creating GOT entries and stubs:\n");

    for (auto *B : Blocks)
      for (auto &E : B->edges())
        if (impl().isGOTEdgeToFix(E)) {
          LLVM_DEBUG({
            dbgs() << "  Updating GOT edge ";
            printEdge(dbgs(), *B, E, "<target GOT>");
            dbgs() << "\n";
          });
          impl().fixGOTEdge(E, getGOTEntrySymbol(E.getTarget()));
        } else if (impl().isExternalBranchEdge(E)) {
          LLVM_DEBUG({
            dbgs() << "  Updating external branch edge ";
            printEdge(dbgs(), *B, E, "<target PC-rel>");
            dbgs() << "\n";
          });
          impl().fixExternalBranchEdge(E, getStubSymbol(E.getTarget()));
        }
  }

protected:
  Symbol &getGOTEntrySymbol(Symbol &Target) {
    assert(Target.hasName() && "GOT edge cannot point to anonymous target");

    auto GOTEntryI = GOTEntries.find(Target.getName());

    // Build the entry if it doesn't exist.
    if (GOTEntryI == GOTEntries.end()) {
      auto &GOTEntry = impl().createGOTEntry(Target);
      LLVM_DEBUG({
        dbgs() << "    Created GOT entry for " << Target.getName() << ": "
               << GOTEntry << "\n";
      });
      GOTEntryI =
          GOTEntries.insert(std::make_pair(Target.getName(), &GOTEntry)).first;
    }

    assert(GOTEntryI != GOTEntries.end() && "Could not get GOT entry symbol");
    LLVM_DEBUG(
        { dbgs() << "    Using GOT entry " << *GOTEntryI->second << "\n"; });
    return *GOTEntryI->second;
  }

  Symbol &getStubSymbol(Symbol &Target) {
    assert(Target.hasName() &&
           "External branch edge can not point to an anonymous target");
    auto StubI = Stubs.find(Target.getName());

    if (StubI == Stubs.end()) {
      auto &StubSymbol = impl().createStub(Target);
      LLVM_DEBUG({
        dbgs() << "    Created stub for " << Target.getName() << ": "
               << StubSymbol << "\n";
      });
      StubI = Stubs.insert(std::make_pair(Target.getName(), &StubSymbol)).first;
    }

    assert(StubI != Stubs.end() && "Count not get stub symbol");
    LLVM_DEBUG({ dbgs() << "    Using stub " << *StubI->second << "\n"; });
    return *StubI->second;
  }

  LinkGraph &G;

private:
  BuilderImpl &impl() { return static_cast<BuilderImpl &>(*this); }

  DenseMap<StringRef, Symbol *> GOTEntries;
  DenseMap<StringRef, Symbol *> Stubs;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_JITLINK_BASICGOTANDSTUBSBUILDER_H
