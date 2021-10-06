//===---------------- PerGraphTLSInfoEntryBuilder.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Construct Thread local storage info entry for each graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_PERGRAPHTLSINFOENTRYBUILDER_H
#define LLVM_EXECUTIONENGINE_JITLINK_PERGRAPHTLSINFOENTRYBUILDER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "jitlink"
namespace llvm {
namespace jitlink {

template <typename BuilderImplT> class PerGraphTLSInfoEntryBuilder {
public:
  PerGraphTLSInfoEntryBuilder(LinkGraph &G) : G(G) {}
  static Error asPass(LinkGraph &G) { return BuilderImplT(G).run(); }

  Error run() {
    LLVM_DEBUG(dbgs() << "Running Per-Graph TLS Info entry builder:\n ");

    std::vector<Block *> Worklist(G.blocks().begin(), G.blocks().end());

    for (auto *B : Worklist)
      for (auto &E : B->edges()) {
        if (impl().isTLSEdgeToFix(E)) {
          LLVM_DEBUG({
            dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind())
                   << " edge at " << formatv("{0:x}", B->getFixupAddress(E))
                   << " (" << formatv("{0:x}", B->getAddress()) << " + "
                   << formatv("{0:x}", E.getOffset()) << ")\n";
          });
          impl().fixTLSEdge(E, getTLSInfoEntry(E.getTarget()));
        }
      }
    return Error::success();
  }

protected:
  LinkGraph &G;

  Symbol &getTLSInfoEntry(Symbol &Target) {
    assert(Target.hasName() && "TLS edge cannot point to anonymous target");
    auto TLSInfoEntryI = TLSInfoEntries.find(Target.getName());
    if (TLSInfoEntryI == TLSInfoEntries.end()) {
      auto &TLSInfoEntry = impl().createTLSInfoEntry(Target);
      LLVM_DEBUG({
        dbgs() << "    Created TLS Info entry for " << Target.getName() << ": "
               << TLSInfoEntry << "\n";
      });
      TLSInfoEntryI =
          TLSInfoEntries.insert(std::make_pair(Target.getName(), &TLSInfoEntry))
              .first;
    }
    assert(TLSInfoEntryI != TLSInfoEntries.end() &&
           "Could not get TLSInfo symbol");
    LLVM_DEBUG({
      dbgs() << "    Using TLS Info entry" << *TLSInfoEntryI->second << "\n";
    });
    return *TLSInfoEntryI->second;
  }

private:
  DenseMap<StringRef, Symbol *> TLSInfoEntries;
  BuilderImplT &impl() { return static_cast<BuilderImplT &>(*this); }
};
} // namespace jitlink
} // namespace llvm
#endif
