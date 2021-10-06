//===---------------------- TableManager.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fix edge for edge that needs an entry to reference the target symbol
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_TABLEMANAGER_H
#define LLVM_EXECUTIONENGINE_JITLINK_TABLEMANAGER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

/// Table like section manager
template <typename TableManagerImplT> class TableManager {
public:
  /// Visit edge, return true if the edge was dealt with, otherwise return
  /// false(let other managers to visit).
  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if (impl().fixEdgeKind(G, B, E)) {
      fixTarget(G, E);
      return true;
    }
    return false;
  }

  /// Return the constructed entry
  ///
  /// Use parameter G to construct the entry for target symbol
  Symbol &getEntryForTarget(LinkGraph &G, Symbol &Target) {
    assert(Target.hasName() && "Edge cannot point to anonymous target");

    auto EntryI = Entries.find(Target.getName());

    // Build the entry if it doesn't exist.
    if (EntryI == Entries.end()) {
      auto &Entry = impl().createEntry(G, Target);
      LLVM_DEBUG({
        dbgs() << "    Created" << impl().getTableName() << "entry for "
               << Target.getName() << ": " << Entry << "\n";
      });
      EntryI = Entries.insert(std::make_pair(Target.getName(), &Entry)).first;
    }

    assert(EntryI != Entries.end() && "Could not get entry symbol");
    LLVM_DEBUG({
      dbgs() << "    Using " << impl().getTableName() << " entry "
             << *EntryI->second << "\n";
    });
    return *EntryI->second;
  }

private:
  void fixTarget(LinkGraph &G, Edge &E) {
    E.setTarget(getEntryForTarget(G, E.getTarget()));
  }

  TableManagerImplT &impl() { return static_cast<TableManagerImplT &>(*this); }
  DenseMap<StringRef, Symbol *> Entries;
};

} // namespace jitlink
} // namespace llvm

#endif
