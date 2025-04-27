//=------- CompactUnwindSupport.cpp - Compact Unwind format support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compact Unwind support.
//
//===----------------------------------------------------------------------===//

#include "CompactUnwindSupport.h"

#include "llvm/ADT/Sequence.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

Error splitCompactUnwindBlocks(LinkGraph &G, Section &CompactUnwindSection,
                               size_t RecordSize) {

  std::vector<Block *> OriginalBlocks(CompactUnwindSection.blocks().begin(),
                                      CompactUnwindSection.blocks().end());
  LLVM_DEBUG({
    dbgs() << "In " << G.getName() << " splitting compact unwind section "
           << CompactUnwindSection.getName() << " containing "
           << OriginalBlocks.size() << " initial blocks...\n";
  });

  while (!OriginalBlocks.empty()) {
    auto *B = OriginalBlocks.back();
    OriginalBlocks.pop_back();

    if (B->getSize() == 0) {
      LLVM_DEBUG({
        dbgs() << "  Skipping empty block at "
               << formatv("{0:x16}", B->getAddress()) << "\n";
      });
      continue;
    }

    unsigned NumBlocks = B->getSize() / RecordSize;

    LLVM_DEBUG({
      dbgs() << "  Splitting block at " << formatv("{0:x16}", B->getAddress())
             << " into " << NumBlocks << " compact unwind record(s)\n";
    });

    if (B->getSize() % RecordSize)
      return make_error<JITLinkError>(
          "Error splitting compact unwind record in " + G.getName() +
          ": block at " + formatv("{0:x}", B->getAddress()) + " has size " +
          formatv("{0:x}", B->getSize()) +
          " (not a multiple of CU record size of " +
          formatv("{0:x}", RecordSize) + ")");

    auto Blocks =
        G.splitBlock(*B, map_range(seq(1U, NumBlocks), [=](Edge::OffsetT Idx) {
          return Idx * RecordSize;
        }));

    for (auto *CURec : Blocks) {
      bool AddedKeepAlive = false;

      for (auto &E : CURec->edges()) {
        if (E.getOffset() == 0) {
          LLVM_DEBUG({
            dbgs() << "    Updating compact unwind record at "
                   << CURec->getAddress() << " to point to "
                   << (E.getTarget().hasName() ? *E.getTarget().getName()
                                               : StringRef())
                   << " (at " << E.getTarget().getAddress() << ")\n";
          });

          if (E.getTarget().isExternal())
            return make_error<JITLinkError>(
                "Error adding keep-alive edge for compact unwind record at " +
                formatv("{0:x}", CURec->getAddress()) + ": target " +
                *E.getTarget().getName() + " is an external symbol");
          auto &TgtBlock = E.getTarget().getBlock();
          auto &CURecSym =
              G.addAnonymousSymbol(*CURec, 0, RecordSize, false, false);
          TgtBlock.addEdge(Edge::KeepAlive, 0, CURecSym, 0);
          AddedKeepAlive = true;
        }
      }

      if (!AddedKeepAlive)
        return make_error<JITLinkError>(
            "Error adding keep-alive edge for compact unwind record at " +
            formatv("{0:x}", CURec->getAddress()) +
            ": no outgoing target edge at offset 0");
    }
  }

  return Error::success();
}

} // end namespace jitlink
} // end namespace llvm
