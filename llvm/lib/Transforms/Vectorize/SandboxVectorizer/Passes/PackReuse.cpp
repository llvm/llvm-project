//===- PackReuse.cpp - A pack de-duplication pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/PackReuse.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm::sandboxir {

bool PackReuse::runOnRegion(Region &Rgn, const Analyses &A) {
  if (Rgn.empty())
    return Change;
  // The key to the map is the ordered operands of the pack.
  // The value is a vector of all Pack Instrs with the same operands.
  DenseMap<std::pair<BasicBlock *, SmallVector<Value *>>,
           SmallVector<SmallVector<Instruction *>>>
      PacksMap;
  // Go over the region and look for pack patterns.
  for (auto *I : Rgn) {
    auto PackOpt = VecUtils::matchPack(I);
    if (PackOpt) {
      // TODO: For now limit pack reuse within a BB.
      BasicBlock *BB = (*PackOpt->Instrs.front()).getParent();
      PacksMap[{BB, PackOpt->Operands}].push_back(PackOpt->Instrs);
    }
  }
  for (auto &Pair : PacksMap) {
    auto &Packs = Pair.second;
    if (Packs.size() <= 1)
      continue;
    // Sort packs by program order.
    sort(Packs, [](const auto &PackInstrs1, const auto &PackInstrs2) {
      return PackInstrs1.front()->comesBefore(PackInstrs2.front());
    });
    Instruction *TopMostPack = Packs[0].front();
    // Replace duplicate packs with the first one.
    for (const auto &PackInstrs :
         make_range(std::next(Packs.begin()), Packs.end())) {
      PackInstrs.front()->replaceAllUsesWith(TopMostPack);
      // Delete the pack instrs bottom-up since they are now dead.
      for (auto *PackI : PackInstrs)
        PackI->eraseFromParent();
    }
    Change = true;
  }
  return Change;
}

} // namespace llvm::sandboxir
