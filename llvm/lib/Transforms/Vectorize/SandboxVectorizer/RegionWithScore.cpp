//===- RegionWithScore.cpp - A Region with score tracking -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/RegionWithScore.h"

namespace llvm::sandboxir {

InstructionCost ScoreBoard::getCost(Instruction *I) const {
  auto *LLVMI = cast<llvm::Instruction>(I->Val);
  SmallVector<const llvm::Value *> Operands(LLVMI->operands());
  return TTI.getInstructionCost(LLVMI, Operands, CostKind);
}

void ScoreBoard::remove(Instruction *I) {
  auto Cost = getCost(I);
  if (Rgn.contains(I))
    // If `I` is one the newly added ones, then we should adjust `AfterCost`
    AfterCost -= Cost;
  else
    // If `I` is one of the original instructions (outside the region) then it
    // is part of the original code, so adjust `BeforeCost`.
    BeforeCost += Cost;
}

#ifndef NDEBUG
void ScoreBoard::dump() const { dump(dbgs()); }
#endif

} // namespace llvm::sandboxir
