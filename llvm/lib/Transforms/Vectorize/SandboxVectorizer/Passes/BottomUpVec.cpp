//===- BottomUpVec.cpp - A bottom-up vectorizer pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/SandboxIR/Instruction.h"

using namespace llvm::sandboxir;

namespace llvm::sandboxir {
// TODO: This is a temporary function that returns some seeds.
//       Replace this with SeedCollector's function when it lands.
static llvm::SmallVector<Value *, 4> collectSeeds(BasicBlock &BB) {
  llvm::SmallVector<Value *, 4> Seeds;
  for (auto &I : BB)
    if (auto *SI = llvm::dyn_cast<StoreInst>(&I))
      Seeds.push_back(SI);
  return Seeds;
}

static SmallVector<Value *, 4> getOperand(ArrayRef<Value *> Bndl,
                                          unsigned OpIdx) {
  SmallVector<Value *, 4> Operands;
  for (Value *BndlV : Bndl) {
    auto *BndlI = cast<Instruction>(BndlV);
    Operands.push_back(BndlI->getOperand(OpIdx));
  }
  return Operands;
}

} // namespace llvm::sandboxir

void BottomUpVec::vectorizeRec(ArrayRef<Value *> Bndl) {
  auto LegalityRes = Legality.canVectorize(Bndl);
  switch (LegalityRes.getSubclassID()) {
  case LegalityResultID::Widen: {
    auto *I = cast<Instruction>(Bndl[0]);
    for (auto OpIdx : seq<unsigned>(I->getNumOperands())) {
      auto OperandBndl = getOperand(Bndl, OpIdx);
      vectorizeRec(OperandBndl);
    }
    break;
  }
  }
}

void BottomUpVec::tryVectorize(ArrayRef<Value *> Bndl) { vectorizeRec(Bndl); }

bool BottomUpVec::runOnFunction(Function &F) {
  Change = false;
  // TODO: Start from innermost BBs first
  for (auto &BB : F) {
    // TODO: Replace with proper SeedCollector function.
    auto Seeds = collectSeeds(BB);
    // TODO: Slice Seeds into smaller chunks.
    if (Seeds.size() >= 2)
      tryVectorize(Seeds);
  }
  return Change;
}
