//===- BottomUpVec.cpp - A bottom-up vectorizer pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

namespace llvm::sandboxir {

BottomUpVec::BottomUpVec(StringRef Pipeline)
    : FunctionPass("bottom-up-vec"),
      RPM("rpm", Pipeline, SandboxVectorizerPassBuilder::createRegionPass) {}

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

void BottomUpVec::vectorizeRec(ArrayRef<Value *> Bndl) {
  const auto &LegalityRes = Legality.canVectorize(Bndl);
  switch (LegalityRes.getSubclassID()) {
  case LegalityResultID::Widen: {
    auto *I = cast<Instruction>(Bndl[0]);
    for (auto OpIdx : seq<unsigned>(I->getNumOperands())) {
      auto OperandBndl = getOperand(Bndl, OpIdx);
      vectorizeRec(OperandBndl);
    }
    break;
  }
  case LegalityResultID::Pack: {
    // TODO: Unimplemented
    llvm_unreachable("Unimplemented");
  }
  }
}

void BottomUpVec::tryVectorize(ArrayRef<Value *> Bndl) { vectorizeRec(Bndl); }

bool BottomUpVec::runOnFunction(Function &F, const Analyses &A) {
  Change = false;
  // TODO: Start from innermost BBs first
  for (auto &BB : F) {
    // TODO: Replace with proper SeedCollector function.
    auto Seeds = collectSeeds(BB);
    // TODO: Slice Seeds into smaller chunks.
    // TODO: If vectorization succeeds, run the RegionPassManager on the
    // resulting region.
    if (Seeds.size() >= 2)
      tryVectorize(Seeds);
  }
  return Change;
}

} // namespace llvm::sandboxir
