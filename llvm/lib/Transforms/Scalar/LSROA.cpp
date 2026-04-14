//===- LSROA.cpp - Logical Scalar Replacement Of Aggregates ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This transformation implements the well known scalar replacement of
/// aggregates transformation. It tries to identify promotable elements of an
/// aggregate alloca, and promote them to registers. It will also try to
/// convert uses of an element (or set of elements) of an alloca into a vector
/// or bitfield-style integer scalar if appropriate.
///
/// It works to do this with minimal slicing of the alloca so that regions
/// which are merely transferred in and out of external memory remain unchanged
/// and are not decomposed to scalar code.
///
/// Because this also performs alloca promotion, it can be thought of as also
/// serving the purpose of SSA formation. The algorithm iterates on the
/// function until all opportunities for promotion have been realized.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LSROA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "lsroa"

namespace {

class LSROA {
public:
  LSROA() {}

  bool runLSROA(Function &F);

private:
  bool runOnStructuredAlloca(StructuredAllocaInst &SAI);
};

} // end anonymous namespace

bool LSROA::runOnStructuredAlloca(StructuredAllocaInst &SAI) {
  // For now, LSROA only handles SGEP on structs.
  StructType *ST = dyn_cast<StructType>(SAI.getAllocationType());
  if (!ST)
    return false;

  SmallVector<SmallVector<StructuredGEPInst *, 4>, 4> FieldUsers(
      ST->getNumElements());
  for (const auto &user : SAI.users()) {
    // Lifetime intrinsincs are handled differently.
    auto II = dyn_cast<IntrinsicInst>(user);
    if (II && II->isLifetimeStartOrEnd())
      continue;

    auto SGEP = dyn_cast<StructuredGEPInst>(user);
    // If any user is not an SGEP, we bail out.
    if (!SGEP) {
      return false;
    }

    // If the SGEP has no indices, this means we have a pointer on the whole
    // struct. For now, we bail out: if it was not used, it would be DCE'd, so
    // there is probably a reference to the whole struct somewhere.
    if (SGEP->getNumIndices() == 0)
      return false;

    // IR rule: SGEP on struct can only use constant int as indices.
    ConstantInt *Index = cast<ConstantInt>(SGEP->getIndexOperand(0));
    assert(Index->getZExtValue() < FieldUsers.size());
    FieldUsers[Index->getZExtValue()].push_back(SGEP);
  }

  bool Changed = false;
  SmallPtrSet<Instruction *, 4> DeadLifetimeInstrs;
  IRBuilder B(&SAI);
  for (size_t I = 0; I < FieldUsers.size(); ++I) {
    if (FieldUsers[I].size() == 0)
      continue;
    Changed = true;

    B.SetInsertPoint(&SAI);
    StructuredAllocaInst *NSAI = cast<StructuredAllocaInst>(
        B.CreateStructuredAlloca(ST->getElementType(I)));

    // Step 1: for each lifetime intrinsic, generate one per newly created NSAI.
    for (const auto &user : SAI.users()) {
      auto II = dyn_cast<IntrinsicInst>(user);
      if (II && II->getIntrinsicID() == Intrinsic::lifetime_start) {
        B.SetInsertPoint(II);
        B.CreateLifetimeStart(NSAI);
        DeadLifetimeInstrs.insert(II);
        continue;
      }

      if (II && II->getIntrinsicID() == Intrinsic::lifetime_end) {
        B.SetInsertPoint(II);
        B.CreateLifetimeEnd(NSAI);
        DeadLifetimeInstrs.insert(II);
        continue;
      }
    }

    // Step 2: replace each SGEP usage with the new alloca
    for (StructuredGEPInst *SGEP : FieldUsers[I]) {
      if (SGEP->getNumIndices() == 1) {
        SGEP->replaceAllUsesWith(NSAI);
        SGEP->eraseFromParent();
        continue;
      }

      SmallVector<Value *, 4> Indices;
      for (unsigned J = 1; J < SGEP->getNumIndices(); ++J)
        Indices.push_back(SGEP->getIndexOperand(J));

      B.SetInsertPoint(SGEP);
      StructuredGEPInst *NSGEP = cast<StructuredGEPInst>(B.CreateStructuredGEP(
          ST->getElementType(I), NSAI, Indices, SGEP->getName()));
      SGEP->replaceAllUsesWith(NSGEP);
      SGEP->eraseFromParent();
    }
  }

  for (Instruction *I : DeadLifetimeInstrs)
    I->eraseFromParent();
  SAI.eraseFromParent();

  return Changed;
}

bool LSROA::runLSROA(Function &F) {
  BasicBlock &EntryBB = F.getEntryBlock();
  SmallVector<StructuredAllocaInst *> Worklist;

  for (BasicBlock::iterator I = EntryBB.begin(), E = std::prev(EntryBB.end());
       I != E; ++I) {
    if (StructuredAllocaInst *SAI = dyn_cast<StructuredAllocaInst>(I))
      Worklist.push_back(SAI);
  }

  bool Changed = false;
  for (StructuredAllocaInst *SAI : Worklist)
    Changed |= runOnStructuredAlloca(*SAI);
  return Changed;
}

PreservedAnalyses LSROAPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!LSROA().runLSROA(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

LSROAPass::LSROAPass() {}

namespace {

/// A legacy pass for the legacy pass manager that wraps the LSROA pass.
class LSROALegacyPass : public FunctionPass {
public:
  static char ID;

  LSROALegacyPass() : FunctionPass(ID) {
    initializeLSROALegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    return LSROA().runLSROA(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  StringRef getPassName() const override { return "LSROA"; }
};

} // end anonymous namespace

char LSROALegacyPass::ID = 0;

FunctionPass *llvm::createLSROAPass() { return new LSROALegacyPass(); }

INITIALIZE_PASS_BEGIN(LSROALegacyPass, "lsroa",
                      "Logical Scalar Replacement Of Aggregates", false, false)
INITIALIZE_PASS_END(LSROALegacyPass, "lsroa",
                    "Logical Scalar Replacement Of Aggregates", false, false)
