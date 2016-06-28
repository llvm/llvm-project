//===- ADCE.cpp - Code to perform dead code elimination -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aggressive Dead Code Elimination pass.  This pass
// optimistically assumes that all instructions are dead until proven otherwise,
// allowing it to eliminate dead computations that other DCE passes do not
// catch, particularly involving loop computations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "adce"

STATISTIC(NumRemoved, "Number of instructions removed");

static void collectLiveScopes(const DILocalScope &LS,
                              SmallPtrSetImpl<const Metadata *> &AliveScopes) {
  if (!AliveScopes.insert(&LS).second)
    return;

  if (isa<DISubprogram>(LS))
    return;

  // Tail-recurse through the scope chain.
  collectLiveScopes(cast<DILocalScope>(*LS.getScope()), AliveScopes);
}

static void collectLiveScopes(const DILocation &DL,
                              SmallPtrSetImpl<const Metadata *> &AliveScopes) {
  // Even though DILocations are not scopes, shove them into AliveScopes so we
  // don't revisit them.
  if (!AliveScopes.insert(&DL).second)
    return;

  // Collect live scopes from the scope chain.
  collectLiveScopes(*DL.getScope(), AliveScopes);

  // Tail-recurse through the inlined-at chain.
  if (const DILocation *IA = DL.getInlinedAt())
    collectLiveScopes(*IA, AliveScopes);
}

// Check if this instruction is a runtime call for value profiling and
// if it's instrumenting a constant.
static bool isInstrumentsConstant(Instruction &I) {
  if (CallInst *CI = dyn_cast<CallInst>(&I))
    if (Function *Callee = CI->getCalledFunction())
      if (Callee->getName().equals(getInstrProfValueProfFuncName()))
        if (isa<Constant>(CI->getArgOperand(0)))
          return true;
  return false;
}

static bool aggressiveDCE(Function& F) {
  SmallPtrSet<Instruction*, 32> Alive;
  SmallVector<Instruction*, 128> Worklist;

  // Collect the set of "root" instructions that are known live.
  for (Instruction &I : instructions(F)) {
    if (isa<TerminatorInst>(I) || I.isEHPad() || I.mayHaveSideEffects()) {
      // Skip any value profile instrumentation calls if they are
      // instrumenting constants.
      if (isInstrumentsConstant(I))
        continue;
      Alive.insert(&I);
      Worklist.push_back(&I);
    }
  }

  // Propagate liveness backwards to operands.  Keep track of live debug info
  // scopes.
  SmallPtrSet<const Metadata *, 32> AliveScopes;
  while (!Worklist.empty()) {
    Instruction *Curr = Worklist.pop_back_val();

    // Collect the live debug info scopes attached to this instruction.
    if (const DILocation *DL = Curr->getDebugLoc())
      collectLiveScopes(*DL, AliveScopes);

    for (Use &OI : Curr->operands()) {
      if (Instruction *Inst = dyn_cast<Instruction>(OI))
        if (Alive.insert(Inst).second)
          Worklist.push_back(Inst);
    }
  }

  // The inverse of the live set is the dead set.  These are those instructions
  // which have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the Worklist vector here for memory efficiency.
  for (Instruction &I : instructions(F)) {
    // Check if the instruction is alive.
    if (Alive.count(&I))
      continue;

    if (auto *DII = dyn_cast<DbgInfoIntrinsic>(&I)) {
      // Check if the scope of this variable location is alive.
      if (AliveScopes.count(DII->getDebugLoc()->getScope()))
        continue;

      // Fallthrough and drop the intrinsic.
      DEBUG({
        // If intrinsic is pointing at a live SSA value, there may be an
        // earlier optimization bug: if we know the location of the variable,
        // why isn't the scope of the location alive?
        if (Value *V = DII->getVariableLocation())
          if (Instruction *II = dyn_cast<Instruction>(V))
            if (Alive.count(II))
              dbgs() << "Dropping debug info for " << *DII << "\n";
      });
    }

    // Prepare to delete.
    Worklist.push_back(&I);
    I.dropAllReferences();
  }

  for (Instruction *&I : Worklist) {
    ++NumRemoved;
    I->eraseFromParent();
  }

  return !Worklist.empty();
}

PreservedAnalyses ADCEPass::run(Function &F, FunctionAnalysisManager &) {
  if (!aggressiveDCE(F))
    return PreservedAnalyses::all();

  // FIXME: This should also 'preserve the CFG'.
  auto PA = PreservedAnalyses();
  PA.preserve<GlobalsAA>();
  return PA;
}

namespace {
struct ADCELegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  ADCELegacyPass() : FunctionPass(ID) {
    initializeADCELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function& F) override {
    if (skipFunction(F))
      return false;
    return aggressiveDCE(F);
  }

  void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char ADCELegacyPass::ID = 0;
INITIALIZE_PASS(ADCELegacyPass, "adce", "Aggressive Dead Code Elimination",
                false, false)

FunctionPass *llvm::createAggressiveDCEPass() { return new ADCELegacyPass(); }
