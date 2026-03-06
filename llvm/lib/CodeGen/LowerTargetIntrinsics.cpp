//===- LowerTargetIntrinsics.cpp - Lower target feature intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers llvm.target.has.feature and llvm.target.is.cpu intrinsics to
// constants using the TargetMachine. It then propagates the constants, folds
// branches, and removes dead blocks. This is a correctness requirement.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LowerTargetIntrinsics.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "lower-target-intrinsics"

STATISTIC(NumHasFeatureLowered, "Number of llvm.target.has.feature lowered");
STATISTIC(NumIsCpuLowered, "Number of llvm.target.is.cpu lowered");

// Extract a metadata string from an intrinsic argument that is a
// MetadataAsValue wrapping an MDString. Returns empty StringRef on failure.
static StringRef extractMetadataStringArg(const CallInst *CI, unsigned ArgNo) {
  auto *MAV = dyn_cast<MetadataAsValue>(CI->getArgOperand(ArgNo));
  if (!MAV)
    return StringRef();
  auto *MDS = dyn_cast<MDString>(MAV->getMetadata());
  return MDS ? MDS->getString() : StringRef();
}

static Constant *resolveTargetIntrinsic(IntrinsicInst *II,
                                        const TargetMachine *TM) {
  StringRef Name = extractMetadataStringArg(II, 0);
  Function *F = II->getFunction();
  LLVMContext &Ctx = II->getContext();

  if (Name.empty())
    return ConstantInt::getFalse(Ctx);

  switch (II->getIntrinsicID()) {
  case Intrinsic::target_has_feature: {
    const TargetSubtargetInfo *STI = TM->getSubtargetImpl(*F);
    bool Has = STI ? STI->hasFeatureString(Name) : false;
    LLVM_DEBUG(dbgs() << "  has.feature(\"" << Name << "\") -> " << Has
                      << "\n");
    ++NumHasFeatureLowered;
    return ConstantInt::getBool(Ctx, Has);
  }
  case Intrinsic::target_is_cpu: {
    const TargetSubtargetInfo *STI = TM->getSubtargetImpl(*F);
    StringRef CPU = STI ? STI->getCPU() : "";
    bool Match = (CPU == Name);
    LLVM_DEBUG(dbgs() << "  is.cpu(\"" << Name << "\") [actual=\"" << CPU
                      << "\"] -> " << Match << "\n");
    ++NumIsCpuLowered;
    return ConstantInt::getBool(Ctx, Match);
  }
  default:
    llvm_unreachable("Not a target reflection intrinsic");
  }
}

static void promoteAllocasToSSA(Function &F, DominatorTree &DT) {
  SmallVector<AllocaInst *, 8> Promotable;
  BasicBlock &Entry = F.getEntryBlock();
  for (Instruction &I : Entry)
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (isAllocaPromotable(AI))
        Promotable.push_back(AI);

  if (!Promotable.empty())
    PromoteMemToReg(Promotable, DT);
}

static SmallVector<IntrinsicInst *, 8> collectTargetIntrinsics(Function &F) {
  SmallVector<IntrinsicInst *, 8> Calls;
  for (Instruction &I : instructions(&F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::target_has_feature ||
          II->getIntrinsicID() == Intrinsic::target_is_cpu)
        Calls.push_back(II);
  return Calls;
}

bool llvm::lowerTargetIntrinsics(Function &F, const TargetMachine &TM) {
  SmallVector<IntrinsicInst *, 8> TargetCalls = collectTargetIntrinsics(F);
  if (TargetCalls.empty())
    return false;

  LLVM_DEBUG(dbgs() << "LowerTargetIntrinsics: processing " << F.getName()
                    << " (" << TargetCalls.size() << " calls)\n");

  // Promote allocas to SSA so constant propagation works through
  // the alloca/store/load patterns Clang emits at -O0.
  DominatorTree DT(F);
  promoteAllocasToSSA(F, DT);

  // Re-collect after mem2reg may have changed things.
  TargetCalls = collectTargetIntrinsics(F);
  if (TargetCalls.empty())
    return true;

  // Resolve each intrinsic to a constant and propagate.
  for (IntrinsicInst *II : TargetCalls) {
    Constant *Val = resolveTargetIntrinsic(II, &TM);
    replaceAndRecursivelySimplify(II, Val);
  }

  // Fold now-constant terminators and remove dead blocks.
  for (BasicBlock &BB : make_early_inc_range(F))
    ConstantFoldTerminator(&BB, true);
  removeUnreachableBlocks(F);

  return true;
}

PreservedAnalyses LowerTargetIntrinsicsPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  Function *HasFeatureDecl =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::target_has_feature);
  Function *IsCpuDecl =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::target_is_cpu);
  if (!HasFeatureDecl && !IsCpuDecl)
    return PreservedAnalyses::all();

  // Collect the set of functions that contain calls to these intrinsics.
  SmallPtrSet<Function *, 8> AffectedFunctions;
  for (Function *Decl : {HasFeatureDecl, IsCpuDecl}) {
    if (!Decl)
      continue;
    for (User *U : Decl->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        AffectedFunctions.insert(CI->getFunction());
  }

  bool Changed = false;
  for (Function *F : AffectedFunctions)
    Changed |= lowerTargetIntrinsics(*F, *TM);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
