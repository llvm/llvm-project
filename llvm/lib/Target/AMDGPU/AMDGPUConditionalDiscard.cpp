//===-- AMDGPUConditionalDiscard.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass transforms conditional discard of the form:
///
///     if (condition)
///       discard (false);
///
/// into:
///
///     discard (!condition);
///
///
/// More specifically,
///
/// ...
/// block:
///   %cond = icmp eq i32 %a, %b
///   br i1 %cond, label %kill_block, label %cont_block
///
/// kill_block:
///   call void @llvm.amdgcn.kill(i1 false)
///   br label %other
/// ...
///
/// gets transformed into:
///
/// ...
/// block:
///   %cond = icmp eq i32 %a, %b
///   %nonkill = not i1 %cond
///   call void @llvm.amdgcn.kill(i1 %nonkill)
///   br i1 %cond, %other, %cont_block
/// ...
///
/// The transformation is useful, because removing basic blocks simplifies CFG
/// and limits the number of phi nodes used.
/// The pass should ideally be placed after code sinking, because some sinking
/// opportunities get lost after the transformation due to the basic block
/// removal.
///
/// Additionally this pass can be used to transform kill intrinsics optimized
/// as above to demote operations.
/// This provides a workaround for applications which perform a non-uniform
/// "kill" and later compute (implicit) derivatives.
/// Note that in Vulkan, such applications should be fixed to use demote
/// (OpDemoteToHelperInvocationEXT) instead of kill (OpKill).
///

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "amdgpu-conditional-discard"

using namespace llvm;
using namespace llvm::AMDGPU;

// Enable conditional discard transformations
static cl::opt<bool> EnableConditionalDiscardTransformations(
  "amdgpu-conditional-discard-transformations",
  cl::desc("Enable conditional discard transformations"),
  cl::init(false),
  cl::Hidden);

// Enable conditional discard to demote transformations
static cl::opt<bool> EnableTransformDiscardToDemote(
  "amdgpu-transform-discard-to-demote",
  cl::desc("Enable transformation of optimized discards to demotes"),
  cl::init(false),
  cl::Hidden);

namespace {

class AMDGPUConditionalDiscard : public FunctionPass {
private:
  SmallVector<BasicBlock *, 4> KillBlocksToRemove;

  const LoopInfo *LI;

public:
  static char ID;

  AMDGPUConditionalDiscard() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
     AU.addRequiredTransitive<LoopInfoWrapperPass>();
  }

  StringRef getPassName() const override { return "AMDGPUConditionalDiscard"; }

  void optimizeBlock(BasicBlock &BB, bool ConvertToDemote);
};

} // namespace

char AMDGPUConditionalDiscard::ID = 0;

char &llvm::AMDGPUConditionalDiscardID = AMDGPUConditionalDiscard::ID;

// Look for a basic block that has only a single predecessor and its
// first instruction is a call to amdgcn_kill, with "false" as argument.
// Transform the branch condition of the block's predecessor and mark
// the block for removal. Clone the call to amdgcn_kill to the predecessor.
void AMDGPUConditionalDiscard::optimizeBlock(BasicBlock &BB, bool ConvertToDemote) {

  if (auto *KillCand = dyn_cast<CallInst>(&BB.front())) {
    auto *Callee = KillCand->getCalledFunction();
    if (!Callee || Callee->getIntrinsicID() != Intrinsic::amdgcn_kill) {
      return;
    }

    ConstantInt *Val = dyn_cast<ConstantInt>(KillCand->getOperand(0));
    if (!Val || !Val->isZero())
      return;

    // Skip if the terminator is not a branch, e.g. an "unreachable".
    if (!isa<BranchInst>(BB.getTerminator()))
      return;

    if (!ConvertToDemote && !BB.hasNPredecessors(1)) {
      LLVM_DEBUG(dbgs() << "Cannot optimize " << BB.getName()
                        << " due to multiple predecessors\n");
      return;
    }

    if (llvm::any_of(predecessors(&BB), [&](const BasicBlock *Pred) {
          return !!LI->getLoopFor(Pred);
        })) {
      // Skip if the kill is in a loop.
      LLVM_DEBUG(dbgs() << "Cannot optimize " << BB.getName()
                        << " due to loop\n");
      return;
    }

    if (llvm::any_of(predecessors(&BB), [](const BasicBlock *Pred) {
          auto *PredBranch = dyn_cast<BranchInst>(Pred->getTerminator());
          return !PredBranch || !PredBranch->isConditional();
        })) {
      LLVM_DEBUG(
          dbgs() << "Cannot optimize " << BB.getName()
                 << " due to lack of conditional branch in predecessor\n");
      return;
    }

    bool FirstPred = true;
    for (auto PredBlock : make_early_inc_range(predecessors(&BB))) {
      auto *PredBranchInst = cast<BranchInst>(PredBlock->getTerminator());

      BasicBlock *LiveBlock = nullptr;
      auto *Cond = PredBranchInst->getCondition();
      if (PredBranchInst->getSuccessor(0) == &BB) {
        // The old kill block could only be reached if
        // the condition was true - negate the condition.
        Cond =
            BinaryOperator::CreateNot(Cond, "", PredBranchInst->getIterator());
        LiveBlock = PredBranchInst->getSuccessor(1);
      } else {
        LiveBlock = PredBranchInst->getSuccessor(0);
      }

      auto *NewKill = cast<CallInst>(KillCand->clone());

      NewKill->setArgOperand(0, Cond);
      NewKill->insertBefore(PredBranchInst);

      if (ConvertToDemote) {
        NewKill->setCalledFunction(Intrinsic::getDeclaration(
            KillCand->getModule(), Intrinsic::amdgcn_wqm_demote));

        // Only mark block for removal once
        if (FirstPred)
          KillBlocksToRemove.push_back(&BB);

        // Change the branch to an unconditional one, targeting the live block.
        auto *NewBranchInst =
            BranchInst::Create(LiveBlock, PredBranchInst->getIterator());
        NewBranchInst->copyMetadata(*PredBranchInst);
        PredBranchInst->eraseFromParent();
      } else {
        assert(FirstPred);

        KillCand->eraseFromParent();

        // Try removing the old kill block.
        if (BB.sizeWithoutDebug() == 1) {
          auto *OldKillBlockBranchInst =
              dyn_cast<BranchInst>(BB.getTerminator());

          if (!OldKillBlockBranchInst->isConditional()) {
            auto *OldKillBlockSucc = OldKillBlockBranchInst->getSuccessor(0);

            // Change the branch to point to the old kill block successor
            // instead of old kill block. This is necessary to remove the old
            // kill block.
            if (PredBranchInst->getSuccessor(0) == &BB)
              PredBranchInst->setSuccessor(0, OldKillBlockSucc);
            else
              PredBranchInst->setSuccessor(1, OldKillBlockSucc);

            // It's possible that the branch became unconditional.
            if (PredBranchInst->getSuccessor(0) ==
                PredBranchInst->getSuccessor(1)) {
              auto *NewBranchInst = BranchInst::Create(
                  OldKillBlockSucc, PredBranchInst->getIterator());
              NewBranchInst->copyMetadata(*PredBranchInst);
              PredBranchInst->eraseFromParent();
            }

            // For OldKillBlockSucc's PHINode, the incoming block should be
            // updated from BB to PredBlock.  If PredBlock is already part
            // of the PHINode then BB can simply be removed.
            for (PHINode &PN : OldKillBlockSucc->phis()) {
              if (PN.getBasicBlockIndex(PredBlock) >= 0) {
                PN.removeIncomingValue(&BB);
              } else {
                PN.replaceIncomingBlockWith(&BB, PredBlock);
              }
            }

            KillBlocksToRemove.push_back(&BB);
          }
        }
      }

      FirstPred = false;
    }
  }
}

bool AMDGPUConditionalDiscard::runOnFunction(Function &F) {
  if (F.getCallingConv() != CallingConv::AMDGPU_PS)
    return false;

  if (skipFunction(F))
    return false;

  if (!(EnableConditionalDiscardTransformations ||
        F.hasFnAttribute("amdgpu-conditional-discard-transformations")))
    return false;

  bool ConvertToDemote =
      (EnableTransformDiscardToDemote ||
       F.hasFnAttribute("amdgpu-transform-discard-to-demote"));

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  for (auto &BB : F)
    optimizeBlock(BB, ConvertToDemote);

  for (auto *BB : KillBlocksToRemove) {
    for (auto *Succ : successors(BB)) {
      for (PHINode &PN : Succ->phis()) {
        if (PN.getBasicBlockIndex(BB) >= 0)
          PN.removeIncomingValue(BB);
      }
    }
    BB->eraseFromParent();
  }
  bool Ret = !KillBlocksToRemove.empty();
  KillBlocksToRemove.clear();

  return Ret;
}

INITIALIZE_PASS_BEGIN(AMDGPUConditionalDiscard, DEBUG_TYPE,
                "AMDGPUConditionalDiscard", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUConditionalDiscard, DEBUG_TYPE,
                "AMDGPUConditionalDiscard", false, false)

FunctionPass *llvm::createAMDGPUConditionalDiscardPass() {
  return new AMDGPUConditionalDiscard();
}
