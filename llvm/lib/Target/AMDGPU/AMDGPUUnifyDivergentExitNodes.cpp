//===- AMDGPUUnifyDivergentExitNodes.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a variant of the UnifyDivergentExitNodes pass. Rather than ensuring
// there is at most one ret and one unreachable instruction, it ensures there is
// at most one divergent exiting block.
//
// StructurizeCFG can't deal with multi-exit regions formed by branches to
// multiple return nodes. It is not desirable to structurize regions with
// uniform branches, so unifying those to the same return block as divergent
// branches inhibits use of scalar branching. It still can't deal with the case
// where one branch goes to return, and one unreachable. Replace unreachable in
// this case with a return.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-unify-divergent-exit-nodes"

namespace {

class AMDGPUUnifyDivergentExitNodes : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  AMDGPUUnifyDivergentExitNodes() : FunctionPass(ID) {
    initializeAMDGPUUnifyDivergentExitNodesPass(*PassRegistry::getPassRegistry());
  }

  // We can preserve non-critical-edgeness when we unify function exit nodes
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

char AMDGPUUnifyDivergentExitNodes::ID = 0;

char &llvm::AMDGPUUnifyDivergentExitNodesID = AMDGPUUnifyDivergentExitNodes::ID;

INITIALIZE_PASS_BEGIN(AMDGPUUnifyDivergentExitNodes, DEBUG_TYPE,
                     "Unify divergent function exit nodes", false, false)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_END(AMDGPUUnifyDivergentExitNodes, DEBUG_TYPE,
                    "Unify divergent function exit nodes", false, false)

void AMDGPUUnifyDivergentExitNodes::getAnalysisUsage(AnalysisUsage &AU) const{
  // TODO: Preserve dominator tree.
  AU.addRequired<PostDominatorTreeWrapperPass>();

  AU.addRequired<LegacyDivergenceAnalysis>();

  // No divergent values are changed, only blocks and branch edges.
  AU.addPreserved<LegacyDivergenceAnalysis>();

  // We preserve the non-critical-edgeness property
  AU.addPreservedID(BreakCriticalEdgesID);

  // This is a cluster of orthogonal Transforms
  AU.addPreservedID(LowerSwitchID);
  FunctionPass::getAnalysisUsage(AU);

  AU.addRequired<TargetTransformInfoWrapperPass>();
}

/// \returns true if \p BB is reachable through only uniform branches.
/// XXX - Is there a more efficient way to find this?
static bool isUniformlyReached(const LegacyDivergenceAnalysis &DA,
                               BasicBlock &BB) {
  SmallVector<BasicBlock *, 8> Stack;
  SmallPtrSet<BasicBlock *, 8> Visited;

  for (BasicBlock *Pred : predecessors(&BB))
    Stack.push_back(Pred);

  while (!Stack.empty()) {
    BasicBlock *Top = Stack.pop_back_val();
    if (!DA.isUniform(Top->getTerminator()))
      return false;

    for (BasicBlock *Pred : predecessors(Top)) {
      if (Visited.insert(Pred).second)
        Stack.push_back(Pred);
    }
  }

  return true;
}

static void removeDoneExport(Function &F) {
  ConstantInt *BoolFalse = ConstantInt::getFalse(F.getContext());
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (IntrinsicInst *Intrin = llvm::dyn_cast<IntrinsicInst>(&I)) {
        if (Intrin->getIntrinsicID() == Intrinsic::amdgcn_exp) {
          Intrin->setArgOperand(6, BoolFalse); // done
        } else if (Intrin->getIntrinsicID() == Intrinsic::amdgcn_exp_compr) {
          Intrin->setArgOperand(4, BoolFalse); // done
        }
      }
    }
  }
}

static BasicBlock *unifyReturnBlockSet(Function &F,
                                       ArrayRef<BasicBlock *> ReturningBlocks,
                                       bool InsertExport,
                                       const TargetTransformInfo &TTI,
                                       StringRef Name) {
  // Otherwise, we need to insert a new basic block into the function, add a PHI
  // nodes (if the function returns values), and convert all of the return
  // instructions into unconditional branches.
  BasicBlock *NewRetBlock = BasicBlock::Create(F.getContext(), Name, &F);
  IRBuilder<> B(NewRetBlock);

  if (InsertExport) {
    // Ensure that there's only one "done" export in the shader by removing the
    // "done" bit set on the original final export. More than one "done" export
    // can lead to undefined behavior.
    removeDoneExport(F);

    Value *Undef = UndefValue::get(B.getFloatTy());
    B.CreateIntrinsic(Intrinsic::amdgcn_exp, { B.getFloatTy() },
                      {
                        B.getInt32(9), // target, SQ_EXP_NULL
                        B.getInt32(0), // enabled channels
                        Undef, Undef, Undef, Undef, // values
                        B.getTrue(), // done
                        B.getTrue(), // valid mask
                      });
  }

  PHINode *PN = nullptr;
  if (F.getReturnType()->isVoidTy()) {
    B.CreateRetVoid();
  } else {
    // If the function doesn't return void... add a PHI node to the block...
    PN = B.CreatePHI(F.getReturnType(), ReturningBlocks.size(),
                     "UnifiedRetVal");
    assert(!InsertExport);
    B.CreateRet(PN);
  }

  // Loop over all of the blocks, replacing the return instruction with an
  // unconditional branch.
  for (BasicBlock *BB : ReturningBlocks) {
    // Add an incoming element to the PHI node for every return instruction that
    // is merging into this new block...
    if (PN)
      PN->addIncoming(BB->getTerminator()->getOperand(0), BB);

    // Remove and delete the return inst.
    BB->getTerminator()->eraseFromParent();
    BranchInst::Create(NewRetBlock, BB);
  }

  for (BasicBlock *BB : ReturningBlocks) {
    // Cleanup possible branch to unconditional branch to the return.
    simplifyCFG(BB, TTI, {2});
  }

  return NewRetBlock;
}

bool AMDGPUUnifyDivergentExitNodes::runOnFunction(Function &F) {
  auto &PDT = getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();

  // If there's only one exit, we don't need to do anything, unless this is a
  // pixel shader and that exit is an infinite loop, since we still have to
  // insert an export in that case.
  if (PDT.root_size() <= 1 && F.getCallingConv() != CallingConv::AMDGPU_PS)
    return false;

  LegacyDivergenceAnalysis &DA = getAnalysis<LegacyDivergenceAnalysis>();

  // Loop over all of the blocks in a function, tracking all of the blocks that
  // return.
  SmallVector<BasicBlock *, 4> ReturningBlocks;
  SmallVector<BasicBlock *, 4> UniformlyReachedRetBlocks;
  SmallVector<BasicBlock *, 4> UnreachableBlocks;

  // Dummy return block for infinite loop.
  BasicBlock *DummyReturnBB = nullptr;

  bool InsertExport = false;

  bool Changed = false;
  for (BasicBlock *BB : PDT.roots()) {
    if (isa<ReturnInst>(BB->getTerminator())) {
      if (!isUniformlyReached(DA, *BB))
        ReturningBlocks.push_back(BB);
      else
        UniformlyReachedRetBlocks.push_back(BB);
    } else if (isa<UnreachableInst>(BB->getTerminator())) {
      if (!isUniformlyReached(DA, *BB))
        UnreachableBlocks.push_back(BB);
    } else if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {

      ConstantInt *BoolTrue = ConstantInt::getTrue(F.getContext());
      if (DummyReturnBB == nullptr) {
        DummyReturnBB = BasicBlock::Create(F.getContext(),
                                           "DummyReturnBlock", &F);
        Type *RetTy = F.getReturnType();
        Value *RetVal = RetTy->isVoidTy() ? nullptr : UndefValue::get(RetTy);

        // For pixel shaders, the producer guarantees that an export is
        // executed before each return instruction. However, if there is an
        // infinite loop and we insert a return ourselves, we need to uphold
        // that guarantee by inserting a null export. This can happen e.g. in
        // an infinite loop with kill instructions, which is supposed to
        // terminate. However, we don't need to do this if there is a non-void
        // return value, since then there is an epilog afterwards which will
        // still export.
        //
        // Note: In the case where only some threads enter the infinite loop,
        // this can result in the null export happening redundantly after the
        // original exports. However, The last "real" export happens after all
        // the threads that didn't enter an infinite loop converged, which
        // means that the only extra threads to execute the null export are
        // threads that entered the infinite loop, and they only could've
        // exited through being killed which sets their exec bit to 0.
        // Therefore, unless there's an actual infinite loop, which can have
        // invalid results, or there's a kill after the last export, which we
        // assume the frontend won't do, this export will have the same exec
        // mask as the last "real" export, and therefore the valid mask will be
        // overwritten with the same value and will still be correct. Also,
        // even though this forces an extra unnecessary export wait, we assume
        // that this happens rare enough in practice to that we don't have to
        // worry about performance.
        if (F.getCallingConv() == CallingConv::AMDGPU_PS &&
            RetTy->isVoidTy()) {
          InsertExport = true;
        }

        ReturnInst::Create(F.getContext(), RetVal, DummyReturnBB);
        ReturningBlocks.push_back(DummyReturnBB);
      }

      if (BI->isUnconditional()) {
        BasicBlock *LoopHeaderBB = BI->getSuccessor(0);
        BI->eraseFromParent(); // Delete the unconditional branch.
        // Add a new conditional branch with a dummy edge to the return block.
        BranchInst::Create(LoopHeaderBB, DummyReturnBB, BoolTrue, BB);
      } else { // Conditional branch.
        // Create a new transition block to hold the conditional branch.
        BasicBlock *TransitionBB = BB->splitBasicBlock(BI, "TransitionBlock");

        // Create a branch that will always branch to the transition block and
        // references DummyReturnBB.
        BB->getTerminator()->eraseFromParent();
        BranchInst::Create(TransitionBB, DummyReturnBB, BoolTrue, BB);
      }
      Changed = true;
    }
  }

  if (!UnreachableBlocks.empty()) {
    BasicBlock *UnreachableBlock = nullptr;

    if (UnreachableBlocks.size() == 1) {
      UnreachableBlock = UnreachableBlocks.front();
    } else {
      UnreachableBlock = BasicBlock::Create(F.getContext(),
                                            "UnifiedUnreachableBlock", &F);
      new UnreachableInst(F.getContext(), UnreachableBlock);

      for (BasicBlock *BB : UnreachableBlocks) {
        // Remove and delete the unreachable inst.
        BB->getTerminator()->eraseFromParent();
        BranchInst::Create(UnreachableBlock, BB);
      }
      Changed = true;
    }

    if (!ReturningBlocks.empty()) {
      // Don't create a new unreachable inst if we have a return. The
      // structurizer/annotator can't handle the multiple exits

      Type *RetTy = F.getReturnType();
      Value *RetVal = RetTy->isVoidTy() ? nullptr : UndefValue::get(RetTy);
      // Remove and delete the unreachable inst.
      UnreachableBlock->getTerminator()->eraseFromParent();

      Function *UnreachableIntrin =
        Intrinsic::getDeclaration(F.getParent(), Intrinsic::amdgcn_unreachable);

      // Insert a call to an intrinsic tracking that this is an unreachable
      // point, in case we want to kill the active lanes or something later.
      CallInst::Create(UnreachableIntrin, {}, "", UnreachableBlock);

      // Don't create a scalar trap. We would only want to trap if this code was
      // really reached, but a scalar trap would happen even if no lanes
      // actually reached here.
      ReturnInst::Create(F.getContext(), RetVal, UnreachableBlock);
      ReturningBlocks.push_back(UnreachableBlock);
      Changed = true;
    }
  }

  // Now handle return blocks.
  if (ReturningBlocks.empty())
    return Changed; // No blocks return

  if (ReturningBlocks.size() == 1 && !InsertExport)
    return Changed; // Already has a single return block

  const TargetTransformInfo &TTI
    = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  // Unify returning blocks. If we are going to insert the export it is also
  // necessary to include blocks that are uniformly reached, because in addition
  // to inserting the export the "done" bits on existing exports will be cleared
  // and we do not want to end up with the normal export in a non-unified,
  // uniformly reached block with the "done" bit cleared.
  auto BlocksToUnify = std::move(ReturningBlocks);
  if (InsertExport) {
    BlocksToUnify.insert(BlocksToUnify.end(), UniformlyReachedRetBlocks.begin(),
                         UniformlyReachedRetBlocks.end());
  }

  unifyReturnBlockSet(F, BlocksToUnify, InsertExport, TTI,
                      "UnifiedReturnBlock");
  return true;
}
