//===- IsolatePath.cpp - Path isolation for undefined behavior -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass identifies undefined behavior (UB) that is reachable via a PHI node
// that can select a null pointer. It then refactors the control-flow graph to
// isolate the UB-triggering path from the safe paths.
//
// Once isolated, the UB path is terminated, either with an 'unreachable'
// instruction or, optionally, with a 'trap' followed by 'unreachable'. This
// prevents the optimizer from making unsafe assumptions based on the presence
// of UB, which could otherwise lead to miscompilations.
//
// For example, a null pointer dereference is transformed from:
//
//   bb:
//     %phi = phi ptr [ %valid_ptr, %pred1 ], [ null, %pred2 ]
//     %val = load i32, ptr %phi
//
// To:
//
//   bb:
//     %phi = phi ptr [ %valid_ptr, %pred1 ]
//     %val = load i32, ptr %phi
//     ...
//
//   bb.ub.path:
//     %phi.ub = phi ptr [ null, %pred2 ]
//     unreachable
//
// Or to this with the optional trap-unreachable flag:
//
//   bb.ub.path:
//     %phi.ub = phi ptr [ null, %pred2 ]
//     %val.ub = load volatile i32, ptr %phi.ub ; Optional trap
//     call void @llvm.trap()
//     unreachable
//
// This ensures that the presence of the null path does not interfere with
// valid code paths.
//
// The pass is conservative and only handles cases where UB is unambiguously
// caused by a null pointer flowing from a PHI node and used within the PHI
// node's block. It does not handle all forms of UB, but could be expanded
// as need be.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/IsolatePath.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "isolate-path"

STATISTIC(NumIsolatedBlocks, "Number of isolated blocks");

static cl::opt<bool> ConvertUBToTrapUnreachable(
    "isolate-ub-path-to-trap-unreachable", cl::init(false), cl::Hidden,
    cl::desc("Isolate the UB path into one with a 'trap-unreachable' pair."));

/// Look through GEPs to see if the nullptr is accessed.
static bool HasUBAccess(BasicBlock *Parent, GetElementPtrInst *GEP) {
  for (Value *V : GEP->materialized_users()) {
    if (auto *G = dyn_cast<GetElementPtrInst>(V)) {
      if (G->getParent() != Parent)
        return false;
      return HasUBAccess(Parent, G);
    } else if (auto *LI = dyn_cast<LoadInst>(V)) {
      if (LI->getParent() != Parent)
        return false;
      if (GEP == LI->getPointerOperand())
        return true;
    } else if (auto *SI = dyn_cast<StoreInst>(V)) {
      if (SI->getParent() != Parent)
        return false;
      if (GEP == SI->getPointerOperand())
        return true;
    }
  }

  return false;
}

static std::pair<PHINode *, Instruction *> GetFirstUBInst(BasicBlock *BB) {
  // Find PHIs that have 'nullptr' inputs.
  SmallPtrSet<PHINode *, 4> NullptrPhis;
  for (PHINode &PN : BB->phis()) {
    if (PN.getType()->isPointerTy() &&
        llvm::any_of(PN.incoming_values(),
                     [](Value *V) { return isa<ConstantPointerNull>(V); }))
      NullptrPhis.insert(&PN);
  }
  if (NullptrPhis.empty())
    return {};

  // Grab instructions that may be UB and map them to the PHI that makes them
  // so.
  SmallDenseMap<Instruction *, PHINode *, 16> UBInstToPhiMap;
  for (PHINode *PN : NullptrPhis) {
    for (Value *V : PN->materialized_users()) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getParent() != BB)
        continue;

      if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        UBInstToPhiMap[I] = PN;
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        if (HasUBAccess(BB, GEP))
          UBInstToPhiMap[GEP] = PN;
      }
    }
  }
  if (UBInstToPhiMap.empty())
    return {};

  // Get the first UB instruction by iterating through the block.
  for (Instruction &I : *BB) {
    if (auto It = UBInstToPhiMap.find(&I); It != UBInstToPhiMap.end())
      return {It->second, &I};
  }

  return {};
}

// Partition the predecessors of the PHI node that can have a null pointer into
// two sets: one for the UB path and one for the non-UB path.
static void PartitionPredecessors(PHINode *PN,
                                  SmallPtrSetImpl<BasicBlock *> &UBPreds,
                                  SmallPtrSetImpl<BasicBlock *> &NonUBPreds) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (isa<ConstantPointerNull>(PN->getIncomingValue(i)))
      UBPreds.insert(PN->getIncomingBlock(i));
    else
      NonUBPreds.insert(PN->getIncomingBlock(i));
  }
}

// Remove the appropriate incoming values from the PHI nodes in the original
// and UB blocks.
static void FixupPHINodes(BasicBlock *OrigBB, BasicBlock *UBBlock,
                          const SmallPtrSetImpl<BasicBlock *> &UBPreds,
                          const SmallPtrSetImpl<BasicBlock *> &NonUBPreds) {
  for (PHINode &PN : OrigBB->phis())
    PN.removeIncomingValueIf(
        [&](unsigned i) { return UBPreds.count(PN.getIncomingBlock(i)); });

  for (PHINode &PN : UBBlock->phis())
    PN.removeIncomingValueIf(
        [&](unsigned i) { return NonUBPreds.count(PN.getIncomingBlock(i)); });
}

// Populate the UB block with either a trap or an unreachable instruction.
static void GenerateUBPathCode(BasicBlock *UBBlock, Instruction *FirstUBInst,
                               ValueToValueMapTy &VMap) {
  if (ConvertUBToTrapUnreachable) {
    // Find the cloned version of the first UB instruction.
    auto *ClonedFirstUBInst =
        cast_or_null<Instruction>(VMap.lookup(FirstUBInst));
    if (!ClonedFirstUBInst) {
      // If the instruction wasn't cloned, it's because it was part of a larger
      // structure that was cloned, and we need to find the new instruction.
      // This can happen with GEPs that are part of a load or store.
      for (auto &I : *UBBlock) {
        if (I.getOpcode() == FirstUBInst->getOpcode() &&
            I.getOperand(0) == VMap.lookup(FirstUBInst->getOperand(0))) {
          ClonedFirstUBInst = &I;
          break;
        }
      }
    }

    // Find the position of the first UB instruction in the new block.
    unsigned UBIndex = 0;
    if (ClonedFirstUBInst) {
      for (auto I = UBBlock->begin(), E = UBBlock->end(); I != E;
           ++I, ++UBIndex) {
        if (&*I == ClonedFirstUBInst) {
          if (isa<LoadInst>(ClonedFirstUBInst) ||
              isa<StoreInst>(ClonedFirstUBInst))
            ++UBIndex;
          break;
        }
      }
    }

    // Remove instructions after the UB instruction.
    while (UBBlock->size() > UBIndex)
      UBBlock->rbegin()->eraseFromParent();

    // Make the faulting instruction volatile to ensure it traps.
    if (auto *LI = dyn_cast<LoadInst>(&*UBBlock->rbegin()))
      LI->setVolatile(true);
    else if (auto *SI = dyn_cast<StoreInst>(&*UBBlock->rbegin()))
      SI->setVolatile(true);

    // Add a trap call.
    IRBuilder<> Builder(UBBlock);
    Function *TrapDecl = Intrinsic::getOrInsertDeclaration(UBBlock->getModule(),
                                                           Intrinsic::trap);
    Builder.CreateCall(TrapDecl);
  } else {
    // Remove all instructions from the UB block.
    while (!UBBlock->empty())
      UBBlock->rbegin()->eraseFromParent();
  }

  // Terminate the UB block with an unreachable instruction.
  IRBuilder<> Builder(UBBlock);
  Builder.CreateUnreachable();
}

/// Convert any accesses of a nullptr within the BB into a trap.
bool IsolatePathPass::ProcessPointerUndefinedBehavior(BasicBlock *BB,
                                                      DomTreeUpdater *DTU,
                                                      LoopInfo *LI) {
  if (!BB->canSplitPredecessors())
    return false;

  // Get the first UB instruction and associated PHI node.
  auto [FirstUBPhiNode, FirstUBInst] = GetFirstUBInst(BB);
  if (!FirstUBInst)
    return false;

  // Partition the predecessors based on whether they feed a null pointer to the
  // PHI.
  SmallPtrSet<BasicBlock *, 4> UBPhiPreds, NonUBPhiPreds;
  PartitionPredecessors(FirstUBPhiNode, UBPhiPreds, NonUBPhiPreds);

  if (NonUBPhiPreds.empty())
    // All paths have undefined behavior. Other passes will deal with this.
    return false;

  // Clone the block to create a dedicated UB path.
  ValueToValueMapTy VMap;
  BasicBlock *UBBlock = CloneBasicBlock(BB, VMap, ".ub.path", BB->getParent());
  VMap[BB] = UBBlock;
  ++NumIsolatedBlocks;

  // Reroute the predecessors that cause UB to the new UB block.
  for (BasicBlock *Pred : UBPhiPreds)
    Pred->getTerminator()->replaceSuccessorWith(BB, UBBlock);

  // Clean up the PHI nodes in both the original and the new UB block.
  FixupPHINodes(BB, UBBlock, UBPhiPreds, NonUBPhiPreds);

  // Remap instructions in the cloned block to use values from the new block.
  for (Instruction &I : *UBBlock) {
    RemapDbgRecordRange(BB->getModule(), I.getDbgRecordRange(), VMap,
                        RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    RemapInstruction(&I, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
  }

  // Update the dominator tree with the new CFG.
  SmallVector<DominatorTree::UpdateType, 8> Updates;
  for (auto *Pred : UBPhiPreds) {
    Updates.push_back({DominatorTree::Insert, Pred, UBBlock});
    Updates.push_back({DominatorTree::Delete, Pred, BB});
  }
  if (!Updates.empty())
    DTU->applyUpdates(Updates);

  // If the original block was in a loop, add the new block to the same loop.
  if (LI)
    if (Loop *L = LI->getLoopFor(BB))
      L->addBasicBlockToLoop(UBBlock, *LI);

  // Populate the UB block with the appropriate terminating code.
  GenerateUBPathCode(UBBlock, FirstUBInst, VMap);

  SplitUBBlocks.insert(UBBlock);
  return true;
}

PreservedAnalyses IsolatePathPass::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  bool Changed = false;

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
  auto *LI = FAM.getCachedResult<LoopAnalysis>(F);
  DomTreeUpdater DTU(&DT, &PDT, DomTreeUpdater::UpdateStrategy::Eager);

  // Use a worklist of blocks because we'll be adding new blocks to the
  // function and potentially processing the same block multiple times.
  std::vector<BasicBlock *> Blocks;
  Blocks.reserve(F.size());
  llvm::transform(F, std::back_inserter(Blocks),
                  [](BasicBlock &BB) { return &BB; });

  while (!Blocks.empty()) {
    BasicBlock *BB = Blocks.back();
    Blocks.pop_back();
    if (SplitUBBlocks.contains(BB))
      continue;

    // No PHI nodes.
    if (BB->phis().empty())
      continue;

    // We don't handle landing pads or EH pads, as splitting them is complex and
    // not a goal of this pass.
    if (BB->isLandingPad() || BB->isEHPad())
      continue;

    // Support some of the more common predecessor terminators.
    if (llvm::any_of(predecessors(BB), [&](BasicBlock *Pred) {
          Instruction *TI = Pred->getTerminator();
          return !isa<BranchInst>(TI) && !isa<ReturnInst>(TI) &&
                 !isa<SwitchInst>(TI);
        }))
      continue;

    if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator()))
      // If a BB has an edge to itself, then duplication of BB could result in
      // reallocation of the BB's PHI nodes.
      if (llvm::any_of(BI->successors(),
                       [&](BasicBlock *B) { return B == BB; }))
        continue;

    if (ProcessPointerUndefinedBehavior(BB, &DTU, LI)) {
      // Reprocess the block to handle further UB instructions.
      Blocks.push_back(BB);
      Changed = true;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
