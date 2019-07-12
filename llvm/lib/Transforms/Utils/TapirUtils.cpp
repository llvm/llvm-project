//===-- TapirUtils.cpp - Utility methods for Tapir -------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file utility methods for handling code containing Tapir instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "tapirutils"

/// Returns true if the given instruction performs a detached rethrow, false
/// otherwise.
bool llvm::isDetachedRethrow(const Instruction *I, const Value *SyncRegion) {
  if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
    if (const Function *Called = II->getCalledFunction())
      if (Intrinsic::detached_rethrow == Called->getIntrinsicID())
        if (!SyncRegion || (SyncRegion == II->getArgOperand(0)))
          return true;
  return false;
}

/// Returns true if the reattach instruction appears to match the given detach
/// instruction, false otherwise.
///
/// If a dominator tree is not given, then this method does a best-effort check.
/// In particular, this function might return true when the reattach instruction
/// does not actually match the detach instruction, but instead matches a
/// sibling detach instruction with the same continuation.  This best-effort
/// check is sufficient in some cases, such as during a traversal of a detached
/// task..
bool llvm::ReattachMatchesDetach(const ReattachInst *RI, const DetachInst *DI,
                                 DominatorTree *DT) {
  // Check that the reattach instruction belonds to the same sync region as the
  // detach instruction.
  if (RI->getSyncRegion() != DI->getSyncRegion())
    return false;

  // Check that the destination of the reattach matches the continue destination
  // of the detach.
  if (RI->getDetachContinue() != DI->getContinue())
    return false;

  // If we have a dominator tree, check that the detach edge dominates the
  // reattach.
  if (DT) {
    BasicBlockEdge DetachEdge(DI->getParent(), DI->getDetached());
    if (!DT->dominates(DetachEdge, RI->getParent()))
      return false;
  }

  return true;
}

/// Return the result of AI->isStaticAlloca() if AI were moved to the entry
/// block. Allocas used in inalloca calls and allocas of dynamic array size
/// cannot be static.
/// (Borrowed from Transforms/Utils/InlineFunction.cpp)
static bool allocaWouldBeStaticInEntry(const AllocaInst *AI) {
  return isa<Constant>(AI->getArraySize()) && !AI->isUsedWithInAlloca();
}

// Check whether this Value is used by a lifetime intrinsic.
static bool isUsedByLifetimeMarker(Value *V) {
  for (User *U : V->users()) {
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
        return true;
      }
    }
  }
  return false;
}

// Check whether the given alloca already has lifetime.start or lifetime.end
// intrinsics.
static bool hasLifetimeMarkers(AllocaInst *AI) {
  Type *Ty = AI->getType();
  Type *Int8PtrTy = Type::getInt8PtrTy(Ty->getContext(),
                                       Ty->getPointerAddressSpace());
  if (Ty == Int8PtrTy)
    return isUsedByLifetimeMarker(AI);

  // Do a scan to find all the casts to i8*.
  for (User *U : AI->users()) {
    if (U->getType() != Int8PtrTy) continue;
    if (U->stripPointerCasts() != AI) continue;
    if (isUsedByLifetimeMarker(U))
      return true;
  }
  return false;
}

// Move static allocas in Block into Entry, which is assumed to do dominate
// Block.  Leave lifetime markers behind in Block and before each instruction in
// ExitPoints for those static allocas.  Returns true if Block still contains
// dynamic allocas, which cannot be moved.
bool llvm::MoveStaticAllocasInBlock(
    BasicBlock *Entry, BasicBlock *Block,
    SmallVectorImpl<Instruction *> &ExitPoints) {
  Function *F = Entry->getParent();
  SmallVector<AllocaInst *, 4> StaticAllocas;
  bool ContainsDynamicAllocas = false;
  BasicBlock::iterator InsertPoint = Entry->begin();
  for (BasicBlock::iterator I = Block->begin(),
         E = Block->end(); I != E; ) {
    AllocaInst *AI = dyn_cast<AllocaInst>(I++);
    if (!AI) continue;

    if (!allocaWouldBeStaticInEntry(AI)) {
      ContainsDynamicAllocas = true;
      continue;
    }

    StaticAllocas.push_back(AI);

    // Scan for the block of allocas that we can move over, and move them all at
    // once.
    while (isa<AllocaInst>(I) &&
           allocaWouldBeStaticInEntry(cast<AllocaInst>(I))) {
      StaticAllocas.push_back(cast<AllocaInst>(I));
      ++I;
    }

    // Transfer all of the allocas over in a block.  Using splice means that the
    // instructions aren't removed from the symbol table, then reinserted.
    Entry->getInstList().splice(
        InsertPoint, Block->getInstList(), AI->getIterator(), I);
  }
  // Move any dbg.declares describing the allocas into the entry basic block.
  DIBuilder DIB(*F->getParent());
  for (auto &AI : StaticAllocas)
    replaceDbgDeclareForAlloca(AI, AI, DIB, DIExpression::NoDeref, 0,
                               DIExpression::NoDeref);

  // Move any syncregion_start's into the entry basic block.
  for (BasicBlock::iterator I = Block->begin(),
         E = Block->end(); I != E; ) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(I++);
    if (!II) continue;
    if (Intrinsic::syncregion_start != II->getIntrinsicID())
      continue;

    while (isa<IntrinsicInst>(I) &&
           Intrinsic::syncregion_start ==
           cast<IntrinsicInst>(I)->getIntrinsicID())
        ++I;

    Entry->getInstList().splice(
        InsertPoint, Block->getInstList(), II->getIterator(), I);
  }

  // Leave lifetime markers for the static alloca's, scoping them to the
  // from cloned block to cloned exit.
  if (!StaticAllocas.empty()) {
    IRBuilder<> Builder(&Block->front());
    for (unsigned ai = 0, ae = StaticAllocas.size(); ai != ae; ++ai) {
      AllocaInst *AI = StaticAllocas[ai];
      // Don't mark swifterror allocas. They can't have bitcast uses.
      if (AI->isSwiftError())
        continue;

      // If the alloca is already scoped to something smaller than the whole
      // function then there's no need to add redundant, less accurate markers.
      if (hasLifetimeMarkers(AI))
        continue;

      // Try to determine the size of the allocation.
      ConstantInt *AllocaSize = nullptr;
      if (ConstantInt *AIArraySize =
          dyn_cast<ConstantInt>(AI->getArraySize())) {
        auto &DL = F->getParent()->getDataLayout();
        Type *AllocaType = AI->getAllocatedType();
        uint64_t AllocaTypeSize = DL.getTypeAllocSize(AllocaType);
        uint64_t AllocaArraySize = AIArraySize->getLimitedValue();

        // Don't add markers for zero-sized allocas.
        if (AllocaArraySize == 0)
          continue;

        // Check that array size doesn't saturate uint64_t and doesn't
        // overflow when it's multiplied by type size.
        if (AllocaArraySize != ~0ULL &&
            UINT64_MAX / AllocaArraySize >= AllocaTypeSize) {
          AllocaSize = ConstantInt::get(Type::getInt64Ty(AI->getContext()),
                                        AllocaArraySize * AllocaTypeSize);
        }
      }

      Builder.CreateLifetimeStart(AI, AllocaSize);
      for (Instruction *ExitPoint : ExitPoints)
        IRBuilder<>(ExitPoint).CreateLifetimeEnd(AI, AllocaSize);
    }
  }

  return ContainsDynamicAllocas;
}


namespace {
/// A class for recording information about inlining a landing pad.
class LandingPadInliningInfo {
  /// Destination of the invoke's unwind.
  BasicBlock *OuterResumeDest;

  /// Destination for the callee's resume.
  BasicBlock *InnerResumeDest = nullptr;

  /// LandingPadInst associated with the detach.
  Value *SpawnerLPad = nullptr;

  /// PHI for EH values from landingpad insts.
  PHINode *InnerEHValuesPHI = nullptr;

  SmallVector<Value*, 8> UnwindDestPHIValues;

  /// Dominator tree to update.
  DominatorTree *DT = nullptr;
public:
  LandingPadInliningInfo(DetachInst *DI, BasicBlock *EHContinue,
                         Value *LPadValInEHContinue,
                         DominatorTree *DT = nullptr)
      : OuterResumeDest(EHContinue), SpawnerLPad(LPadValInEHContinue), DT(DT) {
    // Find the predecessor block of OuterResumeDest.
    BasicBlock *DetachBB = DI->getParent();
    BasicBlock *DetachUnwind = DI->getUnwindDest();
    while (DetachUnwind != OuterResumeDest) {
      DetachBB = DetachUnwind;
      DetachUnwind = DetachUnwind->getUniqueSuccessor();
    }

    // If there are PHI nodes in the unwind destination block, we need to keep
    // track of which values came into them from the detach before removing the
    // edge from this block.
    BasicBlock::iterator I = OuterResumeDest->begin();
    for (; isa<PHINode>(I); ++I) {
      if (&*I == LPadValInEHContinue)
        continue;
      // Save the value to use for this edge.
      PHINode *PHI = cast<PHINode>(I);
      UnwindDestPHIValues.push_back(PHI->getIncomingValueForBlock(DetachBB));
    }
  }

  /// The outer unwind destination is the target of unwind edges introduced for
  /// calls within the inlined function.
  BasicBlock *getOuterResumeDest() const {
    return OuterResumeDest;
  }

  BasicBlock *getInnerResumeDest();

  /// Forward the 'detached_rethrow' instruction to the spawner's landing pad
  /// block.  When the landing pad block has only one predecessor, this is a
  /// simple branch. When there is more than one predecessor, we need to split
  /// the landing pad block after the landingpad instruction and jump to there.
  void forwardDetachedRethrow(InvokeInst *DR);

  /// Add incoming-PHI values to the unwind destination block for the given
  /// basic block, using the values for the original invoke's source block.
  void addIncomingPHIValuesFor(BasicBlock *BB) const {
    addIncomingPHIValuesForInto(BB, OuterResumeDest);
  }

  void addIncomingPHIValuesForInto(BasicBlock *Src, BasicBlock *Dest) const {
    BasicBlock::iterator I = Dest->begin();
    for (unsigned i = 0, e = UnwindDestPHIValues.size(); i != e; ++i, ++I) {
      PHINode *Phi = cast<PHINode>(I);
      Phi->addIncoming(UnwindDestPHIValues[i], Src);
    }
  }
};
} // end anonymous namespace

/// Get or create a target for the branch from ResumeInsts.
BasicBlock *LandingPadInliningInfo::getInnerResumeDest() {
  if (InnerResumeDest) return InnerResumeDest;

  // Split the outer resume destionation.
  BasicBlock::iterator SplitPoint;
  if (isa<LandingPadInst>(SpawnerLPad))
    SplitPoint = ++cast<Instruction>(SpawnerLPad)->getIterator();
  else
    SplitPoint = OuterResumeDest->getFirstNonPHI()->getIterator();
  InnerResumeDest =
    OuterResumeDest->splitBasicBlock(SplitPoint,
                                     OuterResumeDest->getName() + ".body");
  if (DT)
    // OuterResumeDest dominates InnerResumeDest, which dominates all other
    // nodes dominated by OuterResumeDest.
    if (DomTreeNode *OldNode = DT->getNode(OuterResumeDest)) {
      std::vector<DomTreeNode *> Children(OldNode->begin(), OldNode->end());

      DomTreeNode *NewNode = DT->addNewBlock(InnerResumeDest, OuterResumeDest);
      for (DomTreeNode *I : Children)
        DT->changeImmediateDominator(I, NewNode);
    }

  // The number of incoming edges we expect to the inner landing pad.
  const unsigned PHICapacity = 2;

  // Create corresponding new PHIs for all the PHIs in the outer landing pad.
  Instruction *InsertPoint = &InnerResumeDest->front();
  BasicBlock::iterator I = OuterResumeDest->begin();
  for (unsigned i = 0, e = UnwindDestPHIValues.size(); i != e; ++i, ++I) {
    PHINode *OuterPHI = cast<PHINode>(I);
    PHINode *InnerPHI = PHINode::Create(OuterPHI->getType(), PHICapacity,
                                        OuterPHI->getName() + ".lpad-body",
                                        InsertPoint);
    OuterPHI->replaceAllUsesWith(InnerPHI);
    InnerPHI->addIncoming(OuterPHI, OuterResumeDest);
  }

  // Create a PHI for the exception values.
  InnerEHValuesPHI = PHINode::Create(SpawnerLPad->getType(), PHICapacity,
                                     "eh.lpad-body", InsertPoint);
  SpawnerLPad->replaceAllUsesWith(InnerEHValuesPHI);
  InnerEHValuesPHI->addIncoming(SpawnerLPad, OuterResumeDest);

  // All done.
  return InnerResumeDest;
}

/// Forward the 'detached_rethrow' instruction to the spawner's landing pad
/// block.  When the landing pad block has only one predecessor, this is a
/// simple branch. When there is more than one predecessor, we need to split the
/// landing pad block after the landingpad instruction and jump to there.
void LandingPadInliningInfo::forwardDetachedRethrow(InvokeInst *DR) {
  BasicBlock *Dest = getInnerResumeDest();
  BasicBlock *Src = DR->getParent();

  BranchInst::Create(Dest, Src);
  if (DT)
    DT->changeImmediateDominator(
        Dest, DT->findNearestCommonDominator(Dest, Src));

  // Update the PHIs in the destination. They were inserted in an order which
  // makes this work.
  addIncomingPHIValuesForInto(Src, Dest);

  InnerEHValuesPHI->addIncoming(DR->getOperand(1), Src);

  // Update the DT
  BasicBlock *NormalDest = nullptr, *UnwindDest = nullptr;
  if (DT) {
    if (DR->getNormalDest()->getSinglePredecessor()) {
      NormalDest = DR->getNormalDest();
      DT->eraseNode(DR->getNormalDest());
    } else
      DT->deleteEdge(Src, DR->getNormalDest());

    if (DR->getUnwindDest()->getSinglePredecessor()) {
      UnwindDest = DR->getUnwindDest();
      DT->eraseNode(DR->getUnwindDest());
    } else
      DT->deleteEdge(Src, DR->getUnwindDest());
  }

  // Remove the DR
  if (!NormalDest)
    for (PHINode &PN : DR->getNormalDest()->phis())
      PN.removeIncomingValue(Src);
  if (!UnwindDest)
    for (PHINode &PN : DR->getUnwindDest()->phis())
      PN.removeIncomingValue(Src);

  DR->eraseFromParent();
  if (NormalDest)
    NormalDest->eraseFromParent();
  if (UnwindDest)
    UnwindDest->eraseFromParent();
}

static void handleDetachedLandingPads(
    DetachInst *DI, BasicBlock *EHContinue, Value *LPadValInEHContinue,
    SmallPtrSetImpl<LandingPadInst *> &InlinedLPads,
    SmallVectorImpl<Instruction *> &DetachedRethrows,
    DominatorTree *DT = nullptr) {
  LandingPadInliningInfo DetUnwind(DI, EHContinue, LPadValInEHContinue, DT);

  // Append the clauses from the outer landing pad instruction into the inlined
  // landing pad instructions.
  LandingPadInst *OuterLPad = DI->getLandingPadInst();
  for (LandingPadInst *InlinedLPad : InlinedLPads) {
    unsigned OuterNum = OuterLPad->getNumClauses();
    InlinedLPad->reserveClauses(OuterNum);
    for (unsigned OuterIdx = 0; OuterIdx != OuterNum; ++OuterIdx)
      InlinedLPad->addClause(OuterLPad->getClause(OuterIdx));
    if (OuterLPad->isCleanup())
      InlinedLPad->setCleanup(true);
  }

  // Forward the detached rethrows.
  for (Instruction *DR : DetachedRethrows)
    DetUnwind.forwardDetachedRethrow(cast<InvokeInst>(DR));
}

static void cloneEHBlocks(Function *F, Value *SyncRegion,
                          SmallVectorImpl<BasicBlock *> &EHBlocksToClone,
                          SmallPtrSetImpl<BasicBlock *> &EHBlockPreds,
                          SmallPtrSetImpl<LandingPadInst *> *InlinedLPads,
                          SmallVectorImpl<Instruction *> *DetachedRethrows,
                          DominatorTree *DT = nullptr) {
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> NewBlocks;
  SmallPtrSet<BasicBlock *, 8> NewBlocksSet;
  SmallPtrSet<LandingPadInst *, 4> NewInlinedLPads;
  SmallPtrSet<Instruction *, 4> NewDetachedRethrows;
  for (BasicBlock *BB : EHBlocksToClone) {
    BasicBlock *New = CloneBasicBlock(BB, VMap, ".sd", F);
    VMap[BB] = New;
    if (DT)
      DT->addNewBlock(New, DT->getRoot());
    NewBlocks.push_back(New);
    NewBlocksSet.insert(New);
  }

  SmallPtrSet<BasicBlock *, 8> NewSuccSet;
  // For all old successors, remove the predecessors in EHBlockPreds.
  for (BasicBlock *EHPred : EHBlockPreds)
    for (BasicBlock *OldSucc : successors(EHPred))
      if (VMap.count(OldSucc)) {
        OldSucc->removePredecessor(EHPred);
        NewSuccSet.insert(cast<BasicBlock>(VMap[OldSucc]));
      }

  // For all new successors, remove the predecessors not in EHBlockPreds.
  for (BasicBlock *NewSucc : NewSuccSet) {
    for (BasicBlock::iterator I = NewSucc->begin(); isa<PHINode>(I); ) {
      PHINode *PN = cast<PHINode>(I++);

      // NOTE! This loop walks backwards for a reason! First off, this minimizes
      // the cost of removal if we end up removing a large number of values, and
      // second off, this ensures that the indices for the incoming values
      // aren't invalidated when we remove one.
      for (int64_t i = PN->getNumIncomingValues() - 1; i >= 0; --i)
        if (!EHBlockPreds.count(PN->getIncomingBlock(i)))
          PN->removeIncomingValue(i, false);
    }
  }

  for (BasicBlock *EHBlock : EHBlocksToClone) {
    BasicBlock *NewEHBlock = cast<BasicBlock>(VMap[EHBlock]);
    BasicBlock *IDomBB = nullptr;
    if (DT) {
      BasicBlock *IDomBB = DT->getNode(EHBlock)->getIDom()->getBlock();
      if (VMap.lookup(IDomBB))
        DT->changeImmediateDominator(cast<BasicBlock>(VMap[EHBlock]),
                                     cast<BasicBlock>(VMap[IDomBB]));
      else
        DT->changeImmediateDominator(cast<BasicBlock>(VMap[EHBlock]), IDomBB);
    }
    // Move the edges from Preds to point to NewBB instead of BB.
    for (BasicBlock *Pred : EHBlockPreds) {
      // This is slightly more strict than necessary; the minimum requirement is
      // that there be no more than one indirectbr branching to BB. And all
      // BlockAddress uses would need to be updated.
      assert(!isa<IndirectBrInst>(Pred->getTerminator()) &&
             "Cannot split an edge from an IndirectBrInst");
      Pred->getTerminator()->replaceUsesOfWith(EHBlock, NewEHBlock);
      if (DT && Pred == IDomBB)
        DT->deleteEdge(Pred, EHBlock);
    }
  }

  // Remap instructions in the cloned blocks based on VMap.
  remapInstructionsInBlocks(NewBlocks, VMap);

  // Update all successors of the cloned EH blocks.
  for (BasicBlock *BB : EHBlocksToClone) {
    for (BasicBlock *Succ : successors(BB)) {
      if (NewBlocksSet.count(Succ)) continue;
      // Update the PHI's in the successor of the cloned EH block.
      for (PHINode &PN : Succ->phis()) {
        Value *Val = PN.getIncomingValueForBlock(BB);
        Value *NewVal = VMap.count(Val) ? cast<Value>(VMap[Val]) : Val;
        PN.addIncoming(NewVal, cast<BasicBlock>(VMap[BB]));
      }
    }
  }

  // Move the new InlinedLPads and DetachedRethrows to the appropriate
  // set/vector.
  if (InlinedLPads) {
    for (LandingPadInst *LPad : *InlinedLPads) {
      if (VMap.count(LPad))
        NewInlinedLPads.insert(cast<LandingPadInst>(VMap[LPad]));
      else
        NewInlinedLPads.insert(LPad);
    }
    InlinedLPads->clear();
    for (LandingPadInst *LPad : NewInlinedLPads)
      InlinedLPads->insert(LPad);
  }
  if (DetachedRethrows) {
    for (Instruction *DR : *DetachedRethrows) {
      if (VMap.count(DR))
        NewDetachedRethrows.insert(cast<Instruction>(VMap[DR]));
      else
        NewDetachedRethrows.insert(DR);
    }
    DetachedRethrows->clear();
    for (Instruction *DR : NewDetachedRethrows)
      DetachedRethrows->push_back(DR);
  }
}

void llvm::SerializeDetach(DetachInst *DI, BasicBlock *ParentEntry,
                           BasicBlock *EHContinue, Value *LPadValInEHContinue,
                           SmallVectorImpl<Instruction *> &Reattaches,
                           SmallVectorImpl<BasicBlock *> *EHBlocksToClone,
                           SmallPtrSetImpl<BasicBlock *> *EHBlockPreds,
                           SmallPtrSetImpl<LandingPadInst *> *InlinedLPads,
                           SmallVectorImpl<Instruction *> *DetachedRethrows,
                           DominatorTree *DT) {
  BasicBlock *Spawner = DI->getParent();
  BasicBlock *TaskEntry = DI->getDetached();
  BasicBlock *Continue = DI->getContinue();
  Value *SyncRegion = DI->getSyncRegion();

  // Clone any EH blocks that need cloning.
  if (EHBlocksToClone) {
    assert(EHBlockPreds &&
           "Given EH blocks to clone, but not blocks exiting to them.");
    cloneEHBlocks(Spawner->getParent(), SyncRegion, *EHBlocksToClone,
                  *EHBlockPreds, InlinedLPads, DetachedRethrows, DT);
  }

  // Collect the exit points into a single vector.
  SmallVector<Instruction *, 8> ExitPoints;
  for (Instruction *Exit : Reattaches)
    ExitPoints.push_back(Exit);
  if (DetachedRethrows)
    for (Instruction *Exit : *DetachedRethrows)
      ExitPoints.push_back(Exit);

  // Move static alloca instructions in the task entry to the appropriate entry
  // block.
  bool ContainsDynamicAllocas =
    MoveStaticAllocasInBlock(ParentEntry, TaskEntry, ExitPoints);
  // If the cloned loop contained dynamic alloca instructions, wrap the inlined
  // code with llvm.stacksave/llvm.stackrestore intrinsics.
  if (ContainsDynamicAllocas) {
    Module *M = Spawner->getParent()->getParent();
    // Get the two intrinsics we care about.
    Function *StackSave = Intrinsic::getDeclaration(M, Intrinsic::stacksave);
    Function *StackRestore =
      Intrinsic::getDeclaration(M,Intrinsic::stackrestore);

    // Insert the llvm.stacksave.
    CallInst *SavedPtr = IRBuilder<>(TaskEntry, TaskEntry->begin())
      .CreateCall(StackSave, {}, "savedstack");

    // Insert a call to llvm.stackrestore before the reattaches in the original
    // Tapir loop.
    for (Instruction *Exit : ExitPoints)
      IRBuilder<>(Exit).CreateCall(StackRestore, SavedPtr);
  }

  // Handle any detached-rethrows in the task.
  if (DI->hasUnwindDest()) {
    assert(InlinedLPads && "Missing set of landing pads in task.");
    assert(DetachedRethrows && "Missing set of detached rethrows in task.");
    handleDetachedLandingPads(DI, EHContinue, LPadValInEHContinue,
                              *InlinedLPads, *DetachedRethrows, DT);
  }

  // Replace reattaches with unconditional branches to the continuation.
  BasicBlock *ReattachDom = nullptr;
  for (Instruction *I : Reattaches) {
    assert(isa<ReattachInst>(I) && "Recorded reattach is not a reattach");
    assert(cast<ReattachInst>(I)->getSyncRegion() == SyncRegion &&
           "Reattach does not match sync region of detach.");
    if (DT) {
      if (!ReattachDom)
        ReattachDom = I->getParent();
      else
        ReattachDom = DT->findNearestCommonDominator(ReattachDom,
                                                     I->getParent());
    }
    ReplaceInstWithInst(I, BranchInst::Create(Continue));
  }

  // Replace the detach with an unconditional branch to the task entry.
  Continue->removePredecessor(Spawner);
  ReplaceInstWithInst(DI, BranchInst::Create(TaskEntry));

  // Update dominator tree.
  if (DT)
    if (DT->dominates(Spawner, Continue))
      DT->changeImmediateDominator(Continue, ReattachDom);
}

/// Analyze a task for serialization
void llvm::AnalyzeTaskForSerialization(
    Task *T, SmallVectorImpl<Instruction *> &Reattaches,
    SmallVectorImpl<BasicBlock *> &EHBlocksToClone,
    SmallPtrSetImpl<BasicBlock *> &EHBlockPreds,
    SmallPtrSetImpl<LandingPadInst *> &InlinedLPads,
    SmallVectorImpl<Instruction *> &DetachedRethrows) {
  assert(!T->isRootTask() && "Cannot serialize root task.");
  Value *SyncRegion = T->getDetach()->getSyncRegion();
  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    // Look for landing pads in the task (and no subtask) to be merged with a
    // spawner landing pad.
    for (BasicBlock *BB : S->blocks()) {
      // Record any shared-EH blocks that need to be cloned.
      if (S->isSharedEH()) {
        EHBlocksToClone.push_back(BB);
        if (S->getEntry() == BB)
          for (BasicBlock *Pred : predecessors(BB))
            if (T->simplyEncloses(Pred))
              EHBlockPreds.insert(Pred);
      }
      if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
        if (!isDetachedRethrow(BB->getTerminator(), SyncRegion)) {
          assert(!isDetachedRethrow(BB->getTerminator()) &&
                 "Detached rethrow in task does not match sync region.");
          // Record this landing pad to merge with DI's landing pad.
          InlinedLPads.insert(II->getLandingPadInst());
        }
      } else if (DetachInst *SubDI = dyn_cast<DetachInst>(BB->getTerminator()))
        if (SubDI->hasUnwindDest())
          // Record this landing pad to merge with DI's landing pad.
          InlinedLPads.insert(SubDI->getLandingPadInst());
    }

    if (!T->isTaskExiting(S))
      continue;

    // Find the reattach and detached-rethrow exits from this task.
    for (BasicBlock *BB : S->blocks()) {
      if (isa<ReattachInst>(BB->getTerminator())) {
        assert(cast<ReattachInst>(BB->getTerminator())->getSyncRegion() ==
               SyncRegion &&
               "Reattach in task does not match sync region with detach.");
        Reattaches.push_back(BB->getTerminator());
      } else if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
        if (isDetachedRethrow(II, SyncRegion))
          // Get detached rethrows in the task to forward.
          DetachedRethrows.push_back(II);
      }
    }
  }
}

/// Serialize the detach DI that spawns task T.  If provided, the dominator tree
/// DT will be updated to reflect the serialization.
void llvm::SerializeDetach(DetachInst *DI, Task *T, DominatorTree *DT) {
  assert(DI && "SerializeDetach given nullptr for detach.");
  assert(DI == T->getDetach() && "Task and detach arguments do not match.");
  SmallVector<BasicBlock *, 4> EHBlocksToClone;
  SmallPtrSet<BasicBlock *, 4> EHBlockPreds;
  SmallVector<Instruction *, 4> Reattaches;
  SmallPtrSet<LandingPadInst *, 4> InlinedLPads;
  SmallVector<Instruction *, 4> DetachedRethrows;

  AnalyzeTaskForSerialization(T, Reattaches, EHBlocksToClone, EHBlockPreds,
                              InlinedLPads, DetachedRethrows);
  BasicBlock *EHContinue = nullptr;
  Value *LPadVal = nullptr;
  if (DI->hasUnwindDest()) {
    EHContinue = T->getEHContinuationSpindle()->getEntry();
    LPadVal = T->getLPadValueInEHContinuationSpindle();
  }
  SerializeDetach(DI, T->getParentTask()->getEntry(), EHContinue, LPadVal,
                  Reattaches, &EHBlocksToClone, &EHBlockPreds, &InlinedLPads,
                  &DetachedRethrows, DT);
}

/// SerializeDetachedCFG - Serialize the sub-CFG detached by the specified
/// detach instruction.  Removes the detach instruction and returns a pointer to
/// the branch instruction that replaces it.
///
BranchInst *llvm::SerializeDetachedCFG(DetachInst *DI, DominatorTree *DT) {
  // Get the parent of the detach instruction.
  BasicBlock *Detacher = DI->getParent();
  // Get the detached block and continuation of this detach.
  BasicBlock *Detached = DI->getDetached();
  BasicBlock *Continuation = DI->getContinue();
  BasicBlock *Unwind = nullptr;
  if (DI->hasUnwindDest())
    Unwind = DI->getUnwindDest();

  assert(Detached->getSinglePredecessor() &&
         "Detached block has multiple predecessors.");

  // Get the detach edge from DI.
  BasicBlockEdge DetachEdge(Detacher, Detached);

  // Collect the reattaches into the continuation.  If DT is available, verify
  // that all reattaches are dominated by the detach edge from DI.
  SmallVector<ReattachInst *, 8> Reattaches;
  // If we only find a single reattach into the continuation, capture it so we
  // can later update the dominator tree.
  BasicBlock *SingleReattacher = nullptr;
  int ReattachesFound = 0;
  for (auto PI = pred_begin(Continuation), PE = pred_end(Continuation);
       PI != PE; PI++) {
    BasicBlock *Pred = *PI;
    // Skip the detacher.
    if (Detacher == Pred) continue;
    // Record the reattaches found.
    if (isa<ReattachInst>(Pred->getTerminator())) {
      ReattachesFound++;
      if (!SingleReattacher)
        SingleReattacher = Pred;
      if (DT) {
        assert(DT->dominates(DetachEdge, Pred) &&
               "Detach edge does not dominate a reattach "
               "into its continuation.");
      }
      Reattaches.push_back(cast<ReattachInst>(Pred->getTerminator()));
    }
  }
  // TODO: It's possible to detach a CFG that does not terminate with a
  // reattach.  For example, optimizations can create detached CFG's that are
  // terminated by unreachable terminators only.  Some of these special cases
  // lead to problems with other passes, however, and this check will identify
  // those special cases early while we sort out those issues.
  assert(!Reattaches.empty() && "No reattach found for detach.");

  // Replace each reattach with branches to the continuation.
  for (ReattachInst *RI : Reattaches) {
    BranchInst *ReplacementBr = BranchInst::Create(Continuation, RI);
    ReplacementBr->setDebugLoc(RI->getDebugLoc());
    RI->eraseFromParent();
  }

  // Replace the new detach with a branch to the detached CFG.
  Continuation->removePredecessor(DI->getParent());
  if (Unwind)
    Unwind->removePredecessor(DI->getParent());
  BranchInst *ReplacementBr = BranchInst::Create(Detached, DI);
  ReplacementBr->setDebugLoc(DI->getDebugLoc());
  DI->eraseFromParent();

  // Update the dominator tree.
  if (DT)
    if (DT->dominates(Detacher, Continuation) && 1 == ReattachesFound)
      DT->changeImmediateDominator(Continuation, SingleReattacher);

  return ReplacementBr;
}

/// GetDetachedCtx - Get the entry basic block to the detached context
/// that contains the specified block.
///
BasicBlock *llvm::GetDetachedCtx(BasicBlock *BB) {
  return const_cast<BasicBlock *>(
      GetDetachedCtx(const_cast<const BasicBlock *>(BB)));
}

const BasicBlock *llvm::GetDetachedCtx(const BasicBlock *BB) {
  // Traverse the CFG backwards until we either reach the entry block of the
  // function or we find a detach instruction that detaches the current block.
  SmallPtrSet<const BasicBlock *, 32> Visited;
  SmallVector<const BasicBlock *, 32> WorkList;
  WorkList.push_back(BB);
  while (!WorkList.empty()) {
    const BasicBlock *CurrBB = WorkList.pop_back_val();
    if (!Visited.insert(CurrBB).second)
      continue;

    for (auto PI = pred_begin(CurrBB), PE = pred_end(CurrBB);
         PI != PE; ++PI) {
      const BasicBlock *PredBB = *PI;

      // Skip predecessors via reattach instructions.  The detacher block
      // corresponding to this reattach is also a predecessor of the current
      // basic block.
      if (isa<ReattachInst>(PredBB->getTerminator()))
        continue;

      // Skip predecessors via detach rethrows.
      if (isDetachedRethrow(PredBB->getTerminator()))
        continue;

      // If the predecessor is terminated by a detach, check to see if
      // that detach detached the current basic block.
      if (isa<DetachInst>(PredBB->getTerminator())) {
        const DetachInst *DI = cast<DetachInst>(PredBB->getTerminator());
        if (DI->getDetached() == CurrBB)
          // Return the current block, which is the entry of this detached
          // sub-CFG.
          return CurrBB;
      }

      // Otherwise, add the predecessor block to the work list to search.
      WorkList.push_back(PredBB);
    }
  }

  // Our search didn't find anything, so return the entry of the function
  // containing the given block.
  return &(BB->getParent()->getEntryBlock());
}

/// isCriticalContinueEdge - Return true if the specified edge is a critical
/// detach-continue edge.  Critical detach-continue edges are critical edges -
/// from a block with multiple successors to a block with multiple predecessors
/// - even after ignoring all reattach edges.
bool llvm::isCriticalContinueEdge(const Instruction *TI, unsigned SuccNum) {
  assert(SuccNum < TI->getNumSuccessors() && "Illegal edge specification!");
  if (TI->getNumSuccessors() == 1) return false;

  // Edge must come from a detach.
  if (!isa<DetachInst>(TI)) return false;
  // Edge must go to the continuation.
  if (SuccNum != 1) return false;

  const BasicBlock *Dest = TI->getSuccessor(SuccNum);
  const_pred_iterator I = pred_begin(Dest), E = pred_end(Dest);

  // If there is more than one predecessor, this is a critical edge...
  assert(I != E && "No preds, but we have an edge to the block?");
  const BasicBlock *DetachPred = TI->getParent();
  for (; I != E; ++I) {
    if (DetachPred == *I) continue;
    // Even if a reattach instruction isn't associated with the detach
    // instruction TI, we can safely skip it, because it will be associated with
    // a different detach instruction that precedes this block.
    if (isa<ReattachInst>((*I)->getTerminator())) continue;
    return true;
  }
  return false;
}

/// canDetach - Return true if the given function can perform a detach, false
/// otherwise.
bool llvm::canDetach(const Function *F) {
  for (const BasicBlock &BB : *F)
    if (isa<DetachInst>(BB.getTerminator()))
      return true;
  return false;
}


void llvm::GetDetachedCFG(const DetachInst &DI, const DominatorTree &DT,
                          SmallPtrSetImpl<BasicBlock *> &TaskBlocks,
                          SmallPtrSetImpl<BasicBlock *> &EHBlocks,
                          SmallPtrSetImpl<BasicBlock *> &TaskReturns) {
  SmallVector<BasicBlock *, 32> Todo;
  SmallVector<BasicBlock *, 4> WorkListEH;

  LLVM_DEBUG(dbgs() << "Finding CFG detached by " << DI << "\n");

  BasicBlock *Detached = DI.getDetached();
  BasicBlock *Continue = DI.getContinue();
  Value *SyncRegion = DI.getSyncRegion();
  BasicBlockEdge DetachEdge(DI.getParent(), Detached);

  Todo.push_back(Detached);
  while (!Todo.empty()) {
    BasicBlock *BB = Todo.pop_back_val();

    if (!TaskBlocks.insert(BB).second) continue;

    LLVM_DEBUG(dbgs() << "  Found block " << BB->getName() << "\n");

    Instruction *Term = BB->getTerminator();
    if (nullptr == Term)
      llvm_unreachable("BB with null terminator found.");

    if (ReattachInst *RI = dyn_cast<ReattachInst>(Term)) {
      // Either a reattach instruction terminates the detached CFG or it
      // terminates a nested detached CFG.  If it terminates a nested detached
      // CFG, it can simply be ignored, because the corresponding nested detach
      // instruction will be processed later.
      if (RI->getDetachContinue() != Continue) continue;
      assert(RI->getSyncRegion() == SyncRegion &&
             "Reattach terminating detached CFG has nonmatching sync region.");
      TaskReturns.insert(BB);
      continue;
    } else if (DetachInst *NestedDI = dyn_cast<DetachInst>(Term)) {
      assert(NestedDI != &DI && "Found recursive Detach");
      // Add the successors of the nested detach instruction for searching.
      Todo.push_back(NestedDI->getDetached());
      Todo.push_back(NestedDI->getContinue());
      if (NestedDI->hasUnwindDest())
        Todo.push_back(NestedDI->getUnwindDest());
      continue;
    } else if (SyncInst *SI = dyn_cast<SyncInst>(Term)) {
      // A sync instruction should only apply to nested detaches within this
      // task.  Hence it can be treated like a branch.
      assert(SI->getSyncRegion() != SyncRegion &&
             "Sync in detached task applies to parent parallel context.");
      Todo.push_back(SI->getSuccessor(0));
      continue;
    } else if (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
               isa<InvokeInst>(Term)) {
      if (isDetachedRethrow(Term, SyncRegion)) {
        // A detached rethrow terminates this task and is included in the set of
        // exception-handling blocks that might not be unique to this task.
        LLVM_DEBUG(dbgs() << "  Exit block " << BB->getName() << "\n");
        TaskReturns.insert(BB);
        EHBlocks.insert(BB);
      } else {
        for (BasicBlock *Succ : successors(BB)) {
          if (DT.dominates(DetachEdge, Succ)) {
            LLVM_DEBUG(dbgs() <<
                       "Adding successor " << Succ->getName() << "\n");
            Todo.push_back(Succ);
          } else {
            // We assume that this block is an exception-handling block and save
            // it for later processing.
            LLVM_DEBUG(dbgs() <<
                       "  Exit block to search " << Succ->getName() << "\n");
            EHBlocks.insert(Succ);
            WorkListEH.push_back(Succ);
          }
        }
      }
      continue;
    } else if (isa<UnreachableInst>(Term)) {
      // We don't bother cloning unreachable exits from the detached CFG at this
      // point.  We're cloning the entire detached CFG anyway when we outline
      // the function.
      continue;
    } else {
      llvm_unreachable("Detached task does not absolutely terminate in reattach");
    }
  }

  // Find the exception-handling exit blocks.
  {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (!WorkListEH.empty()) {
      BasicBlock *BB = WorkListEH.pop_back_val();
      if (!Visited.insert(BB).second)
        continue;

      // Make sure that the control flow through these exception-handling blocks
      // cannot re-enter the blocks being outlined.
      assert(!TaskBlocks.count(BB) &&
             "EH blocks for a detached task reenter that task.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't perform an ordinary return or resume.
      assert(!isa<ReturnInst>(BB->getTerminator()) &&
             "EH block terminated by return.");
      assert(!isa<ResumeInst>(BB->getTerminator()) &&
             "EH block terminated by resume.");

      // Make sure that the control flow through these exception-handling blocks
      // doesn't reattach to the detached CFG's continuation.
      LLVM_DEBUG({
          if (ReattachInst *RI = dyn_cast<ReattachInst>(BB->getTerminator()))
            assert(RI->getSuccessor(0) != Continue &&
                   "Exit block reaches a reattach to the continuation.");
        });

      // Stop searching down this path upon finding a detached rethrow.
      if (isDetachedRethrow(BB->getTerminator(), SyncRegion)) {
        TaskReturns.insert(BB);
        continue;
      }

      for (BasicBlock *Succ : successors(BB)) {
        EHBlocks.insert(Succ);
        WorkListEH.push_back(Succ);
      }
    }

    // Visited now contains exception-handling blocks that we want to clone as
    // part of outlining.
    for (BasicBlock *EHBlock : Visited)
      TaskBlocks.insert(EHBlock);
  }

  LLVM_DEBUG({
      dbgs() << "Exit blocks:";
      for (BasicBlock *Exit : EHBlocks) {
        if (DT.dominates(DetachEdge, Exit))
          dbgs() << "(dominated)";
        else
          dbgs() << "(shared)";
        dbgs() << *Exit;
      }
      dbgs() << "\n";
    });
}


// Helper function to find PHI nodes that depend on the landing pad in the
// unwind destination of this task's detach.
void llvm::getDetachUnwindPHIUses(DetachInst *DI,
                                  SmallPtrSetImpl<BasicBlock *> &UnwindPHIs) {
  // Get the landing pad of the unwind destination of the detach.
  LandingPadInst *LPad = nullptr;
  if (DI && DI->hasUnwindDest()) {
    BasicBlock *UnwindDest = DI->getUnwindDest();
    LPad = UnwindDest->getLandingPadInst();
    assert(LPad && "Unwind of detach is not a landing pad.");
  }
  if (!LPad) return;

  // Walk the chain of uses of this landing pad to find all PHI nodes that
  // depend on it, directly or indirectly.
  SmallVector<User *, 8> WorkList;
  SmallPtrSet<User *, 8> Visited;
  for (User *U : LPad->users())
    WorkList.push_back(U);

  while (!WorkList.empty()) {
    User *Curr = WorkList.pop_back_val();
    if (!Visited.insert(Curr).second) continue;

    // If we find a PHI-node user, add it to UnwindPHIs
    if (PHINode *PN = dyn_cast<PHINode>(Curr))
      UnwindPHIs.insert(PN->getParent());

    // Queue the successors for processing
    for (User *U : Curr->users())
      WorkList.push_back(U);
  }
}

/// Find hints specified in the loop metadata and update local values.
void llvm::TapirLoopHints::getHintsFromMetadata() {
  MDNode *LoopID = TheLoop->getLoopID();
  if (!LoopID)
    return;

  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
    const MDString *S = nullptr;
    SmallVector<Metadata *, 4> Args;

    // The expected hint is either a MDString or a MDNode with the first
    // operand a MDString.
    if (const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i))) {
      if (!MD || MD->getNumOperands() == 0)
        continue;
      S = dyn_cast<MDString>(MD->getOperand(0));
      for (unsigned i = 1, ie = MD->getNumOperands(); i < ie; ++i)
        Args.push_back(MD->getOperand(i));
    } else {
      S = dyn_cast<MDString>(LoopID->getOperand(i));
      assert(Args.size() == 0 && "too many arguments for MDString");
    }

    if (!S)
      continue;

    // Check if the hint starts with the loop metadata prefix.
    StringRef Name = S->getString();
    if (Args.size() == 1)
      setHint(Name, Args[0]);
  }
}

/// Checks string hint with one operand and set value if valid.
void llvm::TapirLoopHints::setHint(StringRef Name, Metadata *Arg) {
  if (!Name.startswith(Prefix()))
    return;
  Name = Name.substr(Prefix().size(), StringRef::npos);

  const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
  if (!C)
    return;
  unsigned Val = C->getZExtValue();

  Hint *Hints[] = {&Strategy, &Grainsize};
  for (auto H : Hints) {
    if (Name == H->Name) {
      if (H->validate(Val))
        H->Value = Val;
      else
        LLVM_DEBUG(dbgs() << "Tapir: ignoring invalid hint '" <<
                   Name << "'\n");
      break;
    }
  }
}

/// Create a new hint from name / value pair.
MDNode *llvm::TapirLoopHints::createHintMetadata(
    StringRef Name, unsigned V) const {
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  Metadata *MDs[] = {MDString::get(Context, Name),
                     ConstantAsMetadata::get(
                         ConstantInt::get(Type::getInt32Ty(Context), V))};
  return MDNode::get(Context, MDs);
}

/// Matches metadata with hint name.
bool llvm::TapirLoopHints::matchesHintMetadataName(
    MDNode *Node, ArrayRef<Hint> HintTypes) const {
  MDString *Name = dyn_cast<MDString>(Node->getOperand(0));
  if (!Name)
    return false;

  for (auto H : HintTypes)
    if (Name->getString().endswith(H.Name))
      return true;
  return false;
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::writeHintsToMetadata(ArrayRef<Hint> HintTypes) {
  if (HintTypes.size() == 0)
    return;

  // Reserve the first element to LoopID (see below).
  SmallVector<Metadata *, 4> MDs(1);
  // If the loop already has metadata, then ignore the existing operands.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, HintTypes))
        MDs.push_back(Node);
    }
  }

  // Now, add the missing hints.
  for (auto H : HintTypes)
    MDs.push_back(createHintMetadata(Twine(Prefix(), H.Name).str(), H.Value));

  // Replace current metadata node with new one.
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  TheLoop->setLoopID(NewLoopID);
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::writeHintsToClonedMetadata(ArrayRef<Hint> HintTypes,
                                                      ValueToValueMapTy &VMap) {
  if (HintTypes.size() == 0)
    return;

  // Reserve the first element to LoopID (see below).
  SmallVector<Metadata *, 4> MDs(1);
  // If the loop already has metadata, then ignore the existing operands.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, HintTypes))
        MDs.push_back(Node);
    }
  }

  // Now, add the missing hints.
  for (auto H : HintTypes)
    MDs.push_back(createHintMetadata(Twine(Prefix(), H.Name).str(), H.Value));

  // Replace current metadata node with new one.
  LLVMContext &Context =
    cast<BasicBlock>(VMap[TheLoop->getHeader()])->getContext();
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  // Set the metadata on the terminator of the cloned loop's latch.
  BasicBlock *ClonedLatch = cast<BasicBlock>(VMap[TheLoop->getLoopLatch()]);
  assert(ClonedLatch && "Cloned Tapir loop does not have a single latch.");
  ClonedLatch->getTerminator()->setMetadata(LLVMContext::MD_loop, NewLoopID);
}

/// Sets current hints into loop metadata, keeping other values intact.
void llvm::TapirLoopHints::clearHintsMetadata() {
  Hint Hints[] = {Hint("spawn.strategy", ST_SEQ, HK_STRATEGY),
                  Hint("grainsize", 0, HK_GRAINSIZE)};
  // Reserve the first element to LoopID (see below).
  SmallVector<Metadata *, 4> MDs(1);
  // If the loop already has metadata, then ignore the existing operands.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If node in update list, ignore old value.
      if (!matchesHintMetadataName(Node, Hints))
        MDs.push_back(Node);
    }
  }

  // Replace current metadata node with new one.
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);

  TheLoop->setLoopID(NewLoopID);
}

/// Returns true if Tapir-loop hints require loop outlining during lowering.
bool llvm::hintsDemandOutlining(const TapirLoopHints &Hints) {
  switch (Hints.getStrategy()) {
  case TapirLoopHints::ST_DAC: return true;
  default: return false;
  }
}

/// Given an loop id metadata node, returns the loop hint metadata node with the
/// given name (for example, "tapir.loop.stripmine.count"). If no such metadata
/// node exists, then nullptr is returned.
MDNode *llvm::GetStripMineMetadata(MDNode *LoopID, StringRef Name) {
  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;

    if (Name.equals(S->getString()))
      return MD;
  }
  return nullptr;
}

/// Examine a given loop to determine if it is a Tapir loop.  Returns the Task
/// that encodes the loop body if so, or nullptr if not.
Task *llvm::getTaskIfTapirLoop(const Loop *L, TaskInfo *TI) {
  if (!L || !TI)
    return nullptr;

  TapirLoopHints Hints(L);

  LLVM_DEBUG(dbgs() << "Loop hints:"
             << " strategy = " << Hints.printStrategy(Hints.getStrategy())
             << " grainsize = " << Hints.getGrainsize()
             << "\n");

  // Check that this loop has the structure of a Tapir loop.
  Task *T = getTaskIfTapirLoopStructure(L, TI);
  if (!T)
    return nullptr;

  // Check that the loop hints require this loop to be outlined.
  if (!hintsDemandOutlining(Hints))
    return nullptr;

  return T;
}
