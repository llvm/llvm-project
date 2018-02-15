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
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "tapirutils"

/// Returns true if the given instruction performs a detached rethrow, false
/// otherwise.
bool llvm::isDetachedRethrow(const Instruction *I) {
  if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
    if (const Function *Called = II->getCalledFunction())
      if (Intrinsic::detached_rethrow == Called->getIntrinsicID())
        return true;
  return false;
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

// Move static allocas in a cloned block into the entry block of helper.  Leave
// lifetime markers behind for those static allocas.  Returns true if the cloned
// block still contains dynamic allocas, which cannot be moved.
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
      for (Instruction *ExitPoint : ExitPoints) {
        IRBuilder<>(ExitPoint).CreateLifetimeEnd(AI, AllocaSize);
      }
    }
  }

  return ContainsDynamicAllocas;
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
               "Detach edge does not dominate a reattach into its continuation.");
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
bool llvm::isCriticalContinueEdge(const TerminatorInst *TI, unsigned SuccNum) {
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
        DEBUG(dbgs() << "Tapir: ignoring invalid hint '" <<
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
