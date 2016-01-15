//===-- Local.cpp - Functions to perform local transformations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "local"

STATISTIC(NumRemoved, "Number of unreachable basic blocks removed");

//===----------------------------------------------------------------------===//
//  Local constant propagation.
//

/// ConstantFoldTerminator - If a terminator instruction is predicated on a
/// constant value, convert it into an unconditional branch to the constant
/// destination.  This is a nontrivial operation because the successors of this
/// basic block must have their PHI nodes updated.
/// Also calls RecursivelyDeleteTriviallyDeadInstructions() on any branch/switch
/// conditions and indirectbr addresses this might make dead if
/// DeleteDeadConditions is true.
bool llvm::ConstantFoldTerminator(BasicBlock *BB, bool DeleteDeadConditions,
                                  const TargetLibraryInfo *TLI) {
  TerminatorInst *T = BB->getTerminator();
  IRBuilder<> Builder(T);

  // Branch - See if we are conditional jumping on constant
  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = BI->getSuccessor(0);
    BasicBlock *Dest2 = BI->getSuccessor(1);

    if (ConstantInt *Cond = dyn_cast<ConstantInt>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getZExtValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getZExtValue() ? Dest2 : Dest1;

      //cerr << "Function: " << T->getParent()->getParent()
      //     << "\nRemoving branch from " << T->getParent()
      //     << "\n\nTo: " << OldDest << endl;

      // Let the basic block know that we are letting go of it.  Based on this,
      // it will adjust it's PHI nodes.
      OldDest->removePredecessor(BB);

      // Replace the conditional branch with an unconditional one.
      Builder.CreateBr(Destination);
      BI->eraseFromParent();
      return true;
    }

    if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Replace the conditional branch with an unconditional one.
      Builder.CreateBr(Dest1);
      Value *Cond = BI->getCondition();
      BI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Cond, TLI);
      return true;
    }
    return false;
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch to an
    // unconditional branch.
    ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *DefaultDest = SI->getDefaultDest();
    BasicBlock *TheOnlyDest = DefaultDest;

    // If the default is unreachable, ignore it when searching for TheOnlyDest.
    if (isa<UnreachableInst>(DefaultDest->getFirstNonPHIOrDbg()) &&
        SI->getNumCases() > 0) {
      TheOnlyDest = SI->case_begin().getCaseSuccessor();
    }

    // Figure out which case it goes to.
    for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
         i != e; ++i) {
      // Found case matching a constant operand?
      if (i.getCaseValue() == CI) {
        TheOnlyDest = i.getCaseSuccessor();
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (i.getCaseSuccessor() == DefaultDest) {
        MDNode *MD = SI->getMetadata(LLVMContext::MD_prof);
        unsigned NCases = SI->getNumCases();
        // Fold the case metadata into the default if there will be any branches
        // left, unless the metadata doesn't match the switch.
        if (NCases > 1 && MD && MD->getNumOperands() == 2 + NCases) {
          // Collect branch weights into a vector.
          SmallVector<uint32_t, 8> Weights;
          for (unsigned MD_i = 1, MD_e = MD->getNumOperands(); MD_i < MD_e;
               ++MD_i) {
            ConstantInt *CI =
                mdconst::dyn_extract<ConstantInt>(MD->getOperand(MD_i));
            assert(CI);
            Weights.push_back(CI->getValue().getZExtValue());
          }
          // Merge weight of this case to the default weight.
          unsigned idx = i.getCaseIndex();
          Weights[0] += Weights[idx+1];
          // Remove weight for this case.
          std::swap(Weights[idx+1], Weights.back());
          Weights.pop_back();
          SI->setMetadata(LLVMContext::MD_prof,
                          MDBuilder(BB->getContext()).
                          createBranchWeights(Weights));
        }
        // Remove this entry.
        DefaultDest->removePredecessor(SI->getParent());
        SI->removeCase(i);
        --i; --e;
        continue;
      }

      // Otherwise, check to see if the switch only branches to one destination.
      // We do this by reseting "TheOnlyDest" to null when we find two non-equal
      // destinations.
      if (i.getCaseSuccessor() != TheOnlyDest) TheOnlyDest = nullptr;
    }

    if (CI && !TheOnlyDest) {
      // Branching on a constant, but not any of the cases, go to the default
      // successor.
      TheOnlyDest = SI->getDefaultDest();
    }

    // If we found a single destination that we can fold the switch into, do so
    // now.
    if (TheOnlyDest) {
      // Insert the new branch.
      Builder.CreateBr(TheOnlyDest);
      BasicBlock *BB = SI->getParent();

      // Remove entries from PHI nodes which we no longer branch to...
      for (BasicBlock *Succ : SI->successors()) {
        // Found case matching a constant operand?
        if (Succ == TheOnlyDest)
          TheOnlyDest = nullptr; // Don't modify the first branch to TheOnlyDest
        else
          Succ->removePredecessor(BB);
      }

      // Delete the old switch.
      Value *Cond = SI->getCondition();
      SI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Cond, TLI);
      return true;
    }

    if (SI->getNumCases() == 1) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      SwitchInst::CaseIt FirstCase = SI->case_begin();
      Value *Cond = Builder.CreateICmpEQ(SI->getCondition(),
          FirstCase.getCaseValue(), "cond");

      // Insert the new branch.
      BranchInst *NewBr = Builder.CreateCondBr(Cond,
                                               FirstCase.getCaseSuccessor(),
                                               SI->getDefaultDest());
      MDNode *MD = SI->getMetadata(LLVMContext::MD_prof);
      if (MD && MD->getNumOperands() == 3) {
        ConstantInt *SICase =
            mdconst::dyn_extract<ConstantInt>(MD->getOperand(2));
        ConstantInt *SIDef =
            mdconst::dyn_extract<ConstantInt>(MD->getOperand(1));
        assert(SICase && SIDef);
        // The TrueWeight should be the weight for the single case of SI.
        NewBr->setMetadata(LLVMContext::MD_prof,
                        MDBuilder(BB->getContext()).
                        createBranchWeights(SICase->getValue().getZExtValue(),
                                            SIDef->getValue().getZExtValue()));
      }

      // Update make.implicit metadata to the newly-created conditional branch.
      MDNode *MakeImplicitMD = SI->getMetadata(LLVMContext::MD_make_implicit);
      if (MakeImplicitMD)
        NewBr->setMetadata(LLVMContext::MD_make_implicit, MakeImplicitMD);

      // Delete the old switch.
      SI->eraseFromParent();
      return true;
    }
    return false;
  }

  if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(T)) {
    // indirectbr blockaddress(@F, @BB) -> br label @BB
    if (BlockAddress *BA =
          dyn_cast<BlockAddress>(IBI->getAddress()->stripPointerCasts())) {
      BasicBlock *TheOnlyDest = BA->getBasicBlock();
      // Insert the new branch.
      Builder.CreateBr(TheOnlyDest);

      for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
        if (IBI->getDestination(i) == TheOnlyDest)
          TheOnlyDest = nullptr;
        else
          IBI->getDestination(i)->removePredecessor(IBI->getParent());
      }
      Value *Address = IBI->getAddress();
      IBI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Address, TLI);

      // If we didn't find our destination in the IBI successor list, then we
      // have undefined behavior.  Replace the unconditional branch with an
      // 'unreachable' instruction.
      if (TheOnlyDest) {
        BB->getTerminator()->eraseFromParent();
        new UnreachableInst(BB->getContext(), BB);
      }

      return true;
    }
  }

  return false;
}


//===----------------------------------------------------------------------===//
//  Local dead code elimination.
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool llvm::isInstructionTriviallyDead(Instruction *I,
                                      const TargetLibraryInfo *TLI) {
  if (!I->use_empty() || isa<TerminatorInst>(I)) return false;

  // We don't want the landingpad-like instructions removed by anything this
  // general.
  if (I->isEHPad())
    return false;

  // We don't want debug info removed by anything this general, unless
  // debug info is empty.
  if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(I)) {
    if (DDI->getAddress())
      return false;
    return true;
  }
  if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(I)) {
    if (DVI->getValue())
      return false;
    return true;
  }

  if (!I->mayHaveSideEffects()) return true;

  // Special case intrinsics that "may have side effects" but can be deleted
  // when dead.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    // Safe to delete llvm.stacksave if dead.
    if (II->getIntrinsicID() == Intrinsic::stacksave)
      return true;

    // Lifetime intrinsics are dead when their right-hand is undef.
    if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
        II->getIntrinsicID() == Intrinsic::lifetime_end)
      return isa<UndefValue>(II->getArgOperand(1));

    // Assumptions are dead if their condition is trivially true.
    if (II->getIntrinsicID() == Intrinsic::assume) {
      if (ConstantInt *Cond = dyn_cast<ConstantInt>(II->getArgOperand(0)))
        return !Cond->isZero();

      return false;
    }
  }

  if (isAllocLikeFn(I, TLI)) return true;

  if (CallInst *CI = isFreeCall(I, TLI))
    if (Constant *C = dyn_cast<Constant>(CI->getArgOperand(0)))
      return C->isNullValue() || isa<UndefValue>(C);

  return false;
}

/// RecursivelyDeleteTriviallyDeadInstructions - If the specified value is a
/// trivially dead instruction, delete it.  If that makes any of its operands
/// trivially dead, delete them too, recursively.  Return true if any
/// instructions were deleted.
bool
llvm::RecursivelyDeleteTriviallyDeadInstructions(Value *V,
                                                 const TargetLibraryInfo *TLI) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !I->use_empty() || !isInstructionTriviallyDead(I, TLI))
    return false;

  SmallVector<Instruction*, 16> DeadInsts;
  DeadInsts.push_back(I);

  do {
    I = DeadInsts.pop_back_val();

    // Null out all of the instruction's operands to see if any operand becomes
    // dead as we go.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      Value *OpV = I->getOperand(i);
      I->setOperand(i, nullptr);

      if (!OpV->use_empty()) continue;

      // If the operand is an instruction that became dead as we nulled out the
      // operand, and if it is 'trivially' dead, delete it in a future loop
      // iteration.
      if (Instruction *OpI = dyn_cast<Instruction>(OpV))
        if (isInstructionTriviallyDead(OpI, TLI))
          DeadInsts.push_back(OpI);
    }

    I->eraseFromParent();
  } while (!DeadInsts.empty());

  return true;
}

/// areAllUsesEqual - Check whether the uses of a value are all the same.
/// This is similar to Instruction::hasOneUse() except this will also return
/// true when there are no uses or multiple uses that all refer to the same
/// value.
static bool areAllUsesEqual(Instruction *I) {
  Value::user_iterator UI = I->user_begin();
  Value::user_iterator UE = I->user_end();
  if (UI == UE)
    return true;

  User *TheUse = *UI;
  for (++UI; UI != UE; ++UI) {
    if (*UI != TheUse)
      return false;
  }
  return true;
}

/// RecursivelyDeleteDeadPHINode - If the specified value is an effectively
/// dead PHI node, due to being a def-use chain of single-use nodes that
/// either forms a cycle or is terminated by a trivially dead instruction,
/// delete it.  If that makes any of its operands trivially dead, delete them
/// too, recursively.  Return true if a change was made.
bool llvm::RecursivelyDeleteDeadPHINode(PHINode *PN,
                                        const TargetLibraryInfo *TLI) {
  SmallPtrSet<Instruction*, 4> Visited;
  for (Instruction *I = PN; areAllUsesEqual(I) && !I->mayHaveSideEffects();
       I = cast<Instruction>(*I->user_begin())) {
    if (I->use_empty())
      return RecursivelyDeleteTriviallyDeadInstructions(I, TLI);

    // If we find an instruction more than once, we're on a cycle that
    // won't prove fruitful.
    if (!Visited.insert(I).second) {
      // Break the cycle and delete the instruction and its operands.
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      (void)RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
      return true;
    }
  }
  return false;
}

static bool
simplifyAndDCEInstruction(Instruction *I,
                          SmallSetVector<Instruction *, 16> &WorkList,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI) {
  if (isInstructionTriviallyDead(I, TLI)) {
    // Null out all of the instruction's operands to see if any operand becomes
    // dead as we go.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      Value *OpV = I->getOperand(i);
      I->setOperand(i, nullptr);

      if (!OpV->use_empty() || I == OpV)
        continue;

      // If the operand is an instruction that became dead as we nulled out the
      // operand, and if it is 'trivially' dead, delete it in a future loop
      // iteration.
      if (Instruction *OpI = dyn_cast<Instruction>(OpV))
        if (isInstructionTriviallyDead(OpI, TLI))
          WorkList.insert(OpI);
    }

    I->eraseFromParent();

    return true;
  }

  if (Value *SimpleV = SimplifyInstruction(I, DL)) {
    // Add the users to the worklist. CAREFUL: an instruction can use itself,
    // in the case of a phi node.
    for (User *U : I->users())
      if (U != I)
        WorkList.insert(cast<Instruction>(U));

    // Replace the instruction with its simplified value.
    I->replaceAllUsesWith(SimpleV);
    I->eraseFromParent();
    return true;
  }
  return false;
}

/// SimplifyInstructionsInBlock - Scan the specified basic block and try to
/// simplify any instructions in it and recursively delete dead instructions.
///
/// This returns true if it changed the code, note that it can delete
/// instructions in other blocks as well in this block.
bool llvm::SimplifyInstructionsInBlock(BasicBlock *BB,
                                       const TargetLibraryInfo *TLI) {
  bool MadeChange = false;
  const DataLayout &DL = BB->getModule()->getDataLayout();

#ifndef NDEBUG
  // In debug builds, ensure that the terminator of the block is never replaced
  // or deleted by these simplifications. The idea of simplification is that it
  // cannot introduce new instructions, and there is no way to replace the
  // terminator of a block without introducing a new instruction.
  AssertingVH<Instruction> TerminatorVH(&BB->back());
#endif

  SmallSetVector<Instruction *, 16> WorkList;
  // Iterate over the original function, only adding insts to the worklist
  // if they actually need to be revisited. This avoids having to pre-init
  // the worklist with the entire function's worth of instructions.
  for (BasicBlock::iterator BI = BB->begin(), E = std::prev(BB->end()); BI != E;) {
    assert(!BI->isTerminator());
    Instruction *I = &*BI;
    ++BI;

    // We're visiting this instruction now, so make sure it's not in the
    // worklist from an earlier visit.
    if (!WorkList.count(I))
      MadeChange |= simplifyAndDCEInstruction(I, WorkList, DL, TLI);
  }

  while (!WorkList.empty()) {
    Instruction *I = WorkList.pop_back_val();
    MadeChange |= simplifyAndDCEInstruction(I, WorkList, DL, TLI);
  }
  return MadeChange;
}

//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring.
//


/// RemovePredecessorAndSimplify - Like BasicBlock::removePredecessor, this
/// method is called when we're about to delete Pred as a predecessor of BB.  If
/// BB contains any PHI nodes, this drops the entries in the PHI nodes for Pred.
///
/// Unlike the removePredecessor method, this attempts to simplify uses of PHI
/// nodes that collapse into identity values.  For example, if we have:
///   x = phi(1, 0, 0, 0)
///   y = and x, z
///
/// .. and delete the predecessor corresponding to the '1', this will attempt to
/// recursively fold the and to 0.
void llvm::RemovePredecessorAndSimplify(BasicBlock *BB, BasicBlock *Pred) {
  // This only adjusts blocks with PHI nodes.
  if (!isa<PHINode>(BB->begin()))
    return;

  // Remove the entries for Pred from the PHI nodes in BB, but do not simplify
  // them down.  This will leave us with single entry phi nodes and other phis
  // that can be removed.
  BB->removePredecessor(Pred, true);

  WeakVH PhiIt = &BB->front();
  while (PHINode *PN = dyn_cast<PHINode>(PhiIt)) {
    PhiIt = &*++BasicBlock::iterator(cast<Instruction>(PhiIt));
    Value *OldPhiIt = PhiIt;

    if (!recursivelySimplifyInstruction(PN))
      continue;

    // If recursive simplification ended up deleting the next PHI node we would
    // iterate to, then our iterator is invalid, restart scanning from the top
    // of the block.
    if (PhiIt != OldPhiIt) PhiIt = &BB->front();
  }
}


/// MergeBasicBlockIntoOnlyPred - DestBB is a block with one predecessor and its
/// predecessor is known to have one successor (DestBB!).  Eliminate the edge
/// between them, moving the instructions in the predecessor into DestBB and
/// deleting the predecessor block.
///
void llvm::MergeBasicBlockIntoOnlyPred(BasicBlock *DestBB, DominatorTree *DT) {
  // If BB has single-entry PHI nodes, fold them.
  while (PHINode *PN = dyn_cast<PHINode>(DestBB->begin())) {
    Value *NewVal = PN->getIncomingValue(0);
    // Replace self referencing PHI with undef, it must be dead.
    if (NewVal == PN) NewVal = UndefValue::get(PN->getType());
    PN->replaceAllUsesWith(NewVal);
    PN->eraseFromParent();
  }

  BasicBlock *PredBB = DestBB->getSinglePredecessor();
  assert(PredBB && "Block doesn't have a single predecessor!");

  // Zap anything that took the address of DestBB.  Not doing this will give the
  // address an invalid value.
  if (DestBB->hasAddressTaken()) {
    BlockAddress *BA = BlockAddress::get(DestBB);
    Constant *Replacement =
      ConstantInt::get(llvm::Type::getInt32Ty(BA->getContext()), 1);
    BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(Replacement,
                                                     BA->getType()));
    BA->destroyConstant();
  }

  // Anything that branched to PredBB now branches to DestBB.
  PredBB->replaceAllUsesWith(DestBB);

  // Splice all the instructions from PredBB to DestBB.
  PredBB->getTerminator()->eraseFromParent();
  DestBB->getInstList().splice(DestBB->begin(), PredBB->getInstList());

  // If the PredBB is the entry block of the function, move DestBB up to
  // become the entry block after we erase PredBB.
  if (PredBB == &DestBB->getParent()->getEntryBlock())
    DestBB->moveAfter(PredBB);

  if (DT) {
    BasicBlock *PredBBIDom = DT->getNode(PredBB)->getIDom()->getBlock();
    DT->changeImmediateDominator(DestBB, PredBBIDom);
    DT->eraseNode(PredBB);
  }
  // Nuke BB.
  PredBB->eraseFromParent();
}

/// CanMergeValues - Return true if we can choose one of these values to use
/// in place of the other. Note that we will always choose the non-undef
/// value to keep.
static bool CanMergeValues(Value *First, Value *Second) {
  return First == Second || isa<UndefValue>(First) || isa<UndefValue>(Second);
}

/// CanPropagatePredecessorsForPHIs - Return true if we can fold BB, an
/// almost-empty BB ending in an unconditional branch to Succ, into Succ.
///
/// Assumption: Succ is the single successor for BB.
///
static bool CanPropagatePredecessorsForPHIs(BasicBlock *BB, BasicBlock *Succ) {
  assert(*succ_begin(BB) == Succ && "Succ is not successor of BB!");

  DEBUG(dbgs() << "Looking to fold " << BB->getName() << " into "
        << Succ->getName() << "\n");
  // Shortcut, if there is only a single predecessor it must be BB and merging
  // is always safe
  if (Succ->getSinglePredecessor()) return true;

  // Make a list of the predecessors of BB
  SmallPtrSet<BasicBlock*, 16> BBPreds(pred_begin(BB), pred_end(BB));

  // Look at all the phi nodes in Succ, to see if they present a conflict when
  // merging these blocks
  for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);

    // If the incoming value from BB is again a PHINode in
    // BB which has the same incoming value for *PI as PN does, we can
    // merge the phi nodes and then the blocks can still be merged
    PHINode *BBPN = dyn_cast<PHINode>(PN->getIncomingValueForBlock(BB));
    if (BBPN && BBPN->getParent() == BB) {
      for (unsigned PI = 0, PE = PN->getNumIncomingValues(); PI != PE; ++PI) {
        BasicBlock *IBB = PN->getIncomingBlock(PI);
        if (BBPreds.count(IBB) &&
            !CanMergeValues(BBPN->getIncomingValueForBlock(IBB),
                            PN->getIncomingValue(PI))) {
          DEBUG(dbgs() << "Can't fold, phi node " << PN->getName() << " in "
                << Succ->getName() << " is conflicting with "
                << BBPN->getName() << " with regard to common predecessor "
                << IBB->getName() << "\n");
          return false;
        }
      }
    } else {
      Value* Val = PN->getIncomingValueForBlock(BB);
      for (unsigned PI = 0, PE = PN->getNumIncomingValues(); PI != PE; ++PI) {
        // See if the incoming value for the common predecessor is equal to the
        // one for BB, in which case this phi node will not prevent the merging
        // of the block.
        BasicBlock *IBB = PN->getIncomingBlock(PI);
        if (BBPreds.count(IBB) &&
            !CanMergeValues(Val, PN->getIncomingValue(PI))) {
          DEBUG(dbgs() << "Can't fold, phi node " << PN->getName() << " in "
                << Succ->getName() << " is conflicting with regard to common "
                << "predecessor " << IBB->getName() << "\n");
          return false;
        }
      }
    }
  }

  return true;
}

typedef SmallVector<BasicBlock *, 16> PredBlockVector;
typedef DenseMap<BasicBlock *, Value *> IncomingValueMap;

/// \brief Determines the value to use as the phi node input for a block.
///
/// Select between \p OldVal any value that we know flows from \p BB
/// to a particular phi on the basis of which one (if either) is not
/// undef. Update IncomingValues based on the selected value.
///
/// \param OldVal The value we are considering selecting.
/// \param BB The block that the value flows in from.
/// \param IncomingValues A map from block-to-value for other phi inputs
/// that we have examined.
///
/// \returns the selected value.
static Value *selectIncomingValueForBlock(Value *OldVal, BasicBlock *BB,
                                          IncomingValueMap &IncomingValues) {
  if (!isa<UndefValue>(OldVal)) {
    assert((!IncomingValues.count(BB) ||
            IncomingValues.find(BB)->second == OldVal) &&
           "Expected OldVal to match incoming value from BB!");

    IncomingValues.insert(std::make_pair(BB, OldVal));
    return OldVal;
  }

  IncomingValueMap::const_iterator It = IncomingValues.find(BB);
  if (It != IncomingValues.end()) return It->second;

  return OldVal;
}

/// \brief Create a map from block to value for the operands of a
/// given phi.
///
/// Create a map from block to value for each non-undef value flowing
/// into \p PN.
///
/// \param PN The phi we are collecting the map for.
/// \param IncomingValues [out] The map from block to value for this phi.
static void gatherIncomingValuesToPhi(PHINode *PN,
                                      IncomingValueMap &IncomingValues) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *BB = PN->getIncomingBlock(i);
    Value *V = PN->getIncomingValue(i);

    if (!isa<UndefValue>(V))
      IncomingValues.insert(std::make_pair(BB, V));
  }
}

/// \brief Replace the incoming undef values to a phi with the values
/// from a block-to-value map.
///
/// \param PN The phi we are replacing the undefs in.
/// \param IncomingValues A map from block to value.
static void replaceUndefValuesInPhi(PHINode *PN,
                                    const IncomingValueMap &IncomingValues) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *V = PN->getIncomingValue(i);

    if (!isa<UndefValue>(V)) continue;

    BasicBlock *BB = PN->getIncomingBlock(i);
    IncomingValueMap::const_iterator It = IncomingValues.find(BB);
    if (It == IncomingValues.end()) continue;

    PN->setIncomingValue(i, It->second);
  }
}

/// \brief Replace a value flowing from a block to a phi with
/// potentially multiple instances of that value flowing from the
/// block's predecessors to the phi.
///
/// \param BB The block with the value flowing into the phi.
/// \param BBPreds The predecessors of BB.
/// \param PN The phi that we are updating.
static void redirectValuesFromPredecessorsToPhi(BasicBlock *BB,
                                                const PredBlockVector &BBPreds,
                                                PHINode *PN) {
  Value *OldVal = PN->removeIncomingValue(BB, false);
  assert(OldVal && "No entry in PHI for Pred BB!");

  IncomingValueMap IncomingValues;

  // We are merging two blocks - BB, and the block containing PN - and
  // as a result we need to redirect edges from the predecessors of BB
  // to go to the block containing PN, and update PN
  // accordingly. Since we allow merging blocks in the case where the
  // predecessor and successor blocks both share some predecessors,
  // and where some of those common predecessors might have undef
  // values flowing into PN, we want to rewrite those values to be
  // consistent with the non-undef values.

  gatherIncomingValuesToPhi(PN, IncomingValues);

  // If this incoming value is one of the PHI nodes in BB, the new entries
  // in the PHI node are the entries from the old PHI.
  if (isa<PHINode>(OldVal) && cast<PHINode>(OldVal)->getParent() == BB) {
    PHINode *OldValPN = cast<PHINode>(OldVal);
    for (unsigned i = 0, e = OldValPN->getNumIncomingValues(); i != e; ++i) {
      // Note that, since we are merging phi nodes and BB and Succ might
      // have common predecessors, we could end up with a phi node with
      // identical incoming branches. This will be cleaned up later (and
      // will trigger asserts if we try to clean it up now, without also
      // simplifying the corresponding conditional branch).
      BasicBlock *PredBB = OldValPN->getIncomingBlock(i);
      Value *PredVal = OldValPN->getIncomingValue(i);
      Value *Selected = selectIncomingValueForBlock(PredVal, PredBB,
                                                    IncomingValues);

      // And add a new incoming value for this predecessor for the
      // newly retargeted branch.
      PN->addIncoming(Selected, PredBB);
    }
  } else {
    for (unsigned i = 0, e = BBPreds.size(); i != e; ++i) {
      // Update existing incoming values in PN for this
      // predecessor of BB.
      BasicBlock *PredBB = BBPreds[i];
      Value *Selected = selectIncomingValueForBlock(OldVal, PredBB,
                                                    IncomingValues);

      // And add a new incoming value for this predecessor for the
      // newly retargeted branch.
      PN->addIncoming(Selected, PredBB);
    }
  }

  replaceUndefValuesInPhi(PN, IncomingValues);
}

/// TryToSimplifyUncondBranchFromEmptyBlock - BB is known to contain an
/// unconditional branch, and contains no instructions other than PHI nodes,
/// potential side-effect free intrinsics and the branch.  If possible,
/// eliminate BB by rewriting all the predecessors to branch to the successor
/// block and return true.  If we can't transform, return false.
bool llvm::TryToSimplifyUncondBranchFromEmptyBlock(BasicBlock *BB) {
  assert(BB != &BB->getParent()->getEntryBlock() &&
         "TryToSimplifyUncondBranchFromEmptyBlock called on entry block!");

  // We can't eliminate infinite loops.
  BasicBlock *Succ = cast<BranchInst>(BB->getTerminator())->getSuccessor(0);
  if (BB == Succ) return false;

  // Check to see if merging these blocks would cause conflicts for any of the
  // phi nodes in BB or Succ. If not, we can safely merge.
  if (!CanPropagatePredecessorsForPHIs(BB, Succ)) return false;

  // Check for cases where Succ has multiple predecessors and a PHI node in BB
  // has uses which will not disappear when the PHI nodes are merged.  It is
  // possible to handle such cases, but difficult: it requires checking whether
  // BB dominates Succ, which is non-trivial to calculate in the case where
  // Succ has multiple predecessors.  Also, it requires checking whether
  // constructing the necessary self-referential PHI node doesn't introduce any
  // conflicts; this isn't too difficult, but the previous code for doing this
  // was incorrect.
  //
  // Note that if this check finds a live use, BB dominates Succ, so BB is
  // something like a loop pre-header (or rarely, a part of an irreducible CFG);
  // folding the branch isn't profitable in that case anyway.
  if (!Succ->getSinglePredecessor()) {
    BasicBlock::iterator BBI = BB->begin();
    while (isa<PHINode>(*BBI)) {
      for (Use &U : BBI->uses()) {
        if (PHINode* PN = dyn_cast<PHINode>(U.getUser())) {
          if (PN->getIncomingBlock(U) != BB)
            return false;
        } else {
          return false;
        }
      }
      ++BBI;
    }
  }

  DEBUG(dbgs() << "Killing Trivial BB: \n" << *BB);

  if (isa<PHINode>(Succ->begin())) {
    // If there is more than one pred of succ, and there are PHI nodes in
    // the successor, then we need to add incoming edges for the PHI nodes
    //
    const PredBlockVector BBPreds(pred_begin(BB), pred_end(BB));

    // Loop over all of the PHI nodes in the successor of BB.
    for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = cast<PHINode>(I);

      redirectValuesFromPredecessorsToPhi(BB, BBPreds, PN);
    }
  }

  if (Succ->getSinglePredecessor()) {
    // BB is the only predecessor of Succ, so Succ will end up with exactly
    // the same predecessors BB had.

    // Copy over any phi, debug or lifetime instruction.
    BB->getTerminator()->eraseFromParent();
    Succ->getInstList().splice(Succ->getFirstNonPHI()->getIterator(),
                               BB->getInstList());
  } else {
    while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
      // We explicitly check for such uses in CanPropagatePredecessorsForPHIs.
      assert(PN->use_empty() && "There shouldn't be any uses here!");
      PN->eraseFromParent();
    }
  }

  // Everything that jumped to BB now goes to Succ.
  BB->replaceAllUsesWith(Succ);
  if (!Succ->hasName()) Succ->takeName(BB);
  BB->eraseFromParent();              // Delete the old basic block.
  return true;
}

/// EliminateDuplicatePHINodes - Check for and eliminate duplicate PHI
/// nodes in this block. This doesn't try to be clever about PHI nodes
/// which differ only in the order of the incoming values, but instcombine
/// orders them so it usually won't matter.
///
bool llvm::EliminateDuplicatePHINodes(BasicBlock *BB) {
  // This implementation doesn't currently consider undef operands
  // specially. Theoretically, two phis which are identical except for
  // one having an undef where the other doesn't could be collapsed.

  struct PHIDenseMapInfo {
    static PHINode *getEmptyKey() {
      return DenseMapInfo<PHINode *>::getEmptyKey();
    }
    static PHINode *getTombstoneKey() {
      return DenseMapInfo<PHINode *>::getTombstoneKey();
    }
    static unsigned getHashValue(PHINode *PN) {
      // Compute a hash value on the operands. Instcombine will likely have
      // sorted them, which helps expose duplicates, but we have to check all
      // the operands to be safe in case instcombine hasn't run.
      return static_cast<unsigned>(hash_combine(
          hash_combine_range(PN->value_op_begin(), PN->value_op_end()),
          hash_combine_range(PN->block_begin(), PN->block_end())));
    }
    static bool isEqual(PHINode *LHS, PHINode *RHS) {
      if (LHS == getEmptyKey() || LHS == getTombstoneKey() ||
          RHS == getEmptyKey() || RHS == getTombstoneKey())
        return LHS == RHS;
      return LHS->isIdenticalTo(RHS);
    }
  };

  // Set of unique PHINodes.
  DenseSet<PHINode *, PHIDenseMapInfo> PHISet;

  // Examine each PHI.
  bool Changed = false;
  for (auto I = BB->begin(); PHINode *PN = dyn_cast<PHINode>(I++);) {
    auto Inserted = PHISet.insert(PN);
    if (!Inserted.second) {
      // A duplicate. Replace this PHI with its duplicate.
      PN->replaceAllUsesWith(*Inserted.first);
      PN->eraseFromParent();
      Changed = true;

      // The RAUW can change PHIs that we already visited. Start over from the
      // beginning.
      PHISet.clear();
      I = BB->begin();
    }
  }

  return Changed;
}

/// enforceKnownAlignment - If the specified pointer points to an object that
/// we control, modify the object's alignment to PrefAlign. This isn't
/// often possible though. If alignment is important, a more reliable approach
/// is to simply align all global variables and allocation instructions to
/// their preferred alignment from the beginning.
///
static unsigned enforceKnownAlignment(Value *V, unsigned Align,
                                      unsigned PrefAlign,
                                      const DataLayout &DL) {
  V = V->stripPointerCasts();

  if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    // If the preferred alignment is greater than the natural stack alignment
    // then don't round up. This avoids dynamic stack realignment.
    if (DL.exceedsNaturalStackAlignment(PrefAlign))
      return Align;
    // If there is a requested alignment and if this is an alloca, round up.
    if (AI->getAlignment() >= PrefAlign)
      return AI->getAlignment();
    AI->setAlignment(PrefAlign);
    return PrefAlign;
  }

  if (auto *GO = dyn_cast<GlobalObject>(V)) {
    // If there is a large requested alignment and we can, bump up the alignment
    // of the global.  If the memory we set aside for the global may not be the
    // memory used by the final program then it is impossible for us to reliably
    // enforce the preferred alignment.
    if (!GO->isStrongDefinitionForLinker())
      return Align;

    if (GO->getAlignment() >= PrefAlign)
      return GO->getAlignment();
    // We can only increase the alignment of the global if it has no alignment
    // specified or if it is not assigned a section.  If it is assigned a
    // section, the global could be densely packed with other objects in the
    // section, increasing the alignment could cause padding issues.
    if (!GO->hasSection() || GO->getAlignment() == 0)
      GO->setAlignment(PrefAlign);
    return GO->getAlignment();
  }

  return Align;
}

/// getOrEnforceKnownAlignment - If the specified pointer has an alignment that
/// we can determine, return it, otherwise return 0.  If PrefAlign is specified,
/// and it is more than the alignment of the ultimate object, see if we can
/// increase the alignment of the ultimate object, making this check succeed.
unsigned llvm::getOrEnforceKnownAlignment(Value *V, unsigned PrefAlign,
                                          const DataLayout &DL,
                                          const Instruction *CxtI,
                                          AssumptionCache *AC,
                                          const DominatorTree *DT) {
  assert(V->getType()->isPointerTy() &&
         "getOrEnforceKnownAlignment expects a pointer!");
  unsigned BitWidth = DL.getPointerTypeSizeInBits(V->getType());

  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  computeKnownBits(V, KnownZero, KnownOne, DL, 0, AC, CxtI, DT);
  unsigned TrailZ = KnownZero.countTrailingOnes();

  // Avoid trouble with ridiculously large TrailZ values, such as
  // those computed from a null pointer.
  TrailZ = std::min(TrailZ, unsigned(sizeof(unsigned) * CHAR_BIT - 1));

  unsigned Align = 1u << std::min(BitWidth - 1, TrailZ);

  // LLVM doesn't support alignments larger than this currently.
  Align = std::min(Align, +Value::MaximumAlignment);

  if (PrefAlign > Align)
    Align = enforceKnownAlignment(V, Align, PrefAlign, DL);

  // We don't need to make any adjustment.
  return Align;
}

///===---------------------------------------------------------------------===//
///  Dbg Intrinsic utilities
///

/// See if there is a dbg.value intrinsic for DIVar before I.
static bool LdStHasDebugValue(const DILocalVariable *DIVar, Instruction *I) {
  // Since we can't guarantee that the original dbg.declare instrinsic
  // is removed by LowerDbgDeclare(), we need to make sure that we are
  // not inserting the same dbg.value intrinsic over and over.
  llvm::BasicBlock::InstListType::iterator PrevI(I);
  if (PrevI != I->getParent()->getInstList().begin()) {
    --PrevI;
    if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(PrevI))
      if (DVI->getValue() == I->getOperand(0) &&
          DVI->getOffset() == 0 &&
          DVI->getVariable() == DIVar)
        return true;
  }
  return false;
}

/// Inserts a llvm.dbg.value intrinsic before a store to an alloca'd value
/// that has an associated llvm.dbg.decl intrinsic.
bool llvm::ConvertDebugDeclareToDebugValue(DbgDeclareInst *DDI,
                                           StoreInst *SI, DIBuilder &Builder) {
  auto *DIVar = DDI->getVariable();
  auto *DIExpr = DDI->getExpression();
  assert(DIVar && "Missing variable");

  if (LdStHasDebugValue(DIVar, SI))
    return true;

  // If an argument is zero extended then use argument directly. The ZExt
  // may be zapped by an optimization pass in future.
  Argument *ExtendedArg = nullptr;
  if (ZExtInst *ZExt = dyn_cast<ZExtInst>(SI->getOperand(0)))
    ExtendedArg = dyn_cast<Argument>(ZExt->getOperand(0));
  if (SExtInst *SExt = dyn_cast<SExtInst>(SI->getOperand(0)))
    ExtendedArg = dyn_cast<Argument>(SExt->getOperand(0));
  if (ExtendedArg) {
    // We're now only describing a subset of the variable. The piece we're
    // describing will always be smaller than the variable size, because
    // VariableSize == Size of Alloca described by DDI. Since SI stores
    // to the alloca described by DDI, if it's first operand is an extend,
    // we're guaranteed that before extension, the value was narrower than
    // the size of the alloca, hence the size of the described variable.
    SmallVector<uint64_t, 3> NewDIExpr;
    unsigned PieceOffset = 0;
    // If this already is a bit piece, we drop the bit piece from the expression
    // and record the offset.
    if (DIExpr->isBitPiece()) {
      NewDIExpr.append(DIExpr->elements_begin(), DIExpr->elements_end()-3);
      PieceOffset = DIExpr->getBitPieceOffset();
    } else {
      NewDIExpr.append(DIExpr->elements_begin(), DIExpr->elements_end());
    }
    NewDIExpr.push_back(dwarf::DW_OP_bit_piece);
    NewDIExpr.push_back(PieceOffset); //Offset
    const DataLayout &DL = DDI->getModule()->getDataLayout();
    NewDIExpr.push_back(DL.getTypeSizeInBits(ExtendedArg->getType())); // Size
    Builder.insertDbgValueIntrinsic(ExtendedArg, 0, DIVar,
                                    Builder.createExpression(NewDIExpr),
                                    DDI->getDebugLoc(), SI);
  }
  else
    Builder.insertDbgValueIntrinsic(SI->getOperand(0), 0, DIVar, DIExpr,
                                    DDI->getDebugLoc(), SI);
  return true;
}

/// Inserts a llvm.dbg.value intrinsic before a load of an alloca'd value
/// that has an associated llvm.dbg.decl intrinsic.
bool llvm::ConvertDebugDeclareToDebugValue(DbgDeclareInst *DDI,
                                           LoadInst *LI, DIBuilder &Builder) {
  auto *DIVar = DDI->getVariable();
  auto *DIExpr = DDI->getExpression();
  assert(DIVar && "Missing variable");

  if (LdStHasDebugValue(DIVar, LI))
    return true;

  // We are now tracking the loaded value instead of the address. In the
  // future if multi-location support is added to the IR, it might be
  // preferable to keep tracking both the loaded value and the original
  // address in case the alloca can not be elided.
  Instruction *DbgValue = Builder.insertDbgValueIntrinsic(
      LI, 0, DIVar, DIExpr, DDI->getDebugLoc(), (Instruction *)nullptr);
  DbgValue->insertAfter(LI);
  return true;
}

/// Determine whether this alloca is either a VLA or an array.
static bool isArray(AllocaInst *AI) {
  return AI->isArrayAllocation() ||
    AI->getType()->getElementType()->isArrayTy();
}

/// LowerDbgDeclare - Lowers llvm.dbg.declare intrinsics into appropriate set
/// of llvm.dbg.value intrinsics.
bool llvm::LowerDbgDeclare(Function &F) {
  DIBuilder DIB(*F.getParent(), /*AllowUnresolved*/ false);
  SmallVector<DbgDeclareInst *, 4> Dbgs;
  for (auto &FI : F)
    for (Instruction &BI : FI)
      if (auto DDI = dyn_cast<DbgDeclareInst>(&BI))
        Dbgs.push_back(DDI);

  if (Dbgs.empty())
    return false;

  for (auto &I : Dbgs) {
    DbgDeclareInst *DDI = I;
    AllocaInst *AI = dyn_cast_or_null<AllocaInst>(DDI->getAddress());
    // If this is an alloca for a scalar variable, insert a dbg.value
    // at each load and store to the alloca and erase the dbg.declare.
    // The dbg.values allow tracking a variable even if it is not
    // stored on the stack, while the dbg.declare can only describe
    // the stack slot (and at a lexical-scope granularity). Later
    // passes will attempt to elide the stack slot.
    if (AI && !isArray(AI)) {
      for (auto &AIUse : AI->uses()) {
        User *U = AIUse.getUser();
        if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
          if (AIUse.getOperandNo() == 1)
            ConvertDebugDeclareToDebugValue(DDI, SI, DIB);
        } else if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
          ConvertDebugDeclareToDebugValue(DDI, LI, DIB);
        } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
          // This is a call by-value or some other instruction that
          // takes a pointer to the variable. Insert a *value*
          // intrinsic that describes the alloca.
          SmallVector<uint64_t, 1> NewDIExpr;
          auto *DIExpr = DDI->getExpression();
          NewDIExpr.push_back(dwarf::DW_OP_deref);
          NewDIExpr.append(DIExpr->elements_begin(), DIExpr->elements_end());
          DIB.insertDbgValueIntrinsic(AI, 0, DDI->getVariable(),
                                      DIB.createExpression(NewDIExpr),
                                      DDI->getDebugLoc(), CI);
        }
      }
      DDI->eraseFromParent();
    }
  }
  return true;
}

/// FindAllocaDbgDeclare - Finds the llvm.dbg.declare intrinsic describing the
/// alloca 'V', if any.
DbgDeclareInst *llvm::FindAllocaDbgDeclare(Value *V) {
  if (auto *L = LocalAsMetadata::getIfExists(V))
    if (auto *MDV = MetadataAsValue::getIfExists(V->getContext(), L))
      for (User *U : MDV->users())
        if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(U))
          return DDI;

  return nullptr;
}

bool llvm::replaceDbgDeclare(Value *Address, Value *NewAddress,
                             Instruction *InsertBefore, DIBuilder &Builder,
                             bool Deref, int Offset) {
  DbgDeclareInst *DDI = FindAllocaDbgDeclare(Address);
  if (!DDI)
    return false;
  DebugLoc Loc = DDI->getDebugLoc();
  auto *DIVar = DDI->getVariable();
  auto *DIExpr = DDI->getExpression();
  assert(DIVar && "Missing variable");

  if (Deref || Offset) {
    // Create a copy of the original DIDescriptor for user variable, prepending
    // "deref" operation to a list of address elements, as new llvm.dbg.declare
    // will take a value storing address of the memory for variable, not
    // alloca itself.
    SmallVector<uint64_t, 4> NewDIExpr;
    if (Deref)
      NewDIExpr.push_back(dwarf::DW_OP_deref);
    if (Offset > 0) {
      NewDIExpr.push_back(dwarf::DW_OP_plus);
      NewDIExpr.push_back(Offset);
    } else if (Offset < 0) {
      NewDIExpr.push_back(dwarf::DW_OP_minus);
      NewDIExpr.push_back(-Offset);
    }
    if (DIExpr)
      NewDIExpr.append(DIExpr->elements_begin(), DIExpr->elements_end());
    DIExpr = Builder.createExpression(NewDIExpr);
  }

  // Insert llvm.dbg.declare immediately after the original alloca, and remove
  // old llvm.dbg.declare.
  Builder.insertDeclare(NewAddress, DIVar, DIExpr, Loc, InsertBefore);
  DDI->eraseFromParent();
  return true;
}

bool llvm::replaceDbgDeclareForAlloca(AllocaInst *AI, Value *NewAllocaAddress,
                                      DIBuilder &Builder, bool Deref, int Offset) {
  return replaceDbgDeclare(AI, NewAllocaAddress, AI->getNextNode(), Builder,
                           Deref, Offset);
}

void llvm::changeToUnreachable(Instruction *I, bool UseLLVMTrap) {
  BasicBlock *BB = I->getParent();
  // Loop over all of the successors, removing BB's entry from any PHI
  // nodes.
  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    (*SI)->removePredecessor(BB);

  // Insert a call to llvm.trap right before this.  This turns the undefined
  // behavior into a hard fail instead of falling through into random code.
  if (UseLLVMTrap) {
    Function *TrapFn =
      Intrinsic::getDeclaration(BB->getParent()->getParent(), Intrinsic::trap);
    CallInst *CallTrap = CallInst::Create(TrapFn, "", I);
    CallTrap->setDebugLoc(I->getDebugLoc());
  }
  new UnreachableInst(I->getContext(), I);

  // All instructions after this are dead.
  BasicBlock::iterator BBI = I->getIterator(), BBE = BB->end();
  while (BBI != BBE) {
    if (!BBI->use_empty())
      BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
    BB->getInstList().erase(BBI++);
  }
}

/// changeToCall - Convert the specified invoke into a normal call.
static void changeToCall(InvokeInst *II) {
  SmallVector<Value*, 8> Args(II->arg_begin(), II->arg_end());
  SmallVector<OperandBundleDef, 1> OpBundles;
  II->getOperandBundlesAsDefs(OpBundles);
  CallInst *NewCall = CallInst::Create(II->getCalledValue(), Args, OpBundles,
                                       "", II);
  NewCall->takeName(II);
  NewCall->setCallingConv(II->getCallingConv());
  NewCall->setAttributes(II->getAttributes());
  NewCall->setDebugLoc(II->getDebugLoc());
  II->replaceAllUsesWith(NewCall);

  // Follow the call by a branch to the normal destination.
  BranchInst::Create(II->getNormalDest(), II);

  // Update PHI nodes in the unwind destination
  II->getUnwindDest()->removePredecessor(II->getParent());
  II->eraseFromParent();
}

static bool markAliveBlocks(Function &F,
                            SmallPtrSetImpl<BasicBlock*> &Reachable) {

  SmallVector<BasicBlock*, 128> Worklist;
  BasicBlock *BB = &F.front();
  Worklist.push_back(BB);
  Reachable.insert(BB);
  bool Changed = false;
  do {
    BB = Worklist.pop_back_val();

    // Do a quick scan of the basic block, turning any obviously unreachable
    // instructions into LLVM unreachable insts.  The instruction combining pass
    // canonicalizes unreachable insts into stores to null or undef.
    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E;++BBI){
      // Assumptions that are known to be false are equivalent to unreachable.
      // Also, if the condition is undefined, then we make the choice most
      // beneficial to the optimizer, and choose that to also be unreachable.
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(BBI))
        if (II->getIntrinsicID() == Intrinsic::assume) {
          bool MakeUnreachable = false;
          if (isa<UndefValue>(II->getArgOperand(0)))
            MakeUnreachable = true;
          else if (ConstantInt *Cond =
                   dyn_cast<ConstantInt>(II->getArgOperand(0)))
            MakeUnreachable = Cond->isZero();

          if (MakeUnreachable) {
            // Don't insert a call to llvm.trap right before the unreachable.
            changeToUnreachable(&*BBI, false);
            Changed = true;
            break;
          }
        }

      if (CallInst *CI = dyn_cast<CallInst>(BBI)) {
        if (CI->doesNotReturn()) {
          // If we found a call to a no-return function, insert an unreachable
          // instruction after it.  Make sure there isn't *already* one there
          // though.
          ++BBI;
          if (!isa<UnreachableInst>(BBI)) {
            // Don't insert a call to llvm.trap right before the unreachable.
            changeToUnreachable(&*BBI, false);
            Changed = true;
          }
          break;
        }
      }

      // Store to undef and store to null are undefined and used to signal that
      // they should be changed to unreachable by passes that can't modify the
      // CFG.
      if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
        // Don't touch volatile stores.
        if (SI->isVolatile()) continue;

        Value *Ptr = SI->getOperand(1);

        if (isa<UndefValue>(Ptr) ||
            (isa<ConstantPointerNull>(Ptr) &&
             SI->getPointerAddressSpace() == 0)) {
          changeToUnreachable(SI, true);
          Changed = true;
          break;
        }
      }
    }

    TerminatorInst *Terminator = BB->getTerminator();
    if (auto *II = dyn_cast<InvokeInst>(Terminator)) {
      // Turn invokes that call 'nounwind' functions into ordinary calls.
      Value *Callee = II->getCalledValue();
      if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
        changeToUnreachable(II, true);
        Changed = true;
      } else if (II->doesNotThrow() && canSimplifyInvokeNoUnwind(&F)) {
        if (II->use_empty() && II->onlyReadsMemory()) {
          // jump to the normal destination branch.
          BranchInst::Create(II->getNormalDest(), II);
          II->getUnwindDest()->removePredecessor(II->getParent());
          II->eraseFromParent();
        } else
          changeToCall(II);
        Changed = true;
      }
    } else if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(Terminator)) {
      // Remove catchpads which cannot be reached.
      struct CatchPadDenseMapInfo {
        static CatchPadInst *getEmptyKey() {
          return DenseMapInfo<CatchPadInst *>::getEmptyKey();
        }
        static CatchPadInst *getTombstoneKey() {
          return DenseMapInfo<CatchPadInst *>::getTombstoneKey();
        }
        static unsigned getHashValue(CatchPadInst *CatchPad) {
          return static_cast<unsigned>(hash_combine_range(
              CatchPad->value_op_begin(), CatchPad->value_op_end()));
        }
        static bool isEqual(CatchPadInst *LHS, CatchPadInst *RHS) {
          if (LHS == getEmptyKey() || LHS == getTombstoneKey() ||
              RHS == getEmptyKey() || RHS == getTombstoneKey())
            return LHS == RHS;
          return LHS->isIdenticalTo(RHS);
        }
      };

      // Set of unique CatchPads.
      SmallDenseMap<CatchPadInst *, detail::DenseSetEmpty, 4,
                    CatchPadDenseMapInfo, detail::DenseSetPair<CatchPadInst *>>
          HandlerSet;
      detail::DenseSetEmpty Empty;
      for (CatchSwitchInst::handler_iterator I = CatchSwitch->handler_begin(),
                                             E = CatchSwitch->handler_end();
           I != E; ++I) {
        BasicBlock *HandlerBB = *I;
        auto *CatchPad = cast<CatchPadInst>(HandlerBB->getFirstNonPHI());
        if (!HandlerSet.insert({CatchPad, Empty}).second) {
          CatchSwitch->removeHandler(I);
          --I;
          --E;
          Changed = true;
        }
      }
    }

    Changed |= ConstantFoldTerminator(BB, true);
    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
      if (Reachable.insert(*SI).second)
        Worklist.push_back(*SI);
  } while (!Worklist.empty());
  return Changed;
}

void llvm::removeUnwindEdge(BasicBlock *BB) {
  TerminatorInst *TI = BB->getTerminator();

  if (auto *II = dyn_cast<InvokeInst>(TI)) {
    changeToCall(II);
    return;
  }

  TerminatorInst *NewTI;
  BasicBlock *UnwindDest;

  if (auto *CRI = dyn_cast<CleanupReturnInst>(TI)) {
    NewTI = CleanupReturnInst::Create(CRI->getCleanupPad(), nullptr, CRI);
    UnwindDest = CRI->getUnwindDest();
  } else if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(TI)) {
    auto *NewCatchSwitch = CatchSwitchInst::Create(
        CatchSwitch->getParentPad(), nullptr, CatchSwitch->getNumHandlers(),
        CatchSwitch->getName(), CatchSwitch);
    for (BasicBlock *PadBB : CatchSwitch->handlers())
      NewCatchSwitch->addHandler(PadBB);

    NewTI = NewCatchSwitch;
    UnwindDest = CatchSwitch->getUnwindDest();
  } else {
    llvm_unreachable("Could not find unwind successor");
  }

  NewTI->takeName(TI);
  NewTI->setDebugLoc(TI->getDebugLoc());
  UnwindDest->removePredecessor(BB);
  TI->replaceAllUsesWith(NewTI);
  TI->eraseFromParent();
}

/// removeUnreachableBlocksFromFn - Remove blocks that are not reachable, even
/// if they are in a dead cycle.  Return true if a change was made, false
/// otherwise.
bool llvm::removeUnreachableBlocks(Function &F, LazyValueInfo *LVI) {
  SmallPtrSet<BasicBlock*, 128> Reachable;
  bool Changed = markAliveBlocks(F, Reachable);

  // If there are unreachable blocks in the CFG...
  if (Reachable.size() == F.size())
    return Changed;

  assert(Reachable.size() < F.size());
  NumRemoved += F.size()-Reachable.size();

  // Loop over all of the basic blocks that are not reachable, dropping all of
  // their internal references...
  for (Function::iterator BB = ++F.begin(), E = F.end(); BB != E; ++BB) {
    if (Reachable.count(&*BB))
      continue;

    for (succ_iterator SI = succ_begin(&*BB), SE = succ_end(&*BB); SI != SE;
         ++SI)
      if (Reachable.count(*SI))
        (*SI)->removePredecessor(&*BB);
    if (LVI)
      LVI->eraseBlock(&*BB);
    BB->dropAllReferences();
  }

  for (Function::iterator I = ++F.begin(); I != F.end();)
    if (!Reachable.count(&*I))
      I = F.getBasicBlockList().erase(I);
    else
      ++I;

  return true;
}

void llvm::combineMetadata(Instruction *K, const Instruction *J,
                           ArrayRef<unsigned> KnownIDs) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  K->dropUnknownNonDebugMetadata(KnownIDs);
  K->getAllMetadataOtherThanDebugLoc(Metadata);
  for (unsigned i = 0, n = Metadata.size(); i < n; ++i) {
    unsigned Kind = Metadata[i].first;
    MDNode *JMD = J->getMetadata(Kind);
    MDNode *KMD = Metadata[i].second;

    switch (Kind) {
      default:
        K->setMetadata(Kind, nullptr); // Remove unknown metadata
        break;
      case LLVMContext::MD_dbg:
        llvm_unreachable("getAllMetadataOtherThanDebugLoc returned a MD_dbg");
      case LLVMContext::MD_tbaa:
        K->setMetadata(Kind, MDNode::getMostGenericTBAA(JMD, KMD));
        break;
      case LLVMContext::MD_alias_scope:
        K->setMetadata(Kind, MDNode::getMostGenericAliasScope(JMD, KMD));
        break;
      case LLVMContext::MD_noalias:
        K->setMetadata(Kind, MDNode::intersect(JMD, KMD));
        break;
      case LLVMContext::MD_range:
        K->setMetadata(Kind, MDNode::getMostGenericRange(JMD, KMD));
        break;
      case LLVMContext::MD_fpmath:
        K->setMetadata(Kind, MDNode::getMostGenericFPMath(JMD, KMD));
        break;
      case LLVMContext::MD_invariant_load:
        // Only set the !invariant.load if it is present in both instructions.
        K->setMetadata(Kind, JMD);
        break;
      case LLVMContext::MD_nonnull:
        // Only set the !nonnull if it is present in both instructions.
        K->setMetadata(Kind, JMD);
        break;
      case LLVMContext::MD_invariant_group:
        // Preserve !invariant.group in K.
        break;
      case LLVMContext::MD_align:
        K->setMetadata(Kind, 
          MDNode::getMostGenericAlignmentOrDereferenceable(JMD, KMD));
        break;
      case LLVMContext::MD_dereferenceable:
      case LLVMContext::MD_dereferenceable_or_null:
        K->setMetadata(Kind, 
          MDNode::getMostGenericAlignmentOrDereferenceable(JMD, KMD));
        break;
    }
  }
  // Set !invariant.group from J if J has it. If both instructions have it
  // then we will just pick it from J - even when they are different.
  // Also make sure that K is load or store - f.e. combining bitcast with load
  // could produce bitcast with invariant.group metadata, which is invalid.
  // FIXME: we should try to preserve both invariant.group md if they are
  // different, but right now instruction can only have one invariant.group.
  if (auto *JMD = J->getMetadata(LLVMContext::MD_invariant_group))
    if (isa<LoadInst>(K) || isa<StoreInst>(K))
      K->setMetadata(LLVMContext::MD_invariant_group, JMD);
}

unsigned llvm::replaceDominatedUsesWith(Value *From, Value *To,
                                        DominatorTree &DT,
                                        const BasicBlockEdge &Root) {
  assert(From->getType() == To->getType());
  
  unsigned Count = 0;
  for (Value::use_iterator UI = From->use_begin(), UE = From->use_end();
       UI != UE; ) {
    Use &U = *UI++;
    if (DT.dominates(Root, U)) {
      U.set(To);
      DEBUG(dbgs() << "Replace dominated use of '"
            << From->getName() << "' as "
            << *To << " in " << *U << "\n");
      ++Count;
    }
  }
  return Count;
}

unsigned llvm::replaceDominatedUsesWith(Value *From, Value *To,
                                        DominatorTree &DT,
                                        const BasicBlock *BB) {
  assert(From->getType() == To->getType());

  unsigned Count = 0;
  for (Value::use_iterator UI = From->use_begin(), UE = From->use_end();
       UI != UE;) {
    Use &U = *UI++;
    auto *I = cast<Instruction>(U.getUser());
    if (DT.dominates(BB, I->getParent())) {
      U.set(To);
      DEBUG(dbgs() << "Replace dominated use of '" << From->getName() << "' as "
                   << *To << " in " << *U << "\n");
      ++Count;
    }
  }
  return Count;
}

bool llvm::callsGCLeafFunction(ImmutableCallSite CS) {
  if (isa<IntrinsicInst>(CS.getInstruction()))
    // Most LLVM intrinsics are things which can never take a safepoint.
    // As a result, we don't need to have the stack parsable at the
    // callsite.  This is a highly useful optimization since intrinsic
    // calls are fairly prevalent, particularly in debug builds.
    return true;

  // Check if the function is specifically marked as a gc leaf function.
  if (CS.hasFnAttr("gc-leaf-function"))
    return true;
  if (const Function *F = CS.getCalledFunction())
    return F->hasFnAttribute("gc-leaf-function");

  return false;
}

/// A potential constituent of a bitreverse or bswap expression. See
/// collectBitParts for a fuller explanation.
struct BitPart {
  BitPart(Value *P, unsigned BW) : Provider(P) {
    Provenance.resize(BW);
  }

  /// The Value that this is a bitreverse/bswap of.
  Value *Provider;
  /// The "provenance" of each bit. Provenance[A] = B means that bit A
  /// in Provider becomes bit B in the result of this expression.
  SmallVector<int8_t, 32> Provenance; // int8_t means max size is i128.

  enum { Unset = -1 };
};

/// Analyze the specified subexpression and see if it is capable of providing
/// pieces of a bswap or bitreverse. The subexpression provides a potential
/// piece of a bswap or bitreverse if it can be proven that each non-zero bit in
/// the output of the expression came from a corresponding bit in some other
/// value. This function is recursive, and the end result is a mapping of
/// bitnumber to bitnumber. It is the caller's responsibility to validate that
/// the bitnumber to bitnumber mapping is correct for a bswap or bitreverse.
///
/// For example, if the current subexpression if "(shl i32 %X, 24)" then we know
/// that the expression deposits the low byte of %X into the high byte of the
/// result and that all other bits are zero. This expression is accepted and a
/// BitPart is returned with Provider set to %X and Provenance[24-31] set to
/// [0-7].
///
/// To avoid revisiting values, the BitPart results are memoized into the
/// provided map. To avoid unnecessary copying of BitParts, BitParts are
/// constructed in-place in the \c BPS map. Because of this \c BPS needs to
/// store BitParts objects, not pointers. As we need the concept of a nullptr
/// BitParts (Value has been analyzed and the analysis failed), we an Optional
/// type instead to provide the same functionality.
///
/// Because we pass around references into \c BPS, we must use a container that
/// does not invalidate internal references (std::map instead of DenseMap).
///
static const Optional<BitPart> &
collectBitParts(Value *V, bool MatchBSwaps, bool MatchBitReversals,
                std::map<Value *, Optional<BitPart>> &BPS) {
  auto I = BPS.find(V);
  if (I != BPS.end())
    return I->second;

  auto &Result = BPS[V] = None;
  auto BitWidth = cast<IntegerType>(V->getType())->getBitWidth();

  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an or instruction, it may be an inner node of the bswap.
    if (I->getOpcode() == Instruction::Or) {
      auto &A = collectBitParts(I->getOperand(0), MatchBSwaps,
                                MatchBitReversals, BPS);
      auto &B = collectBitParts(I->getOperand(1), MatchBSwaps,
                                MatchBitReversals, BPS);
      if (!A || !B)
        return Result;

      // Try and merge the two together.
      if (!A->Provider || A->Provider != B->Provider)
        return Result;

      Result = BitPart(A->Provider, BitWidth);
      for (unsigned i = 0; i < A->Provenance.size(); ++i) {
        if (A->Provenance[i] != BitPart::Unset &&
            B->Provenance[i] != BitPart::Unset &&
            A->Provenance[i] != B->Provenance[i])
          return Result = None;

        if (A->Provenance[i] == BitPart::Unset)
          Result->Provenance[i] = B->Provenance[i];
        else
          Result->Provenance[i] = A->Provenance[i];
      }

      return Result;
    }

    // If this is a logical shift by a constant, recurse then shift the result.
    if (I->isLogicalShift() && isa<ConstantInt>(I->getOperand(1))) {
      unsigned BitShift =
          cast<ConstantInt>(I->getOperand(1))->getLimitedValue(~0U);
      // Ensure the shift amount is defined.
      if (BitShift > BitWidth)
        return Result;

      auto &Res = collectBitParts(I->getOperand(0), MatchBSwaps,
                                  MatchBitReversals, BPS);
      if (!Res)
        return Result;
      Result = Res;

      // Perform the "shift" on BitProvenance.
      auto &P = Result->Provenance;
      if (I->getOpcode() == Instruction::Shl) {
        P.erase(std::prev(P.end(), BitShift), P.end());
        P.insert(P.begin(), BitShift, BitPart::Unset);
      } else {
        P.erase(P.begin(), std::next(P.begin(), BitShift));
        P.insert(P.end(), BitShift, BitPart::Unset);
      }

      return Result;
    }

    // If this is a logical 'and' with a mask that clears bits, recurse then
    // unset the appropriate bits.
    if (I->getOpcode() == Instruction::And &&
        isa<ConstantInt>(I->getOperand(1))) {
      APInt Bit(I->getType()->getPrimitiveSizeInBits(), 1);
      const APInt &AndMask = cast<ConstantInt>(I->getOperand(1))->getValue();

      // Check that the mask allows a multiple of 8 bits for a bswap, for an
      // early exit.
      unsigned NumMaskedBits = AndMask.countPopulation();
      if (!MatchBitReversals && NumMaskedBits % 8 != 0)
        return Result;
      
      auto &Res = collectBitParts(I->getOperand(0), MatchBSwaps,
                                  MatchBitReversals, BPS);
      if (!Res)
        return Result;
      Result = Res;

      for (unsigned i = 0; i < BitWidth; ++i, Bit <<= 1)
        // If the AndMask is zero for this bit, clear the bit.
        if ((AndMask & Bit) == 0)
          Result->Provenance[i] = BitPart::Unset;

      return Result;
    }
  }

  // Okay, we got to something that isn't a shift, 'or' or 'and'.  This must be
  // the input value to the bswap/bitreverse.
  Result = BitPart(V, BitWidth);
  for (unsigned i = 0; i < BitWidth; ++i)
    Result->Provenance[i] = i;
  return Result;
}

static bool bitTransformIsCorrectForBSwap(unsigned From, unsigned To,
                                          unsigned BitWidth) {
  if (From % 8 != To % 8)
    return false;
  // Convert from bit indices to byte indices and check for a byte reversal.
  From >>= 3;
  To >>= 3;
  BitWidth >>= 3;
  return From == BitWidth - To - 1;
}

static bool bitTransformIsCorrectForBitReverse(unsigned From, unsigned To,
                                               unsigned BitWidth) {
  return From == BitWidth - To - 1;
}

/// Given an OR instruction, check to see if this is a bitreverse
/// idiom. If so, insert the new intrinsic and return true.
bool llvm::recognizeBitReverseOrBSwapIdiom(
    Instruction *I, bool MatchBSwaps, bool MatchBitReversals,
    SmallVectorImpl<Instruction *> &InsertedInsts) {
  if (Operator::getOpcode(I) != Instruction::Or)
    return false;
  if (!MatchBSwaps && !MatchBitReversals)
    return false;
  IntegerType *ITy = dyn_cast<IntegerType>(I->getType());
  if (!ITy || ITy->getBitWidth() > 128)
    return false;   // Can't do vectors or integers > 128 bits.
  unsigned BW = ITy->getBitWidth();

  // Try to find all the pieces corresponding to the bswap.
  std::map<Value *, Optional<BitPart>> BPS;
  auto Res = collectBitParts(I, MatchBSwaps, MatchBitReversals, BPS);
  if (!Res)
    return false;
  auto &BitProvenance = Res->Provenance;

  // Now, is the bit permutation correct for a bswap or a bitreverse? We can
  // only byteswap values with an even number of bytes.
  bool OKForBSwap = BW % 16 == 0, OKForBitReverse = true;
  for (unsigned i = 0; i < BW; ++i) {
    OKForBSwap &= bitTransformIsCorrectForBSwap(BitProvenance[i], i, BW);
    OKForBitReverse &=
        bitTransformIsCorrectForBitReverse(BitProvenance[i], i, BW);
  }

  Intrinsic::ID Intrin;
  if (OKForBSwap && MatchBSwaps)
    Intrin = Intrinsic::bswap;
  else if (OKForBitReverse && MatchBitReversals)
    Intrin = Intrinsic::bitreverse;
  else
    return false;

  Function *F = Intrinsic::getDeclaration(I->getModule(), Intrin, ITy);
  InsertedInsts.push_back(CallInst::Create(F, Res->Provider, "rev", I));
  return true;
}
