//===-- MemorySSAUpdater.cpp - Memory SSA Updater--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the MemorySSAUpdater class.
//
//===----------------------------------------------------------------===//
#include "llvm/Transforms/Utils/MemorySSAUpdater.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include <algorithm>

#define DEBUG_TYPE "memoryssa"
using namespace llvm;
namespace llvm {
// This is the marker algorithm from "Simple and Efficient Construction of
// Static Single Assignment Form"
// The simple, non-marker algorithm places phi nodes at any join
// Here, we place markers, and only place phi nodes if they end up necessary.
// They are only necessary if they break a cycle (IE we recursively visit
// ourselves again), or we discover, while getting the value of the operands,
// that there are two or more definitions needing to be merged.
// This still will leave non-minimal form in the case of irreducible control
// flow, where phi nodes may be in cycles with themselves, but unnecessary.
MemoryAccess *MemorySSAUpdater::getPreviousDefRecursive(BasicBlock *BB) {
  // Single predecessor case, just recurse, we can only have one definition.
  if (BasicBlock *Pred = BB->getSinglePredecessor()) {
    return getPreviousDefFromEnd(Pred);
  } else if (VisitedBlocks.count(BB)) {
    // We hit our node again, meaning we had a cycle, we must insert a phi
    // node to break it so we have an operand. The only case this will
    // insert useless phis is if we have irreducible control flow.
    return MSSA->createMemoryPhi(BB);
  } else if (VisitedBlocks.insert(BB).second) {
    // Mark us visited so we can detect a cycle
    SmallVector<MemoryAccess *, 8> PhiOps;

    // Recurse to get the values in our predecessors for placement of a
    // potential phi node. This will insert phi nodes if we cycle in order to
    // break the cycle and have an operand.
    for (auto *Pred : predecessors(BB))
      PhiOps.push_back(getPreviousDefFromEnd(Pred));

    // Now try to simplify the ops to avoid placing a phi.
    // This may return null if we never created a phi yet, that's okay
    MemoryPhi *Phi = dyn_cast_or_null<MemoryPhi>(MSSA->getMemoryAccess(BB));
    bool PHIExistsButNeedsUpdate = false;
    // See if the existing phi operands match what we need.
    // Unlike normal SSA, we only allow one phi node per block, so we can't just
    // create a new one.
    if (Phi && Phi->getNumOperands() != 0)
      if (!std::equal(Phi->op_begin(), Phi->op_end(), PhiOps.begin())) {
        PHIExistsButNeedsUpdate = true;
      }

    // See if we can avoid the phi by simplifying it.
    auto *Result = tryRemoveTrivialPhi(Phi, PhiOps);
    // If we couldn't simplify, we may have to create a phi
    if (Result == Phi) {
      if (!Phi)
        Phi = MSSA->createMemoryPhi(BB);

      // These will have been filled in by the recursive read we did above.
      if (PHIExistsButNeedsUpdate) {
        std::copy(PhiOps.begin(), PhiOps.end(), Phi->op_begin());
        std::copy(pred_begin(BB), pred_end(BB), Phi->block_begin());
      } else {
        unsigned i = 0;
        for (auto *Pred : predecessors(BB))
          Phi->addIncoming(PhiOps[i++], Pred);
      }

      Result = Phi;
    }
    if (MemoryPhi *MP = dyn_cast<MemoryPhi>(Result))
      InsertedPHIs.push_back(MP);
    // Set ourselves up for the next variable by resetting visited state.
    VisitedBlocks.erase(BB);
    return Result;
  }
  llvm_unreachable("Should have hit one of the three cases above");
}

// This starts at the memory access, and goes backwards in the block to find the
// previous definition. If a definition is not found the block of the access,
// it continues globally, creating phi nodes to ensure we have a single
// definition.
MemoryAccess *MemorySSAUpdater::getPreviousDef(MemoryAccess *MA) {
  auto *LocalResult = getPreviousDefInBlock(MA);

  return LocalResult ? LocalResult : getPreviousDefRecursive(MA->getBlock());
}

// This starts at the memory access, and goes backwards in the block to the find
// the previous definition. If the definition is not found in the block of the
// access, it returns nullptr.
MemoryAccess *MemorySSAUpdater::getPreviousDefInBlock(MemoryAccess *MA) {
  auto *Defs = MSSA->getWritableBlockDefs(MA->getBlock());

  // It's possible there are no defs, or we got handed the first def to start.
  if (Defs) {
    // If this is a def, we can just use the def iterators.
    if (!isa<MemoryUse>(MA)) {
      auto Iter = MA->getReverseDefsIterator();
      ++Iter;
      if (Iter != Defs->rend())
        return &*Iter;
    } else {
      // Otherwise, have to walk the all access iterator.
      auto Iter = MA->getReverseIterator();
      ++Iter;
      while (&*Iter != &*Defs->begin()) {
        if (!isa<MemoryUse>(*Iter))
          return &*Iter;
        --Iter;
      }
      // At this point it must be pointing at firstdef
      assert(&*Iter == &*Defs->begin() &&
             "Should have hit first def walking backwards");
      return &*Iter;
    }
  }
  return nullptr;
}

// This starts at the end of block
MemoryAccess *MemorySSAUpdater::getPreviousDefFromEnd(BasicBlock *BB) {
  auto *Defs = MSSA->getWritableBlockDefs(BB);

  if (Defs)
    return &*Defs->rbegin();

  return getPreviousDefRecursive(BB);
}
// Recurse over a set of phi uses to eliminate the trivial ones
MemoryAccess *MemorySSAUpdater::recursePhi(MemoryAccess *Phi) {
  if (!Phi)
    return nullptr;
  TrackingVH<MemoryAccess> Res(Phi);
  SmallVector<TrackingVH<Value>, 8> Uses;
  std::copy(Phi->user_begin(), Phi->user_end(), std::back_inserter(Uses));
  for (auto &U : Uses) {
    if (MemoryPhi *UsePhi = dyn_cast<MemoryPhi>(&*U)) {
      auto OperRange = UsePhi->operands();
      tryRemoveTrivialPhi(UsePhi, OperRange);
    }
  }
  return Res;
}

// Eliminate trivial phis
// Phis are trivial if they are defined either by themselves, or all the same
// argument.
// IE phi(a, a) or b = phi(a, b) or c = phi(a, a, c)
// We recursively try to remove them.
template <class RangeType>
MemoryAccess *MemorySSAUpdater::tryRemoveTrivialPhi(MemoryPhi *Phi,
                                                    RangeType &Operands) {
  // Detect equal or self arguments
  MemoryAccess *Same = nullptr;
  for (auto &Op : Operands) {
    // If the same or self, good so far
    if (Op == Phi || Op == Same)
      continue;
    // not the same, return the phi since it's not eliminatable by us
    if (Same)
      return Phi;
    Same = cast<MemoryAccess>(Op);
  }
  // Never found a non-self reference, the phi is undef
  if (Same == nullptr)
    return MSSA->getLiveOnEntryDef();
  if (Phi) {
    Phi->replaceAllUsesWith(Same);
    MSSA->removeMemoryAccess(Phi);
  }

  // We should only end up recursing in case we replaced something, in which
  // case, we may have made other Phis trivial.
  return recursePhi(Same);
}

void MemorySSAUpdater::insertUse(MemoryUse *MU) {
  InsertedPHIs.clear();
  MU->setDefiningAccess(getPreviousDef(MU));
  // Unlike for defs, there is no extra work to do.  Because uses do not create
  // new may-defs, there are only two cases:
  //
  // 1. There was a def already below us, and therefore, we should not have
  // created a phi node because it was already needed for the def.
  //
  // 2. There is no def below us, and therefore, there is no extra renaming work
  // to do.
}

// Set every incoming edge {BB, MP->getBlock()} of MemoryPhi MP to NewDef.
void setMemoryPhiValueForBlock(MemoryPhi *MP, const BasicBlock *BB,
                               MemoryAccess *NewDef) {
  // Replace any operand with us an incoming block with the new defining
  // access.
  int i = MP->getBasicBlockIndex(BB);
  assert(i != -1 && "Should have found the basic block in the phi");
  // We can't just compare i against getNumOperands since one is signed and the
  // other not. So use it to index into the block iterator.
  for (auto BBIter = MP->block_begin() + i; BBIter != MP->block_end();
       ++BBIter) {
    if (*BBIter != BB)
      break;
    MP->setIncomingValue(i, NewDef);
    ++i;
  }
}

// A brief description of the algorithm:
// First, we compute what should define the new def, using the SSA
// construction algorithm.
// Then, we update the defs below us (and any new phi nodes) in the graph to
// point to the correct new defs, to ensure we only have one variable, and no
// disconnected stores.
void MemorySSAUpdater::insertDef(MemoryDef *MD, bool RenameUses) {
  InsertedPHIs.clear();

  // See if we had a local def, and if not, go hunting.
  MemoryAccess *DefBefore = getPreviousDefInBlock(MD);
  bool DefBeforeSameBlock = DefBefore != nullptr;
  if (!DefBefore)
    DefBefore = getPreviousDefRecursive(MD->getBlock());

  // There is a def before us, which means we can replace any store/phi uses
  // of that thing with us, since we are in the way of whatever was there
  // before.
  // We now define that def's memorydefs and memoryphis
  if (DefBeforeSameBlock) {
    for (auto UI = DefBefore->use_begin(), UE = DefBefore->use_end();
         UI != UE;) {
      Use &U = *UI++;
      // Leave the uses alone
      if (isa<MemoryUse>(U.getUser()))
        continue;
      U.set(MD);
    }
  }

  // and that def is now our defining access.
  // We change them in this order otherwise we will appear in the use list
  // above and reset ourselves.
  MD->setDefiningAccess(DefBefore);

  SmallVector<MemoryAccess *, 8> FixupList(InsertedPHIs.begin(),
                                           InsertedPHIs.end());
  if (!DefBeforeSameBlock) {
    // If there was a local def before us, we must have the same effect it
    // did. Because every may-def is the same, any phis/etc we would create, it
    // would also have created.  If there was no local def before us, we
    // performed a global update, and have to search all successors and make
    // sure we update the first def in each of them (following all paths until
    // we hit the first def along each path). This may also insert phi nodes.
    // TODO: There are other cases we can skip this work, such as when we have a
    // single successor, and only used a straight line of single pred blocks
    // backwards to find the def.  To make that work, we'd have to track whether
    // getDefRecursive only ever used the single predecessor case.  These types
    // of paths also only exist in between CFG simplifications.
    FixupList.push_back(MD);
  }

  while (!FixupList.empty()) {
    unsigned StartingPHISize = InsertedPHIs.size();
    fixupDefs(FixupList);
    FixupList.clear();
    // Put any new phis on the fixup list, and process them
    FixupList.append(InsertedPHIs.end() - StartingPHISize, InsertedPHIs.end());
  }
  // Now that all fixups are done, rename all uses if we are asked.
  if (RenameUses) {
    SmallPtrSet<BasicBlock *, 16> Visited;
    BasicBlock *StartBlock = MD->getBlock();
    // We are guaranteed there is a def in the block, because we just got it
    // handed to us in this function.
    MemoryAccess *FirstDef = &*MSSA->getWritableBlockDefs(StartBlock)->begin();
    // Convert to incoming value if it's a memorydef. A phi *is* already an
    // incoming value.
    if (auto *MD = dyn_cast<MemoryDef>(FirstDef))
      FirstDef = MD->getDefiningAccess();

    MSSA->renamePass(MD->getBlock(), FirstDef, Visited);
    // We just inserted a phi into this block, so the incoming value will become
    // the phi anyway, so it does not matter what we pass.
    for (auto *MP : InsertedPHIs)
      MSSA->renamePass(MP->getBlock(), nullptr, Visited);
  }
}

void MemorySSAUpdater::fixupDefs(const SmallVectorImpl<MemoryAccess *> &Vars) {
  SmallPtrSet<const BasicBlock *, 8> Seen;
  SmallVector<const BasicBlock *, 16> Worklist;
  for (auto *NewDef : Vars) {
    // First, see if there is a local def after the operand.
    auto *Defs = MSSA->getWritableBlockDefs(NewDef->getBlock());
    auto DefIter = NewDef->getDefsIterator();

    // If there is a local def after us, we only have to rename that.
    if (++DefIter != Defs->end()) {
      cast<MemoryDef>(DefIter)->setDefiningAccess(NewDef);
      continue;
    }

    // Otherwise, we need to search down through the CFG.
    // For each of our successors, handle it directly if their is a phi, or
    // place on the fixup worklist.
    for (const auto *S : successors(NewDef->getBlock())) {
      if (auto *MP = MSSA->getMemoryAccess(S))
        setMemoryPhiValueForBlock(MP, NewDef->getBlock(), NewDef);
      else
        Worklist.push_back(S);
    }

    while (!Worklist.empty()) {
      const BasicBlock *FixupBlock = Worklist.back();
      Worklist.pop_back();

      // Get the first def in the block that isn't a phi node.
      if (auto *Defs = MSSA->getWritableBlockDefs(FixupBlock)) {
        auto *FirstDef = &*Defs->begin();
        // The loop above and below should have taken care of phi nodes
        assert(!isa<MemoryPhi>(FirstDef) &&
               "Should have already handled phi nodes!");
        // We are now this def's defining access, make sure we actually dominate
        // it
        assert(MSSA->dominates(NewDef, FirstDef) &&
               "Should have dominated the new access");

        // This may insert new phi nodes, because we are not guaranteed the
        // block we are processing has a single pred, and depending where the
        // store was inserted, it may require phi nodes below it.
        cast<MemoryDef>(FirstDef)->setDefiningAccess(getPreviousDef(FirstDef));
        return;
      }
      // We didn't find a def, so we must continue.
      for (const auto *S : successors(FixupBlock)) {
        // If there is a phi node, handle it.
        // Otherwise, put the block on the worklist
        if (auto *MP = MSSA->getMemoryAccess(S))
          setMemoryPhiValueForBlock(MP, FixupBlock, NewDef);
        else {
          // If we cycle, we should have ended up at a phi node that we already
          // processed.  FIXME: Double check this
          if (!Seen.insert(S).second)
            continue;
          Worklist.push_back(S);
        }
      }
    }
  }
}

// Move What before Where in the MemorySSA IR.
template <class WhereType>
void MemorySSAUpdater::moveTo(MemoryUseOrDef *What, BasicBlock *BB,
                              WhereType Where) {
  // Replace all our users with our defining access.
  What->replaceAllUsesWith(What->getDefiningAccess());

  // Let MemorySSA take care of moving it around in the lists.
  MSSA->moveTo(What, BB, Where);

  // Now reinsert it into the IR and do whatever fixups needed.
  if (auto *MD = dyn_cast<MemoryDef>(What))
    insertDef(MD);
  else
    insertUse(cast<MemoryUse>(What));
}

// Move What before Where in the MemorySSA IR.
void MemorySSAUpdater::moveBefore(MemoryUseOrDef *What, MemoryUseOrDef *Where) {
  moveTo(What, Where->getBlock(), Where->getIterator());
}

// Move What after Where in the MemorySSA IR.
void MemorySSAUpdater::moveAfter(MemoryUseOrDef *What, MemoryUseOrDef *Where) {
  moveTo(What, Where->getBlock(), ++Where->getIterator());
}

void MemorySSAUpdater::moveToPlace(MemoryUseOrDef *What, BasicBlock *BB,
                                   MemorySSA::InsertionPlace Where) {
  return moveTo(What, BB, Where);
}
} // namespace llvm
