//===- MemorySSAUpdater.h - Memory SSA Updater-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// An automatic updater for MemorySSA that handles arbitrary insertion,
// deletion, and moves.  It performs phi insertion where necessary, and
// automatically updates the MemorySSA IR to be correct.
// While updating loads or removing instructions is often easy enough to not
// need this, updating stores should generally not be attemped outside this
// API.
//
// Basic API usage:
// Create the memory access you want for the instruction (this is mainly so
// we know where it is, without having to duplicate the entire set of create
// functions MemorySSA supports).
// Call insertDef or insertUse depending on whether it's a MemoryUse or a
// MemoryDef.
// That's it.
//
// For moving, first, move the instruction itself using the normal SSA
// instruction moving API, then just call moveBefore, moveAfter,or moveTo with
// the right arguments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYSSAUPDATER_H
#define LLVM_ANALYSIS_MEMORYSSAUPDATER_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class Function;
class Instruction;
class MemoryAccess;
class LLVMContext;
class raw_ostream;

class MemorySSAUpdater {
private:
  MemorySSA *MSSA;

  /// We use WeakVH rather than a costly deletion to deal with dangling pointers.
  /// MemoryPhis are created eagerly and sometimes get zapped shortly afterwards.
  SmallVector<WeakVH, 16> InsertedPHIs;

  SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
  SmallSet<AssertingVH<MemoryPhi>, 8> NonOptPhis;

public:
  MemorySSAUpdater(MemorySSA *MSSA) : MSSA(MSSA) {}
  /// Insert a definition into the MemorySSA IR.  RenameUses will rename any use
  /// below the new def block (and any inserted phis).  RenameUses should be set
  /// to true if the definition may cause new aliases for loads below it.  This
  /// is not the case for hoisting or sinking or other forms of code *movement*.
  /// It *is* the case for straight code insertion.
  /// For example:
  /// store a
  /// if (foo) { }
  /// load a
  ///
  /// Moving the store into the if block, and calling insertDef, does not
  /// require RenameUses.
  /// However, changing it to:
  /// store a
  /// if (foo) { store b }
  /// load a
  /// Where a mayalias b, *does* require RenameUses be set to true.
  void insertDef(MemoryDef *Def, bool RenameUses = false);
  void insertUse(MemoryUse *Use);
  void moveBefore(MemoryUseOrDef *What, MemoryUseOrDef *Where);
  void moveAfter(MemoryUseOrDef *What, MemoryUseOrDef *Where);
  void moveToPlace(MemoryUseOrDef *What, BasicBlock *BB,
                   MemorySSA::InsertionPlace Where);
  /// `From` block was spliced into `From` and `To`.
  /// Move all accesses from `From` to `To` starting at instruction `Start`.
  /// `To` is newly created BB, so empty of MemorySSA::MemoryAccesses.
  /// Edges are already updated, so successors of `To` with MPhi nodes need to
  /// update incoming block.
  /// |------|        |------|
  /// | From |        | From |
  /// |      |        |------|
  /// |      |           ||
  /// |      |   =>      \/
  /// |      |        |------|  <- Start
  /// |      |        |  To  |
  /// |------|        |------|
  void moveAllAfterSpliceBlocks(BasicBlock *From, BasicBlock *To,
                                Instruction *Start);
  /// `From` block was merged into `To`. All instructions were moved and
  /// `From` is an empty block with successor edges; `From` is about to be
  /// deleted. Move all accesses from `From` to `To` starting at instruction
  /// `Start`. `To` may have multiple successors, `From` has a single
  /// predecessor. `From` may have successors with MPhi nodes, replace their
  /// incoming block with `To`.
  /// |------|        |------|
  /// |  To  |        |  To  |
  /// |------|        |      |
  ///    ||      =>   |      |
  ///    \/           |      |
  /// |------|        |      |  <- Start
  /// | From |        |      |
  /// |------|        |------|
  void moveAllAfterMergeBlocks(BasicBlock *From, BasicBlock *To,
                               Instruction *Start);
  /// BasicBlock Old had New, an empty BasicBlock, added directly before it,
  /// and the predecessors in Preds that used to point to Old, now point to
  /// New. If New is the only predecessor, move Old's Phi, if present, to New.
  /// Otherwise, add a new Phi in New with appropriate incoming values, and
  /// update the incoming values in Old's Phi node too, if present.
  void
  wireOldPredecessorsToNewImmediatePredecessor(BasicBlock *Old, BasicBlock *New,
                                               ArrayRef<BasicBlock *> Preds);

  // The below are utility functions. Other than creation of accesses to pass
  // to insertDef, and removeAccess to remove accesses, you should generally
  // not attempt to update memoryssa yourself. It is very non-trivial to get
  // the edge cases right, and the above calls already operate in near-optimal
  // time bounds.

  /// Create a MemoryAccess in MemorySSA at a specified point in a block,
  /// with a specified clobbering definition.
  ///
  /// Returns the new MemoryAccess.
  /// This should be called when a memory instruction is created that is being
  /// used to replace an existing memory instruction. It will *not* create PHI
  /// nodes, or verify the clobbering definition. The insertion place is used
  /// solely to determine where in the memoryssa access lists the instruction
  /// will be placed. The caller is expected to keep ordering the same as
  /// instructions.
  /// It will return the new MemoryAccess.
  /// Note: If a MemoryAccess already exists for I, this function will make it
  /// inaccessible and it *must* have removeMemoryAccess called on it.
  MemoryAccess *createMemoryAccessInBB(Instruction *I, MemoryAccess *Definition,
                                       const BasicBlock *BB,
                                       MemorySSA::InsertionPlace Point);

  /// Create a MemoryAccess in MemorySSA before or after an existing
  /// MemoryAccess.
  ///
  /// Returns the new MemoryAccess.
  /// This should be called when a memory instruction is created that is being
  /// used to replace an existing memory instruction. It will *not* create PHI
  /// nodes, or verify the clobbering definition.
  ///
  /// Note: If a MemoryAccess already exists for I, this function will make it
  /// inaccessible and it *must* have removeMemoryAccess called on it.
  MemoryUseOrDef *createMemoryAccessBefore(Instruction *I,
                                           MemoryAccess *Definition,
                                           MemoryUseOrDef *InsertPt);
  MemoryUseOrDef *createMemoryAccessAfter(Instruction *I,
                                          MemoryAccess *Definition,
                                          MemoryAccess *InsertPt);

  /// Remove a MemoryAccess from MemorySSA, including updating all
  /// definitions and uses.
  /// This should be called when a memory instruction that has a MemoryAccess
  /// associated with it is erased from the program.  For example, if a store or
  /// load is simply erased (not replaced), removeMemoryAccess should be called
  /// on the MemoryAccess for that store/load.
  void removeMemoryAccess(MemoryAccess *);

  /// Remove MemoryAccess for a given instruction, if a MemoryAccess exists.
  /// This should be called when an instruction (load/store) is deleted from
  /// the program.
  void removeMemoryAccess(const Instruction *I) {
    if (MemoryAccess *MA = MSSA->getMemoryAccess(I))
      removeMemoryAccess(MA);
  }

  /// Remove all MemoryAcceses in a set of BasicBlocks about to be deleted.
  /// Assumption we make here: all uses of deleted defs and phi must either
  /// occur in blocks about to be deleted (thus will be deleted as well), or
  /// they occur in phis that will simply lose an incoming value.
  /// Deleted blocks still have successor info, but their predecessor edges and
  /// Phi nodes may already be updated. Instructions in DeadBlocks should be
  /// deleted after this call.
  void removeBlocks(const SmallPtrSetImpl<BasicBlock *> &DeadBlocks);

  /// Get handle on MemorySSA.
  MemorySSA* getMemorySSA() const { return MSSA; }

private:
  // Move What before Where in the MemorySSA IR.
  template <class WhereType>
  void moveTo(MemoryUseOrDef *What, BasicBlock *BB, WhereType Where);
  // Move all memory accesses from `From` to `To` starting at `Start`.
  // Restrictions apply, see public wrappers of this method.
  void moveAllAccesses(BasicBlock *From, BasicBlock *To, Instruction *Start);
  MemoryAccess *getPreviousDef(MemoryAccess *);
  MemoryAccess *getPreviousDefInBlock(MemoryAccess *);
  MemoryAccess *
  getPreviousDefFromEnd(BasicBlock *,
                        DenseMap<BasicBlock *, TrackingVH<MemoryAccess>> &);
  MemoryAccess *
  getPreviousDefRecursive(BasicBlock *,
                          DenseMap<BasicBlock *, TrackingVH<MemoryAccess>> &);
  MemoryAccess *recursePhi(MemoryAccess *Phi);
  template <class RangeType>
  MemoryAccess *tryRemoveTrivialPhi(MemoryPhi *Phi, RangeType &Operands);
  void fixupDefs(const SmallVectorImpl<WeakVH> &);
};
} // end namespace llvm

#endif // LLVM_ANALYSIS_MEMORYSSAUPDATER_H
