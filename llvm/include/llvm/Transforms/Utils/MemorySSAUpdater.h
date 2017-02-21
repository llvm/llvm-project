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
// \brief An automatic updater for MemorySSA that handles arbitrary insertion,
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

#ifndef LLVM_TRANSFORMS_UTILS_MEMORYSSAUPDATER_H
#define LLVM_TRANSFORMS_UTILS_MEMORYSSAUPDATER_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/MemorySSA.h"

namespace llvm {

class Function;
class Instruction;
class MemoryAccess;
class LLVMContext;
class raw_ostream;

class MemorySSAUpdater {
private:
  MemorySSA *MSSA;
  SmallVector<MemoryPhi *, 8> InsertedPHIs;
  SmallPtrSet<BasicBlock *, 8> VisitedBlocks;

public:
  MemorySSAUpdater(MemorySSA *MSSA) : MSSA(MSSA) {}
  // Insert a definition into the MemorySSA IR.  RenameUses will rename any use
  // below the new def block (and any inserted phis).  RenameUses should be set
  // to true if the definition may cause new aliases for loads below it.  This
  // is not the case for hoisting or sinking or other forms of code *movement*.
  // It *is* the case for straight code insertion.
  // For example:
  // store a
  // if (foo) { }
  // load a
  //
  // Moving the store into the if block, and calling insertDef, does not
  // require RenameUses.
  // However, changing it to:
  // store a
  // if (foo) { store b }
  // load a
  // Where a mayalias b, *does* require RenameUses be set to true.
  void insertDef(MemoryDef *Def, bool RenameUses = false);
  void insertUse(MemoryUse *Use);
  void moveBefore(MemoryUseOrDef *What, MemoryUseOrDef *Where);
  void moveAfter(MemoryUseOrDef *What, MemoryUseOrDef *Where);
  void moveToPlace(MemoryUseOrDef *What, BasicBlock *BB,
                   MemorySSA::InsertionPlace Where);

private:
  // Move What before Where in the MemorySSA IR.
  template <class WhereType>
  void moveTo(MemoryUseOrDef *What, BasicBlock *BB,
              WhereType Where);
  MemoryAccess *getPreviousDef(MemoryAccess *);
  MemoryAccess *getPreviousDefInBlock(MemoryAccess *);
  MemoryAccess *getPreviousDefFromEnd(BasicBlock *);
  MemoryAccess *getPreviousDefRecursive(BasicBlock *);
  MemoryAccess *recursePhi(MemoryAccess *Phi);
  template <class RangeType>
  MemoryAccess *tryRemoveTrivialPhi(MemoryPhi *Phi, RangeType &Operands);
  void fixupDefs(const SmallVectorImpl<MemoryAccess *> &);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MEMORYSSAUPDATER_H
