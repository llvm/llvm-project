//===-- ImplicitControlFlowTracking.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This class allows to keep track on instructions with implicit control flow.
// These are instructions that may not pass execution to their successors. For
// example, throwing calls and guards do not always do this. If we need to know
// for sure that some instruction is guaranteed to execute if the given block
// is reached, then we need to make sure that there is no implicit control flow
// instruction (ICFI) preceeding it. For example, this check is required if we
// perform PRE moving non-speculable instruction to other place.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_IMPLICITCONTROLFLOWTRACKING_H
#define LLVM_TRANSFORMS_UTILS_IMPLICITCONTROLFLOWTRACKING_H

#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/OrderedInstructions.h"

namespace llvm {

class ImplicitControlFlowTracking {
public:
  ImplicitControlFlowTracking(DominatorTree *DT)
      : OI(OrderedInstructions(DT)) {}

  // Returns the topmost instruction with implicit control flow from the given
  // basic block. Returns nullptr if there is no such instructions in the block.
  const Instruction *getFirstICFI(const BasicBlock *BB);

  // Returns true if at least one instruction from the given basic block has
  // implicit control flow.
  bool hasICF(const BasicBlock *BB);

  // Returns true if the first ICFI of Insn's block exists and dominates Insn.
  bool isDominatedByICFIFromSameBlock(const Instruction *Insn);

  // Clears information about this particular block.
  void invalidateBlock(const BasicBlock *BB);

  // Invalidates all information from this tracking.
  void clear();

private:
  // Fills information about the given block's implicit control flow.
  void fill(const BasicBlock *BB);

  // Maps a block to the topmost instruction with implicit control flow in it.
  DenseMap<const BasicBlock *, const Instruction *>
      FirstImplicitControlFlowInsts;
  OrderedInstructions OI;
  // Blocks for which we have the actual information.
  SmallPtrSet<const BasicBlock *, 8> KnownBlocks;
};

} // llvm

#endif // LLVM_TRANSFORMS_UTILS_IMPLICITCONTROLFLOWTRACKING_H
