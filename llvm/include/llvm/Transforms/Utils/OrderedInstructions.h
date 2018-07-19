//===- llvm/Transforms/Utils/OrderedInstructions.h -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an efficient way to check for dominance relation between 2
// instructions.
//
// This interface dispatches to appropriate dominance check given 2
// instructions, i.e. in case the instructions are in the same basic block,
// OrderedBasicBlock (with instruction numbering and caching) are used.
// Otherwise, dominator tree is used.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ORDEREDINSTRUCTIONS_H
#define LLVM_TRANSFORMS_UTILS_ORDEREDINSTRUCTIONS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/OrderedBasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Operator.h"

namespace llvm {

class OrderedInstructions {
  /// Used to check dominance for instructions in same basic block.
  mutable DenseMap<const BasicBlock *, std::unique_ptr<OrderedBasicBlock>>
      OBBMap;

  /// The dominator tree of the parent function.
  DominatorTree *DT;

public:
  /// Constructor.
  OrderedInstructions(DominatorTree *DT) : DT(DT) {}

  /// Return true if first instruction dominates the second.
  bool dominates(const Instruction *, const Instruction *) const;

  /// Invalidate the OrderedBasicBlock cache when its basic block changes.
  /// i.e. If an instruction is deleted or added to the basic block, the user
  /// should call this function to invalidate the OrderedBasicBlock cache for
  /// this basic block.
  void invalidateBlock(const BasicBlock *BB) { OBBMap.erase(BB); }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_ORDEREDINSTRUCTIONS_H
