//===- llvm/Transforms/Utils/OrderedInstructions.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an efficient way to check for dominance relation between 2
// instructions.
//
// FIXME: This is really just a convenience wrapper to check dominance between
// two arbitrary instructions in different basic blocks. We should fold it into
// DominatorTree, which is the more widely used interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ORDEREDINSTRUCTIONS_H
#define LLVM_ANALYSIS_ORDEREDINSTRUCTIONS_H

namespace llvm {

class DominatorTree;
class Instruction;

class OrderedInstructions {
  /// The dominator tree of the parent function.
  DominatorTree *DT;

  /// Return true if the first instruction comes before the second in the
  /// same basic block. It will create an ordered basic block, if it does
  /// not yet exist in OBBMap.
  bool localDominates(const Instruction *, const Instruction *) const;

public:
  /// Constructor.
  OrderedInstructions(DominatorTree *DT) : DT(DT) {}

  /// Return true if first instruction dominates the second.
  bool dominates(const Instruction *, const Instruction *) const;

  /// Return true if the first instruction comes before the second in the
  /// dominator tree DFS traversal if they are in different basic blocks,
  /// or if the first instruction comes before the second in the same basic
  /// block.
  bool dfsBefore(const Instruction *, const Instruction *) const;

  // Return true if the first instruction comes before the second in the
  // dominator tree BFS traversal based on the level number of nodes in
  // dominator tree if they are in different basic blocks else if the first
  // instruction comes before the second in the same basic block.
  bool domTreeLevelBefore(const Instruction *, const Instruction *) const;
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_ORDEREDINSTRUCTIONS_H
