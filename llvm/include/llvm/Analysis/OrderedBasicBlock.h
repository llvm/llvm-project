//===- llvm/Analysis/OrderedBasicBlock.h --------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OrderedBasicBlock class. OrderedBasicBlock maintains
// an interface where clients can query if one instruction comes before another
// in a BasicBlock. Since BasicBlock currently lacks a reliable way to query
// relative position between instructions one can use OrderedBasicBlock to do
// such queries. OrderedBasicBlock is lazily built on a source BasicBlock and
// maintains an internal Instruction -> Position map. A OrderedBasicBlock
// instance should be discarded whenever the source BasicBlock changes.
//
// It's currently used by the CaptureTracker in order to find relative
// positions of a pair of instructions inside a BasicBlock.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ORDEREDBASICBLOCK_H
#define LLVM_ANALYSIS_ORDEREDBASICBLOCK_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"

namespace llvm {

class Instruction;
class BasicBlock;

class OrderedBasicBlock {
private:
  /// Map a instruction to its position in a BasicBlock.
  SmallDenseMap<const Instruction *, unsigned, 32> NumberedInsts;

  /// Keep track of last instruction inserted into \p NumberedInsts.
  /// It speeds up queries for uncached instructions by providing a start point
  /// for new queries in OrderedBasicBlock::comesBefore.
  BasicBlock::const_iterator LastInstFound;

  /// The position/number to tag the next instruction to be found.
  unsigned NextInstPos;

  /// The source BasicBlock to map.
  const BasicBlock *BB;

  /// Given no cached results, find if \p A comes before \p B in \p BB.
  /// Cache and number out instruction while walking \p BB.
  bool comesBefore(const Instruction *A, const Instruction *B);

public:
  OrderedBasicBlock(const BasicBlock *BasicB);

  /// Find out whether \p A dominates \p B, meaning whether \p A
  /// comes before \p B in \p BB. This is a simplification that considers
  /// cached instruction positions and ignores other basic blocks, being
  /// only relevant to compare relative instructions positions inside \p BB.
  /// Returns false for A == B.
  bool dominates(const Instruction *A, const Instruction *B);

  /// Remove \p from the ordering, if it is present.
  void eraseInstruction(const Instruction *I);

  /// Replace \p Old with \p New in the ordering. \p New is assigned the
  /// numbering of \p Old, so it must be inserted at the same position in the
  /// IR.
  void replaceInstruction(const Instruction *Old, const Instruction *New);
};

} // End llvm namespace

#endif
