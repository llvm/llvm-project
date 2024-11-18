//===- SpillPlacement.h - Optimal Spill Code Placement ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis computes the optimal spill code placement between basic blocks.
//
// The runOnMachineFunction() method only precomputes some profiling information
// about the CFG. The real work is done by prepare(), addConstraints(), and
// finish() which are called by the register allocator.
//
// Given a variable that is live across multiple basic blocks, and given
// constraints on the basic blocks where the variable is live, determine which
// edge bundles should have the variable in a register and which edge bundles
// should have the variable in a stack slot.
//
// The returned bit vector can be used to place optimal spill code at basic
// block entries and exits. Spill code placement inside a basic block is not
// considered.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SPILLPLACEMENT_H
#define LLVM_LIB_CODEGEN_SPILLPLACEMENT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/BlockFrequency.h"

namespace llvm {

class BitVector;
class EdgeBundles;
class MachineBlockFrequencyInfo;
class MachineFunction;
class SpillPlacementWrapperLegacy;
class SpillPlacementAnalysis;

class SpillPlacement {
  friend class SpillPlacementWrapperLegacy;
  friend class SpillPlacementAnalysis;

  struct Node;

  const MachineFunction *MF = nullptr;
  const EdgeBundles *bundles = nullptr;
  const MachineBlockFrequencyInfo *MBFI = nullptr;

  static void arrayDeleter(Node *N);
  std::unique_ptr<Node, decltype(&arrayDeleter)> nodes;

  // Nodes that are active in the current computation. Owned by the prepare()
  // caller.
  BitVector *ActiveNodes = nullptr;

  // Nodes with active links. Populated by scanActiveBundles.
  SmallVector<unsigned, 8> Linked;

  // Nodes that went positive during the last call to scanActiveBundles or
  // iterate.
  SmallVector<unsigned, 8> RecentPositive;

  // Block frequencies are computed once. Indexed by block number.
  SmallVector<BlockFrequency, 8> BlockFrequencies;

  /// Decision threshold. A node gets the output value 0 if the weighted sum of
  /// its inputs falls in the open interval (-Threshold;Threshold).
  BlockFrequency Threshold;

  /// List of nodes that need to be updated in ::iterate.
  SparseSet<unsigned> TodoList;

public:
  /// BorderConstraint - A basic block has separate constraints for entry and
  /// exit.
  enum BorderConstraint {
    DontCare,  ///< Block doesn't care / variable not live.
    PrefReg,   ///< Block entry/exit prefers a register.
    PrefSpill, ///< Block entry/exit prefers a stack slot.
    PrefBoth,  ///< Block entry prefers both register and stack.
    MustSpill  ///< A register is impossible, variable must be spilled.
  };

  /// BlockConstraint - Entry and exit constraints for a basic block.
  struct BlockConstraint {
    unsigned Number;            ///< Basic block number (from MBB::getNumber()).
    BorderConstraint Entry : 8; ///< Constraint on block entry.
    BorderConstraint Exit : 8;  ///< Constraint on block exit.

    /// True when this block changes the value of the live range. This means
    /// the block has a non-PHI def.  When this is false, a live-in value on
    /// the stack can be live-out on the stack without inserting a spill.
    bool ChangesValue;

    void print(raw_ostream &OS) const;
    void dump() const;
  };

  /// prepare - Reset state and prepare for a new spill placement computation.
  /// @param RegBundles Bit vector to receive the edge bundles where the
  ///                   variable should be kept in a register. Each bit
  ///                   corresponds to an edge bundle, a set bit means the
  ///                   variable should be kept in a register through the
  ///                   bundle. A clear bit means the variable should be
  ///                   spilled. This vector is retained.
  void prepare(BitVector &RegBundles);

  /// addConstraints - Add constraints and biases. This method may be called
  /// more than once to accumulate constraints.
  /// @param LiveBlocks Constraints for blocks that have the variable live in or
  ///                   live out.
  void addConstraints(ArrayRef<BlockConstraint> LiveBlocks);

  /// addPrefSpill - Add PrefSpill constraints to all blocks listed.  This is
  /// equivalent to calling addConstraint with identical BlockConstraints with
  /// Entry = Exit = PrefSpill, and ChangesValue = false.
  ///
  /// @param Blocks Array of block numbers that prefer to spill in and out.
  /// @param Strong When true, double the negative bias for these blocks.
  void addPrefSpill(ArrayRef<unsigned> Blocks, bool Strong);

  /// addLinks - Add transparent blocks with the given numbers.
  void addLinks(ArrayRef<unsigned> Links);

  /// scanActiveBundles - Perform an initial scan of all bundles activated by
  /// addConstraints and addLinks, updating their state. Add all the bundles
  /// that now prefer a register to RecentPositive.
  /// Prepare internal data structures for iterate.
  /// Return true is there are any positive nodes.
  bool scanActiveBundles();

  /// iterate - Update the network iteratively until convergence, or new bundles
  /// are found.
  void iterate();

  /// getRecentPositive - Return an array of bundles that became positive during
  /// the previous call to scanActiveBundles or iterate.
  ArrayRef<unsigned> getRecentPositive() { return RecentPositive; }

  /// finish - Compute the optimal spill code placement given the
  /// constraints. No MustSpill constraints will be violated, and the smallest
  /// possible number of PrefX constraints will be violated, weighted by
  /// expected execution frequencies.
  /// The selected bundles are returned in the bitvector passed to prepare().
  /// @return True if a perfect solution was found, allowing the variable to be
  ///         in a register through all relevant bundles.
  bool finish();

  /// getBlockFrequency - Return the estimated block execution frequency per
  /// function invocation.
  BlockFrequency getBlockFrequency(unsigned Number) const {
    return BlockFrequencies[Number];
  }

  bool invalidate(MachineFunction &MF, const PreservedAnalyses &PA,
                  MachineFunctionAnalysisManager::Invalidator &Inv);

private:
  SpillPlacement(EdgeBundles *Bundles, MachineBlockFrequencyInfo *MBFI)
      : bundles(Bundles), MBFI(MBFI), nodes(nullptr, &arrayDeleter) {}

  void run(MachineFunction &MF);
  void activate(unsigned n);
  void setThreshold(BlockFrequency Entry);

  bool update(unsigned n);
};

class SpillPlacementWrapperLegacy : public MachineFunctionPass {
public:
  static char ID;
  SpillPlacementWrapperLegacy() : MachineFunctionPass(ID) {}

  SpillPlacement &getResult() { return *Impl; }
  const SpillPlacement &getResult() const { return *Impl; }

private:
  std::unique_ptr<SpillPlacement> Impl;
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override { Impl.reset(); }
};

class SpillPlacementAnalysis
    : public AnalysisInfoMixin<SpillPlacementAnalysis> {
  friend AnalysisInfoMixin<SpillPlacementAnalysis>;
  static AnalysisKey Key;

public:
  using Result = SpillPlacement;
  SpillPlacement run(MachineFunction &, MachineFunctionAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_SPILLPLACEMENT_H
