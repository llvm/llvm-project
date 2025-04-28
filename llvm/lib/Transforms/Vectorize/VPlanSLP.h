//===- VPlan.h - VPlan-based SLP ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the declarations for VPlan-based SLP.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANSLP_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANSLP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/VectorUtils.h"

namespace llvm {

class VPBasicBlock;
class VPBlockBase;
class VPRegionBlock;
class VPlan;
class VPValue;
class VPInstruction;

class VPInterleavedAccessInfo {
  DenseMap<VPInstruction *, InterleaveGroup<VPInstruction> *>
      InterleaveGroupMap;

  /// Type for mapping of instruction based interleave groups to VPInstruction
  /// interleave groups
  using Old2NewTy = DenseMap<InterleaveGroup<Instruction> *,
                             InterleaveGroup<VPInstruction> *>;

  /// Recursively \p Region and populate VPlan based interleave groups based on
  /// \p IAI.
  void visitRegion(VPRegionBlock *Region, Old2NewTy &Old2New,
                   InterleavedAccessInfo &IAI);
  /// Recursively traverse \p Block and populate VPlan based interleave groups
  /// based on \p IAI.
  void visitBlock(VPBlockBase *Block, Old2NewTy &Old2New,
                  InterleavedAccessInfo &IAI);

public:
  VPInterleavedAccessInfo(VPlan &Plan, InterleavedAccessInfo &IAI);

  ~VPInterleavedAccessInfo() {
    SmallPtrSet<InterleaveGroup<VPInstruction> *, 4> DelSet;
    // Avoid releasing a pointer twice.
    for (auto &I : InterleaveGroupMap)
      DelSet.insert(I.second);
    for (auto *Ptr : DelSet)
      delete Ptr;
  }

  /// Get the interleave group that \p Instr belongs to.
  ///
  /// \returns nullptr if doesn't have such group.
  InterleaveGroup<VPInstruction> *
  getInterleaveGroup(VPInstruction *Instr) const {
    return InterleaveGroupMap.lookup(Instr);
  }
};

/// Class that maps (parts of) an existing VPlan to trees of combined
/// VPInstructions.
class VPlanSlp {
  enum class OpMode { Failed, Load, Opcode };

  /// A DenseMapInfo implementation for using SmallVector<VPValue *, 4> as
  /// DenseMap keys.
  struct BundleDenseMapInfo {
    static SmallVector<VPValue *, 4> getEmptyKey() {
      return {reinterpret_cast<VPValue *>(-1)};
    }

    static SmallVector<VPValue *, 4> getTombstoneKey() {
      return {reinterpret_cast<VPValue *>(-2)};
    }

    static unsigned getHashValue(const SmallVector<VPValue *, 4> &V) {
      return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
    }

    static bool isEqual(const SmallVector<VPValue *, 4> &LHS,
                        const SmallVector<VPValue *, 4> &RHS) {
      return LHS == RHS;
    }
  };

  /// Mapping of values in the original VPlan to a combined VPInstruction.
  DenseMap<SmallVector<VPValue *, 4>, VPInstruction *, BundleDenseMapInfo>
      BundleToCombined;

  VPInterleavedAccessInfo &IAI;

  /// Basic block to operate on. For now, only instructions in a single BB are
  /// considered.
  const VPBasicBlock &BB;

  /// Indicates whether we managed to combine all visited instructions or not.
  bool CompletelySLP = true;

  /// Width of the widest combined bundle in bits.
  unsigned WidestBundleBits = 0;

  using MultiNodeOpTy =
      typename std::pair<VPInstruction *, SmallVector<VPValue *, 4>>;

  // Input operand bundles for the current multi node. Each multi node operand
  // bundle contains values not matching the multi node's opcode. They will
  // be reordered in reorderMultiNodeOps, once we completed building a
  // multi node.
  SmallVector<MultiNodeOpTy, 4> MultiNodeOps;

  /// Indicates whether we are building a multi node currently.
  bool MultiNodeActive = false;

  /// Check if we can vectorize Operands together.
  bool areVectorizable(ArrayRef<VPValue *> Operands) const;

  /// Add combined instruction \p New for the bundle \p Operands.
  void addCombined(ArrayRef<VPValue *> Operands, VPInstruction *New);

  /// Indicate we hit a bundle we failed to combine. Returns nullptr for now.
  VPInstruction *markFailed();

  /// Reorder operands in the multi node to maximize sequential memory access
  /// and commutative operations.
  SmallVector<MultiNodeOpTy, 4> reorderMultiNodeOps();

  /// Choose the best candidate to use for the lane after \p Last. The set of
  /// candidates to choose from are values with an opcode matching \p Last's
  /// or loads consecutive to \p Last.
  std::pair<OpMode, VPValue *> getBest(OpMode Mode, VPValue *Last,
                                       SmallPtrSetImpl<VPValue *> &Candidates,
                                       VPInterleavedAccessInfo &IAI);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print bundle \p Values to dbgs().
  void dumpBundle(ArrayRef<VPValue *> Values);
#endif

public:
  VPlanSlp(VPInterleavedAccessInfo &IAI, VPBasicBlock &BB) : IAI(IAI), BB(BB) {}

  ~VPlanSlp() = default;

  /// Tries to build an SLP tree rooted at \p Operands and returns a
  /// VPInstruction combining \p Operands, if they can be combined.
  VPInstruction *buildGraph(ArrayRef<VPValue *> Operands);

  /// Return the width of the widest combined bundle in bits.
  unsigned getWidestBundleBits() const { return WidestBundleBits; }

  /// Return true if all visited instruction can be combined.
  bool isCompletelySLP() const { return CompletelySLP; }
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_H
