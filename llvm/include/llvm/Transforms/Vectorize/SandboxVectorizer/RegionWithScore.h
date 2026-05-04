//===- RegionWithScore.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A region with score tracking for added/removed instructions.
//

#ifndef LLVM_SANDBOXIR_REGIONWITHSCORE_H
#define LLVM_SANDBOXIR_REGIONWITHSCORE_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

/// Vectorization Score (cost) tracking class.
class ScoreBoard {
  const Region &Rgn;
  const TargetTransformInfo &TTI;
  constexpr static TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  /// The cost of all instructions added to the region.
  InstructionCost AfterCost = 0;
  /// The cost of all instructions that got removed and replaced by new ones.
  InstructionCost BeforeCost = 0;
  /// Helper for both add() and remove(). \Returns the TTI cost of \p I.
  LLVM_ABI InstructionCost getCost(Instruction *I) const;
  /// No need to allow copies.
  ScoreBoard(const ScoreBoard &) = delete;
  const ScoreBoard &operator=(const ScoreBoard &) = delete;

public:
  ScoreBoard(Region &Rgn, const TargetTransformInfo &TTI)
      : Rgn(Rgn), TTI(TTI) {}
  /// Mark \p I as a newly added instruction to the region.
  void add(Instruction *I) { AfterCost += getCost(I); }
  /// Mark \p I as a deleted instruction from the region.
  LLVM_ABI void remove(Instruction *I);
  /// \Returns the cost of the newly added instructions.
  InstructionCost getAfterCost() const { return AfterCost; }
  /// \Returns the cost of the Removed instructions.
  InstructionCost getBeforeCost() const { return BeforeCost; }

#ifndef NDEBUG
  void dump(raw_ostream &OS) const {
    OS << "BeforeCost: " << BeforeCost << "\n";
    OS << "AfterCost:  " << AfterCost << "\n";
  }
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

/// A Region class that tracks its instructions score.
class RegionWithScore final : public Region {
  /// Keeps track of cost of instructions added and removed.
  ScoreBoard Scoreboard;

  void add(Instruction *I) override {
    addRaw(I);
    // Keep track of the instruction cost.
    Scoreboard.add(I);
  }
  friend class RegionsFromBBs; // For add().

  void remove(Instruction *I) override {
    // Keep track of the instruction cost. This need to be done *before* we
    // remove `I` from the region.
    Scoreboard.remove(I);
    Region::remove(I);
  }

public:
  RegionWithScore(Context &Ctx, const TargetTransformInfo &TTI)
      : Region(Ctx, RegionClassID::RegionWithScoreID), Scoreboard(*this, TTI) {}
  RegionWithScore(Region &&Rgn, const TargetTransformInfo &TTI)
      : Region(std::move(Rgn)), Scoreboard(*this, TTI) {}
  // For isa<> cast<> etc.
  static bool classof(const Region *From) {
    return From->getSubclassID() == RegionClassID::RegionWithScoreID;
  }

  /// \Returns the ScoreBoard data structure that keeps track of instr costs.
  const ScoreBoard &getScoreboard() const { return Scoreboard; }

  LLVM_ABI static SmallVector<std::unique_ptr<RegionWithScore>>
  createRegionsFromMD(Function &F, const TargetTransformInfo &TTI) {
    auto Rgns = Region::createRegionsFromMD(F);
    SmallVector<std::unique_ptr<RegionWithScore>> NewRgns;
    NewRgns.reserve(Rgns.size());
    for (auto &RgnPtr : Rgns)
      NewRgns.push_back(
          std::make_unique<RegionWithScore>(std::move(*RgnPtr.release()), TTI));
    return NewRgns;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_REGIONWITHSCORE_H
