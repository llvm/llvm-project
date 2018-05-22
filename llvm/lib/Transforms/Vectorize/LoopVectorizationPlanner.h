//===- LoopVectorizationPlanner.h - Planner for LoopVectorization ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides a LoopVectorizationPlanner class.
/// InnerLoopVectorizer vectorizes loops which contain only one basic
/// LoopVectorizationPlanner - drives the vectorization process after having
/// passed Legality checks.
/// The planner builds and optimizes the Vectorization Plans which record the
/// decisions how to vectorize the given loop. In particular, represent the
/// control-flow of the vectorized version, the replication of instructions that
/// are to be scalarized, and interleave access groups.
///
/// Also provides a VPlan-based builder utility analogous to IRBuilder.
/// It provides an instruction-level API for generating VPInstructions while
/// abstracting away the Recipe manipulation details.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONPLANNER_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONPLANNER_H

#include "VPlan.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"

namespace llvm {

/// VPlan-based builder utility analogous to IRBuilder.
class VPBuilder {
private:
  VPBasicBlock *BB = nullptr;
  VPBasicBlock::iterator InsertPt = VPBasicBlock::iterator();

  VPInstruction *createInstruction(unsigned Opcode,
                                   ArrayRef<VPValue *> Operands) {
    VPInstruction *Instr = new VPInstruction(Opcode, Operands);
    if (BB)
      BB->insert(Instr, InsertPt);
    return Instr;
  }

  VPInstruction *createInstruction(unsigned Opcode,
                                   std::initializer_list<VPValue *> Operands) {
    return createInstruction(Opcode, ArrayRef<VPValue *>(Operands));
  }

public:
  VPBuilder() {}

  /// Clear the insertion point: created instructions will not be inserted into
  /// a block.
  void clearInsertionPoint() {
    BB = nullptr;
    InsertPt = VPBasicBlock::iterator();
  }

  VPBasicBlock *getInsertBlock() const { return BB; }
  VPBasicBlock::iterator getInsertPoint() const { return InsertPt; }

  /// InsertPoint - A saved insertion point.
  class VPInsertPoint {
    VPBasicBlock *Block = nullptr;
    VPBasicBlock::iterator Point;

  public:
    /// Creates a new insertion point which doesn't point to anything.
    VPInsertPoint() = default;

    /// Creates a new insertion point at the given location.
    VPInsertPoint(VPBasicBlock *InsertBlock, VPBasicBlock::iterator InsertPoint)
        : Block(InsertBlock), Point(InsertPoint) {}

    /// Returns true if this insert point is set.
    bool isSet() const { return Block != nullptr; }

    VPBasicBlock *getBlock() const { return Block; }
    VPBasicBlock::iterator getPoint() const { return Point; }
  };

  /// Sets the current insert point to a previously-saved location.
  void restoreIP(VPInsertPoint IP) {
    if (IP.isSet())
      setInsertPoint(IP.getBlock(), IP.getPoint());
    else
      clearInsertionPoint();
  }

  /// This specifies that created VPInstructions should be appended to the end
  /// of the specified block.
  void setInsertPoint(VPBasicBlock *TheBB) {
    assert(TheBB && "Attempting to set a null insert point");
    BB = TheBB;
    InsertPt = BB->end();
  }

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  void setInsertPoint(VPBasicBlock *TheBB, VPBasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
  }

  /// Insert and return the specified instruction.
  VPInstruction *insert(VPInstruction *I) const {
    BB->insert(I, InsertPt);
    return I;
  }

  /// Create an N-ary operation with \p Opcode, \p Operands and set \p Inst as
  /// its underlying Instruction.
  VPValue *createNaryOp(unsigned Opcode, ArrayRef<VPValue *> Operands,
                        Instruction *Inst = nullptr) {
    VPInstruction *NewVPInst = createInstruction(Opcode, Operands);
    NewVPInst->setUnderlyingValue(Inst);
    return NewVPInst;
  }
  VPValue *createNaryOp(unsigned Opcode,
                        std::initializer_list<VPValue *> Operands,
                        Instruction *Inst = nullptr) {
    return createNaryOp(Opcode, ArrayRef<VPValue *>(Operands), Inst);
  }

  VPValue *createNot(VPValue *Operand) {
    return createInstruction(VPInstruction::Not, {Operand});
  }

  VPValue *createAnd(VPValue *LHS, VPValue *RHS) {
    return createInstruction(Instruction::BinaryOps::And, {LHS, RHS});
  }

  VPValue *createOr(VPValue *LHS, VPValue *RHS) {
    return createInstruction(Instruction::BinaryOps::Or, {LHS, RHS});
  }

  //===--------------------------------------------------------------------===//
  // RAII helpers.
  //===--------------------------------------------------------------------===//

  /// RAII object that stores the current insertion point and restores it when
  /// the object is destroyed.
  class InsertPointGuard {
    VPBuilder &Builder;
    VPBasicBlock *Block;
    VPBasicBlock::iterator Point;

  public:
    InsertPointGuard(VPBuilder &B)
        : Builder(B), Block(B.getInsertBlock()), Point(B.getInsertPoint()) {}

    InsertPointGuard(const InsertPointGuard &) = delete;
    InsertPointGuard &operator=(const InsertPointGuard &) = delete;

    ~InsertPointGuard() { Builder.restoreIP(VPInsertPoint(Block, Point)); }
  };
};

/// TODO: The following VectorizationFactor was pulled out of
/// LoopVectorizationCostModel class. LV also deals with
/// VectorizerParams::VectorizationFactor and VectorizationCostTy.
/// We need to streamline them.

/// Information about vectorization costs
struct VectorizationFactor {
  // Vector width with best cost
  unsigned Width;
  // Cost of the loop with that width
  unsigned Cost;
};

/// Planner drives the vectorization process after having passed
/// Legality checks.
class LoopVectorizationPlanner {
  /// The loop that we evaluate.
  Loop *OrigLoop;

  /// Loop Info analysis.
  LoopInfo *LI;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Target Transform Info.
  const TargetTransformInfo *TTI;

  /// The legality analysis.
  LoopVectorizationLegality *Legal;

  /// The profitablity analysis.
  LoopVectorizationCostModel &CM;

  using VPlanPtr = std::unique_ptr<VPlan>;

  SmallVector<VPlanPtr, 4> VPlans;

  /// This class is used to enable the VPlan to invoke a method of ILV. This is
  /// needed until the method is refactored out of ILV and becomes reusable.
  struct VPCallbackILV : public VPCallback {
    InnerLoopVectorizer &ILV;

    VPCallbackILV(InnerLoopVectorizer &ILV) : ILV(ILV) {}

    Value *getOrCreateVectorValues(Value *V, unsigned Part) override;
  };

  /// A builder used to construct the current plan.
  VPBuilder Builder;

  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR. Note that
  /// if-conversion currently takes place during VPlan-construction, so these
  /// caches are only used at that stage.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<BasicBlock *, BasicBlock *>, VPValue *>;
  using BlockMaskCacheTy = DenseMap<BasicBlock *, VPValue *>;
  EdgeMaskCacheTy EdgeMaskCache;
  BlockMaskCacheTy BlockMaskCache;

  unsigned BestVF = 0;
  unsigned BestUF = 0;

public:
  LoopVectorizationPlanner(Loop *L, LoopInfo *LI, const TargetLibraryInfo *TLI,
                           const TargetTransformInfo *TTI,
                           LoopVectorizationLegality *Legal,
                           LoopVectorizationCostModel &CM)
      : OrigLoop(L), LI(LI), TLI(TLI), TTI(TTI), Legal(Legal), CM(CM) {}

  /// Plan how to best vectorize, return the best VF and its cost.
  VectorizationFactor plan(bool OptForSize, unsigned UserVF);

  /// Use the VPlan-native path to plan how to best vectorize, return the best
  /// VF and its cost.
  VectorizationFactor planInVPlanNativePath(bool OptForSize, unsigned UserVF);

  /// Finalize the best decision and dispose of all other VPlans.
  void setBestPlan(unsigned VF, unsigned UF);

  /// Generate the IR code for the body of the vectorized loop according to the
  /// best selected VPlan.
  void executePlan(InnerLoopVectorizer &LB, DominatorTree *DT);

  void printPlans(raw_ostream &O) {
    for (const auto &Plan : VPlans)
      O << *Plan;
  }

protected:
  /// Collect the instructions from the original loop that would be trivially
  /// dead in the vectorized loop if generated.
  void collectTriviallyDeadInstructions(
      SmallPtrSetImpl<Instruction *> &DeadInstructions);

  /// A range of powers-of-2 vectorization factors with fixed start and
  /// adjustable end. The range includes start and excludes end, e.g.,:
  /// [1, 9) = {1, 2, 4, 8}
  struct VFRange {
    // A power of 2.
    const unsigned Start;

    // Need not be a power of 2. If End <= Start range is empty.
    unsigned End;
  };

  /// Test a \p Predicate on a \p Range of VF's. Return the value of applying
  /// \p Predicate on Range.Start, possibly decreasing Range.End such that the
  /// returned value holds for the entire \p Range.
  bool getDecisionAndClampRange(const std::function<bool(unsigned)> &Predicate,
                                VFRange &Range);

  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop.
  void buildVPlans(unsigned MinVF, unsigned MaxVF);

private:
  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True. It returns the *entry*
  /// mask for the block BB.
  VPValue *createBlockInMask(BasicBlock *BB, VPlanPtr &Plan);

  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  VPValue *createEdgeMask(BasicBlock *Src, BasicBlock *Dst, VPlanPtr &Plan);

  /// Check if \I belongs to an Interleave Group within the given VF \p Range,
  /// \return true in the first returned value if so and false otherwise.
  /// Build a new VPInterleaveGroup Recipe if \I is the primary member of an IG
  /// for \p Range.Start, and provide it as the second returned value.
  /// Note that if \I is an adjunct member of an IG for \p Range.Start, the
  /// \return value is <true, nullptr>, as it is handled by another recipe.
  /// \p Range.End may be decreased to ensure same decision from \p Range.Start
  /// to \p Range.End.
  VPInterleaveRecipe *tryToInterleaveMemory(Instruction *I, VFRange &Range);

  // Check if \I is a memory instruction to be widened for \p Range.Start and
  // potentially masked. Such instructions are handled by a recipe that takes an
  // additional VPInstruction for the mask.
  VPWidenMemoryInstructionRecipe *tryToWidenMemory(Instruction *I,
                                                   VFRange &Range,
                                                   VPlanPtr &Plan);

  /// Check if an induction recipe should be constructed for \I within the given
  /// VF \p Range. If so build and return it. If not, return null. \p Range.End
  /// may be decreased to ensure same decision from \p Range.Start to
  /// \p Range.End.
  VPWidenIntOrFpInductionRecipe *tryToOptimizeInduction(Instruction *I,
                                                        VFRange &Range);

  /// Handle non-loop phi nodes. Currently all such phi nodes are turned into
  /// a sequence of select instructions as the vectorizer currently performs
  /// full if-conversion.
  VPBlendRecipe *tryToBlend(Instruction *I, VPlanPtr &Plan);

  /// Check if \p I can be widened within the given VF \p Range. If \p I can be
  /// widened for \p Range.Start, check if the last recipe of \p VPBB can be
  /// extended to include \p I or else build a new VPWidenRecipe for it and
  /// append it to \p VPBB. Return true if \p I can be widened for Range.Start,
  /// false otherwise. Range.End may be decreased to ensure same decision from
  /// \p Range.Start to \p Range.End.
  bool tryToWiden(Instruction *I, VPBasicBlock *VPBB, VFRange &Range);

  /// Build a VPReplicationRecipe for \p I and enclose it within a Region if it
  /// is predicated. \return \p VPBB augmented with this new recipe if \p I is
  /// not predicated, otherwise \return a new VPBasicBlock that succeeds the new
  /// Region. Update the packing decision of predicated instructions if they
  /// feed \p I. Range.End may be decreased to ensure same recipe behavior from
  /// \p Range.Start to \p Range.End.
  VPBasicBlock *handleReplication(
      Instruction *I, VFRange &Range, VPBasicBlock *VPBB,
      DenseMap<Instruction *, VPReplicateRecipe *> &PredInst2Recipe,
      VPlanPtr &Plan);

  /// Create a replicating region for instruction \p I that requires
  /// predication. \p PredRecipe is a VPReplicateRecipe holding \p I.
  VPRegionBlock *createReplicateRegion(Instruction *I, VPRecipeBase *PredRecipe,
                                       VPlanPtr &Plan);

  /// Build a VPlan according to the information gathered by Legal. \return a
  /// VPlan for vectorization factors \p Range.Start and up to \p Range.End
  /// exclusive, possibly decreasing \p Range.End.
  VPlanPtr buildVPlan(VFRange &Range,
                                    const SmallPtrSetImpl<Value *> &NeedDef);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONPLANNER_H
