//===- LoopVectorizationPlanner.h - Planner for LoopVectorization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class PredicatedScalarEvolution;
class VPRecipeBuilder;

/// VPlan-based builder utility analogous to IRBuilder.
class VPBuilder {
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
  ElementCount Width;
  // Cost of the loop with that width
  unsigned Cost;

  // Width 1 means no vectorization, cost 0 means uncomputed cost.
  static VectorizationFactor Disabled() {
    return {ElementCount::getFixed(1), 0};
  }

  bool operator==(const VectorizationFactor &rhs) const {
    return Width == rhs.Width && Cost == rhs.Cost;
  }

  bool operator!=(const VectorizationFactor &rhs) const {
    return !(*this == rhs);
  }
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

  /// The profitability analysis.
  LoopVectorizationCostModel &CM;

  /// The interleaved access analysis.
  InterleavedAccessInfo &IAI;

  PredicatedScalarEvolution &PSE;

  SmallVector<VPlanPtr, 4> VPlans;

  /// This class is used to enable the VPlan to invoke a method of ILV. This is
  /// needed until the method is refactored out of ILV and becomes reusable.
  struct VPCallbackILV : public VPCallback {
    InnerLoopVectorizer &ILV;

    VPCallbackILV(InnerLoopVectorizer &ILV) : ILV(ILV) {}

    Value *getOrCreateVectorValues(Value *V, unsigned Part) override;
    Value *getOrCreateScalarValue(Value *V,
                                  const VPIteration &Instance) override;
  };

  /// A builder used to construct the current plan.
  VPBuilder Builder;

  /// The best number of elements of the vector types used in the
  /// transformed loop. BestVF = None means that vectorization is
  /// disabled.
  Optional<ElementCount> BestVF = None;
  unsigned BestUF = 0;

public:
  LoopVectorizationPlanner(Loop *L, LoopInfo *LI, const TargetLibraryInfo *TLI,
                           const TargetTransformInfo *TTI,
                           LoopVectorizationLegality *Legal,
                           LoopVectorizationCostModel &CM,
                           InterleavedAccessInfo &IAI,
                           PredicatedScalarEvolution &PSE)
      : OrigLoop(L), LI(LI), TLI(TLI), TTI(TTI), Legal(Legal), CM(CM), IAI(IAI),
        PSE(PSE) {}

  /// Plan how to best vectorize, return the best VF and its cost, or None if
  /// vectorization and interleaving should be avoided up front.
  Optional<VectorizationFactor> plan(ElementCount UserVF, unsigned UserIC);

  /// Use the VPlan-native path to plan how to best vectorize, return the best
  /// VF and its cost.
  VectorizationFactor planInVPlanNativePath(ElementCount UserVF);

  /// Finalize the best decision and dispose of all other VPlans.
  void setBestPlan(ElementCount VF, unsigned UF);

  /// Generate the IR code for the body of the vectorized loop according to the
  /// best selected VPlan.
  void executePlan(InnerLoopVectorizer &LB, DominatorTree *DT);

  void printPlans(raw_ostream &O) {
    for (const auto &Plan : VPlans)
      O << *Plan;
  }

  /// Look through the existing plans and return true if we have one with all
  /// the vectorization factors in question.
  bool hasPlanWithVFs(const ArrayRef<ElementCount> VFs) const {
    return any_of(VPlans, [&](const VPlanPtr &Plan) {
      return all_of(VFs, [&](const ElementCount &VF) {
        return Plan->hasVF(VF);
      });
    });
  }

  /// Test a \p Predicate on a \p Range of VF's. Return the value of applying
  /// \p Predicate on Range.Start, possibly decreasing Range.End such that the
  /// returned value holds for the entire \p Range.
  static bool
  getDecisionAndClampRange(const std::function<bool(ElementCount)> &Predicate,
                           VFRange &Range);

protected:
  /// Collect the instructions from the original loop that would be trivially
  /// dead in the vectorized loop if generated.
  void collectTriviallyDeadInstructions(
      SmallPtrSetImpl<Instruction *> &DeadInstructions);

  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop.
  void buildVPlans(ElementCount MinVF, ElementCount MaxVF);

private:
  /// Build a VPlan according to the information gathered by Legal. \return a
  /// VPlan for vectorization factors \p Range.Start and up to \p Range.End
  /// exclusive, possibly decreasing \p Range.End.
  VPlanPtr buildVPlan(VFRange &Range);

  /// Build a VPlan using VPRecipes according to the information gather by
  /// Legal. This method is only used for the legacy inner loop vectorizer.
  VPlanPtr buildVPlanWithVPRecipes(
      VFRange &Range, SmallPtrSetImpl<Instruction *> &DeadInstructions,
      const DenseMap<Instruction *, Instruction *> &SinkAfter);

  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop. This method creates VPlans using VPRecipes.
  void buildVPlansWithVPRecipes(ElementCount MinVF, ElementCount MaxVF);

  /// Adjust the recipes for any inloop reductions. The chain of instructions
  /// leading from the loop exit instr to the phi need to be converted to
  /// reductions, with one operand being vector and the other being the scalar
  /// reduction chain.
  void adjustRecipesForInLoopReductions(VPlanPtr &Plan,
                                        VPRecipeBuilder &RecipeBuilder);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONPLANNER_H
