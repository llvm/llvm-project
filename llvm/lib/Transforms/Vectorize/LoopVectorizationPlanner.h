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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/InstructionCost.h"

namespace {
class GeneratedRTChecks;
}

namespace llvm {

class LoopInfo;
class DominatorTree;
class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class PredicatedScalarEvolution;
class LoopVectorizeHints;
class RecurrenceDescriptor;
class LoopVersioning;
class OptimizationRemarkEmitter;
class TargetLibraryInfo;
class VPRecipeBuilder;
struct VPRegisterUsage;
struct VFRange;

extern cl::opt<bool> EnableVPlanNativePath;
extern cl::opt<unsigned> ForceTargetInstructionCost;
extern cl::opt<bool> PreferInLoopReductions;

/// \return An upper bound for vscale based on TTI or the vscale_range
/// attribute.
std::optional<unsigned> getMaxVScale(const Function &F,
                                     const TargetTransformInfo &TTI);

/// Reports an informative message: print \p Msg for debugging purposes as well
/// as an optimization remark. Uses either \p I as location of the remark, or
/// otherwise \p TheLoop. If \p DL is passed, use it as debug location for the
/// remark.
void reportVectorizationInfo(const StringRef Msg, const StringRef ORETag,
                             OptimizationRemarkEmitter *ORE,
                             const Loop *TheLoop, Instruction *I = nullptr,
                             DebugLoc DL = {});

/// VPlan-based builder utility analogous to IRBuilder.
class VPBuilder {
  VPBasicBlock *BB = nullptr;
  VPBasicBlock::iterator InsertPt = VPBasicBlock::iterator();

  /// Insert \p VPI in BB at InsertPt if BB is set.
  template <typename T> T *tryInsertInstruction(T *R) {
    if (BB)
      BB->insert(R, InsertPt);
    return R;
  }

  VPInstruction *createInstruction(unsigned Opcode,
                                   ArrayRef<VPValue *> Operands,
                                   const VPIRMetadata &MD, DebugLoc DL,
                                   const Twine &Name = "") {
    return tryInsertInstruction(
        new VPInstruction(Opcode, Operands, {}, MD, DL, Name));
  }

public:
  VPBuilder() = default;
  VPBuilder(VPBasicBlock *InsertBB) { setInsertPoint(InsertBB); }
  VPBuilder(VPRecipeBase *InsertPt) { setInsertPoint(InsertPt); }
  VPBuilder(VPBasicBlock *TheBB, VPBasicBlock::iterator IP) {
    setInsertPoint(TheBB, IP);
  }

  /// Clear the insertion point: created instructions will not be inserted into
  /// a block.
  void clearInsertionPoint() {
    BB = nullptr;
    InsertPt = VPBasicBlock::iterator();
  }

  VPBasicBlock *getInsertBlock() const { return BB; }
  VPBasicBlock::iterator getInsertPoint() const { return InsertPt; }

  /// Create a VPBuilder to insert after \p R.
  static VPBuilder getToInsertAfter(VPRecipeBase *R) {
    VPBuilder B;
    B.setInsertPoint(R->getParent(), std::next(R->getIterator()));
    return B;
  }

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

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  void setInsertPoint(VPRecipeBase *IP) {
    BB = IP->getParent();
    InsertPt = IP->getIterator();
  }

  /// Insert \p R at the current insertion point. Returns \p R unchanged.
  template <typename T> [[maybe_unused]] T *insert(T *R) {
    BB->insert(R, InsertPt);
    return R;
  }

  /// Create an N-ary operation with \p Opcode, \p Operands and set \p Inst as
  /// its underlying Instruction.
  VPInstruction *createNaryOp(unsigned Opcode, ArrayRef<VPValue *> Operands,
                              Instruction *Inst = nullptr,
                              const VPIRFlags &Flags = {},
                              const VPIRMetadata &MD = {},
                              DebugLoc DL = DebugLoc::getUnknown(),
                              const Twine &Name = "") {
    VPInstruction *NewVPInst = tryInsertInstruction(
        new VPInstruction(Opcode, Operands, Flags, MD, DL, Name));
    NewVPInst->setUnderlyingValue(Inst);
    return NewVPInst;
  }
  VPInstruction *createNaryOp(unsigned Opcode, ArrayRef<VPValue *> Operands,
                              DebugLoc DL, const Twine &Name = "") {
    return createInstruction(Opcode, Operands, {}, DL, Name);
  }
  VPInstruction *createNaryOp(unsigned Opcode, ArrayRef<VPValue *> Operands,
                              const VPIRFlags &Flags,
                              DebugLoc DL = DebugLoc::getUnknown(),
                              const Twine &Name = "") {
    return tryInsertInstruction(
        new VPInstruction(Opcode, Operands, Flags, {}, DL, Name));
  }

  VPInstruction *createNaryOp(unsigned Opcode, ArrayRef<VPValue *> Operands,
                              Type *ResultTy, const VPIRFlags &Flags = {},
                              DebugLoc DL = DebugLoc::getUnknown(),
                              const Twine &Name = "") {
    return tryInsertInstruction(new VPInstructionWithType(
        Opcode, Operands, ResultTy, Flags, {}, DL, Name));
  }

  VPInstruction *createOverflowingOp(
      unsigned Opcode, ArrayRef<VPValue *> Operands,
      VPRecipeWithIRFlags::WrapFlagsTy WrapFlags = {false, false},
      DebugLoc DL = DebugLoc::getUnknown(), const Twine &Name = "") {
    return tryInsertInstruction(
        new VPInstruction(Opcode, Operands, WrapFlags, {}, DL, Name));
  }

  VPInstruction *createNot(VPValue *Operand,
                           DebugLoc DL = DebugLoc::getUnknown(),
                           const Twine &Name = "") {
    return createInstruction(VPInstruction::Not, {Operand}, {}, DL, Name);
  }

  VPInstruction *createAnd(VPValue *LHS, VPValue *RHS,
                           DebugLoc DL = DebugLoc::getUnknown(),
                           const Twine &Name = "") {
    return createInstruction(Instruction::BinaryOps::And, {LHS, RHS}, {}, DL,
                             Name);
  }

  VPInstruction *createOr(VPValue *LHS, VPValue *RHS,
                          DebugLoc DL = DebugLoc::getUnknown(),
                          const Twine &Name = "") {

    return tryInsertInstruction(new VPInstruction(
        Instruction::BinaryOps::Or, {LHS, RHS},
        VPRecipeWithIRFlags::DisjointFlagsTy(false), {}, DL, Name));
  }

  VPInstruction *
  createAdd(VPValue *LHS, VPValue *RHS, DebugLoc DL = DebugLoc::getUnknown(),
            const Twine &Name = "",
            VPRecipeWithIRFlags::WrapFlagsTy WrapFlags = {false, false}) {
    return createOverflowingOp(Instruction::Add, {LHS, RHS}, WrapFlags, DL,
                               Name);
  }

  VPInstruction *
  createSub(VPValue *LHS, VPValue *RHS, DebugLoc DL = DebugLoc::getUnknown(),
            const Twine &Name = "",
            VPRecipeWithIRFlags::WrapFlagsTy WrapFlags = {false, false}) {
    return createOverflowingOp(Instruction::Sub, {LHS, RHS}, WrapFlags, DL,
                               Name);
  }

  VPInstruction *createLogicalAnd(VPValue *LHS, VPValue *RHS,
                                  DebugLoc DL = DebugLoc::getUnknown(),
                                  const Twine &Name = "") {
    return createNaryOp(VPInstruction::LogicalAnd, {LHS, RHS}, DL, Name);
  }

  VPInstruction *createLogicalOr(VPValue *LHS, VPValue *RHS,
                                 DebugLoc DL = DebugLoc::getUnknown(),
                                 const Twine &Name = "") {
    return createNaryOp(VPInstruction::LogicalOr, {LHS, RHS}, DL, Name);
  }

  VPInstruction *createSelect(VPValue *Cond, VPValue *TrueVal,
                              VPValue *FalseVal,
                              DebugLoc DL = DebugLoc::getUnknown(),
                              const Twine &Name = "",
                              const VPIRFlags &Flags = {}) {
    return tryInsertInstruction(new VPInstruction(
        Instruction::Select, {Cond, TrueVal, FalseVal}, Flags, {}, DL, Name));
  }

  /// Create a new ICmp VPInstruction with predicate \p Pred and operands \p A
  /// and \p B.
  VPInstruction *createICmp(CmpInst::Predicate Pred, VPValue *A, VPValue *B,
                            DebugLoc DL = DebugLoc::getUnknown(),
                            const Twine &Name = "") {
    assert(Pred >= CmpInst::FIRST_ICMP_PREDICATE &&
           Pred <= CmpInst::LAST_ICMP_PREDICATE && "invalid predicate");
    return tryInsertInstruction(
        new VPInstruction(Instruction::ICmp, {A, B}, Pred, {}, DL, Name));
  }

  /// Create a new FCmp VPInstruction with predicate \p Pred and operands \p A
  /// and \p B.
  VPInstruction *createFCmp(CmpInst::Predicate Pred, VPValue *A, VPValue *B,
                            DebugLoc DL = DebugLoc::getUnknown(),
                            const Twine &Name = "") {
    assert(Pred >= CmpInst::FIRST_FCMP_PREDICATE &&
           Pred <= CmpInst::LAST_FCMP_PREDICATE && "invalid predicate");
    return tryInsertInstruction(
        new VPInstruction(Instruction::FCmp, {A, B},
                          VPIRFlags(Pred, FastMathFlags()), {}, DL, Name));
  }

  /// Create an AnyOf reduction pattern: or-reduce \p ChainOp, freeze the
  /// result, then select between \p TrueVal and \p FalseVal.
  VPInstruction *createAnyOfReduction(VPValue *ChainOp, VPValue *TrueVal,
                                      VPValue *FalseVal,
                                      DebugLoc DL = DebugLoc::getUnknown());

  VPInstruction *createPtrAdd(VPValue *Ptr, VPValue *Offset,
                              DebugLoc DL = DebugLoc::getUnknown(),
                              const Twine &Name = "") {
    return tryInsertInstruction(
        new VPInstruction(VPInstruction::PtrAdd, {Ptr, Offset},
                          GEPNoWrapFlags::none(), {}, DL, Name));
  }

  VPInstruction *createNoWrapPtrAdd(VPValue *Ptr, VPValue *Offset,
                                    GEPNoWrapFlags GEPFlags,
                                    DebugLoc DL = DebugLoc::getUnknown(),
                                    const Twine &Name = "") {
    return tryInsertInstruction(new VPInstruction(
        VPInstruction::PtrAdd, {Ptr, Offset}, GEPFlags, {}, DL, Name));
  }

  VPInstruction *createWidePtrAdd(VPValue *Ptr, VPValue *Offset,
                                  DebugLoc DL = DebugLoc::getUnknown(),
                                  const Twine &Name = "") {
    return tryInsertInstruction(
        new VPInstruction(VPInstruction::WidePtrAdd, {Ptr, Offset},
                          GEPNoWrapFlags::none(), {}, DL, Name));
  }

  VPPhi *createScalarPhi(ArrayRef<VPValue *> IncomingValues,
                         DebugLoc DL = DebugLoc::getUnknown(),
                         const Twine &Name = "", const VPIRFlags &Flags = {}) {
    return tryInsertInstruction(new VPPhi(IncomingValues, Flags, DL, Name));
  }

  VPWidenPHIRecipe *createWidenPhi(ArrayRef<VPValue *> IncomingValues,
                                   DebugLoc DL = DebugLoc::getUnknown(),
                                   const Twine &Name = "") {
    return tryInsertInstruction(new VPWidenPHIRecipe(IncomingValues, DL, Name));
  }

  VPValue *createElementCount(Type *Ty, ElementCount EC) {
    VPlan &Plan = *getInsertBlock()->getPlan();
    VPValue *RuntimeEC = Plan.getConstantInt(Ty, EC.getKnownMinValue());
    if (EC.isScalable()) {
      VPValue *VScale = createNaryOp(VPInstruction::VScale, {}, Ty);
      RuntimeEC = EC.getKnownMinValue() == 1
                      ? VScale
                      : createOverflowingOp(Instruction::Mul,
                                            {VScale, RuntimeEC}, {true, false});
    }
    return RuntimeEC;
  }

  /// Convert the input value \p Current to the corresponding value of an
  /// induction with \p Start and \p Step values, using \p Start + \p Current *
  /// \p Step.
  VPDerivedIVRecipe *createDerivedIV(InductionDescriptor::InductionKind Kind,
                                     FPMathOperator *FPBinOp, VPIRValue *Start,
                                     VPValue *Current, VPValue *Step,
                                     const Twine &Name = "") {
    return tryInsertInstruction(
        new VPDerivedIVRecipe(Kind, FPBinOp, Start, Current, Step, Name));
  }

  VPInstructionWithType *createScalarLoad(Type *ResultTy, VPValue *Addr,
                                          DebugLoc DL,
                                          const VPIRMetadata &Metadata = {}) {
    return tryInsertInstruction(new VPInstructionWithType(
        Instruction::Load, Addr, ResultTy, {}, Metadata, DL));
  }

  VPInstruction *createScalarCast(Instruction::CastOps Opcode, VPValue *Op,
                                  Type *ResultTy, DebugLoc DL,
                                  const VPIRMetadata &Metadata = {}) {
    return tryInsertInstruction(new VPInstructionWithType(
        Opcode, Op, ResultTy, VPIRFlags::getDefaultFlags(Opcode), Metadata,
        DL));
  }

  VPInstruction *createScalarCast(Instruction::CastOps Opcode, VPValue *Op,
                                  Type *ResultTy, DebugLoc DL,
                                  const VPIRFlags &Flags,
                                  const VPIRMetadata &Metadata = {}) {
    return tryInsertInstruction(
        new VPInstructionWithType(Opcode, Op, ResultTy, Flags, Metadata, DL));
  }

  VPValue *createScalarZExtOrTrunc(VPValue *Op, Type *ResultTy, Type *SrcTy,
                                   DebugLoc DL) {
    if (ResultTy == SrcTy)
      return Op;
    Instruction::CastOps CastOp =
        ResultTy->getScalarSizeInBits() < SrcTy->getScalarSizeInBits()
            ? Instruction::Trunc
            : Instruction::ZExt;
    return createScalarCast(CastOp, Op, ResultTy, DL);
  }

  VPValue *createScalarSExtOrTrunc(VPValue *Op, Type *ResultTy, Type *SrcTy,
                                   DebugLoc DL) {
    if (ResultTy == SrcTy)
      return Op;
    Instruction::CastOps CastOp =
        ResultTy->getScalarSizeInBits() < SrcTy->getScalarSizeInBits()
            ? Instruction::Trunc
            : Instruction::SExt;
    return createScalarCast(CastOp, Op, ResultTy, DL);
  }

  VPWidenCastRecipe *createWidenCast(Instruction::CastOps Opcode, VPValue *Op,
                                     Type *ResultTy) {
    return tryInsertInstruction(new VPWidenCastRecipe(
        Opcode, Op, ResultTy, nullptr, VPIRFlags::getDefaultFlags(Opcode)));
  }

  VPScalarIVStepsRecipe *
  createScalarIVSteps(Instruction::BinaryOps InductionOpcode,
                      FPMathOperator *FPBinOp, VPValue *IV, VPValue *Step,
                      VPValue *VF, DebugLoc DL) {
    return tryInsertInstruction(new VPScalarIVStepsRecipe(
        IV, Step, VF, InductionOpcode,
        FPBinOp ? FPBinOp->getFastMathFlags() : FastMathFlags(), DL));
  }

  VPExpandSCEVRecipe *createExpandSCEV(const SCEV *Expr) {
    return tryInsertInstruction(new VPExpandSCEVRecipe(Expr));
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
/// VectorizerParams::VectorizationFactor.
/// We need to streamline them.

/// Information about vectorization costs.
struct VectorizationFactor {
  /// Vector width with best cost.
  ElementCount Width;

  /// Cost of the loop with that width.
  InstructionCost Cost;

  /// Cost of the scalar loop.
  InstructionCost ScalarCost;

  /// The minimum trip count required to make vectorization profitable, e.g. due
  /// to runtime checks.
  ElementCount MinProfitableTripCount;

  VectorizationFactor(ElementCount Width, InstructionCost Cost,
                      InstructionCost ScalarCost)
      : Width(Width), Cost(Cost), ScalarCost(ScalarCost) {}

  /// Width 1 means no vectorization, cost 0 means uncomputed cost.
  static VectorizationFactor Disabled() {
    return {ElementCount::getFixed(1), 0, 0};
  }

  bool operator==(const VectorizationFactor &rhs) const {
    return Width == rhs.Width && Cost == rhs.Cost;
  }

  bool operator!=(const VectorizationFactor &rhs) const {
    return !(*this == rhs);
  }
};

/// A class that represents two vectorization factors (initialized with 0 by
/// default). One for fixed-width vectorization and one for scalable
/// vectorization. This can be used by the vectorizer to choose from a range of
/// fixed and/or scalable VFs in order to find the most cost-effective VF to
/// vectorize with.
struct FixedScalableVFPair {
  ElementCount FixedVF;
  ElementCount ScalableVF;

  FixedScalableVFPair()
      : FixedVF(ElementCount::getFixed(0)),
        ScalableVF(ElementCount::getScalable(0)) {}
  FixedScalableVFPair(const ElementCount &Max) : FixedScalableVFPair() {
    *(Max.isScalable() ? &ScalableVF : &FixedVF) = Max;
  }
  FixedScalableVFPair(const ElementCount &FixedVF,
                      const ElementCount &ScalableVF)
      : FixedVF(FixedVF), ScalableVF(ScalableVF) {
    assert(!FixedVF.isScalable() && ScalableVF.isScalable() &&
           "Invalid scalable properties");
  }

  static FixedScalableVFPair getNone() { return FixedScalableVFPair(); }

  /// \return true if either fixed- or scalable VF is non-zero.
  explicit operator bool() const { return FixedVF || ScalableVF; }

  /// \return true if either fixed- or scalable VF is a valid vector VF.
  bool hasVector() const { return FixedVF.isVector() || ScalableVF.isVector(); }
};

/// Holds state needed to make cost decisions before computing costs per-VF,
/// including the maximum VFs.
class VFSelectionContext {
  /// \return True if maximizing vector bandwidth is enabled by the target or
  /// user options, for the given register kind (scalable or fixed-width).
  bool useMaxBandwidth(bool IsScalable) const;

  /// \return the maximized element count based on the targets vector
  /// registers and the loop trip-count, but limited to a maximum safe VF.
  /// This is a helper function of computeFeasibleMaxVF.
  ElementCount getMaximizedVFForTarget(unsigned MaxTripCount,
                                       unsigned SmallestType,
                                       unsigned WidestType,
                                       ElementCount MaxSafeVF, unsigned UserIC,
                                       bool FoldTailByMasking,
                                       bool RequiresScalarEpilogue);

  /// If \p VF * \p UserIC > MaxTripcount, clamps VF to the next lower VF
  /// that results in VF * UserIC <= MaxTripCount.
  ElementCount clampVFByMaxTripCount(ElementCount VF, unsigned MaxTripCount,
                                     unsigned UserIC, bool FoldTailByMasking,
                                     bool RequiresScalarEpilogue) const;

  /// Checks if scalable vectorization is supported and enabled. Caches the
  /// result to avoid repeated debug dumps for repeated queries.
  bool isScalableVectorizationAllowed();

  /// \return the maximum legal scalable VF, based on the safe max number
  /// of elements.
  ElementCount getMaxLegalScalableVF(unsigned MaxSafeElements);

  /// Initializes the value of vscale used for tuning the cost model. If
  /// vscale_range.min == vscale_range.max then return vscale_range.max, else
  /// return the value returned by the corresponding TTI method.
  void initializeVScaleForTuning();

  const TargetTransformInfo &TTI;
  const LoopVectorizationLegality *Legal;
  const Loop *TheLoop;
  const Function &F;
  PredicatedScalarEvolution &PSE;
  OptimizationRemarkEmitter *ORE;
  const LoopVectorizeHints *Hints;

  /// Cached result of isScalableVectorizationAllowed.
  std::optional<bool> IsScalableVectorizationAllowed;

  /// Used to store the value of vscale used for tuning the cost model. It is
  /// initialized during object construction.
  std::optional<unsigned> VScaleForTuning;

  /// The highest VF possible for this loop, without using MaxBandwidth.
  FixedScalableVFPair MaxPermissibleVFWithoutMaxBW;

  /// All element types found in the loop.
  SmallPtrSet<Type *, 16> ElementTypesInLoop;

  /// PHINodes of the reductions that should be expanded in-loop. Set by
  /// collectInLoopReductions.
  SmallPtrSet<PHINode *, 4> InLoopReductions;

  /// A Map of inloop reduction operations and their immediate chain operand.
  /// FIXME: This can be removed once reductions can be costed correctly in
  /// VPlan. This was added to allow quick lookup of the inloop operations.
  /// Set by collectInLoopReductions.
  DenseMap<Instruction *, Instruction *> InLoopReductionImmediateChains;

  /// Maximum safe number of elements to be processed per vector iteration,
  /// which do not prevent store-load forwarding and are safe with regard to the
  /// memory dependencies. Required for EVL-based vectorization, where this
  /// value is used as the upper bound of the safe AVL. Set by
  /// computeFeasibleMaxVF.
  std::optional<unsigned> MaxSafeElements;

public:
  /// The kind of cost that we are calculating.
  const TTI::TargetCostKind CostKind;

  /// Whether this loop should be optimized for size based on function attribute
  /// or profile information.
  const bool OptForSize;

  VFSelectionContext(const TargetTransformInfo &TTI,
                     const LoopVectorizationLegality *Legal,
                     const Loop *TheLoop, const Function &F,
                     PredicatedScalarEvolution &PSE,
                     OptimizationRemarkEmitter *ORE,
                     const LoopVectorizeHints *Hints, bool OptForSize)
      : TTI(TTI), Legal(Legal), TheLoop(TheLoop), F(F), PSE(PSE), ORE(ORE),
        Hints(Hints),
        CostKind(F.hasMinSize() ? TTI::TCK_CodeSize : TTI::TCK_RecipThroughput),
        OptForSize(OptForSize) {
    initializeVScaleForTuning();
  }

  /// \return The vscale value used for tuning the cost model.
  std::optional<unsigned> getVScaleForTuning() const { return VScaleForTuning; }

  /// \return True if register pressure should be considered for the given VF.
  bool shouldConsiderRegPressureForVF(ElementCount VF) const;

  /// \return True if scalable vectors are supported by the target or forced.
  bool supportsScalableVectors() const;

  /// Collect element types in the loop that need widening.
  void collectElementTypesForWidening(
      const SmallPtrSetImpl<const Value *> *ValuesToIgnore = nullptr);

  /// \return The size (in bits) of the smallest and widest types in the code
  /// that need to be vectorized. We ignore values that remain scalar such as
  /// 64 bit loop indices.
  std::pair<unsigned, unsigned> getSmallestAndWidestTypes() const;

  /// \return An upper bound for the vectorization factors for both
  /// fixed and scalable vectorization, where the minimum-known number of
  /// elements is a power-of-2 larger than zero. If scalable vectorization is
  /// disabled or unsupported, then the scalable part will be equal to
  /// ElementCount::getScalable(0). Also sets MaxSafeElements.
  FixedScalableVFPair computeFeasibleMaxVF(unsigned MaxTripCount,
                                           ElementCount UserVF, unsigned UserIC,
                                           bool FoldTailByMasking,
                                           bool RequiresScalarEpilogue);

  /// Return maximum safe number of elements to be processed per vector
  /// iteration, which do not prevent store-load forwarding and are safe with
  /// regard to the memory dependencies. Required for EVL-based VPlans to
  /// correctly calculate AVL (application vector length) as min(remaining AVL,
  /// MaxSafeElements). Set by computeFeasibleMaxVF.
  /// TODO: need to consider adjusting cost model to use this value as a
  /// vectorization factor for EVL-based vectorization.
  std::optional<unsigned> getMaxSafeElements() const { return MaxSafeElements; }

  /// Returns true if we should use strict in-order reductions for the given
  /// RdxDesc. This is true if the -enable-strict-reductions flag is passed,
  /// the IsOrdered flag of RdxDesc is set and we do not allow reordering
  /// of FP operations.
  bool useOrderedReductions(const RecurrenceDescriptor &RdxDesc) const;

  /// Returns true if the target machine supports masked store operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedStore(Type *DataType, Value *Ptr, Align Alignment,
                          unsigned AddressSpace) const;

  /// Returns true if the target machine supports masked load operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedLoad(Type *DataType, Value *Ptr, Align Alignment,
                         unsigned AddressSpace) const;

  /// Returns true if the target machine can represent \p V as a masked gather
  /// or scatter operation.
  bool isLegalGatherOrScatter(Value *V, ElementCount VF) const;

  /// Split reductions into those that happen in the loop, and those that
  /// happen outside. In-loop reductions are collected into InLoopReductions.
  /// InLoopReductionImmediateChains is filled with each in-loop reduction
  /// operation and its immediate chain operand for use during cost modelling.
  void collectInLoopReductions();

  /// Returns true if the Phi is part of an inloop reduction.
  bool isInLoopReduction(PHINode *Phi) const {
    return InLoopReductions.contains(Phi);
  }

  /// Returns the set of in-loop reduction PHIs.
  const SmallPtrSetImpl<PHINode *> &getInLoopReductions() const {
    return InLoopReductions;
  }

  /// Returns the immediate chain operand of in-loop reduction operation \p I,
  /// or nullptr if \p I is not an in-loop reduction operation.
  Instruction *getInLoopReductionImmediateChain(Instruction *I) const {
    return InLoopReductionImmediateChains.lookup(I);
  }

  /// Check whether vectorization would require runtime checks. When optimizing
  /// for size, returning true here aborts vectorization.
  bool runtimeChecksRequired();
};

/// Planner drives the vectorization process after having passed
/// Legality checks.
class LoopVectorizationPlanner {
  /// The loop that we evaluate.
  Loop *OrigLoop;

  /// Loop Info analysis.
  LoopInfo *LI;

  /// The dominator tree.
  DominatorTree *DT;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Target Transform Info.
  const TargetTransformInfo &TTI;

  /// The legality analysis.
  LoopVectorizationLegality *Legal;

  /// The profitability analysis.
  LoopVectorizationCostModel &CM;

  /// VF selection state independent of cost-modeling decisions.
  VFSelectionContext &Config;

  /// The interleaved access analysis.
  InterleavedAccessInfo &IAI;

  PredicatedScalarEvolution &PSE;

  const LoopVectorizeHints &Hints;

  OptimizationRemarkEmitter *ORE;

  SmallVector<VPlanPtr, 4> VPlans;

  /// Profitable vector factors.
  SmallVector<VectorizationFactor, 8> ProfitableVFs;

  /// A builder used to construct the current plan.
  VPBuilder Builder;

  /// Computes the cost of \p Plan for vectorization factor \p VF.
  ///
  /// The current implementation requires access to the
  /// LoopVectorizationLegality to handle inductions and reductions, which is
  /// why it is kept separate from the VPlan-only cost infrastructure.
  ///
  /// TODO: Move to VPlan::cost once the use of LoopVectorizationLegality has
  /// been retired.
  InstructionCost cost(VPlan &Plan, ElementCount VF, VPRegisterUsage *RU) const;

  /// Precompute costs for certain instructions using the legacy cost model. The
  /// function is used to bring up the VPlan-based cost model to initially avoid
  /// taking different decisions due to inaccuracies in the legacy cost model.
  InstructionCost precomputeCosts(VPlan &Plan, ElementCount VF,
                                  VPCostContext &CostCtx) const;

public:
  LoopVectorizationPlanner(
      Loop *L, LoopInfo *LI, DominatorTree *DT, const TargetLibraryInfo *TLI,
      const TargetTransformInfo &TTI, LoopVectorizationLegality *Legal,
      LoopVectorizationCostModel &CM, VFSelectionContext &Config,
      InterleavedAccessInfo &IAI, PredicatedScalarEvolution &PSE,
      const LoopVectorizeHints &Hints, OptimizationRemarkEmitter *ORE)
      : OrigLoop(L), LI(LI), DT(DT), TLI(TLI), TTI(TTI), Legal(Legal), CM(CM),
        Config(Config), IAI(IAI), PSE(PSE), Hints(Hints), ORE(ORE) {}

  /// Build VPlans for the specified \p UserVF and \p UserIC if they are
  /// non-zero or all applicable candidate VFs otherwise. If vectorization and
  /// interleaving should be avoided up-front, no plans are generated.
  void plan(ElementCount UserVF, unsigned UserIC);

  /// Use the VPlan-native path to plan how to best vectorize, return the best
  /// VF and its cost.
  VectorizationFactor planInVPlanNativePath(ElementCount UserVF);

  /// Return the VPlan for \p VF. At the moment, there is always a single VPlan
  /// for each VF.
  VPlan &getPlanFor(ElementCount VF) const;

  /// Compute and return the most profitable vectorization factor and the
  /// corresponding best VPlan. Also collect all profitable VFs in
  /// ProfitableVFs.
  std::pair<VectorizationFactor, VPlan *> computeBestVF();

  /// \return The desired interleave count.
  /// If interleave count has been specified by metadata it will be returned.
  /// Otherwise, the interleave count is computed and returned. VF and LoopCost
  /// are the selected vectorization factor and the cost of the selected VF.
  unsigned selectInterleaveCount(VPlan &Plan, ElementCount VF,
                                 InstructionCost LoopCost);

  /// Generate the IR code for the vectorized loop captured in VPlan \p BestPlan
  /// according to the best selected \p VF and  \p UF.
  ///
  /// TODO: \p EpilogueVecKind should be removed once the re-use issue has been
  /// fixed.
  ///
  /// Returns a mapping of SCEVs to their expanded IR values.
  /// Note that this is a temporary workaround needed due to the current
  /// epilogue handling.
  enum class EpilogueVectorizationKind {
    None,     ///< Not part of epilogue vectorization.
    MainLoop, ///< Vectorizing the main loop of epilogue vectorization.
    Epilogue  ///< Vectorizing the epilogue loop.
  };
  DenseMap<const SCEV *, Value *>
  executePlan(ElementCount VF, unsigned UF, VPlan &BestPlan,
              InnerLoopVectorizer &LB, DominatorTree *DT,
              EpilogueVectorizationKind EpilogueVecKind =
                  EpilogueVectorizationKind::None);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printPlans(raw_ostream &O);
#endif

  /// Look through the existing plans and return true if we have one with
  /// vectorization factor \p VF.
  bool hasPlanWithVF(ElementCount VF) const {
    return any_of(VPlans,
                  [&](const VPlanPtr &Plan) { return Plan->hasVF(VF); });
  }

  /// Test a \p Predicate on a \p Range of VF's. Return the value of applying
  /// \p Predicate on Range.Start, possibly decreasing Range.End such that the
  /// returned value holds for the entire \p Range.
  static bool
  getDecisionAndClampRange(const std::function<bool(ElementCount)> &Predicate,
                           VFRange &Range);

  /// \return A VPlan for the most profitable epilogue vectorization, with its
  /// VF narrowed to the chosen factor. The returned plan is a duplicate.
  /// Returns nullptr if epilogue vectorization is not supported or not
  /// profitable for the loop.
  std::unique_ptr<VPlan>
  selectBestEpiloguePlan(VPlan &MainPlan, ElementCount MainLoopVF, unsigned IC);

  /// Emit remarks for recipes with invalid costs in the available VPlans.
  void emitInvalidCostRemarks(OptimizationRemarkEmitter *ORE);

  /// Create a check to \p Plan to see if the vector loop should be executed
  /// based on its trip count.
  void addMinimumIterationCheck(VPlan &Plan, ElementCount VF, unsigned UF,
                                ElementCount MinProfitableTripCount) const;

  /// Attach the runtime checks of \p RTChecks to \p Plan.
  void attachRuntimeChecks(VPlan &Plan, GeneratedRTChecks &RTChecks,
                           bool HasBranchWeights) const;

  /// Update loop metadata and profile info for both the scalar remainder loop
  /// and \p VectorLoop, if it exists. Keeps all loop hints from the original
  /// loop on the vector loop and replaces vectorizer-specific metadata. The
  /// loop ID of the original loop \p OrigLoopID must be passed, together with
  /// the average trip count and invocation weight of the original loop (\p
  /// OrigAverageTripCount and \p OrigLoopInvocationWeight respectively). They
  /// cannot be retrieved after the plan has been executed, as the original loop
  /// may have been removed.
  void updateLoopMetadataAndProfileInfo(
      Loop *VectorLoop, VPBasicBlock *HeaderVPBB, const VPlan &Plan,
      bool VectorizingEpilogue, MDNode *OrigLoopID,
      std::optional<unsigned> OrigAverageTripCount,
      unsigned OrigLoopInvocationWeight, unsigned EstimatedVFxUF,
      bool DisableRuntimeUnroll);

protected:
  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop.
  void buildVPlans(ElementCount MinVF, ElementCount MaxVF);

private:
  /// Build a VPlan according to the information gathered by Legal. \return a
  /// VPlan for vectorization factors \p Range.Start and up to \p Range.End
  /// exclusive, possibly decreasing \p Range.End. If no VPlan can be built for
  /// the input range, set the largest included VF to the maximum VF for which
  /// no plan could be built.
  VPlanPtr tryToBuildVPlan(VFRange &Range);

  /// Build a VPlan using VPRecipes according to the information gather by
  /// Legal. This method is only used for the legacy inner loop vectorizer.
  /// \p Range's largest included VF is restricted to the maximum VF the
  /// returned VPlan is valid for. If no VPlan can be built for the input range,
  /// set the largest included VF to the maximum VF for which no plan could be
  /// built. Each VPlan is built starting from a copy of \p InitialPlan, which
  /// is a plain CFG VPlan wrapping the original scalar loop.
  VPlanPtr tryToBuildVPlanWithVPRecipes(VPlanPtr InitialPlan, VFRange &Range);

  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop. This method creates VPlans using VPRecipes.
  void buildVPlansWithVPRecipes(ElementCount MinVF, ElementCount MaxVF);

  /// Add ComputeReductionResult recipes to the middle block to compute the
  /// final reduction results. Add Select recipes to the latch block when
  /// folding tail, to feed ComputeReductionResult with the last or penultimate
  /// iteration values according to the header mask.
  void addReductionResultComputation(VPlanPtr &Plan,
                                     VPRecipeBuilder &RecipeBuilder,
                                     ElementCount MinVF);

  /// Returns true if the per-lane cost of VectorizationFactor A is lower than
  /// that of B.
  bool isMoreProfitable(const VectorizationFactor &A,
                        const VectorizationFactor &B, bool HasTail,
                        bool IsEpilogue = false) const;

  /// Returns true if the per-lane cost of VectorizationFactor A is lower than
  /// that of B in the context of vectorizing a loop with known \p MaxTripCount.
  bool isMoreProfitable(const VectorizationFactor &A,
                        const VectorizationFactor &B,
                        const unsigned MaxTripCount, bool HasTail,
                        bool IsEpilogue = false) const;

  /// Determines if we have the infrastructure to vectorize the loop and its
  /// epilogue, assuming the main loop is vectorized by \p MainPlan.
  bool isCandidateForEpilogueVectorization(VPlan &MainPlan) const;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONPLANNER_H
