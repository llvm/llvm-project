//===- llvm/Transforms/Utils/LoopUtils.h - Loop utilities -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPUTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Casting.h"

namespace llvm {

class AliasSet;
class AliasSetTracker;
class BasicBlock;
class DataLayout;
class Loop;
class LoopInfo;
class OptimizationRemarkEmitter;
class PredicatedScalarEvolution;
class PredIteratorCache;
class ScalarEvolution;
class SCEV;
class TargetLibraryInfo;
class TargetTransformInfo;


/// The RecurrenceDescriptor is used to identify recurrences variables in a
/// loop. Reduction is a special case of recurrence that has uses of the
/// recurrence variable outside the loop. The method isReductionPHI identifies
/// reductions that are basic recurrences.
///
/// Basic recurrences are defined as the summation, product, OR, AND, XOR, min,
/// or max of a set of terms. For example: for(i=0; i<n; i++) { total +=
/// array[i]; } is a summation of array elements. Basic recurrences are a
/// special case of chains of recurrences (CR). See ScalarEvolution for CR
/// references.

/// This struct holds information about recurrence variables.
class RecurrenceDescriptor {
public:
  /// This enum represents the kinds of recurrences that we support.
  enum RecurrenceKind {
    RK_NoRecurrence,  ///< Not a recurrence.
    RK_IntegerAdd,    ///< Sum of integers.
    RK_IntegerMult,   ///< Product of integers.
    RK_IntegerOr,     ///< Bitwise or logical OR of numbers.
    RK_IntegerAnd,    ///< Bitwise or logical AND of numbers.
    RK_IntegerXor,    ///< Bitwise or logical XOR of numbers.
    RK_IntegerMinMax, ///< Min/max implemented in terms of select(cmp()).
    RK_FloatAdd,      ///< Sum of floats.
    RK_FloatMult,     ///< Product of floats.
    RK_FloatMinMax    ///< Min/max implemented in terms of select(cmp()).
  };

  // This enum represents the kind of minmax recurrence.
  enum MinMaxRecurrenceKind {
    MRK_Invalid,
    MRK_UIntMin,
    MRK_UIntMax,
    MRK_SIntMin,
    MRK_SIntMax,
    MRK_FloatMin,
    MRK_FloatMax
  };

  RecurrenceDescriptor() = default;

  RecurrenceDescriptor(Value *Start, Instruction *Exit, RecurrenceKind K,
                       MinMaxRecurrenceKind MK, Instruction *UAI, Type *RT,
                       bool Signed, SmallPtrSetImpl<Instruction *> &CI)
      : StartValue(Start), LoopExitInstr(Exit), Kind(K), MinMaxKind(MK),
        UnsafeAlgebraInst(UAI), RecurrenceType(RT), IsSigned(Signed) {
    CastInsts.insert(CI.begin(), CI.end());
  }

  /// This POD struct holds information about a potential recurrence operation.
  class InstDesc {
  public:
    InstDesc(bool IsRecur, Instruction *I, Instruction *UAI = nullptr)
        : IsRecurrence(IsRecur), PatternLastInst(I), MinMaxKind(MRK_Invalid),
          UnsafeAlgebraInst(UAI) {}

    InstDesc(Instruction *I, MinMaxRecurrenceKind K, Instruction *UAI = nullptr)
        : IsRecurrence(true), PatternLastInst(I), MinMaxKind(K),
          UnsafeAlgebraInst(UAI) {}

    bool isRecurrence() { return IsRecurrence; }

    bool hasUnsafeAlgebra() { return UnsafeAlgebraInst != nullptr; }

    Instruction *getUnsafeAlgebraInst() { return UnsafeAlgebraInst; }

    MinMaxRecurrenceKind getMinMaxKind() { return MinMaxKind; }

    Instruction *getPatternInst() { return PatternLastInst; }

  private:
    // Is this instruction a recurrence candidate.
    bool IsRecurrence;
    // The last instruction in a min/max pattern (select of the select(icmp())
    // pattern), or the current recurrence instruction otherwise.
    Instruction *PatternLastInst;
    // If this is a min/max pattern the comparison predicate.
    MinMaxRecurrenceKind MinMaxKind;
    // Recurrence has unsafe algebra.
    Instruction *UnsafeAlgebraInst;
  };

  /// Returns a struct describing if the instruction 'I' can be a recurrence
  /// variable of type 'Kind'. If the recurrence is a min/max pattern of
  /// select(icmp()) this function advances the instruction pointer 'I' from the
  /// compare instruction to the select instruction and stores this pointer in
  /// 'PatternLastInst' member of the returned struct.
  static InstDesc isRecurrenceInstr(Instruction *I, RecurrenceKind Kind,
                                    InstDesc &Prev, bool HasFunNoNaNAttr);

  /// Returns true if instruction I has multiple uses in Insts
  static bool hasMultipleUsesOf(Instruction *I,
                                SmallPtrSetImpl<Instruction *> &Insts);

  /// Returns true if all uses of the instruction I is within the Set.
  static bool areAllUsesIn(Instruction *I, SmallPtrSetImpl<Instruction *> &Set);

  /// Returns a struct describing if the instruction if the instruction is a
  /// Select(ICmp(X, Y), X, Y) instruction pattern corresponding to a min(X, Y)
  /// or max(X, Y).
  static InstDesc isMinMaxSelectCmpPattern(Instruction *I, InstDesc &Prev);

  /// Returns identity corresponding to the RecurrenceKind.
  static Constant *getRecurrenceIdentity(RecurrenceKind K, Type *Tp);

  /// Returns the opcode of binary operation corresponding to the
  /// RecurrenceKind.
  static unsigned getRecurrenceBinOp(RecurrenceKind Kind);

  /// Returns a Min/Max operation corresponding to MinMaxRecurrenceKind.
  static Value *createMinMaxOp(IRBuilder<> &Builder, MinMaxRecurrenceKind RK,
                               Value *Left, Value *Right);

  /// Returns true if Phi is a reduction of type Kind and adds it to the
  /// RecurrenceDescriptor. If either \p DB is non-null or \p AC and \p DT are
  /// non-null, the minimal bit width needed to compute the reduction will be
  /// computed.
  static bool AddReductionVar(PHINode *Phi, RecurrenceKind Kind, Loop *TheLoop,
                              bool HasFunNoNaNAttr,
                              RecurrenceDescriptor &RedDes,
                              DemandedBits *DB = nullptr,
                              AssumptionCache *AC = nullptr,
                              DominatorTree *DT = nullptr);

  /// Returns true if Phi is a reduction in TheLoop. The RecurrenceDescriptor
  /// is returned in RedDes. If either \p DB is non-null or \p AC and \p DT are
  /// non-null, the minimal bit width needed to compute the reduction will be
  /// computed.
  static bool isReductionPHI(PHINode *Phi, Loop *TheLoop,
                             RecurrenceDescriptor &RedDes,
                             DemandedBits *DB = nullptr,
                             AssumptionCache *AC = nullptr,
                             DominatorTree *DT = nullptr);

  /// Returns true if Phi is a first-order recurrence. A first-order recurrence
  /// is a non-reduction recurrence relation in which the value of the
  /// recurrence in the current loop iteration equals a value defined in the
  /// previous iteration. \p SinkAfter includes pairs of instructions where the
  /// first will be rescheduled to appear after the second if/when the loop is
  /// vectorized. It may be augmented with additional pairs if needed in order
  /// to handle Phi as a first-order recurrence.
  static bool
  isFirstOrderRecurrence(PHINode *Phi, Loop *TheLoop,
                         DenseMap<Instruction *, Instruction *> &SinkAfter,
                         DominatorTree *DT);

  RecurrenceKind getRecurrenceKind() { return Kind; }

  MinMaxRecurrenceKind getMinMaxRecurrenceKind() { return MinMaxKind; }

  TrackingVH<Value> getRecurrenceStartValue() { return StartValue; }

  Instruction *getLoopExitInstr() { return LoopExitInstr; }

  /// Returns true if the recurrence has unsafe algebra which requires a relaxed
  /// floating-point model.
  bool hasUnsafeAlgebra() { return UnsafeAlgebraInst != nullptr; }

  /// Returns first unsafe algebra instruction in the PHI node's use-chain.
  Instruction *getUnsafeAlgebraInst() { return UnsafeAlgebraInst; }

  /// Returns true if the recurrence kind is an integer kind.
  static bool isIntegerRecurrenceKind(RecurrenceKind Kind);

  /// Returns true if the recurrence kind is a floating point kind.
  static bool isFloatingPointRecurrenceKind(RecurrenceKind Kind);

  /// Returns true if the recurrence kind is an arithmetic kind.
  static bool isArithmeticRecurrenceKind(RecurrenceKind Kind);

  /// Returns the type of the recurrence. This type can be narrower than the
  /// actual type of the Phi if the recurrence has been type-promoted.
  Type *getRecurrenceType() { return RecurrenceType; }

  /// Returns a reference to the instructions used for type-promoting the
  /// recurrence.
  SmallPtrSet<Instruction *, 8> &getCastInsts() { return CastInsts; }

  /// Returns true if all source operands of the recurrence are SExtInsts.
  bool isSigned() { return IsSigned; }

private:
  // The starting value of the recurrence.
  // It does not have to be zero!
  TrackingVH<Value> StartValue;
  // The instruction who's value is used outside the loop.
  Instruction *LoopExitInstr = nullptr;
  // The kind of the recurrence.
  RecurrenceKind Kind = RK_NoRecurrence;
  // If this a min/max recurrence the kind of recurrence.
  MinMaxRecurrenceKind MinMaxKind = MRK_Invalid;
  // First occurrence of unasfe algebra in the PHI's use-chain.
  Instruction *UnsafeAlgebraInst = nullptr;
  // The type of the recurrence.
  Type *RecurrenceType = nullptr;
  // True if all source operands of the recurrence are SExtInsts.
  bool IsSigned = false;
  // Instructions used for type-promoting the recurrence.
  SmallPtrSet<Instruction *, 8> CastInsts;
};

/// A struct for saving information about induction variables.
class InductionDescriptor {
public:
  /// This enum represents the kinds of inductions that we support.
  enum InductionKind {
    IK_NoInduction,  ///< Not an induction variable.
    IK_IntInduction, ///< Integer induction variable. Step = C.
    IK_PtrInduction, ///< Pointer induction var. Step = C / sizeof(elem).
    IK_FpInduction   ///< Floating point induction variable.
  };

public:
  /// Default constructor - creates an invalid induction.
  InductionDescriptor() = default;

  /// Get the consecutive direction. Returns:
  ///   0 - unknown or non-consecutive.
  ///   1 - consecutive and increasing.
  ///  -1 - consecutive and decreasing.
  int getConsecutiveDirection() const;

  /// Compute the transformed value of Index at offset StartValue using step
  /// StepValue.
  /// For integer induction, returns StartValue + Index * StepValue.
  /// For pointer induction, returns StartValue[Index * StepValue].
  /// FIXME: The newly created binary instructions should contain nsw/nuw
  /// flags, which can be found from the original scalar operations.
  Value *transform(IRBuilder<> &B, Value *Index, ScalarEvolution *SE,
                   const DataLayout& DL) const;

  Value *getStartValue() const { return StartValue; }
  InductionKind getKind() const { return IK; }
  const SCEV *getStep() const { return Step; }
  ConstantInt *getConstIntStepValue() const;

  /// Returns true if \p Phi is an induction in the loop \p L. If \p Phi is an
  /// induction, the induction descriptor \p D will contain the data describing
  /// this induction. If by some other means the caller has a better SCEV
  /// expression for \p Phi than the one returned by the ScalarEvolution
  /// analysis, it can be passed through \p Expr. If the def-use chain
  /// associated with the phi includes casts (that we know we can ignore
  /// under proper runtime checks), they are passed through \p CastsToIgnore.
  static bool
  isInductionPHI(PHINode *Phi, const Loop* L, ScalarEvolution *SE,
                 InductionDescriptor &D, const SCEV *Expr = nullptr,
                 SmallVectorImpl<Instruction *> *CastsToIgnore = nullptr);

  /// Returns true if \p Phi is a floating point induction in the loop \p L.
  /// If \p Phi is an induction, the induction descriptor \p D will contain
  /// the data describing this induction.
  static bool isFPInductionPHI(PHINode *Phi, const Loop* L,
                               ScalarEvolution *SE, InductionDescriptor &D);

  /// Returns true if \p Phi is a loop \p L induction, in the context associated
  /// with the run-time predicate of PSE. If \p Assume is true, this can add
  /// further SCEV predicates to \p PSE in order to prove that \p Phi is an
  /// induction.
  /// If \p Phi is an induction, \p D will contain the data describing this
  /// induction.
  static bool isInductionPHI(PHINode *Phi, const Loop* L,
                             PredicatedScalarEvolution &PSE,
                             InductionDescriptor &D, bool Assume = false);

  /// Returns true if the induction type is FP and the binary operator does
  /// not have the "fast-math" property. Such operation requires a relaxed FP
  /// mode.
  bool hasUnsafeAlgebra() {
    return InductionBinOp && !cast<FPMathOperator>(InductionBinOp)->isFast();
  }

  /// Returns induction operator that does not have "fast-math" property
  /// and requires FP unsafe mode.
  Instruction *getUnsafeAlgebraInst() {
    if (!InductionBinOp || cast<FPMathOperator>(InductionBinOp)->isFast())
      return nullptr;
    return InductionBinOp;
  }

  /// Returns binary opcode of the induction operator.
  Instruction::BinaryOps getInductionOpcode() const {
    return InductionBinOp ? InductionBinOp->getOpcode() :
      Instruction::BinaryOpsEnd;
  }

  /// Returns a reference to the type cast instructions in the induction
  /// update chain, that are redundant when guarded with a runtime
  /// SCEV overflow check.
  const SmallVectorImpl<Instruction *> &getCastInsts() const {
    return RedundantCasts;
  }

private:
  /// Private constructor - used by \c isInductionPHI.
  InductionDescriptor(Value *Start, InductionKind K, const SCEV *Step,
                      BinaryOperator *InductionBinOp = nullptr,
                      SmallVectorImpl<Instruction *> *Casts = nullptr);

  /// Start value.
  TrackingVH<Value> StartValue;
  /// Induction kind.
  InductionKind IK = IK_NoInduction;
  /// Step value.
  const SCEV *Step = nullptr;
  // Instruction that advances induction variable.
  BinaryOperator *InductionBinOp = nullptr;
  // Instructions used for type-casts of the induction variable,
  // that are redundant when guarded with a runtime SCEV overflow check.
  SmallVector<Instruction *, 2> RedundantCasts;
};

BasicBlock *InsertPreheaderForLoop(Loop *L, DominatorTree *DT, LoopInfo *LI,
                                   bool PreserveLCSSA);

/// Ensure that all exit blocks of the loop are dedicated exits.
///
/// For any loop exit block with non-loop predecessors, we split the loop
/// predecessors to use a dedicated loop exit block. We update the dominator
/// tree and loop info if provided, and will preserve LCSSA if requested.
bool formDedicatedExitBlocks(Loop *L, DominatorTree *DT, LoopInfo *LI,
                             bool PreserveLCSSA);

/// Ensures LCSSA form for every instruction from the Worklist in the scope of
/// innermost containing loop.
///
/// For the given instruction which have uses outside of the loop, an LCSSA PHI
/// node is inserted and the uses outside the loop are rewritten to use this
/// node.
///
/// LoopInfo and DominatorTree are required and, since the routine makes no
/// changes to CFG, preserved.
///
/// Returns true if any modifications are made.
bool formLCSSAForInstructions(SmallVectorImpl<Instruction *> &Worklist,
                              DominatorTree &DT, LoopInfo &LI);

/// \brief Put loop into LCSSA form.
///
/// Looks at all instructions in the loop which have uses outside of the
/// current loop. For each, an LCSSA PHI node is inserted and the uses outside
/// the loop are rewritten to use this node.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSA(Loop &L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution *SE);

/// \brief Put a loop nest into LCSSA form.
///
/// This recursively forms LCSSA for a loop nest.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSARecursively(Loop &L, DominatorTree &DT, LoopInfo *LI,
                          ScalarEvolution *SE);

/// \brief Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in
/// reverse depth first order w.r.t the DominatorTree. This allows us to visit
/// uses before definitions, allowing us to sink a loop body in one pass without
/// iteration. Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree,
/// DataLayout, TargetLibraryInfo, Loop, AliasSet information for all
/// instructions of the loop and loop safety information as
/// arguments. Diagnostics is emitted via \p ORE. It returns changed status.
bool sinkRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                TargetLibraryInfo *, TargetTransformInfo *, Loop *,
                AliasSetTracker *, LoopSafetyInfo *,
                OptimizationRemarkEmitter *ORE);

/// \brief Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
/// Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree, DataLayout,
/// TargetLibraryInfo, Loop, AliasSet information for all instructions of the
/// loop and loop safety information as arguments. Diagnostics is emitted via \p
/// ORE. It returns changed status.
bool hoistRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                 TargetLibraryInfo *, Loop *, AliasSetTracker *,
                 LoopSafetyInfo *, OptimizationRemarkEmitter *ORE);

/// This function deletes dead loops. The caller of this function needs to
/// guarantee that the loop is infact dead.
/// The function requires a bunch or prerequisites to be present:
///   - The loop needs to be in LCSSA form
///   - The loop needs to have a Preheader
///   - A unique dedicated exit block must exist
///
/// This also updates the relevant analysis information in \p DT, \p SE, and \p
/// LI if pointers to those are provided.
/// It also updates the loop PM if an updater struct is provided.

void deleteDeadLoop(Loop *L, DominatorTree *DT, ScalarEvolution *SE,
                    LoopInfo *LI);

/// \brief Try to promote memory values to scalars by sinking stores out of
/// the loop and moving loads to before the loop.  We do this by looping over
/// the stores in the loop, looking for stores to Must pointers which are
/// loop invariant. It takes a set of must-alias values, Loop exit blocks
/// vector, loop exit blocks insertion point vector, PredIteratorCache,
/// LoopInfo, DominatorTree, Loop, AliasSet information for all instructions
/// of the loop and loop safety information as arguments.
/// Diagnostics is emitted via \p ORE. It returns changed status.
bool promoteLoopAccessesToScalars(const SmallSetVector<Value *, 8> &,
                                  SmallVectorImpl<BasicBlock *> &,
                                  SmallVectorImpl<Instruction *> &,
                                  PredIteratorCache &, LoopInfo *,
                                  DominatorTree *, const TargetLibraryInfo *,
                                  Loop *, AliasSetTracker *, LoopSafetyInfo *,
                                  OptimizationRemarkEmitter *);

/// Does a BFS from a given node to all of its children inside a given loop.
/// The returned vector of nodes includes the starting point.
SmallVector<DomTreeNode *, 16> collectChildrenInLoop(DomTreeNode *N,
                                                     const Loop *CurLoop);

/// \brief Returns the instructions that use values defined in the loop.
SmallVector<Instruction *, 8> findDefsUsedOutsideOfLoop(Loop *L);

/// \brief Find string metadata for loop
///
/// If it has a value (e.g. {"llvm.distribute", 1} return the value as an
/// operand or null otherwise.  If the string metadata is not found return
/// Optional's not-a-value.
Optional<const MDOperand *> findStringMetadataForLoop(Loop *TheLoop,
                                                      StringRef Name);

/// \brief Set input string into loop metadata by keeping other values intact.
void addStringMetadataToLoop(Loop *TheLoop, const char *MDString,
                             unsigned V = 0);

/// \brief Get a loop's estimated trip count based on branch weight metadata.
/// Returns 0 when the count is estimated to be 0, or None when a meaningful
/// estimate can not be made.
Optional<unsigned> getLoopEstimatedTripCount(Loop *L);

/// Helper to consistently add the set of standard passes to a loop pass's \c
/// AnalysisUsage.
///
/// All loop passes should call this as part of implementing their \c
/// getAnalysisUsage.
void getLoopAnalysisUsage(AnalysisUsage &AU);

/// Returns true if the hoister and sinker can handle this instruction.
/// If SafetyInfo is null, we are checking for sinking instructions from
/// preheader to loop body (no speculation).
/// If SafetyInfo is not null, we are checking for hoisting/sinking
/// instructions from loop body to preheader/exit. Check if the instruction
/// can execute speculatively.
/// If \p ORE is set use it to emit optimization remarks.
bool canSinkOrHoistInst(Instruction &I, AAResults *AA, DominatorTree *DT,
                        Loop *CurLoop, AliasSetTracker *CurAST,
                        LoopSafetyInfo *SafetyInfo,
                        OptimizationRemarkEmitter *ORE = nullptr);

/// Generates an ordered vector reduction using extracts to reduce the value.
Value *
getOrderedReduction(IRBuilder<> &Builder, Value *Acc, Value *Src, unsigned Op,
                    RecurrenceDescriptor::MinMaxRecurrenceKind MinMaxKind =
                        RecurrenceDescriptor::MRK_Invalid,
                    ArrayRef<Value *> RedOps = None);

/// Generates a vector reduction using shufflevectors to reduce the value.
Value *getShuffleReduction(IRBuilder<> &Builder, Value *Src, unsigned Op,
                           RecurrenceDescriptor::MinMaxRecurrenceKind
                               MinMaxKind = RecurrenceDescriptor::MRK_Invalid,
                           ArrayRef<Value *> RedOps = None);

/// Create a target reduction of the given vector. The reduction operation
/// is described by the \p Opcode parameter. min/max reductions require
/// additional information supplied in \p Flags.
/// The target is queried to determine if intrinsics or shuffle sequences are
/// required to implement the reduction.
Value *
createSimpleTargetReduction(IRBuilder<> &B, const TargetTransformInfo *TTI,
                            unsigned Opcode, Value *Src,
                            TargetTransformInfo::ReductionFlags Flags =
                                TargetTransformInfo::ReductionFlags(),
                            ArrayRef<Value *> RedOps = None);

/// Create a generic target reduction using a recurrence descriptor \p Desc
/// The target is queried to determine if intrinsics or shuffle sequences are
/// required to implement the reduction.
Value *createTargetReduction(IRBuilder<> &B, const TargetTransformInfo *TTI,
                             RecurrenceDescriptor &Desc, Value *Src,
                             bool NoNaN = false);

/// Get the intersection (logical and) of all of the potential IR flags
/// of each scalar operation (VL) that will be converted into a vector (I).
/// If OpValue is non-null, we only consider operations similar to OpValue
/// when intersecting.
/// Flag set: NSW, NUW, exact, and all of fast-math.
void propagateIRFlags(Value *I, ArrayRef<Value *> VL, Value *OpValue = nullptr);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
