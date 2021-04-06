//===- llvm/Analysis/IVDescriptors.h - IndVar Descriptors -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file "describes" induction and recurrence variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IVDESCRIPTORS_H
#define LLVM_ANALYSIS_IVDESCRIPTORS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Casting.h"

namespace llvm {

class DemandedBits;
class AssumptionCache;
class Loop;
class PredicatedScalarEvolution;
class ScalarEvolution;
class SCEV;
class DominatorTree;

/// These are the kinds of recurrences that we support.
enum class RecurKind {
  None,   ///< Not a recurrence.
  Add,    ///< Sum of integers.
  Mul,    ///< Product of integers.
  Or,     ///< Bitwise or logical OR of integers.
  And,    ///< Bitwise or logical AND of integers.
  Xor,    ///< Bitwise or logical XOR of integers.
  SMin,   ///< Signed integer min implemented in terms of select(cmp()).
  SMax,   ///< Signed integer max implemented in terms of select(cmp()).
  UMin,   ///< Unisgned integer min implemented in terms of select(cmp()).
  UMax,   ///< Unsigned integer max implemented in terms of select(cmp()).
  FAdd,   ///< Sum of floats.
  FMul,   ///< Product of floats.
  FMin,   ///< FP min implemented in terms of select(cmp()).
  FMax    ///< FP max implemented in terms of select(cmp()).
};

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
  RecurrenceDescriptor() = default;

  RecurrenceDescriptor(Value *Start, Instruction *Exit, RecurKind K,
                       FastMathFlags FMF, Instruction *ExactFP, Type *RT,
                       bool Signed, SmallPtrSetImpl<Instruction *> &CI)
      : StartValue(Start), LoopExitInstr(Exit), Kind(K), FMF(FMF),
        ExactFPMathInst(ExactFP), RecurrenceType(RT), IsSigned(Signed) {
    CastInsts.insert(CI.begin(), CI.end());
  }

  /// This POD struct holds information about a potential recurrence operation.
  class InstDesc {
  public:
    InstDesc(bool IsRecur, Instruction *I, Instruction *ExactFP = nullptr)
        : IsRecurrence(IsRecur), PatternLastInst(I),
          RecKind(RecurKind::None), ExactFPMathInst(ExactFP) {}

    InstDesc(Instruction *I, RecurKind K, Instruction *ExactFP = nullptr)
        : IsRecurrence(true), PatternLastInst(I), RecKind(K),
          ExactFPMathInst(ExactFP) {}

    bool isRecurrence() const { return IsRecurrence; }

    bool needsExactFPMath() const { return ExactFPMathInst != nullptr; }

    Instruction *getExactFPMathInst() const { return ExactFPMathInst; }

    RecurKind getRecKind() const { return RecKind; }

    Instruction *getPatternInst() const { return PatternLastInst; }

  private:
    // Is this instruction a recurrence candidate.
    bool IsRecurrence;
    // The last instruction in a min/max pattern (select of the select(icmp())
    // pattern), or the current recurrence instruction otherwise.
    Instruction *PatternLastInst;
    // If this is a min/max pattern.
    RecurKind RecKind;
    // Recurrence does not allow floating-point reassociation.
    Instruction *ExactFPMathInst;
  };

  /// Returns a struct describing if the instruction 'I' can be a recurrence
  /// variable of type 'Kind'. If the recurrence is a min/max pattern of
  /// select(icmp()) this function advances the instruction pointer 'I' from the
  /// compare instruction to the select instruction and stores this pointer in
  /// 'PatternLastInst' member of the returned struct.
  static InstDesc isRecurrenceInstr(Instruction *I, RecurKind Kind,
                                    InstDesc &Prev, FastMathFlags FMF);

  /// Returns true if instruction I has multiple uses in Insts
  static bool hasMultipleUsesOf(Instruction *I,
                                SmallPtrSetImpl<Instruction *> &Insts,
                                unsigned MaxNumUses);

  /// Returns true if all uses of the instruction I is within the Set.
  static bool areAllUsesIn(Instruction *I, SmallPtrSetImpl<Instruction *> &Set);

  /// Returns a struct describing if the instruction is a
  /// Select(ICmp(X, Y), X, Y) instruction pattern corresponding to a min(X, Y)
  /// or max(X, Y). \p Prev specifies the description of an already processed
  /// select instruction, so its corresponding cmp can be matched to it.
  static InstDesc isMinMaxSelectCmpPattern(Instruction *I,
                                           const InstDesc &Prev);

  /// Returns a struct describing if the instruction is a
  /// Select(FCmp(X, Y), (Z = X op PHINode), PHINode) instruction pattern.
  static InstDesc isConditionalRdxPattern(RecurKind Kind, Instruction *I);

  /// Returns identity corresponding to the RecurrenceKind.
  static Constant *getRecurrenceIdentity(RecurKind K, Type *Tp,
                                         FastMathFlags FMF);

  /// Returns the opcode corresponding to the RecurrenceKind.
  static unsigned getOpcode(RecurKind Kind);

  /// Returns true if Phi is a reduction of type Kind and adds it to the
  /// RecurrenceDescriptor. If either \p DB is non-null or \p AC and \p DT are
  /// non-null, the minimal bit width needed to compute the reduction will be
  /// computed.
  static bool AddReductionVar(PHINode *Phi, RecurKind Kind, Loop *TheLoop,
                              FastMathFlags FMF,
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

  RecurKind getRecurrenceKind() const { return Kind; }

  unsigned getOpcode() const { return getOpcode(getRecurrenceKind()); }

  FastMathFlags getFastMathFlags() const { return FMF; }

  TrackingVH<Value> getRecurrenceStartValue() const { return StartValue; }

  Instruction *getLoopExitInstr() const { return LoopExitInstr; }

  /// Returns true if the recurrence has floating-point math that requires
  /// precise (ordered) operations.
  bool hasExactFPMath() const { return ExactFPMathInst != nullptr; }

  /// Returns 1st non-reassociative FP instruction in the PHI node's use-chain.
  Instruction *getExactFPMathInst() const { return ExactFPMathInst; }

  /// Returns true if the recurrence kind is an integer kind.
  static bool isIntegerRecurrenceKind(RecurKind Kind);

  /// Returns true if the recurrence kind is a floating point kind.
  static bool isFloatingPointRecurrenceKind(RecurKind Kind);

  /// Returns true if the recurrence kind is an arithmetic kind.
  static bool isArithmeticRecurrenceKind(RecurKind Kind);

  /// Returns true if the recurrence kind is an integer min/max kind.
  static bool isIntMinMaxRecurrenceKind(RecurKind Kind) {
    return Kind == RecurKind::UMin || Kind == RecurKind::UMax ||
           Kind == RecurKind::SMin || Kind == RecurKind::SMax;
  }

  /// Returns true if the recurrence kind is a floating-point min/max kind.
  static bool isFPMinMaxRecurrenceKind(RecurKind Kind) {
    return Kind == RecurKind::FMin || Kind == RecurKind::FMax;
  }

  /// Returns true if the recurrence kind is any min/max kind.
  static bool isMinMaxRecurrenceKind(RecurKind Kind) {
    return isIntMinMaxRecurrenceKind(Kind) || isFPMinMaxRecurrenceKind(Kind);
  }

  /// Returns the type of the recurrence. This type can be narrower than the
  /// actual type of the Phi if the recurrence has been type-promoted.
  Type *getRecurrenceType() const { return RecurrenceType; }

  /// Returns a reference to the instructions used for type-promoting the
  /// recurrence.
  const SmallPtrSet<Instruction *, 8> &getCastInsts() const { return CastInsts; }

  /// Returns true if all source operands of the recurrence are SExtInsts.
  bool isSigned() const { return IsSigned; }

  /// Attempts to find a chain of operations from Phi to LoopExitInst that can
  /// be treated as a set of reductions instructions for in-loop reductions.
  SmallVector<Instruction *, 4> getReductionOpChain(PHINode *Phi,
                                                    Loop *L) const;

private:
  // The starting value of the recurrence.
  // It does not have to be zero!
  TrackingVH<Value> StartValue;
  // The instruction who's value is used outside the loop.
  Instruction *LoopExitInstr = nullptr;
  // The kind of the recurrence.
  RecurKind Kind = RecurKind::None;
  // The fast-math flags on the recurrent instructions.  We propagate these
  // fast-math flags into the vectorized FP instructions we generate.
  FastMathFlags FMF;
  // First instance of non-reassociative floating-point in the PHI's use-chain.
  Instruction *ExactFPMathInst = nullptr;
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

  Value *getStartValue() const { return StartValue; }
  InductionKind getKind() const { return IK; }
  const SCEV *getStep() const { return Step; }
  BinaryOperator *getInductionBinOp() const { return InductionBinOp; }
  ConstantInt *getConstIntStepValue() const;

  /// Returns true if \p Phi is an induction in the loop \p L. If \p Phi is an
  /// induction, the induction descriptor \p D will contain the data describing
  /// this induction. If by some other means the caller has a better SCEV
  /// expression for \p Phi than the one returned by the ScalarEvolution
  /// analysis, it can be passed through \p Expr. If the def-use chain
  /// associated with the phi includes casts (that we know we can ignore
  /// under proper runtime checks), they are passed through \p CastsToIgnore.
  static bool
  isInductionPHI(PHINode *Phi, const Loop *L, ScalarEvolution *SE,
                 InductionDescriptor &D, const SCEV *Expr = nullptr,
                 SmallVectorImpl<Instruction *> *CastsToIgnore = nullptr);

  /// Returns true if \p Phi is a floating point induction in the loop \p L.
  /// If \p Phi is an induction, the induction descriptor \p D will contain
  /// the data describing this induction.
  static bool isFPInductionPHI(PHINode *Phi, const Loop *L, ScalarEvolution *SE,
                               InductionDescriptor &D);

  /// Returns true if \p Phi is a loop \p L induction, in the context associated
  /// with the run-time predicate of PSE. If \p Assume is true, this can add
  /// further SCEV predicates to \p PSE in order to prove that \p Phi is an
  /// induction.
  /// If \p Phi is an induction, \p D will contain the data describing this
  /// induction.
  static bool isInductionPHI(PHINode *Phi, const Loop *L,
                             PredicatedScalarEvolution &PSE,
                             InductionDescriptor &D, bool Assume = false);

  /// Returns floating-point induction operator that does not allow
  /// reassociation (transforming the induction requires an override of normal
  /// floating-point rules).
  Instruction *getExactFPMathInst() {
    if (IK == IK_FpInduction && InductionBinOp &&
        !InductionBinOp->hasAllowReassoc())
      return InductionBinOp;
    return nullptr;
  }

  /// Returns binary opcode of the induction operator.
  Instruction::BinaryOps getInductionOpcode() const {
    return InductionBinOp ? InductionBinOp->getOpcode()
                          : Instruction::BinaryOpsEnd;
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

} // end namespace llvm

#endif // LLVM_ANALYSIS_IVDESCRIPTORS_H
