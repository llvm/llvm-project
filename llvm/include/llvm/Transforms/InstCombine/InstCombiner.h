//===- InstCombiner.h - InstCombine implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the interface for the instcombine pass implementation.
/// The interface is used for generic transformations in this folder and
/// target specific combinations in the targets.
/// The visitor implementation is in \c InstCombinerImpl in
/// \c InstCombineInternal.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINER_H
#define LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINER_H

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"
#include <cassert>

#define DEBUG_TYPE "instcombine"

namespace llvm {

class AAResults;
class AssumptionCache;
class ProfileSummaryInfo;
class TargetLibraryInfo;
class TargetTransformInfo;

/// The core instruction combiner logic.
///
/// This class provides both the logic to recursively visit instructions and
/// combine them.
class LLVM_LIBRARY_VISIBILITY InstCombiner {
  /// Only used to call target specific inst combining.
  TargetTransformInfo &TTI;

public:
  /// Maximum size of array considered when transforming.
  uint64_t MaxArraySizeForCombine = 0;

  /// An IRBuilder that automatically inserts new instructions into the
  /// worklist.
  using BuilderTy = IRBuilder<TargetFolder, IRBuilderCallbackInserter>;
  BuilderTy &Builder;

protected:
  /// A worklist of the instructions that need to be simplified.
  InstCombineWorklist &Worklist;

  // Mode in which we are running the combiner.
  const bool MinimizeSize;

  AAResults *AA;

  // Required analyses.
  AssumptionCache &AC;
  TargetLibraryInfo &TLI;
  DominatorTree &DT;
  const DataLayout &DL;
  const SimplifyQuery SQ;
  OptimizationRemarkEmitter &ORE;
  BlockFrequencyInfo *BFI;
  ProfileSummaryInfo *PSI;

  // Optional analyses. When non-null, these can both be used to do better
  // combining and will be updated to reflect any changes.
  LoopInfo *LI;

  bool MadeIRChange = false;

public:
  InstCombiner(InstCombineWorklist &Worklist, BuilderTy &Builder,
               bool MinimizeSize, AAResults *AA, AssumptionCache &AC,
               TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
               DominatorTree &DT, OptimizationRemarkEmitter &ORE,
               BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
               const DataLayout &DL, LoopInfo *LI)
      : TTI(TTI), Builder(Builder), Worklist(Worklist),
        MinimizeSize(MinimizeSize), AA(AA), AC(AC), TLI(TLI), DT(DT), DL(DL),
        SQ(DL, &TLI, &DT, &AC), ORE(ORE), BFI(BFI), PSI(PSI), LI(LI) {}

  virtual ~InstCombiner() {}

  /// Return the source operand of a potentially bitcasted value while
  /// optionally checking if it has one use. If there is no bitcast or the one
  /// use check is not met, return the input value itself.
  static Value *peekThroughBitcast(Value *V, bool OneUseOnly = false) {
    if (auto *BitCast = dyn_cast<BitCastInst>(V))
      if (!OneUseOnly || BitCast->hasOneUse())
        return BitCast->getOperand(0);

    // V is not a bitcast or V has more than one use and OneUseOnly is true.
    return V;
  }

  /// Assign a complexity or rank value to LLVM Values. This is used to reduce
  /// the amount of pattern matching needed for compares and commutative
  /// instructions. For example, if we have:
  ///   icmp ugt X, Constant
  /// or
  ///   xor (add X, Constant), cast Z
  ///
  /// We do not have to consider the commuted variants of these patterns because
  /// canonicalization based on complexity guarantees the above ordering.
  ///
  /// This routine maps IR values to various complexity ranks:
  ///   0 -> undef
  ///   1 -> Constants
  ///   2 -> Other non-instructions
  ///   3 -> Arguments
  ///   4 -> Cast and (f)neg/not instructions
  ///   5 -> Other instructions
  static unsigned getComplexity(Value *V) {
    if (isa<Instruction>(V)) {
      if (isa<CastInst>(V) || match(V, m_Neg(PatternMatch::m_Value())) ||
          match(V, m_Not(PatternMatch::m_Value())) ||
          match(V, m_FNeg(PatternMatch::m_Value())))
        return 4;
      return 5;
    }
    if (isa<Argument>(V))
      return 3;
    return isa<Constant>(V) ? (isa<UndefValue>(V) ? 0 : 1) : 2;
  }

  /// Predicate canonicalization reduces the number of patterns that need to be
  /// matched by other transforms. For example, we may swap the operands of a
  /// conditional branch or select to create a compare with a canonical
  /// (inverted) predicate which is then more likely to be matched with other
  /// values.
  static bool isCanonicalPredicate(CmpInst::Predicate Pred) {
    switch (Pred) {
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SLE:
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_SGE:
    // TODO: There are 16 FCMP predicates. Should others be (not) canonical?
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_OGE:
      return false;
    default:
      return true;
    }
  }

  /// Given an exploded icmp instruction, return true if the comparison only
  /// checks the sign bit. If it only checks the sign bit, set TrueIfSigned if
  /// the result of the comparison is true when the input value is signed.
  static bool isSignBitCheck(ICmpInst::Predicate Pred, const APInt &RHS,
                             bool &TrueIfSigned) {
    switch (Pred) {
    case ICmpInst::ICMP_SLT: // True if LHS s< 0
      TrueIfSigned = true;
      return RHS.isNullValue();
    case ICmpInst::ICMP_SLE: // True if LHS s<= -1
      TrueIfSigned = true;
      return RHS.isAllOnesValue();
    case ICmpInst::ICMP_SGT: // True if LHS s> -1
      TrueIfSigned = false;
      return RHS.isAllOnesValue();
    case ICmpInst::ICMP_SGE: // True if LHS s>= 0
      TrueIfSigned = false;
      return RHS.isNullValue();
    case ICmpInst::ICMP_UGT:
      // True if LHS u> RHS and RHS == sign-bit-mask - 1
      TrueIfSigned = true;
      return RHS.isMaxSignedValue();
    case ICmpInst::ICMP_UGE:
      // True if LHS u>= RHS and RHS == sign-bit-mask (2^7, 2^15, 2^31, etc)
      TrueIfSigned = true;
      return RHS.isMinSignedValue();
    case ICmpInst::ICMP_ULT:
      // True if LHS u< RHS and RHS == sign-bit-mask (2^7, 2^15, 2^31, etc)
      TrueIfSigned = false;
      return RHS.isMinSignedValue();
    case ICmpInst::ICMP_ULE:
      // True if LHS u<= RHS and RHS == sign-bit-mask - 1
      TrueIfSigned = false;
      return RHS.isMaxSignedValue();
    default:
      return false;
    }
  }

  /// Add one to a Constant
  static Constant *AddOne(Constant *C) {
    return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
  }

  /// Subtract one from a Constant
  static Constant *SubOne(Constant *C) {
    return ConstantExpr::getSub(C, ConstantInt::get(C->getType(), 1));
  }

  llvm::Optional<std::pair<
      CmpInst::Predicate,
      Constant *>> static getFlippedStrictnessPredicateAndConstant(CmpInst::
                                                                       Predicate
                                                                           Pred,
                                                                   Constant *C);

  /// Return true if the specified value is free to invert (apply ~ to).
  /// This happens in cases where the ~ can be eliminated.  If WillInvertAllUses
  /// is true, work under the assumption that the caller intends to remove all
  /// uses of V and only keep uses of ~V.
  ///
  /// See also: canFreelyInvertAllUsersOf()
  static bool isFreeToInvert(Value *V, bool WillInvertAllUses) {
    // ~(~(X)) -> X.
    if (match(V, m_Not(PatternMatch::m_Value())))
      return true;

    // Constants can be considered to be not'ed values.
    if (match(V, PatternMatch::m_AnyIntegralConstant()))
      return true;

    // Compares can be inverted if all of their uses are being modified to use
    // the ~V.
    if (isa<CmpInst>(V))
      return WillInvertAllUses;

    // If `V` is of the form `A + Constant` then `-1 - V` can be folded into
    // `(-1 - Constant) - A` if we are willing to invert all of the uses.
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V))
      if (BO->getOpcode() == Instruction::Add ||
          BO->getOpcode() == Instruction::Sub)
        if (isa<Constant>(BO->getOperand(0)) ||
            isa<Constant>(BO->getOperand(1)))
          return WillInvertAllUses;

    // Selects with invertible operands are freely invertible
    if (match(V,
              m_Select(PatternMatch::m_Value(), m_Not(PatternMatch::m_Value()),
                       m_Not(PatternMatch::m_Value()))))
      return WillInvertAllUses;

    return false;
  }

  /// Given i1 V, can every user of V be freely adapted if V is changed to !V ?
  /// InstCombine's canonicalizeICmpPredicate() must be kept in sync with this
  /// fn.
  ///
  /// See also: isFreeToInvert()
  static bool canFreelyInvertAllUsersOf(Value *V, Value *IgnoredUser) {
    // Look at every user of V.
    for (Use &U : V->uses()) {
      if (U.getUser() == IgnoredUser)
        continue; // Don't consider this user.

      auto *I = cast<Instruction>(U.getUser());
      switch (I->getOpcode()) {
      case Instruction::Select:
        if (U.getOperandNo() != 0) // Only if the value is used as select cond.
          return false;
        break;
      case Instruction::Br:
        assert(U.getOperandNo() == 0 && "Must be branching on that value.");
        break; // Free to invert by swapping true/false values/destinations.
      case Instruction::Xor: // Can invert 'xor' if it's a 'not', by ignoring
                             // it.
        if (!match(I, m_Not(PatternMatch::m_Value())))
          return false; // Not a 'not'.
        break;
      default:
        return false; // Don't know, likely not freely invertible.
      }
      // So far all users were free to invert...
    }
    return true; // Can freely invert all users!
  }

  /// Some binary operators require special handling to avoid poison and
  /// undefined behavior. If a constant vector has undef elements, replace those
  /// undefs with identity constants if possible because those are always safe
  /// to execute. If no identity constant exists, replace undef with some other
  /// safe constant.
  static Constant *
  getSafeVectorConstantForBinop(BinaryOperator::BinaryOps Opcode, Constant *In,
                                bool IsRHSConstant) {
    auto *InVTy = cast<FixedVectorType>(In->getType());

    Type *EltTy = InVTy->getElementType();
    auto *SafeC = ConstantExpr::getBinOpIdentity(Opcode, EltTy, IsRHSConstant);
    if (!SafeC) {
      // TODO: Should this be available as a constant utility function? It is
      // similar to getBinOpAbsorber().
      if (IsRHSConstant) {
        switch (Opcode) {
        case Instruction::SRem: // X % 1 = 0
        case Instruction::URem: // X %u 1 = 0
          SafeC = ConstantInt::get(EltTy, 1);
          break;
        case Instruction::FRem: // X % 1.0 (doesn't simplify, but it is safe)
          SafeC = ConstantFP::get(EltTy, 1.0);
          break;
        default:
          llvm_unreachable(
              "Only rem opcodes have no identity constant for RHS");
        }
      } else {
        switch (Opcode) {
        case Instruction::Shl:  // 0 << X = 0
        case Instruction::LShr: // 0 >>u X = 0
        case Instruction::AShr: // 0 >> X = 0
        case Instruction::SDiv: // 0 / X = 0
        case Instruction::UDiv: // 0 /u X = 0
        case Instruction::SRem: // 0 % X = 0
        case Instruction::URem: // 0 %u X = 0
        case Instruction::Sub:  // 0 - X (doesn't simplify, but it is safe)
        case Instruction::FSub: // 0.0 - X (doesn't simplify, but it is safe)
        case Instruction::FDiv: // 0.0 / X (doesn't simplify, but it is safe)
        case Instruction::FRem: // 0.0 % X = 0
          SafeC = Constant::getNullValue(EltTy);
          break;
        default:
          llvm_unreachable("Expected to find identity constant for opcode");
        }
      }
    }
    assert(SafeC && "Must have safe constant for binop");
    unsigned NumElts = InVTy->getNumElements();
    SmallVector<Constant *, 16> Out(NumElts);
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *C = In->getAggregateElement(i);
      Out[i] = isa<UndefValue>(C) ? SafeC : C;
    }
    return ConstantVector::get(Out);
  }

  /// Create and insert the idiom we use to indicate a block is unreachable
  /// without having to rewrite the CFG from within InstCombine.
  static void CreateNonTerminatorUnreachable(Instruction *InsertAt) {
    auto &Ctx = InsertAt->getContext();
    new StoreInst(ConstantInt::getTrue(Ctx),
                  UndefValue::get(Type::getInt1PtrTy(Ctx)), InsertAt);
  }

  void addToWorklist(Instruction *I) { Worklist.push(I); }

  AssumptionCache &getAssumptionCache() const { return AC; }
  TargetLibraryInfo &getTargetLibraryInfo() const { return TLI; }
  DominatorTree &getDominatorTree() const { return DT; }
  const DataLayout &getDataLayout() const { return DL; }
  const SimplifyQuery &getSimplifyQuery() const { return SQ; }
  OptimizationRemarkEmitter &getOptimizationRemarkEmitter() const {
    return ORE;
  }
  BlockFrequencyInfo *getBlockFrequencyInfo() const { return BFI; }
  ProfileSummaryInfo *getProfileSummaryInfo() const { return PSI; }
  LoopInfo *getLoopInfo() const { return LI; }

  // Call target specific combiners
  Optional<Instruction *> targetInstCombineIntrinsic(IntrinsicInst &II);
  Optional<Value *>
  targetSimplifyDemandedUseBitsIntrinsic(IntrinsicInst &II, APInt DemandedMask,
                                         KnownBits &Known,
                                         bool &KnownBitsComputed);
  Optional<Value *> targetSimplifyDemandedVectorEltsIntrinsic(
      IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp);

  /// Inserts an instruction \p New before instruction \p Old
  ///
  /// Also adds the new instruction to the worklist and returns \p New so that
  /// it is suitable for use as the return from the visitation patterns.
  Instruction *InsertNewInstBefore(Instruction *New, Instruction &Old) {
    assert(New && !New->getParent() &&
           "New instruction already inserted into a basic block!");
    BasicBlock *BB = Old.getParent();
    BB->getInstList().insert(Old.getIterator(), New); // Insert inst
    Worklist.push(New);
    return New;
  }

  /// Same as InsertNewInstBefore, but also sets the debug loc.
  Instruction *InsertNewInstWith(Instruction *New, Instruction &Old) {
    New->setDebugLoc(Old.getDebugLoc());
    return InsertNewInstBefore(New, Old);
  }

  /// A combiner-aware RAUW-like routine.
  ///
  /// This method is to be used when an instruction is found to be dead,
  /// replaceable with another preexisting expression. Here we add all uses of
  /// I to the worklist, replace all uses of I with the new value, then return
  /// I, so that the inst combiner will know that I was modified.
  Instruction *replaceInstUsesWith(Instruction &I, Value *V) {
    // If there are no uses to replace, then we return nullptr to indicate that
    // no changes were made to the program.
    if (I.use_empty())
      return nullptr;

    Worklist.pushUsersToWorkList(I); // Add all modified instrs to worklist.

    // If we are replacing the instruction with itself, this must be in a
    // segment of unreachable code, so just clobber the instruction.
    if (&I == V)
      V = UndefValue::get(I.getType());

    LLVM_DEBUG(dbgs() << "IC: Replacing " << I << "\n"
                      << "    with " << *V << '\n');

    I.replaceAllUsesWith(V);
    return &I;
  }

  /// Replace operand of instruction and add old operand to the worklist.
  Instruction *replaceOperand(Instruction &I, unsigned OpNum, Value *V) {
    Worklist.addValue(I.getOperand(OpNum));
    I.setOperand(OpNum, V);
    return &I;
  }

  /// Replace use and add the previously used value to the worklist.
  void replaceUse(Use &U, Value *NewValue) {
    Worklist.addValue(U);
    U = NewValue;
  }

  /// Combiner aware instruction erasure.
  ///
  /// When dealing with an instruction that has side effects or produces a void
  /// value, we can't rely on DCE to delete the instruction. Instead, visit
  /// methods should return the value returned by this function.
  virtual Instruction *eraseInstFromFunction(Instruction &I) = 0;

  void computeKnownBits(const Value *V, KnownBits &Known, unsigned Depth,
                        const Instruction *CxtI) const {
    llvm::computeKnownBits(V, Known, DL, Depth, &AC, CxtI, &DT);
  }

  KnownBits computeKnownBits(const Value *V, unsigned Depth,
                             const Instruction *CxtI) const {
    return llvm::computeKnownBits(V, DL, Depth, &AC, CxtI, &DT);
  }

  bool isKnownToBeAPowerOfTwo(const Value *V, bool OrZero = false,
                              unsigned Depth = 0,
                              const Instruction *CxtI = nullptr) {
    return llvm::isKnownToBeAPowerOfTwo(V, DL, OrZero, Depth, &AC, CxtI, &DT);
  }

  bool MaskedValueIsZero(const Value *V, const APInt &Mask, unsigned Depth = 0,
                         const Instruction *CxtI = nullptr) const {
    return llvm::MaskedValueIsZero(V, Mask, DL, Depth, &AC, CxtI, &DT);
  }

  unsigned ComputeNumSignBits(const Value *Op, unsigned Depth = 0,
                              const Instruction *CxtI = nullptr) const {
    return llvm::ComputeNumSignBits(Op, DL, Depth, &AC, CxtI, &DT);
  }

  OverflowResult computeOverflowForUnsignedMul(const Value *LHS,
                                               const Value *RHS,
                                               const Instruction *CxtI) const {
    return llvm::computeOverflowForUnsignedMul(LHS, RHS, DL, &AC, CxtI, &DT);
  }

  OverflowResult computeOverflowForSignedMul(const Value *LHS, const Value *RHS,
                                             const Instruction *CxtI) const {
    return llvm::computeOverflowForSignedMul(LHS, RHS, DL, &AC, CxtI, &DT);
  }

  OverflowResult computeOverflowForUnsignedAdd(const Value *LHS,
                                               const Value *RHS,
                                               const Instruction *CxtI) const {
    return llvm::computeOverflowForUnsignedAdd(LHS, RHS, DL, &AC, CxtI, &DT);
  }

  OverflowResult computeOverflowForSignedAdd(const Value *LHS, const Value *RHS,
                                             const Instruction *CxtI) const {
    return llvm::computeOverflowForSignedAdd(LHS, RHS, DL, &AC, CxtI, &DT);
  }

  OverflowResult computeOverflowForUnsignedSub(const Value *LHS,
                                               const Value *RHS,
                                               const Instruction *CxtI) const {
    return llvm::computeOverflowForUnsignedSub(LHS, RHS, DL, &AC, CxtI, &DT);
  }

  OverflowResult computeOverflowForSignedSub(const Value *LHS, const Value *RHS,
                                             const Instruction *CxtI) const {
    return llvm::computeOverflowForSignedSub(LHS, RHS, DL, &AC, CxtI, &DT);
  }

  virtual bool SimplifyDemandedBits(Instruction *I, unsigned OpNo,
                                    const APInt &DemandedMask, KnownBits &Known,
                                    unsigned Depth = 0) = 0;
  virtual Value *
  SimplifyDemandedVectorElts(Value *V, APInt DemandedElts, APInt &UndefElts,
                             unsigned Depth = 0,
                             bool AllowMultipleUsers = false) = 0;
};

} // namespace llvm

#undef DEBUG_TYPE

#endif
