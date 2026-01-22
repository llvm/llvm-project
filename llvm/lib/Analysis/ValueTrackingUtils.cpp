//===- ValueTrackingUtils.cpp - ValueTracking utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueTrackingUtils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"

using namespace llvm;
using namespace llvm::vtutils;
using namespace llvm::PatternMatch;

// Controls the number of uses of the value searched for possible
// dominating comparisons.
static cl::opt<unsigned> DomConditionsMaxUses("dom-conditions-max-uses",
                                              cl::Hidden, cl::init(20));

/// Maximum number of instructions to check between assume and context
/// instruction.
static constexpr unsigned MaxInstrsToCheckForFree = 16;

void llvm::computeKnownBitsFromRangeMetadata(const MDNode &Ranges,
                                             KnownBits &Known) {
  unsigned BitWidth = Known.getBitWidth();
  unsigned NumRanges = Ranges.getNumOperands() / 2;
  assert(NumRanges >= 1);

  Known.setAllConflict();

  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());
    // BitWidth must equal the Ranges BitWidth for the correct number of high
    // bits to be set.
    assert(BitWidth == Range.getBitWidth() &&
           "Known bit width must match range bit width!");

    // The first CommonPrefixBits of all values in Range are equal.
    unsigned CommonPrefixBits =
        (Range.getUnsignedMax() ^ Range.getUnsignedMin()).countl_zero();
    APInt Mask = APInt::getHighBitsSet(BitWidth, CommonPrefixBits);
    APInt UnsignedMax = Range.getUnsignedMax().zextOrTrunc(BitWidth);
    Known.One &= UnsignedMax & Mask;
    Known.Zero &= ~UnsignedMax & Mask;
  }
}

bool llvm::mayHaveNonDefUseDependency(const Instruction &I) {
  if (I.mayReadOrWriteMemory())
    // Memory dependency possible
    return true;
  if (!isSafeToSpeculativelyExecute(&I))
    // Can't move above a maythrow call or infinite loop.  Or if an
    // inalloca alloca, above a stacksave call.
    return true;
  if (!isGuaranteedToTransferExecutionToSuccessor(&I))
    // 1) Can't reorder two inf-loop calls, even if readonly
    // 2) Also can't reorder an inf-loop call below a instruction which isn't
    //    safe to speculative execute.  (Inverse of above)
    return true;
  return false;
}

// Is this an intrinsic that cannot be speculated but also cannot trap?
bool llvm::isAssumeLikeIntrinsic(const Instruction *I) {
  if (const IntrinsicInst *CI = dyn_cast<IntrinsicInst>(I))
    return CI->isAssumeLikeIntrinsic();

  return false;
}

bool llvm::isValidAssumeForContext(const Instruction *Inv,
                                   const Instruction *CxtI,
                                   const DominatorTree *DT,
                                   bool AllowEphemerals) {
  // There are two restrictions on the use of an assume:
  //  1. The assume must dominate the context (or the control flow must
  //     reach the assume whenever it reaches the context).
  //  2. The context must not be in the assume's set of ephemeral values
  //     (otherwise we will use the assume to prove that the condition
  //     feeding the assume is trivially true, thus causing the removal of
  //     the assume).

  if (Inv->getParent() == CxtI->getParent()) {
    // If Inv and CtxI are in the same block, check if the assume (Inv) is first
    // in the BB.
    if (Inv->comesBefore(CxtI))
      return true;

    // Don't let an assume affect itself - this would cause the problems
    // `isEphemeralValueOf` is trying to prevent, and it would also make
    // the loop below go out of bounds.
    if (!AllowEphemerals && Inv == CxtI)
      return false;

    // The context comes first, but they're both in the same block.
    // Make sure there is nothing in between that might interrupt
    // the control flow, not even CxtI itself.
    // We limit the scan distance between the assume and its context instruction
    // to avoid a compile-time explosion. This limit is chosen arbitrarily, so
    // it can be adjusted if needed (could be turned into a cl::opt).
    auto Range = make_range(CxtI->getIterator(), Inv->getIterator());
    if (!isGuaranteedToTransferExecutionToSuccessor(Range, 15))
      return false;

    return AllowEphemerals || !isEphemeralValueOf(Inv, CxtI);
  }

  // Inv and CxtI are in different blocks.
  if (DT) {
    if (DT->dominates(Inv, CxtI))
      return true;
  } else if (Inv->getParent() == CxtI->getParent()->getSinglePredecessor() ||
             Inv->getParent()->isEntryBlock()) {
    // We don't have a DT, but this trivially dominates.
    return true;
  }

  return false;
}

bool llvm::willNotFreeBetween(const Instruction *Assume,
                              const Instruction *CtxI) {
  // Helper to check if there are any calls in the range that may free memory.
  auto hasNoFreeCalls = [](auto Range) {
    for (const auto &[Idx, I] : enumerate(Range)) {
      if (Idx > MaxInstrsToCheckForFree)
        return false;
      if (const auto *CB = dyn_cast<CallBase>(&I))
        if (!CB->hasFnAttr(Attribute::NoFree))
          return false;
    }
    return true;
  };

  // Make sure the current function cannot arrange for another thread to free on
  // its behalf.
  if (!CtxI->getFunction()->hasNoSync())
    return false;

  // Handle cross-block case: CtxI in a successor of Assume's block.
  const BasicBlock *CtxBB = CtxI->getParent();
  const BasicBlock *AssumeBB = Assume->getParent();
  BasicBlock::const_iterator CtxIter = CtxI->getIterator();
  if (CtxBB != AssumeBB) {
    if (CtxBB->getSinglePredecessor() != AssumeBB)
      return false;

    if (!hasNoFreeCalls(make_range(CtxBB->begin(), CtxIter)))
      return false;

    CtxIter = AssumeBB->end();
  } else {
    // Same block case: check that Assume comes before CtxI.
    if (!Assume->comesBefore(CtxI))
      return false;
  }

  // Check if there are any calls between Assume and CtxIter that may free
  // memory.
  return hasNoFreeCalls(make_range(Assume->getIterator(), CtxIter));
}

bool llvm::isSafeToSpeculativelyExecute(
    const Instruction *Inst, const Instruction *CtxI, AssumptionCache *AC,
    const DominatorTree *DT, const TargetLibraryInfo *TLI, bool UseVariableInfo,
    bool IgnoreUBImplyingAttrs) {
  return isSafeToSpeculativelyExecuteWithOpcode(Inst->getOpcode(), Inst, CtxI,
                                                AC, DT, TLI, UseVariableInfo,
                                                IgnoreUBImplyingAttrs);
}

bool llvm::isSafeToSpeculativelyExecuteWithOpcode(
    unsigned Opcode, const Instruction *Inst, const Instruction *CtxI,
    AssumptionCache *AC, const DominatorTree *DT, const TargetLibraryInfo *TLI,
    bool UseVariableInfo, bool IgnoreUBImplyingAttrs) {
#ifndef NDEBUG
  if (Inst->getOpcode() != Opcode) {
    // Check that the operands are actually compatible with the Opcode override.
    auto hasEqualReturnAndLeadingOperandTypes =
        [](const Instruction *Inst, unsigned NumLeadingOperands) {
          if (Inst->getNumOperands() < NumLeadingOperands)
            return false;
          const Type *ExpectedType = Inst->getType();
          for (unsigned ItOp = 0; ItOp < NumLeadingOperands; ++ItOp)
            if (Inst->getOperand(ItOp)->getType() != ExpectedType)
              return false;
          return true;
        };
    assert(!Instruction::isBinaryOp(Opcode) ||
           hasEqualReturnAndLeadingOperandTypes(Inst, 2));
    assert(!Instruction::isUnaryOp(Opcode) ||
           hasEqualReturnAndLeadingOperandTypes(Inst, 1));
  }
#endif

  switch (Opcode) {
  default:
    return true;
  case Instruction::UDiv:
  case Instruction::URem: {
    // x / y is undefined if y == 0.
    const APInt *V;
    if (match(Inst->getOperand(1), m_APInt(V)))
      return *V != 0;
    return false;
  }
  case Instruction::SDiv:
  case Instruction::SRem: {
    // x / y is undefined if y == 0 or x == INT_MIN and y == -1
    const APInt *Numerator, *Denominator;
    if (!match(Inst->getOperand(1), m_APInt(Denominator)))
      return false;
    // We cannot hoist this division if the denominator is 0.
    if (*Denominator == 0)
      return false;
    // It's safe to hoist if the denominator is not 0 or -1.
    if (!Denominator->isAllOnes())
      return true;
    // At this point we know that the denominator is -1.  It is safe to hoist as
    // long we know that the numerator is not INT_MIN.
    if (match(Inst->getOperand(0), m_APInt(Numerator)))
      return !Numerator->isMinSignedValue();
    // The numerator *might* be MinSignedValue.
    return false;
  }
  case Instruction::Load: {
    if (!UseVariableInfo)
      return false;

    const LoadInst *LI = dyn_cast<LoadInst>(Inst);
    if (!LI)
      return false;
    if (mustSuppressSpeculation(*LI))
      return false;
    const DataLayout &DL = LI->getDataLayout();
    return isDereferenceableAndAlignedPointer(LI->getPointerOperand(),
                                              LI->getType(), LI->getAlign(), DL,
                                              CtxI, AC, DT, TLI);
  }
  case Instruction::Call: {
    auto *CI = dyn_cast<const CallInst>(Inst);
    if (!CI)
      return false;
    const Function *Callee = CI->getCalledFunction();

    // The called function could have undefined behavior or side-effects, even
    // if marked readnone nounwind.
    if (!Callee || !Callee->isSpeculatable())
      return false;
    // Since the operands may be changed after hoisting, undefined behavior may
    // be triggered by some UB-implying attributes.
    return IgnoreUBImplyingAttrs || !CI->hasUBImplyingAttrs();
  }
  case Instruction::VAArg:
  case Instruction::Alloca:
  case Instruction::Invoke:
  case Instruction::CallBr:
  case Instruction::PHI:
  case Instruction::Store:
  case Instruction::Ret:
  case Instruction::Br:
  case Instruction::IndirectBr:
  case Instruction::Switch:
  case Instruction::Unreachable:
  case Instruction::Fence:
  case Instruction::AtomicRMW:
  case Instruction::AtomicCmpXchg:
  case Instruction::LandingPad:
  case Instruction::Resume:
  case Instruction::CatchSwitch:
  case Instruction::CatchPad:
  case Instruction::CatchRet:
  case Instruction::CleanupPad:
  case Instruction::CleanupRet:
    return false; // Misc instructions which have effects
  }
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(const Instruction *I) {
  // Note: An atomic operation isn't guaranteed to return in a reasonable amount
  // of time because it's possible for another thread to interfere with it for
  // an arbitrary length of time, but programs aren't allowed to rely on that.

  // If there is no successor, then execution can't transfer to it.
  if (isa<ReturnInst>(I))
    return false;
  if (isa<UnreachableInst>(I))
    return false;

  // Note: Do not add new checks here; instead, change Instruction::mayThrow or
  // Instruction::willReturn.
  //
  // FIXME: Move this check into Instruction::willReturn.
  if (isa<CatchPadInst>(I)) {
    switch (classifyEHPersonality(I->getFunction()->getPersonalityFn())) {
    default:
      // A catchpad may invoke exception object constructors and such, which
      // in some languages can be arbitrary code, so be conservative by default.
      return false;
    case EHPersonality::CoreCLR:
      // For CoreCLR, it just involves a type test.
      return true;
    }
  }

  // An instruction that returns without throwing must transfer control flow
  // to a successor.
  return !I->mayThrow() && I->willReturn();
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(const BasicBlock *BB) {
  // TODO: This is slightly conservative for invoke instruction since exiting
  // via an exception *is* normal control for them.
  for (const Instruction &I : *BB)
    if (!isGuaranteedToTransferExecutionToSuccessor(&I))
      return false;
  return true;
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(
    BasicBlock::const_iterator Begin, BasicBlock::const_iterator End,
    unsigned ScanLimit) {
  return isGuaranteedToTransferExecutionToSuccessor(make_range(Begin, End),
                                                    ScanLimit);
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(
    iterator_range<BasicBlock::const_iterator> Range, unsigned ScanLimit) {
  assert(ScanLimit && "scan limit must be non-zero");
  for (const Instruction &I : Range) {
    if (--ScanLimit == 0)
      return false;
    if (!isGuaranteedToTransferExecutionToSuccessor(&I))
      return false;
  }
  return true;
}

bool llvm::isGuaranteedToExecuteForEveryIteration(const Instruction *I,
                                                  const Loop *L) {
  // The loop header is guaranteed to be executed for every iteration.
  //
  // FIXME: Relax this constraint to cover all basic blocks that are
  // guaranteed to be executed at every iteration.
  if (I->getParent() != L->getHeader())
    return false;

  for (const Instruction &LI : *L->getHeader()) {
    if (&LI == I)
      return true;
    if (!isGuaranteedToTransferExecutionToSuccessor(&LI))
      return false;
  }
  llvm_unreachable("Instruction not contained in its own parent basic block.");
}

bool llvm::intrinsicPropagatesPoison(Intrinsic::ID IID) {
  switch (IID) {
  // TODO: Add more intrinsics.
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::umul_with_overflow:
    // If an input is a vector containing a poison element, the
    // two output vectors (calculated results, overflow bits)'
    // corresponding lanes are poison.
    return true;
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::abs:
  case Intrinsic::smax:
  case Intrinsic::smin:
  case Intrinsic::umax:
  case Intrinsic::umin:
  case Intrinsic::scmp:
  case Intrinsic::is_fpclass:
  case Intrinsic::ptrmask:
  case Intrinsic::ucmp:
  case Intrinsic::bitreverse:
  case Intrinsic::bswap:
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat:
  case Intrinsic::sshl_sat:
  case Intrinsic::uadd_sat:
  case Intrinsic::usub_sat:
  case Intrinsic::ushl_sat:
  case Intrinsic::smul_fix:
  case Intrinsic::smul_fix_sat:
  case Intrinsic::umul_fix:
  case Intrinsic::umul_fix_sat:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::sin:
  case Intrinsic::sinh:
  case Intrinsic::cos:
  case Intrinsic::cosh:
  case Intrinsic::sincos:
  case Intrinsic::sincospi:
  case Intrinsic::tan:
  case Intrinsic::tanh:
  case Intrinsic::asin:
  case Intrinsic::acos:
  case Intrinsic::atan:
  case Intrinsic::atan2:
  case Intrinsic::canonicalize:
  case Intrinsic::sqrt:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::exp10:
  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::modf:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::lrint:
  case Intrinsic::llrint:
  case Intrinsic::fshl:
  case Intrinsic::fshr:
    return true;
  default:
    return false;
  }
}

bool llvm::propagatesPoison(const Use &PoisonOp) {
  const Operator *I = cast<Operator>(PoisonOp.getUser());
  switch (I->getOpcode()) {
  case Instruction::Freeze:
  case Instruction::PHI:
  case Instruction::Invoke:
    return false;
  case Instruction::Select:
    return PoisonOp.getOperandNo() == 0;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(I))
      return intrinsicPropagatesPoison(II->getIntrinsicID());
    return false;
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::GetElementPtr:
    return true;
  default:
    if (isa<BinaryOperator>(I) || isa<UnaryOperator>(I) || isa<CastInst>(I))
      return true;

    // Be conservative and return false.
    return false;
  }
}

/// Enumerates all operands of \p I that are guaranteed to not be undef or
/// poison. If the callback \p Handle returns true, stop processing and return
/// true. Otherwise, return false.
template <typename CallableT>
static bool handleGuaranteedWellDefinedOps(const Instruction *I,
                                           const CallableT &Handle) {
  switch (I->getOpcode()) {
  case Instruction::Store:
    if (Handle(cast<StoreInst>(I)->getPointerOperand()))
      return true;
    break;

  case Instruction::Load:
    if (Handle(cast<LoadInst>(I)->getPointerOperand()))
      return true;
    break;

  // Since dereferenceable attribute imply noundef, atomic operations
  // also implicitly have noundef pointers too
  case Instruction::AtomicCmpXchg:
    if (Handle(cast<AtomicCmpXchgInst>(I)->getPointerOperand()))
      return true;
    break;

  case Instruction::AtomicRMW:
    if (Handle(cast<AtomicRMWInst>(I)->getPointerOperand()))
      return true;
    break;

  case Instruction::Call:
  case Instruction::Invoke: {
    const CallBase *CB = cast<CallBase>(I);
    if (CB->isIndirectCall() && Handle(CB->getCalledOperand()))
      return true;
    for (unsigned i = 0; i < CB->arg_size(); ++i)
      if ((CB->paramHasAttr(i, Attribute::NoUndef) ||
           CB->paramHasAttr(i, Attribute::Dereferenceable) ||
           CB->paramHasAttr(i, Attribute::DereferenceableOrNull)) &&
          Handle(CB->getArgOperand(i)))
        return true;
    break;
  }
  case Instruction::Ret:
    if (I->getFunction()->hasRetAttribute(Attribute::NoUndef) &&
        Handle(I->getOperand(0)))
      return true;
    break;
  case Instruction::Switch:
    if (Handle(cast<SwitchInst>(I)->getCondition()))
      return true;
    break;
  case Instruction::Br: {
    auto *BR = cast<BranchInst>(I);
    if (BR->isConditional() && Handle(BR->getCondition()))
      return true;
    break;
  }
  default:
    break;
  }

  return false;
}

/// Enumerates all operands of \p I that are guaranteed to not be poison.
template <typename CallableT>
static bool handleGuaranteedNonPoisonOps(const Instruction *I,
                                         const CallableT &Handle) {
  if (handleGuaranteedWellDefinedOps(I, Handle))
    return true;
  switch (I->getOpcode()) {
  // Divisors of these operations are allowed to be partially undef.
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    return Handle(I->getOperand(1));
  default:
    return false;
  }
}

bool llvm::mustTriggerUB(const Instruction *I,
                         const SmallPtrSetImpl<const Value *> &KnownPoison) {
  return handleGuaranteedNonPoisonOps(
      I, [&](const Value *V) { return KnownPoison.count(V); });
}

/// Return true if undefined behavior would provably be executed on the path to
/// OnPathTo if Root produced a posion result.  Note that this doesn't say
/// anything about whether OnPathTo is actually executed or whether Root is
/// actually poison.  This can be used to assess whether a new use of Root can
/// be added at a location which is control equivalent with OnPathTo (such as
/// immediately before it) without introducing UB which didn't previously
/// exist.  Note that a false result conveys no information.
bool llvm::mustExecuteUBIfPoisonOnPathTo(Instruction *Root,
                                         Instruction *OnPathTo,
                                         DominatorTree *DT) {
  // Basic approach is to assume Root is poison, propagate poison forward
  // through all users we can easily track, and then check whether any of those
  // users are provable UB and must execute before out exiting block might
  // exit.

  // The set of all recursive users we've visited (which are assumed to all be
  // poison because of said visit)
  SmallPtrSet<const Value *, 16> KnownPoison;
  SmallVector<const Instruction *, 16> Worklist;
  Worklist.push_back(Root);
  while (!Worklist.empty()) {
    const Instruction *I = Worklist.pop_back_val();

    // If we know this must trigger UB on a path leading our target.
    if (mustTriggerUB(I, KnownPoison) && DT->dominates(I, OnPathTo))
      return true;

    // If we can't analyze propagation through this instruction, just skip it
    // and transitive users.  Safe as false is a conservative result.
    if (I != Root && !any_of(I->operands(), [&KnownPoison](const Use &U) {
          return KnownPoison.contains(U) && propagatesPoison(U);
        }))
      continue;

    if (KnownPoison.insert(I).second)
      for (const User *User : I->users())
        Worklist.push_back(cast<Instruction>(User));
  }

  // Might be non-UB, or might have a path we couldn't prove must execute on
  // way to exiting bb.
  return false;
}

std::optional<std::pair<CmpPredicate, Constant *>>
llvm::getFlippedStrictnessPredicateAndConstant(CmpPredicate Pred, Constant *C) {
  assert(ICmpInst::isRelational(Pred) && ICmpInst::isIntPredicate(Pred) &&
         "Only for relational integer predicates.");
  if (isa<UndefValue>(C))
    return std::nullopt;

  Type *Type = C->getType();
  bool IsSigned = ICmpInst::isSigned(Pred);

  CmpInst::Predicate UnsignedPred = ICmpInst::getUnsignedPredicate(Pred);
  bool WillIncrement =
      UnsignedPred == ICmpInst::ICMP_ULE || UnsignedPred == ICmpInst::ICMP_UGT;

  // Check if the constant operand can be safely incremented/decremented
  // without overflowing/underflowing.
  auto ConstantIsOk = [WillIncrement, IsSigned](ConstantInt *C) {
    return WillIncrement ? !C->isMaxValue(IsSigned) : !C->isMinValue(IsSigned);
  };

  Constant *SafeReplacementConstant = nullptr;
  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    // Bail out if the constant can't be safely incremented/decremented.
    if (!ConstantIsOk(CI))
      return std::nullopt;
  } else if (auto *FVTy = dyn_cast<FixedVectorType>(Type)) {
    unsigned NumElts = FVTy->getNumElements();
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *Elt = C->getAggregateElement(i);
      if (!Elt)
        return std::nullopt;

      if (isa<UndefValue>(Elt))
        continue;

      // Bail out if we can't determine if this constant is min/max or if we
      // know that this constant is min/max.
      auto *CI = dyn_cast<ConstantInt>(Elt);
      if (!CI || !ConstantIsOk(CI))
        return std::nullopt;

      if (!SafeReplacementConstant)
        SafeReplacementConstant = CI;
    }
  } else if (isa<VectorType>(C->getType())) {
    // Handle scalable splat
    Value *SplatC = C->getSplatValue();
    auto *CI = dyn_cast_or_null<ConstantInt>(SplatC);
    // Bail out if the constant can't be safely incremented/decremented.
    if (!CI || !ConstantIsOk(CI))
      return std::nullopt;
  } else {
    // ConstantExpr?
    return std::nullopt;
  }

  // It may not be safe to change a compare predicate in the presence of
  // undefined elements, so replace those elements with the first safe constant
  // that we found.
  // TODO: in case of poison, it is safe; let's replace undefs only.
  if (C->containsUndefOrPoisonElement()) {
    assert(SafeReplacementConstant && "Replacement constant not set");
    C = Constant::replaceUndefsWith(C, SafeReplacementConstant);
  }

  CmpInst::Predicate NewPred = CmpInst::getFlippedStrictnessPredicate(Pred);

  // Increment or decrement the constant.
  Constant *OneOrNegOne = ConstantInt::get(Type, WillIncrement ? 1 : -1, true);
  Constant *NewC = ConstantExpr::getAdd(C, OneOrNegOne);

  return std::make_pair(NewPred, NewC);
}

Intrinsic::ID llvm::getIntrinsicForCallSite(const CallBase &CB,
                                            const TargetLibraryInfo *TLI) {
  const Function *F = CB.getCalledFunction();
  if (!F)
    return Intrinsic::not_intrinsic;

  if (F->isIntrinsic())
    return F->getIntrinsicID();

  // We are going to infer semantics of a library function based on mapping it
  // to an LLVM intrinsic. Check that the library function is available from
  // this callbase and in this environment.
  LibFunc Func;
  if (F->hasLocalLinkage() || !TLI || !TLI->getLibFunc(CB, Func) ||
      !CB.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  switch (Func) {
  default:
    break;
  case LibFunc_sin:
  case LibFunc_sinf:
  case LibFunc_sinl:
    return Intrinsic::sin;
  case LibFunc_cos:
  case LibFunc_cosf:
  case LibFunc_cosl:
    return Intrinsic::cos;
  case LibFunc_tan:
  case LibFunc_tanf:
  case LibFunc_tanl:
    return Intrinsic::tan;
  case LibFunc_asin:
  case LibFunc_asinf:
  case LibFunc_asinl:
    return Intrinsic::asin;
  case LibFunc_acos:
  case LibFunc_acosf:
  case LibFunc_acosl:
    return Intrinsic::acos;
  case LibFunc_atan:
  case LibFunc_atanf:
  case LibFunc_atanl:
    return Intrinsic::atan;
  case LibFunc_atan2:
  case LibFunc_atan2f:
  case LibFunc_atan2l:
    return Intrinsic::atan2;
  case LibFunc_sinh:
  case LibFunc_sinhf:
  case LibFunc_sinhl:
    return Intrinsic::sinh;
  case LibFunc_cosh:
  case LibFunc_coshf:
  case LibFunc_coshl:
    return Intrinsic::cosh;
  case LibFunc_tanh:
  case LibFunc_tanhf:
  case LibFunc_tanhl:
    return Intrinsic::tanh;
  case LibFunc_exp:
  case LibFunc_expf:
  case LibFunc_expl:
    return Intrinsic::exp;
  case LibFunc_exp2:
  case LibFunc_exp2f:
  case LibFunc_exp2l:
    return Intrinsic::exp2;
  case LibFunc_exp10:
  case LibFunc_exp10f:
  case LibFunc_exp10l:
    return Intrinsic::exp10;
  case LibFunc_log:
  case LibFunc_logf:
  case LibFunc_logl:
    return Intrinsic::log;
  case LibFunc_log10:
  case LibFunc_log10f:
  case LibFunc_log10l:
    return Intrinsic::log10;
  case LibFunc_log2:
  case LibFunc_log2f:
  case LibFunc_log2l:
    return Intrinsic::log2;
  case LibFunc_fabs:
  case LibFunc_fabsf:
  case LibFunc_fabsl:
    return Intrinsic::fabs;
  case LibFunc_fmin:
  case LibFunc_fminf:
  case LibFunc_fminl:
    return Intrinsic::minnum;
  case LibFunc_fmax:
  case LibFunc_fmaxf:
  case LibFunc_fmaxl:
    return Intrinsic::maxnum;
  case LibFunc_copysign:
  case LibFunc_copysignf:
  case LibFunc_copysignl:
    return Intrinsic::copysign;
  case LibFunc_floor:
  case LibFunc_floorf:
  case LibFunc_floorl:
    return Intrinsic::floor;
  case LibFunc_ceil:
  case LibFunc_ceilf:
  case LibFunc_ceill:
    return Intrinsic::ceil;
  case LibFunc_trunc:
  case LibFunc_truncf:
  case LibFunc_truncl:
    return Intrinsic::trunc;
  case LibFunc_rint:
  case LibFunc_rintf:
  case LibFunc_rintl:
    return Intrinsic::rint;
  case LibFunc_nearbyint:
  case LibFunc_nearbyintf:
  case LibFunc_nearbyintl:
    return Intrinsic::nearbyint;
  case LibFunc_round:
  case LibFunc_roundf:
  case LibFunc_roundl:
    return Intrinsic::round;
  case LibFunc_roundeven:
  case LibFunc_roundevenf:
  case LibFunc_roundevenl:
    return Intrinsic::roundeven;
  case LibFunc_pow:
  case LibFunc_powf:
  case LibFunc_powl:
    return Intrinsic::pow;
  case LibFunc_sqrt:
  case LibFunc_sqrtf:
  case LibFunc_sqrtl:
    return Intrinsic::sqrt;
  }

  return Intrinsic::not_intrinsic;
}

/// Given an exploded icmp instruction, return true if the comparison only
/// checks the sign bit. If it only checks the sign bit, set TrueIfSigned if
/// the result of the comparison is true when the input value is signed.
bool llvm::isSignBitCheck(ICmpInst::Predicate Pred, const APInt &RHS,
                          bool &TrueIfSigned) {
  switch (Pred) {
  case ICmpInst::ICMP_SLT: // True if LHS s< 0
    TrueIfSigned = true;
    return RHS.isZero();
  case ICmpInst::ICMP_SLE: // True if LHS s<= -1
    TrueIfSigned = true;
    return RHS.isAllOnes();
  case ICmpInst::ICMP_SGT: // True if LHS s> -1
    TrueIfSigned = false;
    return RHS.isAllOnes();
  case ICmpInst::ICMP_SGE: // True if LHS s>= 0
    TrueIfSigned = false;
    return RHS.isZero();
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

Value *llvm::isBytewiseValue(Value *V, const DataLayout &DL) {

  // All byte-wide stores are splatable, even of arbitrary variables.
  if (V->getType()->isIntegerTy(8))
    return V;

  LLVMContext &Ctx = V->getContext();

  // Undef don't care.
  auto *UndefInt8 = UndefValue::get(Type::getInt8Ty(Ctx));
  if (isa<UndefValue>(V))
    return UndefInt8;

  // Return poison for zero-sized type.
  if (DL.getTypeStoreSize(V->getType()).isZero())
    return PoisonValue::get(Type::getInt8Ty(Ctx));

  Constant *C = dyn_cast<Constant>(V);
  if (!C) {
    // Conceptually, we could handle things like:
    //   %a = zext i8 %X to i16
    //   %b = shl i16 %a, 8
    //   %c = or i16 %a, %b
    // but until there is an example that actually needs this, it doesn't seem
    // worth worrying about.
    return nullptr;
  }

  // Handle 'null' ConstantArrayZero etc.
  if (C->isNullValue())
    return Constant::getNullValue(Type::getInt8Ty(Ctx));

  // Constant floating-point values can be handled as integer values if the
  // corresponding integer value is "byteable".  An important case is 0.0.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    Type *Ty = nullptr;
    if (CFP->getType()->isHalfTy())
      Ty = Type::getInt16Ty(Ctx);
    else if (CFP->getType()->isFloatTy())
      Ty = Type::getInt32Ty(Ctx);
    else if (CFP->getType()->isDoubleTy())
      Ty = Type::getInt64Ty(Ctx);
    // Don't handle long double formats, which have strange constraints.
    return Ty ? isBytewiseValue(ConstantExpr::getBitCast(CFP, Ty), DL)
              : nullptr;
  }

  // We can handle constant integers that are multiple of 8 bits.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
    if (CI->getBitWidth() % 8 == 0) {
      assert(CI->getBitWidth() > 8 && "8 bits should be handled above!");
      if (!CI->getValue().isSplat(8))
        return nullptr;
      return ConstantInt::get(Ctx, CI->getValue().trunc(8));
    }
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getOpcode() == Instruction::IntToPtr) {
      if (auto *PtrTy = dyn_cast<PointerType>(CE->getType())) {
        unsigned BitWidth = DL.getPointerSizeInBits(PtrTy->getAddressSpace());
        if (Constant *Op = ConstantFoldIntegerCast(
                CE->getOperand(0), Type::getIntNTy(Ctx, BitWidth), false, DL))
          return isBytewiseValue(Op, DL);
      }
    }
  }

  auto Merge = [&](Value *LHS, Value *RHS) -> Value * {
    if (LHS == RHS)
      return LHS;
    if (!LHS || !RHS)
      return nullptr;
    if (LHS == UndefInt8)
      return RHS;
    if (RHS == UndefInt8)
      return LHS;
    return nullptr;
  };

  if (ConstantDataSequential *CA = dyn_cast<ConstantDataSequential>(C)) {
    Value *Val = UndefInt8;
    for (uint64_t I = 0, E = CA->getNumElements(); I != E; ++I)
      if (!(Val = Merge(Val, isBytewiseValue(CA->getElementAsConstant(I), DL))))
        return nullptr;
    return Val;
  }

  if (isa<ConstantAggregate>(C)) {
    Value *Val = UndefInt8;
    for (Value *Op : C->operands())
      if (!(Val = Merge(Val, isBytewiseValue(Op, DL))))
        return nullptr;
    return Val;
  }

  // Don't try to handle the handful of other constants.
  return nullptr;
}

// This is the recursive version of BuildSubAggregate. It takes a few different
// arguments. Idxs is the index within the nested struct From that we are
// looking at now (which is of type IndexedType). IdxSkip is the number of
// indices from Idxs that should be left out when inserting into the resulting
// struct. To is the result struct built so far, new insertvalue instructions
// build on that.
static Value *BuildSubAggregate(Value *From, Value *To, Type *IndexedType,
                                SmallVectorImpl<unsigned> &Idxs,
                                unsigned IdxSkip,
                                BasicBlock::iterator InsertBefore) {
  StructType *STy = dyn_cast<StructType>(IndexedType);
  if (STy) {
    // Save the original To argument so we can modify it
    Value *OrigTo = To;
    // General case, the type indexed by Idxs is a struct
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      // Process each struct element recursively
      Idxs.push_back(i);
      Value *PrevTo = To;
      To = BuildSubAggregate(From, To, STy->getElementType(i), Idxs, IdxSkip,
                             InsertBefore);
      Idxs.pop_back();
      if (!To) {
        // Couldn't find any inserted value for this index? Cleanup
        while (PrevTo != OrigTo) {
          InsertValueInst *Del = cast<InsertValueInst>(PrevTo);
          PrevTo = Del->getAggregateOperand();
          Del->eraseFromParent();
        }
        // Stop processing elements
        break;
      }
    }
    // If we successfully found a value for each of our subaggregates
    if (To)
      return To;
  }
  // Base case, the type indexed by SourceIdxs is not a struct, or not all of
  // the struct's elements had a value that was inserted directly. In the latter
  // case, perhaps we can't determine each of the subelements individually, but
  // we might be able to find the complete struct somewhere.

  // Find the value that is at that particular spot
  Value *V = FindInsertedValue(From, Idxs);

  if (!V)
    return nullptr;

  // Insert the value in the new (sub) aggregate
  return InsertValueInst::Create(To, V, ArrayRef(Idxs).slice(IdxSkip), "tmp",
                                 InsertBefore);
}

// This helper takes a nested struct and extracts a part of it (which is again a
// struct) into a new value. For example, given the struct:
// { a, { b, { c, d }, e } }
// and the indices "1, 1" this returns
// { c, d }.
//
// It does this by inserting an insertvalue for each element in the resulting
// struct, as opposed to just inserting a single struct. This will only work if
// each of the elements of the substruct are known (ie, inserted into From by an
// insertvalue instruction somewhere).
//
// All inserted insertvalue instructions are inserted before InsertBefore
static Value *BuildSubAggregate(Value *From, ArrayRef<unsigned> idx_range,
                                BasicBlock::iterator InsertBefore) {
  Type *IndexedType =
      ExtractValueInst::getIndexedType(From->getType(), idx_range);
  Value *To = PoisonValue::get(IndexedType);
  SmallVector<unsigned, 10> Idxs(idx_range);
  unsigned IdxSkip = Idxs.size();

  return BuildSubAggregate(From, To, IndexedType, Idxs, IdxSkip, InsertBefore);
}

/// Given an aggregate and a sequence of indices, see if the scalar value
/// indexed is already around as a register, for example if it was inserted
/// directly into the aggregate.
///
/// If InsertBefore is not null, this function will duplicate (modified)
/// insertvalues when a part of a nested struct is extracted.
Value *
llvm::FindInsertedValue(Value *V, ArrayRef<unsigned> idx_range,
                        std::optional<BasicBlock::iterator> InsertBefore) {
  // Nothing to index? Just return V then (this is useful at the end of our
  // recursion).
  if (idx_range.empty())
    return V;
  // We have indices, so V should have an indexable type.
  assert((V->getType()->isStructTy() || V->getType()->isArrayTy()) &&
         "Not looking at a struct or array?");
  assert(ExtractValueInst::getIndexedType(V->getType(), idx_range) &&
         "Invalid indices for type?");

  if (Constant *C = dyn_cast<Constant>(V)) {
    C = C->getAggregateElement(idx_range[0]);
    if (!C)
      return nullptr;
    return FindInsertedValue(C, idx_range.slice(1), InsertBefore);
  }

  if (InsertValueInst *I = dyn_cast<InsertValueInst>(V)) {
    // Loop the indices for the insertvalue instruction in parallel with the
    // requested indices
    const unsigned *req_idx = idx_range.begin();
    for (const unsigned *i = I->idx_begin(), *e = I->idx_end(); i != e;
         ++i, ++req_idx) {
      if (req_idx == idx_range.end()) {
        // We can't handle this without inserting insertvalues
        if (!InsertBefore)
          return nullptr;

        // The requested index identifies a part of a nested aggregate. Handle
        // this specially. For example,
        // %A = insertvalue { i32, {i32, i32 } } undef, i32 10, 1, 0
        // %B = insertvalue { i32, {i32, i32 } } %A, i32 11, 1, 1
        // %C = extractvalue {i32, { i32, i32 } } %B, 1
        // This can be changed into
        // %A = insertvalue {i32, i32 } undef, i32 10, 0
        // %C = insertvalue {i32, i32 } %A, i32 11, 1
        // which allows the unused 0,0 element from the nested struct to be
        // removed.
        return BuildSubAggregate(V, ArrayRef(idx_range.begin(), req_idx),
                                 *InsertBefore);
      }

      // This insert value inserts something else than what we are looking for.
      // See if the (aggregate) value inserted into has the value we are
      // looking for, then.
      if (*req_idx != *i)
        return FindInsertedValue(I->getAggregateOperand(), idx_range,
                                 InsertBefore);
    }
    // If we end up here, the indices of the insertvalue match with those
    // requested (though possibly only partially). Now we recursively look at
    // the inserted value, passing any remaining indices.
    return FindInsertedValue(I->getInsertedValueOperand(),
                             ArrayRef(req_idx, idx_range.end()), InsertBefore);
  }

  if (ExtractValueInst *I = dyn_cast<ExtractValueInst>(V)) {
    // If we're extracting a value from an aggregate that was extracted from
    // something else, we can extract from that something else directly instead.
    // However, we will need to chain I's indices with the requested indices.

    // Calculate the number of indices required
    unsigned size = I->getNumIndices() + idx_range.size();
    // Allocate some space to put the new indices in
    SmallVector<unsigned, 5> Idxs;
    Idxs.reserve(size);
    // Add indices from the extract value instruction
    Idxs.append(I->idx_begin(), I->idx_end());

    // Add requested indices
    Idxs.append(idx_range.begin(), idx_range.end());

    assert(Idxs.size() == size && "Number of indices added not correct?");

    return FindInsertedValue(I->getAggregateOperand(), Idxs, InsertBefore);
  }
  // Otherwise, we don't know (such as, extracting from a function return value
  // or load instruction)
  return nullptr;
}

// If V refers to an initialized global constant, set Slice either to
// its initializer if the size of its elements equals ElementSize, or,
// for ElementSize == 8, to its representation as an array of unsiged
// char. Return true on success.
// Offset is in the unit "nr of ElementSize sized elements".
bool llvm::getConstantDataArrayInfo(const Value *V,
                                    ConstantDataArraySlice &Slice,
                                    unsigned ElementSize, uint64_t Offset) {
  assert(V && "V should not be null.");
  assert((ElementSize % 8) == 0 &&
         "ElementSize expected to be a multiple of the size of a byte.");
  unsigned ElementSizeInBytes = ElementSize / 8;

  // Drill down into the pointer expression V, ignoring any intervening
  // casts, and determine the identity of the object it references along
  // with the cumulative byte offset into it.
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(V));
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
    // Fail if V is not based on constant global object.
    return false;

  const DataLayout &DL = GV->getDataLayout();
  APInt Off(DL.getIndexTypeSizeInBits(V->getType()), 0);

  if (GV != V->stripAndAccumulateConstantOffsets(DL, Off,
                                                 /*AllowNonInbounds*/ true))
    // Fail if a constant offset could not be determined.
    return false;

  uint64_t StartIdx = Off.getLimitedValue();
  if (StartIdx == UINT64_MAX)
    // Fail if the constant offset is excessive.
    return false;

  // Off/StartIdx is in the unit of bytes. So we need to convert to number of
  // elements. Simply bail out if that isn't possible.
  if ((StartIdx % ElementSizeInBytes) != 0)
    return false;

  Offset += StartIdx / ElementSizeInBytes;
  ConstantDataArray *Array = nullptr;
  ArrayType *ArrayTy = nullptr;

  if (GV->getInitializer()->isNullValue()) {
    Type *GVTy = GV->getValueType();
    uint64_t SizeInBytes = DL.getTypeStoreSize(GVTy).getFixedValue();
    uint64_t Length = SizeInBytes / ElementSizeInBytes;

    Slice.Array = nullptr;
    Slice.Offset = 0;
    // Return an empty Slice for undersized constants to let callers
    // transform even undefined library calls into simpler, well-defined
    // expressions.  This is preferable to making the calls although it
    // prevents sanitizers from detecting such calls.
    Slice.Length = Length < Offset ? 0 : Length - Offset;
    return true;
  }

  auto *Init = const_cast<Constant *>(GV->getInitializer());
  if (auto *ArrayInit = dyn_cast<ConstantDataArray>(Init)) {
    Type *InitElTy = ArrayInit->getElementType();
    if (InitElTy->isIntegerTy(ElementSize)) {
      // If Init is an initializer for an array of the expected type
      // and size, use it as is.
      Array = ArrayInit;
      ArrayTy = ArrayInit->getType();
    }
  }

  if (!Array) {
    if (ElementSize != 8)
      // TODO: Handle conversions to larger integral types.
      return false;

    // Otherwise extract the portion of the initializer starting
    // at Offset as an array of bytes, and reset Offset.
    Init = ReadByteArrayFromGlobal(GV, Offset);
    if (!Init)
      return false;

    Offset = 0;
    Array = dyn_cast<ConstantDataArray>(Init);
    ArrayTy = dyn_cast<ArrayType>(Init->getType());
  }

  uint64_t NumElts = ArrayTy->getArrayNumElements();
  if (Offset > NumElts)
    return false;

  Slice.Array = Array;
  Slice.Offset = Offset;
  Slice.Length = NumElts - Offset;
  return true;
}

/// Extract bytes from the initializer of the constant array V, which need
/// not be a nul-terminated string.  On success, store the bytes in Str and
/// return true.  When TrimAtNul is set, Str will contain only the bytes up
/// to but not including the first nul.  Return false on failure.
bool llvm::getConstantStringInfo(const Value *V, StringRef &Str,
                                 bool TrimAtNul) {
  ConstantDataArraySlice Slice;
  if (!getConstantDataArrayInfo(V, Slice, 8))
    return false;

  if (Slice.Array == nullptr) {
    if (TrimAtNul) {
      // Return a nul-terminated string even for an empty Slice.  This is
      // safe because all existing SimplifyLibcalls callers require string
      // arguments and the behavior of the functions they fold is undefined
      // otherwise.  Folding the calls this way is preferable to making
      // the undefined library calls, even though it prevents sanitizers
      // from reporting such calls.
      Str = StringRef();
      return true;
    }
    if (Slice.Length == 1) {
      Str = StringRef("", 1);
      return true;
    }
    // We cannot instantiate a StringRef as we do not have an appropriate string
    // of 0s at hand.
    return false;
  }

  // Start out with the entire array in the StringRef.
  Str = Slice.Array->getAsString();
  // Skip over 'offset' bytes.
  Str = Str.substr(Slice.Offset);

  if (TrimAtNul) {
    // Trim off the \0 and anything after it.  If the array is not nul
    // terminated, we just return the whole end of string.  The client may know
    // some other way that the string is length-bound.
    Str = Str.substr(0, Str.find('\0'));
  }
  return true;
}

// These next two are very similar to the above, but also look through PHI
// nodes.
// TODO: See if we can integrate these two together.

/// If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(const Value *V,
                                 SmallPtrSetImpl<const PHINode *> &PHIs,
                                 unsigned CharSize) {
  // Look through noop bitcast instructions.
  V = V->stripPointerCasts();

  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (const PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN).second)
      return ~0ULL; // already in the set.

    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (Value *IncValue : PN->incoming_values()) {
      uint64_t Len = GetStringLengthH(IncValue, PHIs, CharSize);
      if (Len == 0)
        return 0; // Unknown length -> unknown.

      if (Len == ~0ULL)
        continue;

      if (Len != LenSoFar && LenSoFar != ~0ULL)
        return 0; // Disagree -> unknown.
      LenSoFar = Len;
    }

    // Success, all agree.
    return LenSoFar;
  }

  // strlen(select(c,x,y)) -> strlen(x) ^ strlen(y)
  if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
    uint64_t Len1 = GetStringLengthH(SI->getTrueValue(), PHIs, CharSize);
    if (Len1 == 0)
      return 0;
    uint64_t Len2 = GetStringLengthH(SI->getFalseValue(), PHIs, CharSize);
    if (Len2 == 0)
      return 0;
    if (Len1 == ~0ULL)
      return Len2;
    if (Len2 == ~0ULL)
      return Len1;
    if (Len1 != Len2)
      return 0;
    return Len1;
  }

  // Otherwise, see if we can read the string.
  ConstantDataArraySlice Slice;
  if (!getConstantDataArrayInfo(V, Slice, CharSize))
    return 0;

  if (Slice.Array == nullptr)
    // Zeroinitializer (including an empty one).
    return 1;

  // Search for the first nul character.  Return a conservative result even
  // when there is no nul.  This is safe since otherwise the string function
  // being folded such as strlen is undefined, and can be preferable to
  // making the undefined library call.
  unsigned NullIndex = 0;
  for (unsigned E = Slice.Length; NullIndex < E; ++NullIndex) {
    if (Slice.Array->getElementAsInteger(Slice.Offset + NullIndex) == 0)
      break;
  }

  return NullIndex + 1;
}

/// If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
uint64_t llvm::GetStringLength(const Value *V, unsigned CharSize) {
  if (!V->getType()->isPointerTy())
    return 0;

  SmallPtrSet<const PHINode *, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs, CharSize);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

const Value *
llvm::getArgumentAliasingToReturnedPointer(const CallBase *Call,
                                           bool MustPreserveNullness) {
  assert(Call &&
         "getArgumentAliasingToReturnedPointer only works on nonnull calls");
  if (const Value *RV = Call->getReturnedArgOperand())
    return RV;
  // This can be used only as a aliasing property.
  if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(
          Call, MustPreserveNullness))
    return Call->getArgOperand(0);
  return nullptr;
}

bool llvm::isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(
    const CallBase *Call, bool MustPreserveNullness) {
  switch (Call->getIntrinsicID()) {
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group:
  case Intrinsic::aarch64_irg:
  case Intrinsic::aarch64_tagp:
  // The amdgcn_make_buffer_rsrc function does not alter the address of the
  // input pointer (and thus preserve null-ness for the purposes of escape
  // analysis, which is where the MustPreserveNullness flag comes in to play).
  // However, it will not necessarily map ptr addrspace(N) null to ptr
  // addrspace(8) null, aka the "null descriptor", which has "all loads return
  // 0, all stores are dropped" semantics. Given the context of this intrinsic
  // list, no one should be relying on such a strict interpretation of
  // MustPreserveNullness (and, at time of writing, they are not), but we
  // document this fact out of an abundance of caution.
  case Intrinsic::amdgcn_make_buffer_rsrc:
    return true;
  case Intrinsic::ptrmask:
    return !MustPreserveNullness;
  case Intrinsic::threadlocal_address:
    // The underlying variable changes with thread ID. The Thread ID may change
    // at coroutine suspend points.
    return !Call->getParent()->getParent()->isPresplitCoroutine();
  default:
    return false;
  }
}

const Value *llvm::getUnderlyingObject(const Value *V, unsigned MaxLookup) {
  for (unsigned Count = 0; MaxLookup == 0 || Count < MaxLookup; ++Count) {
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      const Value *PtrOp = GEP->getPointerOperand();
      if (!PtrOp->getType()->isPointerTy()) // Only handle scalar pointer base.
        return V;
      V = PtrOp;
    } else if (Operator::getOpcode(V) == Instruction::BitCast ||
               Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
      Value *NewV = cast<Operator>(V)->getOperand(0);
      if (!NewV->getType()->isPointerTy())
        return V;
      V = NewV;
    } else if (auto *GA = dyn_cast<GlobalAlias>(V)) {
      if (GA->isInterposable())
        return V;
      V = GA->getAliasee();
    } else {
      if (auto *PHI = dyn_cast<PHINode>(V)) {
        // Look through single-arg phi nodes created by LCSSA.
        if (PHI->getNumIncomingValues() == 1) {
          V = PHI->getIncomingValue(0);
          continue;
        }
      } else if (auto *Call = dyn_cast<CallBase>(V)) {
        // CaptureTracking can know about special capturing properties of some
        // intrinsics like launder.invariant.group, that can't be expressed with
        // the attributes, but have properties like returning aliasing pointer.
        // Because some analysis may assume that nocaptured pointer is not
        // returned from some special intrinsic (because function would have to
        // be marked with returns attribute), it is crucial to use this function
        // because it should be in sync with CaptureTracking. Not using it may
        // cause weird miscompilations where 2 aliasing pointers are assumed to
        // noalias.
        if (auto *RP = getArgumentAliasingToReturnedPointer(Call, false)) {
          V = RP;
          continue;
        }
      }

      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  }
  return V;
}

/// \p PN defines a loop-variant pointer to an object.  Check if the
/// previous iteration of the loop was referring to the same object as \p PN.
static bool isSameUnderlyingObjectInLoop(const PHINode *PN,
                                         const LoopInfo *LI) {
  // Find the loop-defined value.
  Loop *L = LI->getLoopFor(PN->getParent());
  if (PN->getNumIncomingValues() != 2)
    return true;

  // Find the value from previous iteration.
  auto *PrevValue = dyn_cast<Instruction>(PN->getIncomingValue(0));
  if (!PrevValue || LI->getLoopFor(PrevValue->getParent()) != L)
    PrevValue = dyn_cast<Instruction>(PN->getIncomingValue(1));
  if (!PrevValue || LI->getLoopFor(PrevValue->getParent()) != L)
    return true;

  // If a new pointer is loaded in the loop, the pointer references a different
  // object in every iteration.  E.g.:
  //    for (i)
  //       int *p = a[i];
  //       ...
  if (auto *Load = dyn_cast<LoadInst>(PrevValue))
    if (!L->isLoopInvariant(Load->getPointerOperand()))
      return false;
  return true;
}

void llvm::getUnderlyingObjects(const Value *V,
                                SmallVectorImpl<const Value *> &Objects,
                                const LoopInfo *LI, unsigned MaxLookup) {
  SmallPtrSet<const Value *, 4> Visited;
  SmallVector<const Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    const Value *P = Worklist.pop_back_val();
    P = getUnderlyingObject(P, MaxLookup);

    if (!Visited.insert(P).second)
      continue;

    if (auto *SI = dyn_cast<SelectInst>(P)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    if (auto *PN = dyn_cast<PHINode>(P)) {
      // If this PHI changes the underlying object in every iteration of the
      // loop, don't look through it.  Consider:
      //   int **A;
      //   for (i) {
      //     Prev = Curr;     // Prev = PHI (Prev_0, Curr)
      //     Curr = A[i];
      //     *Prev, *Curr;
      //
      // Prev is tracking Curr one iteration behind so they refer to different
      // underlying objects.
      if (!LI || !LI->isLoopHeader(PN->getParent()) ||
          isSameUnderlyingObjectInLoop(PN, LI))
        append_range(Worklist, PN->incoming_values());
      else
        Objects.push_back(P);
      continue;
    }

    Objects.push_back(P);
  } while (!Worklist.empty());
}

const Value *llvm::getUnderlyingObjectAggressive(const Value *V) {
  const unsigned MaxVisited = 8;

  SmallPtrSet<const Value *, 8> Visited;
  SmallVector<const Value *, 8> Worklist;
  Worklist.push_back(V);
  const Value *Object = nullptr;
  // Used as fallback if we can't find a common underlying object through
  // recursion.
  bool First = true;
  const Value *FirstObject = getUnderlyingObject(V);
  do {
    const Value *P = Worklist.pop_back_val();
    P = First ? FirstObject : getUnderlyingObject(P);
    First = false;

    if (!Visited.insert(P).second)
      continue;

    if (Visited.size() == MaxVisited)
      return FirstObject;

    if (auto *SI = dyn_cast<SelectInst>(P)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    if (auto *PN = dyn_cast<PHINode>(P)) {
      append_range(Worklist, PN->incoming_values());
      continue;
    }

    if (!Object)
      Object = P;
    else if (Object != P)
      return FirstObject;
  } while (!Worklist.empty());

  return Object ? Object : FirstObject;
}

/// This is the function that does the work of looking through basic
/// ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObjectFromInt(const Value *V) {
  do {
    if (const Operator *U = dyn_cast<Operator>(V)) {
      // If we find a ptrtoint, we can transfer control back to the
      // regular getUnderlyingObjectFromInt.
      if (U->getOpcode() == Instruction::PtrToInt)
        return U->getOperand(0);
      // If we find an add of a constant, a multiplied value, or a phi, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifiable object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul &&
           !isa<PHINode>(U->getOperand(1))))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (true);
}

/// This is a wrapper around getUnderlyingObjects and adds support for basic
/// ptrtoint+arithmetic+inttoptr sequences.
/// It returns false if unidentified object is found in getUnderlyingObjects.
bool llvm::getUnderlyingObjectsForCodeGen(const Value *V,
                                          SmallVectorImpl<Value *> &Objects) {
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 4> Working(1, V);
  do {
    V = Working.pop_back_val();

    SmallVector<const Value *, 4> Objs;
    getUnderlyingObjects(V, Objs);

    for (const Value *V : Objs) {
      if (!Visited.insert(V).second)
        continue;
      if (Operator::getOpcode(V) == Instruction::IntToPtr) {
        const Value *O =
            getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
        if (O->getType()->isPointerTy()) {
          Working.push_back(O);
          continue;
        }
      }
      // If getUnderlyingObjects fails to find an identifiable object,
      // getUnderlyingObjectsForCodeGen also fails for safety.
      if (!isIdentifiedObject(V)) {
        Objects.clear();
        return false;
      }
      Objects.push_back(const_cast<Value *>(V));
    }
  } while (!Working.empty());
  return true;
}

AllocaInst *llvm::findAllocaForValue(Value *V, bool OffsetZero) {
  AllocaInst *Result = nullptr;
  SmallPtrSet<Value *, 4> Visited;
  SmallVector<Value *, 4> Worklist;

  auto AddWork = [&](Value *V) {
    if (Visited.insert(V).second)
      Worklist.push_back(V);
  };

  AddWork(V);
  do {
    V = Worklist.pop_back_val();
    assert(Visited.count(V));

    if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      if (Result && Result != AI)
        return nullptr;
      Result = AI;
    } else if (CastInst *CI = dyn_cast<CastInst>(V)) {
      AddWork(CI->getOperand(0));
    } else if (PHINode *PN = dyn_cast<PHINode>(V)) {
      for (Value *IncValue : PN->incoming_values())
        AddWork(IncValue);
    } else if (auto *SI = dyn_cast<SelectInst>(V)) {
      AddWork(SI->getTrueValue());
      AddWork(SI->getFalseValue());
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
      if (OffsetZero && !GEP->hasAllZeroIndices())
        return nullptr;
      AddWork(GEP->getPointerOperand());
    } else if (CallBase *CB = dyn_cast<CallBase>(V)) {
      Value *Returned = CB->getReturnedArgOperand();
      if (Returned)
        AddWork(Returned);
      else
        return nullptr;
    } else {
      return nullptr;
    }
  } while (!Worklist.empty());

  return Result;
}

static bool onlyUsedByLifetimeMarkersOrDroppableInstsHelper(
    const Value *V, bool AllowLifetime, bool AllowDroppable) {
  for (const User *U : V->users()) {
    const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
    if (!II)
      return false;

    if (AllowLifetime && II->isLifetimeStartOrEnd())
      continue;

    if (AllowDroppable && II->isDroppable())
      continue;

    return false;
  }
  return true;
}

bool llvm::onlyUsedByLifetimeMarkers(const Value *V) {
  return onlyUsedByLifetimeMarkersOrDroppableInstsHelper(
      V, /* AllowLifetime */ true, /* AllowDroppable */ false);
}
bool llvm::onlyUsedByLifetimeMarkersOrDroppableInsts(const Value *V) {
  return onlyUsedByLifetimeMarkersOrDroppableInstsHelper(
      V, /* AllowLifetime */ true, /* AllowDroppable */ true);
}

bool llvm::isNotCrossLaneOperation(const Instruction *I) {
  if (auto *II = dyn_cast<IntrinsicInst>(I))
    return isTriviallyVectorizable(II->getIntrinsicID());
  auto *Shuffle = dyn_cast<ShuffleVectorInst>(I);
  return (!Shuffle || Shuffle->isSelect()) &&
         !isa<CallBase, BitCastInst, ExtractElementInst>(I);
}

bool llvm::isKnownNegation(const Value *X, const Value *Y, bool NeedNSW,
                           bool AllowPoison) {
  assert(X && Y && "Invalid operand");

  auto IsNegationOf = [&](const Value *X, const Value *Y) {
    if (!match(X, m_Neg(m_Specific(Y))))
      return false;

    auto *BO = cast<BinaryOperator>(X);
    if (NeedNSW && !BO->hasNoSignedWrap())
      return false;

    auto *Zero = cast<Constant>(BO->getOperand(0));
    if (!AllowPoison && !Zero->isNullValue())
      return false;

    return true;
  };

  // X = -Y or Y = -X
  if (IsNegationOf(X, Y) || IsNegationOf(Y, X))
    return true;

  // X = sub (A, B), Y = sub (B, A) || X = sub nsw (A, B), Y = sub nsw (B, A)
  Value *A, *B;
  return (!NeedNSW && (match(X, m_Sub(m_Value(A), m_Value(B))) &&
                       match(Y, m_Sub(m_Specific(B), m_Specific(A))))) ||
         (NeedNSW && (match(X, m_NSWSub(m_Value(A), m_Value(B))) &&
                      match(Y, m_NSWSub(m_Specific(B), m_Specific(A)))));
}

bool llvm::isKnownInversion(const Value *X, const Value *Y) {
  // Handle X = icmp pred A, B, Y = icmp pred A, C.
  Value *A, *B, *C;
  CmpPredicate Pred1, Pred2;
  if (!match(X, m_ICmp(Pred1, m_Value(A), m_Value(B))) ||
      !match(Y, m_c_ICmp(Pred2, m_Specific(A), m_Value(C))))
    return false;

  // They must both have samesign flag or not.
  if (Pred1.hasSameSign() != Pred2.hasSameSign())
    return false;

  if (B == C)
    return Pred1 == ICmpInst::getInversePredicate(Pred2);

  // Try to infer the relationship from constant ranges.
  const APInt *RHSC1, *RHSC2;
  if (!match(B, m_APInt(RHSC1)) || !match(C, m_APInt(RHSC2)))
    return false;

  // Sign bits of two RHSCs should match.
  if (Pred1.hasSameSign() && RHSC1->isNonNegative() != RHSC2->isNonNegative())
    return false;

  const auto CR1 = ConstantRange::makeExactICmpRegion(Pred1, *RHSC1);
  const auto CR2 = ConstantRange::makeExactICmpRegion(Pred2, *RHSC2);

  return CR1.inverse() == CR2;
}

SelectPatternResult llvm::getSelectPattern(CmpInst::Predicate Pred,
                                           SelectPatternNaNBehavior NaNBehavior,
                                           bool Ordered) {
  switch (Pred) {
  default:
    return {SPF_UNKNOWN, SPNB_NA, false}; // Equality.
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_UGE:
    return {SPF_UMAX, SPNB_NA, false};
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_SGE:
    return {SPF_SMAX, SPNB_NA, false};
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_ULE:
    return {SPF_UMIN, SPNB_NA, false};
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_SLE:
    return {SPF_SMIN, SPNB_NA, false};
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_UGE:
  case FCmpInst::FCMP_OGT:
  case FCmpInst::FCMP_OGE:
    return {SPF_FMAXNUM, NaNBehavior, Ordered};
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_ULE:
  case FCmpInst::FCMP_OLT:
  case FCmpInst::FCMP_OLE:
    return {SPF_FMINNUM, NaNBehavior, Ordered};
  }
}

CmpInst::Predicate llvm::getMinMaxPred(SelectPatternFlavor SPF, bool Ordered) {
  if (SPF == SPF_SMIN)
    return ICmpInst::ICMP_SLT;
  if (SPF == SPF_UMIN)
    return ICmpInst::ICMP_ULT;
  if (SPF == SPF_SMAX)
    return ICmpInst::ICMP_SGT;
  if (SPF == SPF_UMAX)
    return ICmpInst::ICMP_UGT;
  if (SPF == SPF_FMINNUM)
    return Ordered ? FCmpInst::FCMP_OLT : FCmpInst::FCMP_ULT;
  if (SPF == SPF_FMAXNUM)
    return Ordered ? FCmpInst::FCMP_OGT : FCmpInst::FCMP_UGT;
  llvm_unreachable("unhandled!");
}

Intrinsic::ID llvm::getMinMaxIntrinsic(SelectPatternFlavor SPF) {
  switch (SPF) {
  case SelectPatternFlavor::SPF_UMIN:
    return Intrinsic::umin;
  case SelectPatternFlavor::SPF_UMAX:
    return Intrinsic::umax;
  case SelectPatternFlavor::SPF_SMIN:
    return Intrinsic::smin;
  case SelectPatternFlavor::SPF_SMAX:
    return Intrinsic::smax;
  default:
    llvm_unreachable("Unexpected SPF");
  }
}

SelectPatternFlavor llvm::getInverseMinMaxFlavor(SelectPatternFlavor SPF) {
  if (SPF == SPF_SMIN)
    return SPF_SMAX;
  if (SPF == SPF_UMIN)
    return SPF_UMAX;
  if (SPF == SPF_SMAX)
    return SPF_SMIN;
  if (SPF == SPF_UMAX)
    return SPF_UMIN;
  llvm_unreachable("unhandled!");
}

Intrinsic::ID llvm::getInverseMinMaxIntrinsic(Intrinsic::ID MinMaxID) {
  switch (MinMaxID) {
  case Intrinsic::smax:
    return Intrinsic::smin;
  case Intrinsic::smin:
    return Intrinsic::smax;
  case Intrinsic::umax:
    return Intrinsic::umin;
  case Intrinsic::umin:
    return Intrinsic::umax;
  // Please note that next four intrinsics may produce the same result for
  // original and inverted case even if X != Y due to NaN is handled specially.
  case Intrinsic::maximum:
    return Intrinsic::minimum;
  case Intrinsic::minimum:
    return Intrinsic::maximum;
  case Intrinsic::maxnum:
    return Intrinsic::minnum;
  case Intrinsic::minnum:
    return Intrinsic::maxnum;
  case Intrinsic::maximumnum:
    return Intrinsic::minimumnum;
  case Intrinsic::minimumnum:
    return Intrinsic::maximumnum;
  default:
    llvm_unreachable("Unexpected intrinsic");
  }
}

APInt llvm::getMinMaxLimit(SelectPatternFlavor SPF, unsigned BitWidth) {
  switch (SPF) {
  case SPF_SMAX:
    return APInt::getSignedMaxValue(BitWidth);
  case SPF_SMIN:
    return APInt::getSignedMinValue(BitWidth);
  case SPF_UMAX:
    return APInt::getMaxValue(BitWidth);
  case SPF_UMIN:
    return APInt::getMinValue(BitWidth);
  default:
    llvm_unreachable("Unexpected flavor");
  }
}

std::pair<Intrinsic::ID, bool>
llvm::canConvertToMinOrMaxIntrinsic(ArrayRef<Value *> VL) {
  // Check if VL contains select instructions that can be folded into a min/max
  // vector intrinsic and return the intrinsic if it is possible.
  // TODO: Support floating point min/max.
  bool AllCmpSingleUse = true;
  SelectPatternResult SelectPattern;
  SelectPattern.Flavor = SPF_UNKNOWN;
  if (all_of(VL, [&SelectPattern, &AllCmpSingleUse](Value *I) {
        Value *LHS, *RHS;
        auto CurrentPattern = matchSelectPattern(I, LHS, RHS);
        if (!SelectPatternResult::isMinOrMax(CurrentPattern.Flavor))
          return false;
        if (SelectPattern.Flavor != SPF_UNKNOWN &&
            SelectPattern.Flavor != CurrentPattern.Flavor)
          return false;
        SelectPattern = CurrentPattern;
        AllCmpSingleUse &=
            match(I, m_Select(m_OneUse(m_Value()), m_Value(), m_Value()));
        return true;
      })) {
    switch (SelectPattern.Flavor) {
    case SPF_SMIN:
      return {Intrinsic::smin, AllCmpSingleUse};
    case SPF_UMIN:
      return {Intrinsic::umin, AllCmpSingleUse};
    case SPF_SMAX:
      return {Intrinsic::smax, AllCmpSingleUse};
    case SPF_UMAX:
      return {Intrinsic::umax, AllCmpSingleUse};
    case SPF_FMAXNUM:
      return {Intrinsic::maxnum, AllCmpSingleUse};
    case SPF_FMINNUM:
      return {Intrinsic::minnum, AllCmpSingleUse};
    default:
      llvm_unreachable("unexpected select pattern flavor");
    }
  }
  return {Intrinsic::not_intrinsic, false};
}

template <typename InstTy>
static bool matchTwoInputRecurrence(const PHINode *PN, InstTy *&Inst,
                                    Value *&Init, Value *&OtherOp) {
  // Handle the case of a simple two-predecessor recurrence PHI.
  // There's a lot more that could theoretically be done here, but
  // this is sufficient to catch some interesting cases.
  // TODO: Expand list -- gep, uadd.sat etc.
  if (PN->getNumIncomingValues() != 2)
    return false;

  for (unsigned I = 0; I != 2; ++I) {
    if (auto *Operation = dyn_cast<InstTy>(PN->getIncomingValue(I));
        Operation && Operation->getNumOperands() >= 2) {
      Value *LHS = Operation->getOperand(0);
      Value *RHS = Operation->getOperand(1);
      if (LHS != PN && RHS != PN)
        continue;

      Inst = Operation;
      Init = PN->getIncomingValue(!I);
      OtherOp = (LHS == PN) ? RHS : LHS;
      return true;
    }
  }
  return false;
}

bool llvm::matchSimpleRecurrence(const PHINode *P, BinaryOperator *&BO,
                                 Value *&Start, Value *&Step) {
  // We try to match a recurrence of the form:
  //   %iv = [Start, %entry], [%iv.next, %backedge]
  //   %iv.next = binop %iv, Step
  // Or:
  //   %iv = [Start, %entry], [%iv.next, %backedge]
  //   %iv.next = binop Step, %iv
  return matchTwoInputRecurrence(P, BO, Start, Step);
}

bool llvm::matchSimpleRecurrence(const BinaryOperator *I, PHINode *&P,
                                 Value *&Start, Value *&Step) {
  BinaryOperator *BO = nullptr;
  P = dyn_cast<PHINode>(I->getOperand(0));
  if (!P)
    P = dyn_cast<PHINode>(I->getOperand(1));
  return P && matchSimpleRecurrence(P, BO, Start, Step) && BO == I;
}

bool llvm::matchSimpleBinaryIntrinsicRecurrence(const IntrinsicInst *I,
                                                PHINode *&P, Value *&Init,
                                                Value *&OtherOp) {
  // Binary intrinsics only supported for now.
  if (I->arg_size() != 2 || I->getType() != I->getArgOperand(0)->getType() ||
      I->getType() != I->getArgOperand(1)->getType())
    return false;

  IntrinsicInst *II = nullptr;
  P = dyn_cast<PHINode>(I->getArgOperand(0));
  if (!P)
    P = dyn_cast<PHINode>(I->getArgOperand(1));

  return P && matchTwoInputRecurrence(P, II, Init, OtherOp) && II == I;
}

const Value *llvm::stripNullTest(const Value *V) {
  // (X >> C) or/add (X & mask(C) != 0)
  if (const auto *BO = dyn_cast<BinaryOperator>(V)) {
    if (BO->getOpcode() == Instruction::Add ||
        BO->getOpcode() == Instruction::Or) {
      const Value *X;
      const APInt *C1, *C2;
      if (match(BO, m_c_BinOp(m_LShr(m_Value(X), m_APInt(C1)),
                              m_ZExt(m_SpecificICmp(
                                  ICmpInst::ICMP_NE,
                                  m_And(m_Deferred(X), m_LowBitMask(C2)),
                                  m_Zero())))) &&
          C2->popcount() == C1->getZExtValue())
        return X;
    }
  }
  return nullptr;
}

Value *llvm::stripNullTest(Value *V) {
  return const_cast<Value *>(stripNullTest(const_cast<const Value *>(V)));
}

bool llvm::collectPossibleValues(const Value *V,
                                 SmallPtrSetImpl<const Constant *> &Constants,
                                 unsigned MaxCount, bool AllowUndefOrPoison) {
  SmallPtrSet<const Instruction *, 8> Visited;
  SmallVector<const Instruction *, 8> Worklist;
  auto Push = [&](const Value *V) -> bool {
    Constant *C;
    if (match(const_cast<Value *>(V), m_ImmConstant(C))) {
      if (!AllowUndefOrPoison && !isGuaranteedNotToBeUndefOrPoison(C))
        return false;
      // Check existence first to avoid unnecessary allocations.
      if (Constants.contains(C))
        return true;
      if (Constants.size() == MaxCount)
        return false;
      Constants.insert(C);
      return true;
    }

    if (auto *Inst = dyn_cast<Instruction>(V)) {
      if (Visited.insert(Inst).second)
        Worklist.push_back(Inst);
      return true;
    }
    return false;
  };
  if (!Push(V))
    return false;
  while (!Worklist.empty()) {
    const Instruction *CurInst = Worklist.pop_back_val();
    switch (CurInst->getOpcode()) {
    case Instruction::Select:
      if (!Push(CurInst->getOperand(1)))
        return false;
      if (!Push(CurInst->getOperand(2)))
        return false;
      break;
    case Instruction::PHI:
      for (Value *IncomingValue : cast<PHINode>(CurInst)->incoming_values()) {
        // Fast path for recurrence PHI.
        if (IncomingValue == CurInst)
          continue;
        if (!Push(IncomingValue))
          return false;
      }
      break;
    default:
      return false;
    }
  }
  return true;
}

static void
addValueAffectedByCondition(Value *V,
                            function_ref<void(Value *)> InsertAffected) {
  assert(V != nullptr);
  if (isa<Argument>(V) || isa<GlobalValue>(V)) {
    InsertAffected(V);
  } else if (auto *I = dyn_cast<Instruction>(V)) {
    InsertAffected(V);

    // Peek through unary operators to find the source of the condition.
    Value *Op;
    if (match(I, m_CombineOr(m_PtrToIntOrAddr(m_Value(Op)),
                             m_Trunc(m_Value(Op))))) {
      if (isa<Instruction>(Op) || isa<Argument>(Op))
        InsertAffected(Op);
    }
  }
}

void llvm::findValuesAffectedByCondition(
    Value *Cond, bool IsAssume, function_ref<void(Value *)> InsertAffected) {
  auto AddAffected = [&InsertAffected](Value *V) {
    addValueAffectedByCondition(V, InsertAffected);
  };

  auto AddCmpOperands = [&AddAffected, IsAssume](Value *LHS, Value *RHS) {
    if (IsAssume) {
      AddAffected(LHS);
      AddAffected(RHS);
    } else if (match(RHS, m_Constant()))
      AddAffected(LHS);
  };

  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 8> Visited;
  Worklist.push_back(Cond);
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    CmpPredicate Pred;
    Value *A, *B, *X;

    if (IsAssume) {
      AddAffected(V);
      if (match(V, m_Not(m_Value(X))))
        AddAffected(X);
    }

    if (match(V, m_LogicalOp(m_Value(A), m_Value(B)))) {
      // assume(A && B) is split to -> assume(A); assume(B);
      // assume(!(A || B)) is split to -> assume(!A); assume(!B);
      // Finally, assume(A || B) / assume(!(A && B)) generally don't provide
      // enough information to be worth handling (intersection of information as
      // opposed to union).
      if (!IsAssume) {
        Worklist.push_back(A);
        Worklist.push_back(B);
      }
    } else if (match(V, m_ICmp(Pred, m_Value(A), m_Value(B)))) {
      bool HasRHSC = match(B, m_ConstantInt());
      if (ICmpInst::isEquality(Pred)) {
        AddAffected(A);
        if (IsAssume)
          AddAffected(B);
        if (HasRHSC) {
          Value *Y;
          // (X << C) or (X >>_s C) or (X >>_u C).
          if (match(A, m_Shift(m_Value(X), m_ConstantInt())))
            AddAffected(X);
          // (X & C) or (X | C).
          else if (match(A, m_And(m_Value(X), m_Value(Y))) ||
                   match(A, m_Or(m_Value(X), m_Value(Y)))) {
            AddAffected(X);
            AddAffected(Y);
          }
          // X - Y
          else if (match(A, m_Sub(m_Value(X), m_Value(Y)))) {
            AddAffected(X);
            AddAffected(Y);
          }
        }
      } else {
        AddCmpOperands(A, B);
        if (HasRHSC) {
          // Handle (A + C1) u< C2, which is the canonical form of
          // A > C3 && A < C4.
          if (match(A, m_AddLike(m_Value(X), m_ConstantInt())))
            AddAffected(X);

          if (ICmpInst::isUnsigned(Pred)) {
            Value *Y;
            // X & Y u> C    -> X >u C && Y >u C
            // X | Y u< C    -> X u< C && Y u< C
            // X nuw+ Y u< C -> X u< C && Y u< C
            if (match(A, m_And(m_Value(X), m_Value(Y))) ||
                match(A, m_Or(m_Value(X), m_Value(Y))) ||
                match(A, m_NUWAdd(m_Value(X), m_Value(Y)))) {
              AddAffected(X);
              AddAffected(Y);
            }
            // X nuw- Y u> C -> X u> C
            if (match(A, m_NUWSub(m_Value(X), m_Value())))
              AddAffected(X);
          }
        }

        // Handle icmp slt/sgt (bitcast X to int), 0/-1, which is supported
        // by computeKnownFPClass().
        if (match(A, m_ElementWiseBitCast(m_Value(X)))) {
          if (Pred == ICmpInst::ICMP_SLT && match(B, m_Zero()))
            InsertAffected(X);
          else if (Pred == ICmpInst::ICMP_SGT && match(B, m_AllOnes()))
            InsertAffected(X);
        }
      }

      if (HasRHSC && match(A, m_Intrinsic<Intrinsic::ctpop>(m_Value(X))))
        AddAffected(X);
    } else if (match(V, m_FCmp(Pred, m_Value(A), m_Value(B)))) {
      AddCmpOperands(A, B);

      // fcmp fneg(x), y
      // fcmp fabs(x), y
      // fcmp fneg(fabs(x)), y
      if (match(A, m_FNeg(m_Value(A))))
        AddAffected(A);
      if (match(A, m_FAbs(m_Value(A))))
        AddAffected(A);

    } else if (match(V, m_Intrinsic<Intrinsic::is_fpclass>(m_Value(A),
                                                           m_Value()))) {
      // Handle patterns that computeKnownFPClass() support.
      AddAffected(A);
    } else if (!IsAssume && match(V, m_Trunc(m_Value(X)))) {
      // Assume is checked here as X is already added above for assumes in
      // addValueAffectedByCondition
      AddAffected(X);
    } else if (!IsAssume && match(V, m_Not(m_Value(X)))) {
      // Assume is checked here to avoid issues with ephemeral values
      Worklist.push_back(X);
    }
  }
}

bool llvm::canIgnoreSignBitOfNaN(const Use &U) {
  auto *User = cast<Instruction>(U.getUser());
  if (auto *FPOp = dyn_cast<FPMathOperator>(User)) {
    if (FPOp->hasNoNaNs())
      return true;
  }

  switch (User->getOpcode()) {
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    return true;
  // Proper FP math operations ignore the sign bit of NaN.
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FCmp:
    return true;
  // Bitwise FP operations should preserve the sign bit of NaN.
  case Instruction::FNeg:
  case Instruction::Select:
  case Instruction::PHI:
    return false;
  case Instruction::Ret:
    return User->getFunction()->getAttributes().getRetNoFPClass() &
           FPClassTest::fcNan;
  case Instruction::Call:
  case Instruction::Invoke: {
    if (auto *II = dyn_cast<IntrinsicInst>(User)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::fabs:
        return true;
      case Intrinsic::copysign:
        return U.getOperandNo() == 0;
      // Other proper FP math intrinsics ignore the sign bit of NaN.
      case Intrinsic::maxnum:
      case Intrinsic::minnum:
      case Intrinsic::maximum:
      case Intrinsic::minimum:
      case Intrinsic::maximumnum:
      case Intrinsic::minimumnum:
      case Intrinsic::canonicalize:
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
      case Intrinsic::sqrt:
      case Intrinsic::pow:
      case Intrinsic::powi:
      case Intrinsic::fptoui_sat:
      case Intrinsic::fptosi_sat:
      case Intrinsic::is_fpclass:
      case Intrinsic::vp_is_fpclass:
        return true;
      default:
        return false;
      }
    }

    FPClassTest NoFPClass =
        cast<CallBase>(User)->getParamNoFPClass(U.getOperandNo());
    return NoFPClass & FPClassTest::fcNan;
  }
  default:
    return false;
  }
}

bool llvm::isOverflowIntrinsicNoWrap(const WithOverflowInst *WO,
                                     const DominatorTree &DT) {
  SmallVector<const BranchInst *, 2> GuardingBranches;
  SmallVector<const ExtractValueInst *, 2> Results;

  for (const User *U : WO->users()) {
    if (const auto *EVI = dyn_cast<ExtractValueInst>(U)) {
      assert(EVI->getNumIndices() == 1 && "Obvious from CI's type");

      if (EVI->getIndices()[0] == 0)
        Results.push_back(EVI);
      else {
        assert(EVI->getIndices()[0] == 1 && "Obvious from CI's type");

        for (const auto *U : EVI->users())
          if (const auto *B = dyn_cast<BranchInst>(U)) {
            assert(B->isConditional() && "How else is it using an i1?");
            GuardingBranches.push_back(B);
          }
      }
    } else {
      // We are using the aggregate directly in a way we don't want to analyze
      // here (storing it to a global, say).
      return false;
    }
  }

  auto AllUsesGuardedByBranch = [&](const BranchInst *BI) {
    BasicBlockEdge NoWrapEdge(BI->getParent(), BI->getSuccessor(1));
    if (!NoWrapEdge.isSingleEdge())
      return false;

    // Check if all users of the add are provably no-wrap.
    for (const auto *Result : Results) {
      // If the extractvalue itself is not executed on overflow, the we don't
      // need to check each use separately, since domination is transitive.
      if (DT.dominates(NoWrapEdge, Result->getParent()))
        continue;

      for (const auto &RU : Result->uses())
        if (!DT.dominates(NoWrapEdge, RU))
          return false;
    }

    return true;
  };

  return llvm::any_of(GuardingBranches, AllUsesGuardedByBranch);
}

namespace llvm::vtutils {
// Returns the bitwidth of the given scalar or pointer type. For vector types,
/// returns the element type's bitwidth.
unsigned getBitWidth(Type *Ty, const DataLayout &DL) {
  if (unsigned BitWidth = Ty->getScalarSizeInBits())
    return BitWidth;

  return DL.getPointerTypeSizeInBits(Ty);
}

// Given the provided Value and, potentially, a context instruction, return
// the preferred context instruction (if any).
const Instruction *safeCxtI(const Value *V, const Instruction *CxtI) {
  // If we've been provided with a context instruction, then use that (provided
  // it has been inserted).
  if (CxtI && CxtI->getParent())
    return CxtI;

  // If the value is really an already-inserted instruction, then use that.
  CxtI = dyn_cast<Instruction>(V);
  if (CxtI && CxtI->getParent())
    return CxtI;

  return nullptr;
}

bool getShuffleDemandedElts(const ShuffleVectorInst *Shuf,
                                   const APInt &DemandedElts,
                                   APInt &DemandedLHS, APInt &DemandedRHS) {
  if (isa<ScalableVectorType>(Shuf->getType())) {
    assert(DemandedElts == APInt(1, 1));
    DemandedLHS = DemandedRHS = DemandedElts;
    return true;
  }

  int NumElts =
      cast<FixedVectorType>(Shuf->getOperand(0)->getType())->getNumElements();
  return llvm::getShuffleDemandedElts(NumElts, Shuf->getShuffleMask(),
                                      DemandedElts, DemandedLHS, DemandedRHS);
}

bool isEphemeralValueOf(const Instruction *I, const Value *E) {
  SmallVector<const Instruction *, 16> WorkSet(1, I);
  SmallPtrSet<const Instruction *, 32> Visited;
  SmallPtrSet<const Instruction *, 16> EphValues;

  // The instruction defining an assumption's condition itself is always
  // considered ephemeral to that assumption (even if it has other
  // non-ephemeral users). See r246696's test case for an example.
  if (is_contained(I->operands(), E))
    return true;

  while (!WorkSet.empty()) {
    const Instruction *V = WorkSet.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    // If all uses of this value are ephemeral, then so is this value.
    if (all_of(V->users(), [&](const User *U) {
          return EphValues.count(cast<Instruction>(U));
        })) {
      if (V == E)
        return true;

      if (V == I || (!V->mayHaveSideEffects() && !V->isTerminator())) {
        EphValues.insert(V);

        if (const User *U = dyn_cast<User>(V)) {
          for (const Use &U : U->operands()) {
            if (const auto *I = dyn_cast<Instruction>(U.get()))
              WorkSet.push_back(I);
          }
        }
      }
    }
  }

  return false;
}

bool programUndefinedIfUndefOrPoison(const Value *V, bool PoisonOnly) {
  // We currently only look for uses of values within the same basic
  // block, as that makes it easier to guarantee that the uses will be
  // executed given that Inst is executed.
  //
  // FIXME: Expand this to consider uses beyond the same basic block. To do
  // this, look out for the distinction between post-dominance and strong
  // post-dominance.
  const BasicBlock *BB = nullptr;
  BasicBlock::const_iterator Begin;
  if (const auto *Inst = dyn_cast<Instruction>(V)) {
    BB = Inst->getParent();
    Begin = Inst->getIterator();
    Begin++;
  } else if (const auto *Arg = dyn_cast<Argument>(V)) {
    if (Arg->getParent()->isDeclaration())
      return false;
    BB = &Arg->getParent()->getEntryBlock();
    Begin = BB->begin();
  } else {
    return false;
  }

  // Limit number of instructions we look at, to avoid scanning through large
  // blocks. The current limit is chosen arbitrarily.
  unsigned ScanLimit = 32;
  BasicBlock::const_iterator End = BB->end();

  if (!PoisonOnly) {
    // Since undef does not propagate eagerly, be conservative & just check
    // whether a value is directly passed to an instruction that must take
    // well-defined operands.

    for (const auto &I : make_range(Begin, End)) {
      if (--ScanLimit == 0)
        break;

      if (handleGuaranteedWellDefinedOps(&I, [V](const Value *WellDefinedOp) {
            return WellDefinedOp == V;
          }))
        return true;

      if (!isGuaranteedToTransferExecutionToSuccessor(&I))
        break;
    }
    return false;
  }

  // Set of instructions that we have proved will yield poison if Inst
  // does.
  SmallPtrSet<const Value *, 16> YieldsPoison;
  SmallPtrSet<const BasicBlock *, 4> Visited;

  YieldsPoison.insert(V);
  Visited.insert(BB);

  while (true) {
    for (const auto &I : make_range(Begin, End)) {
      if (--ScanLimit == 0)
        return false;
      if (mustTriggerUB(&I, YieldsPoison))
        return true;
      if (!isGuaranteedToTransferExecutionToSuccessor(&I))
        return false;

      // If an operand is poison and propagates it, mark I as yielding poison.
      for (const Use &Op : I.operands()) {
        if (YieldsPoison.count(Op) && propagatesPoison(Op)) {
          YieldsPoison.insert(&I);
          break;
        }
      }

      // Special handling for select, which returns poison if its operand 0 is
      // poison (handled in the loop above) *or* if both its true/false operands
      // are poison (handled here).
      if (I.getOpcode() == Instruction::Select &&
          YieldsPoison.count(I.getOperand(1)) &&
          YieldsPoison.count(I.getOperand(2))) {
        YieldsPoison.insert(&I);
      }
    }

    BB = BB->getSingleSuccessor();
    if (!BB || !Visited.insert(BB).second)
      break;

    Begin = BB->getFirstNonPHIIt();
    End = BB->end();
  }
  return false;
}

bool includesPoison(UndefPoisonKind Kind) {
  return (unsigned(Kind) & unsigned(UndefPoisonKind::PoisonOnly)) != 0;
}

bool includesUndef(UndefPoisonKind Kind) {
  return (unsigned(Kind) & unsigned(UndefPoisonKind::UndefOnly)) != 0;
}

/// Shifts return poison if shiftwidth is larger than the bitwidth.
static bool shiftAmountKnownInRange(const Value *ShiftAmount) {
  auto *C = dyn_cast<Constant>(ShiftAmount);
  if (!C)
    return false;

  // Shifts return poison if shiftwidth is larger than the bitwidth.
  SmallVector<const Constant *, 4> ShiftAmounts;
  if (auto *FVTy = dyn_cast<FixedVectorType>(C->getType())) {
    unsigned NumElts = FVTy->getNumElements();
    for (unsigned i = 0; i < NumElts; ++i)
      ShiftAmounts.push_back(C->getAggregateElement(i));
  } else if (isa<ScalableVectorType>(C->getType()))
    return false; // Can't tell, just return false to be safe
  else
    ShiftAmounts.push_back(C);

  bool Safe = llvm::all_of(ShiftAmounts, [](const Constant *C) {
    auto *CI = dyn_cast_or_null<ConstantInt>(C);
    return CI && CI->getValue().ult(C->getType()->getIntegerBitWidth());
  });

  return Safe;
}

bool canCreateUndefOrPoison(const Operator *Op, UndefPoisonKind Kind,
                            bool ConsiderFlagsAndMetadata) {

  if (ConsiderFlagsAndMetadata && includesPoison(Kind) &&
      Op->hasPoisonGeneratingAnnotations())
    return true;

  unsigned Opcode = Op->getOpcode();

  // Check whether opcode is a poison/undef-generating operation
  switch (Opcode) {
  case Instruction::Shl:
  case Instruction::AShr:
  case Instruction::LShr:
    return includesPoison(Kind) && !shiftAmountKnownInRange(Op->getOperand(1));
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    // fptosi/ui yields poison if the resulting value does not fit in the
    // destination type.
    return true;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(Op)) {
      switch (II->getIntrinsicID()) {
      // TODO: Add more intrinsics.
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
      case Intrinsic::abs:
        // We're not considering flags so it is safe to just return false.
        return false;
      case Intrinsic::sshl_sat:
      case Intrinsic::ushl_sat:
        if (!includesPoison(Kind) ||
            shiftAmountKnownInRange(II->getArgOperand(1)))
          return false;
        break;
      }
    }
    [[fallthrough]];
  case Instruction::CallBr:
  case Instruction::Invoke: {
    const auto *CB = cast<CallBase>(Op);
    return !CB->hasRetAttr(Attribute::NoUndef) &&
           !CB->hasFnAttr(Attribute::NoCreateUndefOrPoison);
  }
  case Instruction::InsertElement:
  case Instruction::ExtractElement: {
    // If index exceeds the length of the vector, it returns poison
    auto *VTy = cast<VectorType>(Op->getOperand(0)->getType());
    unsigned IdxOp = Op->getOpcode() == Instruction::InsertElement ? 2 : 1;
    auto *Idx = dyn_cast<ConstantInt>(Op->getOperand(IdxOp));
    if (includesPoison(Kind))
      return !Idx ||
             Idx->getValue().uge(VTy->getElementCount().getKnownMinValue());
    return false;
  }
  case Instruction::ShuffleVector: {
    ArrayRef<int> Mask = isa<ConstantExpr>(Op)
                             ? cast<ConstantExpr>(Op)->getShuffleMask()
                             : cast<ShuffleVectorInst>(Op)->getShuffleMask();
    return includesPoison(Kind) && is_contained(Mask, PoisonMaskElem);
  }
  case Instruction::FNeg:
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
  case Instruction::Freeze:
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::GetElementPtr:
    return false;
  case Instruction::AddrSpaceCast:
    return true;
  default: {
    const auto *CE = dyn_cast<ConstantExpr>(Op);
    if (isa<CastInst>(Op) || (CE && CE->isCast()))
      return false;
    else if (Instruction::isBinaryOp(Opcode))
      return false;
    // Be conservative and return true.
    return true;
  }
  }
}

// TODO: cmpExcludesZero misses many cases where `RHS` is non-constant but
// we still have enough information about `RHS` to conclude non-zero. For
// example Pred=EQ, RHS=isKnownNonZero. cmpExcludesZero is called in loops
// so the extra compile time may not be worth it, but possibly a second API
// should be created for use outside of loops.
bool cmpExcludesZero(CmpInst::Predicate Pred, const Value *RHS) {
  // v u> y implies v != 0.
  if (Pred == ICmpInst::ICMP_UGT)
    return true;

  // Special-case v != 0 to also handle v != null.
  if (Pred == ICmpInst::ICMP_NE)
    return match(RHS, m_Zero());

  // All other predicates - rely on generic ConstantRange handling.
  const APInt *C;
  auto Zero = APInt::getZero(RHS->getType()->getScalarSizeInBits());
  if (match(RHS, m_APInt(C))) {
    ConstantRange TrueValues = ConstantRange::makeExactICmpRegion(Pred, *C);
    return !TrueValues.contains(Zero);
  }

  auto *VC = dyn_cast<ConstantDataVector>(RHS);
  if (VC == nullptr)
    return false;

  for (unsigned ElemIdx = 0, NElem = VC->getNumElements(); ElemIdx < NElem;
       ++ElemIdx) {
    ConstantRange TrueValues = ConstantRange::makeExactICmpRegion(
        Pred, VC->getElementAsAPInt(ElemIdx));
    if (TrueValues.contains(Zero))
      return false;
  }
  return true;
}

void breakSelfRecursivePHI(const Use *U, const PHINode *PHI,
                                  Value *&ValOut, Instruction *&CtxIOut,
                                  const PHINode **PhiOut) {
  ValOut = U->get();
  if (ValOut == PHI)
    return;
  CtxIOut = PHI->getIncomingBlock(*U)->getTerminator();
  if (PhiOut)
    *PhiOut = PHI;
  Value *V;
  // If the Use is a select of this phi, compute analysis on other arm to break
  // recursion.
  // TODO: Min/Max
  if (match(ValOut, m_Select(m_Value(), m_Specific(PHI), m_Value(V))) ||
      match(ValOut, m_Select(m_Value(), m_Value(V), m_Specific(PHI))))
    ValOut = V;

  // Same for select, if this phi is 2-operand phi, compute analysis on other
  // incoming value to break recursion.
  // TODO: We could handle any number of incoming edges as long as we only have
  // two unique values.
  if (auto *IncPhi = dyn_cast<PHINode>(ValOut);
      IncPhi && IncPhi->getNumIncomingValues() == 2) {
    for (int Idx = 0; Idx < 2; ++Idx) {
      if (IncPhi->getIncomingValue(Idx) == PHI) {
        ValOut = IncPhi->getIncomingValue(1 - Idx);
        if (PhiOut)
          *PhiOut = IncPhi;
        CtxIOut = IncPhi->getIncomingBlock(1 - Idx)->getTerminator();
        break;
      }
    }
  }
}

bool isKnownNonZeroFromAssume(const Value *V, const SimplifyQuery &Q) {
  // Use of assumptions is context-sensitive. If we don't have a context, we
  // cannot use them!
  if (!Q.AC || !Q.CxtI)
    return false;

  for (AssumptionCache::ResultElem &Elem : Q.AC->assumptionsFor(V)) {
    if (!Elem.Assume)
      continue;

    AssumeInst *I = cast<AssumeInst>(Elem.Assume);
    assert(I->getFunction() == Q.CxtI->getFunction() &&
           "Got assumption for the wrong function!");

    if (Elem.Index != AssumptionCache::ExprResultIdx) {
      if (!V->getType()->isPointerTy())
        continue;
      if (RetainedKnowledge RK = getKnowledgeFromBundle(
              *I, I->bundle_op_info_begin()[Elem.Index])) {
        if (RK.WasOn != V)
          continue;
        bool AssumeImpliesNonNull = [&]() {
          if (RK.AttrKind == Attribute::NonNull)
            return true;

          if (RK.AttrKind == Attribute::Dereferenceable) {
            if (NullPointerIsDefined(Q.CxtI->getFunction(),
                                     V->getType()->getPointerAddressSpace()))
              return false;
            assert(RK.IRArgValue &&
                   "Dereferenceable attribute without IR argument?");

            auto *CI = dyn_cast<ConstantInt>(RK.IRArgValue);
            return CI && !CI->isZero();
          }

          return false;
        }();
        if (AssumeImpliesNonNull && isValidAssumeForContext(I, Q.CxtI, Q.DT))
          return true;
      }
      continue;
    }

    // Warning: This loop can end up being somewhat performance sensitive.
    // We're running this loop for once for each value queried resulting in a
    // runtime of ~O(#assumes * #values).

    Value *RHS;
    CmpPredicate Pred;
    auto m_V = m_CombineOr(m_Specific(V), m_PtrToInt(m_Specific(V)));
    if (!match(I->getArgOperand(0), m_c_ICmp(Pred, m_V, m_Value(RHS))))
      continue;

    if (cmpExcludesZero(Pred, RHS) && isValidAssumeForContext(I, Q.CxtI, Q.DT))
      return true;
  }

  return false;
}

void computeKnownBitsFromCmp(const Value *V, CmpInst::Predicate Pred,
                                    Value *LHS, Value *RHS, KnownBits &Known,
                                    const SimplifyQuery &Q) {
  if (RHS->getType()->isPointerTy()) {
    // Handle comparison of pointer to null explicitly, as it will not be
    // covered by the m_APInt() logic below.
    if (LHS == V && match(RHS, m_Zero())) {
      switch (Pred) {
      case ICmpInst::ICMP_EQ:
        Known.setAllZero();
        break;
      case ICmpInst::ICMP_SGE:
      case ICmpInst::ICMP_SGT:
        Known.makeNonNegative();
        break;
      case ICmpInst::ICMP_SLT:
        Known.makeNegative();
        break;
      default:
        break;
      }
    }
    return;
  }

  unsigned BitWidth = Known.getBitWidth();
  auto m_V =
      m_CombineOr(m_Specific(V), m_PtrToIntSameSize(Q.DL, m_Specific(V)));

  Value *Y;
  const APInt *Mask, *C;
  if (!match(RHS, m_APInt(C)))
    return;

  uint64_t ShAmt;
  switch (Pred) {
  case ICmpInst::ICMP_EQ:
    // assume(V = C)
    if (match(LHS, m_V)) {
      Known = Known.unionWith(KnownBits::makeConstant(*C));
      // assume(V & Mask = C)
    } else if (match(LHS, m_c_And(m_V, m_Value(Y)))) {
      // For one bits in Mask, we can propagate bits from C to V.
      Known.One |= *C;
      if (match(Y, m_APInt(Mask)))
        Known.Zero |= ~*C & *Mask;
      // assume(V | Mask = C)
    } else if (match(LHS, m_c_Or(m_V, m_Value(Y)))) {
      // For zero bits in Mask, we can propagate bits from C to V.
      Known.Zero |= ~*C;
      if (match(Y, m_APInt(Mask)))
        Known.One |= *C & ~*Mask;
      // assume(V << ShAmt = C)
    } else if (match(LHS, m_Shl(m_V, m_ConstantInt(ShAmt))) &&
               ShAmt < BitWidth) {
      // For those bits in C that are known, we can propagate them to known
      // bits in V shifted to the right by ShAmt.
      KnownBits RHSKnown = KnownBits::makeConstant(*C);
      RHSKnown >>= ShAmt;
      Known = Known.unionWith(RHSKnown);
      // assume(V >> ShAmt = C)
    } else if (match(LHS, m_Shr(m_V, m_ConstantInt(ShAmt))) &&
               ShAmt < BitWidth) {
      // For those bits in RHS that are known, we can propagate them to known
      // bits in V shifted to the right by C.
      KnownBits RHSKnown = KnownBits::makeConstant(*C);
      RHSKnown <<= ShAmt;
      Known = Known.unionWith(RHSKnown);
    }
    break;
  case ICmpInst::ICMP_NE: {
    // assume (V & B != 0) where B is a power of 2
    const APInt *BPow2;
    if (C->isZero() && match(LHS, m_And(m_V, m_Power2(BPow2))))
      Known.One |= *BPow2;
    break;
  }
  default: {
    const APInt *Offset = nullptr;
    if (match(LHS, m_CombineOr(m_V, m_AddLike(m_V, m_APInt(Offset))))) {
      ConstantRange LHSRange = ConstantRange::makeAllowedICmpRegion(Pred, *C);
      if (Offset)
        LHSRange = LHSRange.sub(*Offset);
      Known = Known.unionWith(LHSRange.toKnownBits());
    }
    if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE) {
      // X & Y u> C     -> X u> C && Y u> C
      // X nuw- Y u> C  -> X u> C
      if (match(LHS, m_c_And(m_V, m_Value())) ||
          match(LHS, m_NUWSub(m_V, m_Value())))
        Known.One.setHighBits(
            (*C + (Pred == ICmpInst::ICMP_UGT)).countLeadingOnes());
    }
    if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_ULE) {
      // X | Y u< C    -> X u< C && Y u< C
      // X nuw+ Y u< C -> X u< C && Y u< C
      if (match(LHS, m_c_Or(m_V, m_Value())) ||
          match(LHS, m_c_NUWAdd(m_V, m_Value()))) {
        Known.Zero.setHighBits(
            (*C - (Pred == ICmpInst::ICMP_ULT)).countLeadingZeros());
      }
    }
  } break;
  }
}

void computeKnownBitsFromICmpCond(const Value *V, ICmpInst *Cmp,
                                         KnownBits &Known,
                                         const SimplifyQuery &SQ, bool Invert) {
  ICmpInst::Predicate Pred =
      Invert ? Cmp->getInversePredicate() : Cmp->getPredicate();
  Value *LHS = Cmp->getOperand(0);
  Value *RHS = Cmp->getOperand(1);

  // Handle icmp pred (trunc V), C
  if (match(LHS, m_Trunc(m_Specific(V)))) {
    KnownBits DstKnown(LHS->getType()->getScalarSizeInBits());
    computeKnownBitsFromCmp(LHS, Pred, LHS, RHS, DstKnown, SQ);
    if (cast<TruncInst>(LHS)->hasNoUnsignedWrap())
      Known = Known.unionWith(DstKnown.zext(Known.getBitWidth()));
    else
      Known = Known.unionWith(DstKnown.anyext(Known.getBitWidth()));
    return;
  }

  computeKnownBitsFromCmp(V, Pred, LHS, RHS, Known, SQ);
}

bool isKnownNonNullFromDominatingCondition(const Value *V,
                                                  const Instruction *CtxI,
                                                  const DominatorTree *DT) {
  assert(!isa<Constant>(V) && "Called for constant?");

  if (!CtxI || !DT)
    return false;

  unsigned NumUsesExplored = 0;
  for (auto &U : V->uses()) {
    // Avoid massive lists
    if (NumUsesExplored >= DomConditionsMaxUses)
      break;
    NumUsesExplored++;

    const Instruction *UI = cast<Instruction>(U.getUser());
    // If the value is used as an argument to a call or invoke, then argument
    // attributes may provide an answer about null-ness.
    if (V->getType()->isPointerTy()) {
      if (const auto *CB = dyn_cast<CallBase>(UI)) {
        if (CB->isArgOperand(&U) &&
            CB->paramHasNonNullAttr(CB->getArgOperandNo(&U),
                                    /*AllowUndefOrPoison=*/false) &&
            DT->dominates(CB, CtxI))
          return true;
      }
    }

    // If the value is used as a load/store, then the pointer must be non null.
    if (V == getLoadStorePointerOperand(UI)) {
      if (!NullPointerIsDefined(UI->getFunction(),
                                V->getType()->getPointerAddressSpace()) &&
          DT->dominates(UI, CtxI))
        return true;
    }

    if ((match(UI, m_IDiv(m_Value(), m_Specific(V))) ||
         match(UI, m_IRem(m_Value(), m_Specific(V)))) &&
        isValidAssumeForContext(UI, CtxI, DT))
      return true;

    // Consider only compare instructions uniquely controlling a branch
    Value *RHS;
    CmpPredicate Pred;
    if (!match(UI, m_c_ICmp(Pred, m_Specific(V), m_Value(RHS))))
      continue;

    bool NonNullIfTrue;
    if (cmpExcludesZero(Pred, RHS))
      NonNullIfTrue = true;
    else if (cmpExcludesZero(CmpInst::getInversePredicate(Pred), RHS))
      NonNullIfTrue = false;
    else
      continue;

    SmallVector<const User *, 4> WorkList;
    SmallPtrSet<const User *, 4> Visited;
    for (const auto *CmpU : UI->users()) {
      assert(WorkList.empty() && "Should be!");
      if (Visited.insert(CmpU).second)
        WorkList.push_back(CmpU);

      while (!WorkList.empty()) {
        auto *Curr = WorkList.pop_back_val();

        // If a user is an AND, add all its users to the work list. We only
        // propagate "pred != null" condition through AND because it is only
        // correct to assume that all conditions of AND are met in true branch.
        // TODO: Support similar logic of OR and EQ predicate?
        if (NonNullIfTrue)
          if (match(Curr, m_LogicalAnd(m_Value(), m_Value()))) {
            for (const auto *CurrU : Curr->users())
              if (Visited.insert(CurrU).second)
                WorkList.push_back(CurrU);
            continue;
          }

        if (const BranchInst *BI = dyn_cast<BranchInst>(Curr)) {
          assert(BI->isConditional() && "uses a comparison!");

          BasicBlock *NonNullSuccessor =
              BI->getSuccessor(NonNullIfTrue ? 0 : 1);
          BasicBlockEdge Edge(BI->getParent(), NonNullSuccessor);
          if (Edge.isSingleEdge() && DT->dominates(Edge, CtxI->getParent()))
            return true;
        } else if (NonNullIfTrue && isGuard(Curr) &&
                   DT->dominates(cast<Instruction>(Curr), CtxI)) {
          return true;
        }
      }
    }
  }

  return false;
}

// Match a signed min+max clamp pattern like smax(smin(In, CHigh), CLow).
// Returns the input and lower/upper bounds.
bool isSignedMinMaxClamp(const Value *Select, const Value *&In,
                                const APInt *&CLow, const APInt *&CHigh) {
  assert(isa<Operator>(Select) &&
         cast<Operator>(Select)->getOpcode() == Instruction::Select &&
         "Input should be a Select!");

  const Value *LHS = nullptr, *RHS = nullptr;
  SelectPatternFlavor SPF = matchSelectPattern(Select, LHS, RHS).Flavor;
  if (SPF != SPF_SMAX && SPF != SPF_SMIN)
    return false;

  if (!match(RHS, m_APInt(CLow)))
    return false;

  const Value *LHS2 = nullptr, *RHS2 = nullptr;
  SelectPatternFlavor SPF2 = matchSelectPattern(LHS, LHS2, RHS2).Flavor;
  if (getInverseMinMaxFlavor(SPF) != SPF2)
    return false;

  if (!match(RHS2, m_APInt(CHigh)))
    return false;

  if (SPF == SPF_SMIN)
    std::swap(CLow, CHigh);

  In = LHS2;
  return CLow->sle(*CHigh);
}

bool isSignedMinMaxIntrinsicClamp(const IntrinsicInst *II,
                                         const APInt *&CLow,
                                         const APInt *&CHigh) {
  assert((II->getIntrinsicID() == Intrinsic::smin ||
          II->getIntrinsicID() == Intrinsic::smax) &&
         "Must be smin/smax");

  Intrinsic::ID InverseID = getInverseMinMaxIntrinsic(II->getIntrinsicID());
  auto *InnerII = dyn_cast<IntrinsicInst>(II->getArgOperand(0));
  if (!InnerII || InnerII->getIntrinsicID() != InverseID ||
      !match(II->getArgOperand(1), m_APInt(CLow)) ||
      !match(InnerII->getArgOperand(1), m_APInt(CHigh)))
    return false;

  if (II->getIntrinsicID() == Intrinsic::smin)
    std::swap(CLow, CHigh);
  return CLow->sle(*CHigh);
}

void unionWithMinMaxIntrinsicClamp(const IntrinsicInst *II,
                                          KnownBits &Known) {
  const APInt *CLow, *CHigh;
  if (isSignedMinMaxIntrinsicClamp(II, CLow, CHigh))
    Known = Known.unionWith(
        ConstantRange::getNonEmpty(*CLow, *CHigh + 1).toKnownBits());
}

/// Return true if we can infer that \p V is known to be a power of 2 from
/// dominating condition \p Cond (e.g., ctpop(V) == 1).
bool isImpliedToBeAPowerOfTwoFromCond(const Value *V, bool OrZero,
                                             const Value *Cond,
                                             bool CondIsTrue) {
  CmpPredicate Pred;
  const APInt *RHSC;
  if (!match(Cond, m_ICmp(Pred, m_Intrinsic<Intrinsic::ctpop>(m_Specific(V)),
                          m_APInt(RHSC))))
    return false;
  if (!CondIsTrue)
    Pred = ICmpInst::getInversePredicate(Pred);
  // ctpop(V) u< 2
  if (OrZero && Pred == ICmpInst::ICMP_ULT && *RHSC == 2)
    return true;
  // ctpop(V) == 1
  return Pred == ICmpInst::ICMP_EQ && *RHSC == 1;
}

/// Try to detect a recurrence that monotonically increases/decreases from a
/// non-zero starting value. These are common as induction variables.
bool isNonZeroRecurrence(const PHINode *PN) {
  BinaryOperator *BO = nullptr;
  Value *Start = nullptr, *Step = nullptr;
  const APInt *StartC, *StepC;
  if (!matchSimpleRecurrence(PN, BO, Start, Step) ||
      !match(Start, m_APInt(StartC)) || StartC->isZero())
    return false;

  switch (BO->getOpcode()) {
  case Instruction::Add:
    // Starting from non-zero and stepping away from zero can never wrap back
    // to zero.
    return BO->hasNoUnsignedWrap() ||
           (BO->hasNoSignedWrap() && match(Step, m_APInt(StepC)) &&
            StartC->isNegative() == StepC->isNegative());
  case Instruction::Mul:
    return (BO->hasNoUnsignedWrap() || BO->hasNoSignedWrap()) &&
           match(Step, m_APInt(StepC)) && !StepC->isZero();
  case Instruction::Shl:
    return BO->hasNoUnsignedWrap() || BO->hasNoSignedWrap();
  case Instruction::AShr:
  case Instruction::LShr:
    return BO->isExact();
  default:
    return false;
  }
}

bool matchOpWithOpEqZero(Value *Op0, Value *Op1) {
  return match(Op0, m_ZExtOrSExt(m_SpecificICmp(ICmpInst::ICMP_EQ,
                                                m_Specific(Op1), m_Zero()))) ||
         match(Op1, m_ZExtOrSExt(m_SpecificICmp(ICmpInst::ICMP_EQ,
                                                m_Specific(Op0), m_Zero())));
}

// Check to see if A is both a GEP and is the incoming value for a PHI in the
// loop, and B is either a ptr or another GEP. If the PHI has 2 incoming values,
// one of them being the recursive GEP A and the other a ptr at same base and at
// the same/higher offset than B we are only incrementing the pointer further in
// loop if offset of recursive GEP is greater than 0.
bool isNonEqualPointersWithRecursiveGEP(const Value *A, const Value *B,
                                               const SimplifyQuery &Q) {
  if (!A->getType()->isPointerTy() || !B->getType()->isPointerTy())
    return false;

  auto *GEPA = dyn_cast<GEPOperator>(A);
  if (!GEPA || GEPA->getNumIndices() != 1 || !isa<Constant>(GEPA->idx_begin()))
    return false;

  // Handle 2 incoming PHI values with one being a recursive GEP.
  auto *PN = dyn_cast<PHINode>(GEPA->getPointerOperand());
  if (!PN || PN->getNumIncomingValues() != 2)
    return false;

  // Search for the recursive GEP as an incoming operand, and record that as
  // Step.
  Value *Start = nullptr;
  Value *Step = const_cast<Value *>(A);
  if (PN->getIncomingValue(0) == Step)
    Start = PN->getIncomingValue(1);
  else if (PN->getIncomingValue(1) == Step)
    Start = PN->getIncomingValue(0);
  else
    return false;

  // Other incoming node base should match the B base.
  // StartOffset >= OffsetB && StepOffset > 0?
  // StartOffset <= OffsetB && StepOffset < 0?
  // Is non-equal if above are true.
  // We use stripAndAccumulateInBoundsConstantOffsets to restrict the
  // optimisation to inbounds GEPs only.
  unsigned IndexWidth = Q.DL.getIndexTypeSizeInBits(Start->getType());
  APInt StartOffset(IndexWidth, 0);
  Start = Start->stripAndAccumulateInBoundsConstantOffsets(Q.DL, StartOffset);
  APInt StepOffset(IndexWidth, 0);
  Step = Step->stripAndAccumulateInBoundsConstantOffsets(Q.DL, StepOffset);

  // Check if Base Pointer of Step matches the PHI.
  if (Step != PN)
    return false;
  APInt OffsetB(IndexWidth, 0);
  B = B->stripAndAccumulateInBoundsConstantOffsets(Q.DL, OffsetB);
  return Start == B &&
         ((StartOffset.sge(OffsetB) && StepOffset.isStrictlyPositive()) ||
          (StartOffset.sle(OffsetB) && StepOffset.isNegative()));
}

bool isKnownNonNaN(const Value *V, FastMathFlags FMF) {
  if (FMF.noNaNs())
    return true;

  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isNaN();

  if (auto *C = dyn_cast<ConstantDataVector>(V)) {
    if (!C->getElementType()->isFloatingPointTy())
      return false;
    for (unsigned I = 0, E = C->getNumElements(); I < E; ++I) {
      if (C->getElementAsAPFloat(I).isNaN())
        return false;
    }
    return true;
  }

  if (isa<ConstantAggregateZero>(V))
    return true;

  return false;
}

bool isKnownNonZero(const Value *V) {
  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isZero();

  if (auto *C = dyn_cast<ConstantDataVector>(V)) {
    if (!C->getElementType()->isFloatingPointTy())
      return false;
    for (unsigned I = 0, E = C->getNumElements(); I < E; ++I) {
      if (C->getElementAsAPFloat(I).isZero())
        return false;
    }
    return true;
  }

  return false;
}
/// Return true if "icmp Pred LHS RHS" is always true.
static bool isTruePredicate(CmpInst::Predicate Pred, const Value *LHS,
                            const Value *RHS) {
  if (ICmpInst::isTrueWhenEqual(Pred) && LHS == RHS)
    return true;

  switch (Pred) {
  default:
    return false;

  case CmpInst::ICMP_SLE: {
    const APInt *C;

    // LHS s<= LHS +_{nsw} C   if C >= 0
    // LHS s<= LHS | C         if C >= 0
    if (match(RHS, m_NSWAdd(m_Specific(LHS), m_APInt(C))) ||
        match(RHS, m_Or(m_Specific(LHS), m_APInt(C))))
      return !C->isNegative();

    // LHS s<= smax(LHS, V) for any V
    if (match(RHS, m_c_SMax(m_Specific(LHS), m_Value())))
      return true;

    // smin(RHS, V) s<= RHS for any V
    if (match(LHS, m_c_SMin(m_Specific(RHS), m_Value())))
      return true;

    // Match A to (X +_{nsw} CA) and B to (X +_{nsw} CB)
    const Value *X;
    const APInt *CLHS, *CRHS;
    if (match(LHS, m_NSWAddLike(m_Value(X), m_APInt(CLHS))) &&
        match(RHS, m_NSWAddLike(m_Specific(X), m_APInt(CRHS))))
      return CLHS->sle(*CRHS);

    return false;
  }

  case CmpInst::ICMP_ULE: {
    // LHS u<= LHS +_{nuw} V for any V
    if (match(RHS, m_c_Add(m_Specific(LHS), m_Value())) &&
        cast<OverflowingBinaryOperator>(RHS)->hasNoUnsignedWrap())
      return true;

    // LHS u<= LHS | V for any V
    if (match(RHS, m_c_Or(m_Specific(LHS), m_Value())))
      return true;

    // LHS u<= umax(LHS, V) for any V
    if (match(RHS, m_c_UMax(m_Specific(LHS), m_Value())))
      return true;

    // RHS >> V u<= RHS for any V
    if (match(LHS, m_LShr(m_Specific(RHS), m_Value())))
      return true;

    // RHS u/ C_ugt_1 u<= RHS
    const APInt *C;
    if (match(LHS, m_UDiv(m_Specific(RHS), m_APInt(C))) && C->ugt(1))
      return true;

    // RHS & V u<= RHS for any V
    if (match(LHS, m_c_And(m_Specific(RHS), m_Value())))
      return true;

    // umin(RHS, V) u<= RHS for any V
    if (match(LHS, m_c_UMin(m_Specific(RHS), m_Value())))
      return true;

    // Match A to (X +_{nuw} CA) and B to (X +_{nuw} CB)
    const Value *X;
    const APInt *CLHS, *CRHS;
    if (match(LHS, m_NUWAddLike(m_Value(X), m_APInt(CLHS))) &&
        match(RHS, m_NUWAddLike(m_Specific(X), m_APInt(CRHS))))
      return CLHS->ule(*CRHS);

    return false;
  }
  }
}

static Value *lookThroughCastConst(CmpInst *CmpI, Type *SrcTy, Constant *C,
                                   Instruction::CastOps *CastOp) {
  const DataLayout &DL = CmpI->getDataLayout();

  Constant *CastedTo = nullptr;
  switch (*CastOp) {
  case Instruction::ZExt:
    if (CmpI->isUnsigned())
      CastedTo = ConstantExpr::getTrunc(C, SrcTy);
    break;
  case Instruction::SExt:
    if (CmpI->isSigned())
      CastedTo = ConstantExpr::getTrunc(C, SrcTy, true);
    break;
  case Instruction::Trunc:
    Constant *CmpConst;
    if (match(CmpI->getOperand(1), m_Constant(CmpConst)) &&
        CmpConst->getType() == SrcTy) {
      // Here we have the following case:
      //
      //   %cond = cmp iN %x, CmpConst
      //   %tr = trunc iN %x to iK
      //   %narrowsel = select i1 %cond, iK %t, iK C
      //
      // We can always move trunc after select operation:
      //
      //   %cond = cmp iN %x, CmpConst
      //   %widesel = select i1 %cond, iN %x, iN CmpConst
      //   %tr = trunc iN %widesel to iK
      //
      // Note that C could be extended in any way because we don't care about
      // upper bits after truncation. It can't be abs pattern, because it would
      // look like:
      //
      //   select i1 %cond, x, -x.
      //
      // So only min/max pattern could be matched. Such match requires widened C
      // == CmpConst. That is why set widened C = CmpConst, condition trunc
      // CmpConst == C is checked below.
      CastedTo = CmpConst;
    } else {
      unsigned ExtOp = CmpI->isSigned() ? Instruction::SExt : Instruction::ZExt;
      CastedTo = ConstantFoldCastOperand(ExtOp, C, SrcTy, DL);
    }
    break;
  case Instruction::FPTrunc:
    CastedTo = ConstantFoldCastOperand(Instruction::FPExt, C, SrcTy, DL);
    break;
  case Instruction::FPExt:
    CastedTo = ConstantFoldCastOperand(Instruction::FPTrunc, C, SrcTy, DL);
    break;
  case Instruction::FPToUI:
    CastedTo = ConstantFoldCastOperand(Instruction::UIToFP, C, SrcTy, DL);
    break;
  case Instruction::FPToSI:
    CastedTo = ConstantFoldCastOperand(Instruction::SIToFP, C, SrcTy, DL);
    break;
  case Instruction::UIToFP:
    CastedTo = ConstantFoldCastOperand(Instruction::FPToUI, C, SrcTy, DL);
    break;
  case Instruction::SIToFP:
    CastedTo = ConstantFoldCastOperand(Instruction::FPToSI, C, SrcTy, DL);
    break;
  default:
    break;
  }

  if (!CastedTo)
    return nullptr;

  // Make sure the cast doesn't lose any information.
  Constant *CastedBack =
      ConstantFoldCastOperand(*CastOp, CastedTo, C->getType(), DL);
  if (CastedBack && CastedBack != C)
    return nullptr;

  return CastedTo;
}

/// Helps to match a select pattern in case of a type mismatch.
///
/// The function processes the case when type of true and false values of a
/// select instruction differs from type of the cmp instruction operands because
/// of a cast instruction. The function checks if it is legal to move the cast
/// operation after "select". If yes, it returns the new second value of
/// "select" (with the assumption that cast is moved):
/// 1. As operand of cast instruction when both values of "select" are same cast
/// instructions.
/// 2. As restored constant (by applying reverse cast operation) when the first
/// value of the "select" is a cast operation and the second value is a
/// constant. It is implemented in lookThroughCastConst().
/// 3. As one operand is cast instruction and the other is not. The operands in
/// sel(cmp) are in different type integer.
/// NOTE: We return only the new second value because the first value could be
/// accessed as operand of cast instruction.
Value *lookThroughCast(CmpInst *CmpI, Value *V1, Value *V2,
                              Instruction::CastOps *CastOp) {
  auto *Cast1 = dyn_cast<CastInst>(V1);
  if (!Cast1)
    return nullptr;

  *CastOp = Cast1->getOpcode();
  Type *SrcTy = Cast1->getSrcTy();
  if (auto *Cast2 = dyn_cast<CastInst>(V2)) {
    // If V1 and V2 are both the same cast from the same type, look through V1.
    if (*CastOp == Cast2->getOpcode() && SrcTy == Cast2->getSrcTy())
      return Cast2->getOperand(0);
    return nullptr;
  }

  auto *C = dyn_cast<Constant>(V2);
  if (C)
    return lookThroughCastConst(CmpI, SrcTy, C, CastOp);

  Value *CastedTo = nullptr;
  if (*CastOp == Instruction::Trunc) {
    if (match(CmpI->getOperand(1), m_ZExtOrSExt(m_Specific(V2)))) {
      // Here we have the following case:
      //   %y_ext = sext iK %y to iN
      //   %cond = cmp iN %x, %y_ext
      //   %tr = trunc iN %x to iK
      //   %narrowsel = select i1 %cond, iK %tr, iK %y
      //
      // We can always move trunc after select operation:
      //   %y_ext = sext iK %y to iN
      //   %cond = cmp iN %x, %y_ext
      //   %widesel = select i1 %cond, iN %x, iN %y_ext
      //   %tr = trunc iN %widesel to iK
      assert(V2->getType() == Cast1->getType() &&
             "V2 and Cast1 should be the same type.");
      CastedTo = CmpI->getOperand(1);
    }
  }

  return CastedTo;
}

void setLimitForFPToI(const Instruction *I, APInt &Lower, APInt &Upper) {
  // The maximum representable value of a half is 65504. For floats the maximum
  // value is 3.4e38 which requires roughly 129 bits.
  unsigned BitWidth = I->getType()->getScalarSizeInBits();
  if (!I->getOperand(0)->getType()->getScalarType()->isHalfTy())
    return;
  if (isa<FPToSIInst>(I) && BitWidth >= 17) {
    Lower = APInt(BitWidth, -65504, true);
    Upper = APInt(BitWidth, 65505);
  }

  if (isa<FPToUIInst>(I) && BitWidth >= 16) {
    // For a fptoui the lower limit is left as 0.
    Upper = APInt(BitWidth, 65505);
  }
}

ConstantRange getRangeForIntrinsic(const IntrinsicInst &II,
                                          bool UseInstrInfo) {
  unsigned Width = II.getType()->getScalarSizeInBits();
  const APInt *C;
  switch (II.getIntrinsicID()) {
  case Intrinsic::ctlz:
  case Intrinsic::cttz: {
    APInt Upper(Width, Width);
    if (!UseInstrInfo || !match(II.getArgOperand(1), m_One()))
      Upper += 1;
    // Maximum of set/clear bits is the bit width.
    return ConstantRange::getNonEmpty(APInt::getZero(Width), Upper);
  }
  case Intrinsic::ctpop:
    // Maximum of set/clear bits is the bit width.
    return ConstantRange::getNonEmpty(APInt::getZero(Width),
                                      APInt(Width, Width) + 1);
  case Intrinsic::uadd_sat:
    // uadd.sat(x, C) produces [C, UINT_MAX].
    if (match(II.getOperand(0), m_APInt(C)) ||
        match(II.getOperand(1), m_APInt(C)))
      return ConstantRange::getNonEmpty(*C, APInt::getZero(Width));
    break;
  case Intrinsic::sadd_sat:
    if (match(II.getOperand(0), m_APInt(C)) ||
        match(II.getOperand(1), m_APInt(C))) {
      if (C->isNegative())
        // sadd.sat(x, -C) produces [SINT_MIN, SINT_MAX + (-C)].
        return ConstantRange::getNonEmpty(APInt::getSignedMinValue(Width),
                                          APInt::getSignedMaxValue(Width) + *C +
                                              1);

      // sadd.sat(x, +C) produces [SINT_MIN + C, SINT_MAX].
      return ConstantRange::getNonEmpty(APInt::getSignedMinValue(Width) + *C,
                                        APInt::getSignedMaxValue(Width) + 1);
    }
    break;
  case Intrinsic::usub_sat:
    // usub.sat(C, x) produces [0, C].
    if (match(II.getOperand(0), m_APInt(C)))
      return ConstantRange::getNonEmpty(APInt::getZero(Width), *C + 1);

    // usub.sat(x, C) produces [0, UINT_MAX - C].
    if (match(II.getOperand(1), m_APInt(C)))
      return ConstantRange::getNonEmpty(APInt::getZero(Width),
                                        APInt::getMaxValue(Width) - *C + 1);
    break;
  case Intrinsic::ssub_sat:
    if (match(II.getOperand(0), m_APInt(C))) {
      if (C->isNegative())
        // ssub.sat(-C, x) produces [SINT_MIN, -SINT_MIN + (-C)].
        return ConstantRange::getNonEmpty(APInt::getSignedMinValue(Width),
                                          *C - APInt::getSignedMinValue(Width) +
                                              1);

      // ssub.sat(+C, x) produces [-SINT_MAX + C, SINT_MAX].
      return ConstantRange::getNonEmpty(*C - APInt::getSignedMaxValue(Width),
                                        APInt::getSignedMaxValue(Width) + 1);
    } else if (match(II.getOperand(1), m_APInt(C))) {
      if (C->isNegative())
        // ssub.sat(x, -C) produces [SINT_MIN - (-C), SINT_MAX]:
        return ConstantRange::getNonEmpty(APInt::getSignedMinValue(Width) - *C,
                                          APInt::getSignedMaxValue(Width) + 1);

      // ssub.sat(x, +C) produces [SINT_MIN, SINT_MAX - C].
      return ConstantRange::getNonEmpty(APInt::getSignedMinValue(Width),
                                        APInt::getSignedMaxValue(Width) - *C +
                                            1);
    }
    break;
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::smin:
  case Intrinsic::smax:
    if (!match(II.getOperand(0), m_APInt(C)) &&
        !match(II.getOperand(1), m_APInt(C)))
      break;

    switch (II.getIntrinsicID()) {
    case Intrinsic::umin:
      return ConstantRange::getNonEmpty(APInt::getZero(Width), *C + 1);
    case Intrinsic::umax:
      return ConstantRange::getNonEmpty(*C, APInt::getZero(Width));
    case Intrinsic::smin:
      return ConstantRange::getNonEmpty(APInt::getSignedMinValue(Width),
                                        *C + 1);
    case Intrinsic::smax:
      return ConstantRange::getNonEmpty(*C,
                                        APInt::getSignedMaxValue(Width) + 1);
    default:
      llvm_unreachable("Must be min/max intrinsic");
    }
    break;
  case Intrinsic::abs:
    // If abs of SIGNED_MIN is poison, then the result is [0..SIGNED_MAX],
    // otherwise it is [0..SIGNED_MIN], as -SIGNED_MIN == SIGNED_MIN.
    if (match(II.getOperand(1), m_One()))
      return ConstantRange::getNonEmpty(APInt::getZero(Width),
                                        APInt::getSignedMaxValue(Width) + 1);

    return ConstantRange::getNonEmpty(APInt::getZero(Width),
                                      APInt::getSignedMinValue(Width) + 1);
  case Intrinsic::vscale:
    if (!II.getParent() || !II.getFunction())
      break;
    return getVScaleRange(II.getFunction(), Width);
  default:
    break;
  }

  return ConstantRange::getFull(Width);
}

ConstantRange getRangeForSelectPattern(const SelectInst &SI,
                                              const InstrInfoQuery &IIQ) {
  unsigned BitWidth = SI.getType()->getScalarSizeInBits();
  const Value *LHS = nullptr, *RHS = nullptr;
  SelectPatternResult R = matchSelectPattern(&SI, LHS, RHS);
  if (R.Flavor == SPF_UNKNOWN)
    return ConstantRange::getFull(BitWidth);

  if (R.Flavor == SelectPatternFlavor::SPF_ABS) {
    // If the negation part of the abs (in RHS) has the NSW flag,
    // then the result of abs(X) is [0..SIGNED_MAX],
    // otherwise it is [0..SIGNED_MIN], as -SIGNED_MIN == SIGNED_MIN.
    if (match(RHS, m_Neg(m_Specific(LHS))) &&
        IIQ.hasNoSignedWrap(cast<Instruction>(RHS)))
      return ConstantRange::getNonEmpty(APInt::getZero(BitWidth),
                                        APInt::getSignedMaxValue(BitWidth) + 1);

    return ConstantRange::getNonEmpty(APInt::getZero(BitWidth),
                                      APInt::getSignedMinValue(BitWidth) + 1);
  }

  if (R.Flavor == SelectPatternFlavor::SPF_NABS) {
    // The result of -abs(X) is <= 0.
    return ConstantRange::getNonEmpty(APInt::getSignedMinValue(BitWidth),
                                      APInt(BitWidth, 1));
  }

  const APInt *C;
  if (!match(LHS, m_APInt(C)) && !match(RHS, m_APInt(C)))
    return ConstantRange::getFull(BitWidth);

  switch (R.Flavor) {
  case SPF_UMIN:
    return ConstantRange::getNonEmpty(APInt::getZero(BitWidth), *C + 1);
  case SPF_UMAX:
    return ConstantRange::getNonEmpty(*C, APInt::getZero(BitWidth));
  case SPF_SMIN:
    return ConstantRange::getNonEmpty(APInt::getSignedMinValue(BitWidth),
                                      *C + 1);
  case SPF_SMAX:
    return ConstantRange::getNonEmpty(*C,
                                      APInt::getSignedMaxValue(BitWidth) + 1);
  default:
    return ConstantRange::getFull(BitWidth);
  }
}

void setLimitsForBinOp(const BinaryOperator &BO, APInt &Lower,
                              APInt &Upper, const InstrInfoQuery &IIQ,
                              bool PreferSignedRange) {
  unsigned Width = Lower.getBitWidth();
  const APInt *C;
  switch (BO.getOpcode()) {
  case Instruction::Sub:
    if (match(BO.getOperand(0), m_APInt(C))) {
      bool HasNSW = IIQ.hasNoSignedWrap(&BO);
      bool HasNUW = IIQ.hasNoUnsignedWrap(&BO);

      // If the caller expects a signed compare, then try to use a signed range.
      // Otherwise if both no-wraps are set, use the unsigned range because it
      // is never larger than the signed range. Example:
      // "sub nuw nsw i8 -2, x" is unsigned [0, 254] vs. signed [-128, 126].
      // "sub nuw nsw i8 2, x" is unsigned [0, 2] vs. signed [-125, 127].
      if (PreferSignedRange && HasNSW && HasNUW)
        HasNUW = false;

      if (HasNUW) {
        // 'sub nuw c, x' produces [0, C].
        Upper = *C + 1;
      } else if (HasNSW) {
        if (C->isNegative()) {
          // 'sub nsw -C, x' produces [SINT_MIN, -C - SINT_MIN].
          Lower = APInt::getSignedMinValue(Width);
          Upper = *C - APInt::getSignedMaxValue(Width);
        } else {
          // Note that sub 0, INT_MIN is not NSW. It techically is a signed wrap
          // 'sub nsw C, x' produces [C - SINT_MAX, SINT_MAX].
          Lower = *C - APInt::getSignedMaxValue(Width);
          Upper = APInt::getSignedMinValue(Width);
        }
      }
    }
    break;
  case Instruction::Add:
    if (match(BO.getOperand(1), m_APInt(C)) && !C->isZero()) {
      bool HasNSW = IIQ.hasNoSignedWrap(&BO);
      bool HasNUW = IIQ.hasNoUnsignedWrap(&BO);

      // If the caller expects a signed compare, then try to use a signed
      // range. Otherwise if both no-wraps are set, use the unsigned range
      // because it is never larger than the signed range. Example: "add nuw
      // nsw i8 X, -2" is unsigned [254,255] vs. signed [-128, 125].
      if (PreferSignedRange && HasNSW && HasNUW)
        HasNUW = false;

      if (HasNUW) {
        // 'add nuw x, C' produces [C, UINT_MAX].
        Lower = *C;
      } else if (HasNSW) {
        if (C->isNegative()) {
          // 'add nsw x, -C' produces [SINT_MIN, SINT_MAX - C].
          Lower = APInt::getSignedMinValue(Width);
          Upper = APInt::getSignedMaxValue(Width) + *C + 1;
        } else {
          // 'add nsw x, +C' produces [SINT_MIN + C, SINT_MAX].
          Lower = APInt::getSignedMinValue(Width) + *C;
          Upper = APInt::getSignedMaxValue(Width) + 1;
        }
      }
    }
    break;

  case Instruction::And:
    if (match(BO.getOperand(1), m_APInt(C)))
      // 'and x, C' produces [0, C].
      Upper = *C + 1;
    // X & -X is a power of two or zero. So we can cap the value at max power of
    // two.
    if (match(BO.getOperand(0), m_Neg(m_Specific(BO.getOperand(1)))) ||
        match(BO.getOperand(1), m_Neg(m_Specific(BO.getOperand(0)))))
      Upper = APInt::getSignedMinValue(Width) + 1;
    break;

  case Instruction::Or:
    if (match(BO.getOperand(1), m_APInt(C)))
      // 'or x, C' produces [C, UINT_MAX].
      Lower = *C;
    break;

  case Instruction::AShr:
    if (match(BO.getOperand(1), m_APInt(C)) && C->ult(Width)) {
      // 'ashr x, C' produces [INT_MIN >> C, INT_MAX >> C].
      Lower = APInt::getSignedMinValue(Width).ashr(*C);
      Upper = APInt::getSignedMaxValue(Width).ashr(*C) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      unsigned ShiftAmount = Width - 1;
      if (!C->isZero() && IIQ.isExact(&BO))
        ShiftAmount = C->countr_zero();
      if (C->isNegative()) {
        // 'ashr C, x' produces [C, C >> (Width-1)]
        Lower = *C;
        Upper = C->ashr(ShiftAmount) + 1;
      } else {
        // 'ashr C, x' produces [C >> (Width-1), C]
        Lower = C->ashr(ShiftAmount);
        Upper = *C + 1;
      }
    }
    break;

  case Instruction::LShr:
    if (match(BO.getOperand(1), m_APInt(C)) && C->ult(Width)) {
      // 'lshr x, C' produces [0, UINT_MAX >> C].
      Upper = APInt::getAllOnes(Width).lshr(*C) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      // 'lshr C, x' produces [C >> (Width-1), C].
      unsigned ShiftAmount = Width - 1;
      if (!C->isZero() && IIQ.isExact(&BO))
        ShiftAmount = C->countr_zero();
      Lower = C->lshr(ShiftAmount);
      Upper = *C + 1;
    }
    break;

  case Instruction::Shl:
    if (match(BO.getOperand(0), m_APInt(C))) {
      if (IIQ.hasNoUnsignedWrap(&BO)) {
        // 'shl nuw C, x' produces [C, C << CLZ(C)]
        Lower = *C;
        Upper = Lower.shl(Lower.countl_zero()) + 1;
      } else if (BO.hasNoSignedWrap()) { // TODO: What if both nuw+nsw?
        if (C->isNegative()) {
          // 'shl nsw C, x' produces [C << CLO(C)-1, C]
          unsigned ShiftAmount = C->countl_one() - 1;
          Lower = C->shl(ShiftAmount);
          Upper = *C + 1;
        } else {
          // 'shl nsw C, x' produces [C, C << CLZ(C)-1]
          unsigned ShiftAmount = C->countl_zero() - 1;
          Lower = *C;
          Upper = C->shl(ShiftAmount) + 1;
        }
      } else {
        // If lowbit is set, value can never be zero.
        if ((*C)[0])
          Lower = APInt::getOneBitSet(Width, 0);
        // If we are shifting a constant the largest it can be is if the longest
        // sequence of consecutive ones is shifted to the highbits (breaking
        // ties for which sequence is higher). At the moment we take a liberal
        // upper bound on this by just popcounting the constant.
        // TODO: There may be a bitwise trick for it longest/highest
        // consecutative sequence of ones (naive method is O(Width) loop).
        Upper = APInt::getHighBitsSet(Width, C->popcount()) + 1;
      }
    } else if (match(BO.getOperand(1), m_APInt(C)) && C->ult(Width)) {
      Upper = APInt::getBitsSetFrom(Width, C->getZExtValue()) + 1;
    }
    break;

  case Instruction::SDiv:
    if (match(BO.getOperand(1), m_APInt(C))) {
      APInt IntMin = APInt::getSignedMinValue(Width);
      APInt IntMax = APInt::getSignedMaxValue(Width);
      if (C->isAllOnes()) {
        // 'sdiv x, -1' produces [INT_MIN + 1, INT_MAX]
        //    where C != -1 and C != 0 and C != 1
        Lower = IntMin + 1;
        Upper = IntMax + 1;
      } else if (C->countl_zero() < Width - 1) {
        // 'sdiv x, C' produces [INT_MIN / C, INT_MAX / C]
        //    where C != -1 and C != 0 and C != 1
        Lower = IntMin.sdiv(*C);
        Upper = IntMax.sdiv(*C);
        if (Lower.sgt(Upper))
          std::swap(Lower, Upper);
        Upper = Upper + 1;
        assert(Upper != Lower && "Upper part of range has wrapped!");
      }
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      if (C->isMinSignedValue()) {
        // 'sdiv INT_MIN, x' produces [INT_MIN, INT_MIN / -2].
        Lower = *C;
        Upper = Lower.lshr(1) + 1;
      } else {
        // 'sdiv C, x' produces [-|C|, |C|].
        Upper = C->abs() + 1;
        Lower = (-Upper) + 1;
      }
    }
    break;

  case Instruction::UDiv:
    if (match(BO.getOperand(1), m_APInt(C)) && !C->isZero()) {
      // 'udiv x, C' produces [0, UINT_MAX / C].
      Upper = APInt::getMaxValue(Width).udiv(*C) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      // 'udiv C, x' produces [0, C].
      Upper = *C + 1;
    }
    break;

  case Instruction::SRem:
    if (match(BO.getOperand(1), m_APInt(C))) {
      // 'srem x, C' produces (-|C|, |C|).
      Upper = C->abs();
      Lower = (-Upper) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      if (C->isNegative()) {
        // 'srem -|C|, x' produces [-|C|, 0].
        Upper = 1;
        Lower = *C;
      } else {
        // 'srem |C|, x' produces [0, |C|].
        Upper = *C + 1;
      }
    }
    break;

  case Instruction::URem:
    if (match(BO.getOperand(1), m_APInt(C)))
      // 'urem x, C' produces [0, C).
      Upper = *C;
    else if (match(BO.getOperand(0), m_APInt(C)))
      // 'urem C, x' produces [0, C].
      Upper = *C + 1;
    break;

  default:
    break;
  }
}

/// Return true if "icmp Pred BLHS BRHS" is true whenever "icmp Pred
/// ALHS ARHS" is true.  Otherwise, return std::nullopt.
std::optional<bool>
isImpliedCondOperands(CmpInst::Predicate Pred, const Value *ALHS,
                      const Value *ARHS, const Value *BLHS, const Value *BRHS) {
  switch (Pred) {
  default:
    return std::nullopt;

  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
    if (isTruePredicate(CmpInst::ICMP_SLE, BLHS, ALHS) &&
        isTruePredicate(CmpInst::ICMP_SLE, ARHS, BRHS))
      return true;
    return std::nullopt;

  case CmpInst::ICMP_SGT:
  case CmpInst::ICMP_SGE:
    if (isTruePredicate(CmpInst::ICMP_SLE, ALHS, BLHS) &&
        isTruePredicate(CmpInst::ICMP_SLE, BRHS, ARHS))
      return true;
    return std::nullopt;

  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
    if (isTruePredicate(CmpInst::ICMP_ULE, BLHS, ALHS) &&
        isTruePredicate(CmpInst::ICMP_ULE, ARHS, BRHS))
      return true;
    return std::nullopt;

  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE:
    if (isTruePredicate(CmpInst::ICMP_ULE, ALHS, BLHS) &&
        isTruePredicate(CmpInst::ICMP_ULE, BRHS, ARHS))
      return true;
    return std::nullopt;
  }
}

/// Return true if "icmp LPred X, LCR" implies "icmp RPred X, RCR" is true.
/// Return false if "icmp LPred X, LCR" implies "icmp RPred X, RCR" is false.
/// Otherwise, return std::nullopt if we can't infer anything.
std::optional<bool>
isImpliedCondCommonOperandWithCR(CmpPredicate LPred, const ConstantRange &LCR,
                                 CmpPredicate RPred, const ConstantRange &RCR) {
  auto CRImpliesPred = [&](ConstantRange CR,
                           CmpInst::Predicate Pred) -> std::optional<bool> {
    // If all true values for lhs and true for rhs, lhs implies rhs
    if (CR.icmp(Pred, RCR))
      return true;

    // If there is no overlap, lhs implies not rhs
    if (CR.icmp(CmpInst::getInversePredicate(Pred), RCR))
      return false;

    return std::nullopt;
  };
  if (auto Res = CRImpliesPred(ConstantRange::makeAllowedICmpRegion(LPred, LCR),
                               RPred))
    return Res;
  if (LPred.hasSameSign() ^ RPred.hasSameSign()) {
    LPred = LPred.hasSameSign() ? ICmpInst::getFlippedSignednessPredicate(LPred)
                                : LPred.dropSameSign();
    RPred = RPred.hasSameSign() ? ICmpInst::getFlippedSignednessPredicate(RPred)
                                : RPred.dropSameSign();
    return CRImpliesPred(ConstantRange::makeAllowedICmpRegion(LPred, LCR),
                         RPred);
  }
  return std::nullopt;
}

/// If the pair of operators are the same invertible function, return the
/// the operands of the function corresponding to each input. Otherwise,
/// return std::nullopt.  An invertible function is one that is 1-to-1 and maps
/// every input value to exactly one output value.  This is equivalent to
/// saying that Op1 and Op2 are equal exactly when the specified pair of
/// operands are equal, (except that Op1 and Op2 may be poison more often.)
std::optional<std::pair<Value *, Value *>>
getInvertibleOperands(const Operator *Op1, const Operator *Op2) {
  if (Op1->getOpcode() != Op2->getOpcode())
    return std::nullopt;

  auto getOperands = [&](unsigned OpNum) -> auto {
    return std::make_pair(Op1->getOperand(OpNum), Op2->getOperand(OpNum));
  };

  switch (Op1->getOpcode()) {
  default:
    break;
  case Instruction::Or:
    if (!cast<PossiblyDisjointInst>(Op1)->isDisjoint() ||
        !cast<PossiblyDisjointInst>(Op2)->isDisjoint())
      break;
    [[fallthrough]];
  case Instruction::Xor:
  case Instruction::Add: {
    Value *Other;
    if (match(Op2, m_c_BinOp(m_Specific(Op1->getOperand(0)), m_Value(Other))))
      return std::make_pair(Op1->getOperand(1), Other);
    if (match(Op2, m_c_BinOp(m_Specific(Op1->getOperand(1)), m_Value(Other))))
      return std::make_pair(Op1->getOperand(0), Other);
    break;
  }
  case Instruction::Sub:
    if (Op1->getOperand(0) == Op2->getOperand(0))
      return getOperands(1);
    if (Op1->getOperand(1) == Op2->getOperand(1))
      return getOperands(0);
    break;
  case Instruction::Mul: {
    // invertible if A * B == (A * B) mod 2^N where A, and B are integers
    // and N is the bitwdith.  The nsw case is non-obvious, but proven by
    // alive2: https://alive2.llvm.org/ce/z/Z6D5qK
    auto *OBO1 = cast<OverflowingBinaryOperator>(Op1);
    auto *OBO2 = cast<OverflowingBinaryOperator>(Op2);
    if ((!OBO1->hasNoUnsignedWrap() || !OBO2->hasNoUnsignedWrap()) &&
        (!OBO1->hasNoSignedWrap() || !OBO2->hasNoSignedWrap()))
      break;

    // Assume operand order has been canonicalized
    if (Op1->getOperand(1) == Op2->getOperand(1) &&
        isa<ConstantInt>(Op1->getOperand(1)) &&
        !cast<ConstantInt>(Op1->getOperand(1))->isZero())
      return getOperands(0);
    break;
  }
  case Instruction::Shl: {
    // Same as multiplies, with the difference that we don't need to check
    // for a non-zero multiply. Shifts always multiply by non-zero.
    auto *OBO1 = cast<OverflowingBinaryOperator>(Op1);
    auto *OBO2 = cast<OverflowingBinaryOperator>(Op2);
    if ((!OBO1->hasNoUnsignedWrap() || !OBO2->hasNoUnsignedWrap()) &&
        (!OBO1->hasNoSignedWrap() || !OBO2->hasNoSignedWrap()))
      break;

    if (Op1->getOperand(1) == Op2->getOperand(1))
      return getOperands(0);
    break;
  }
  case Instruction::AShr:
  case Instruction::LShr: {
    auto *PEO1 = cast<PossiblyExactOperator>(Op1);
    auto *PEO2 = cast<PossiblyExactOperator>(Op2);
    if (!PEO1->isExact() || !PEO2->isExact())
      break;

    if (Op1->getOperand(1) == Op2->getOperand(1))
      return getOperands(0);
    break;
  }
  case Instruction::SExt:
  case Instruction::ZExt:
    if (Op1->getOperand(0)->getType() == Op2->getOperand(0)->getType())
      return getOperands(0);
    break;
  case Instruction::PHI: {
    const PHINode *PN1 = cast<PHINode>(Op1);
    const PHINode *PN2 = cast<PHINode>(Op2);

    // If PN1 and PN2 are both recurrences, can we prove the entire recurrences
    // are a single invertible function of the start values? Note that repeated
    // application of an invertible function is also invertible
    BinaryOperator *BO1 = nullptr;
    Value *Start1 = nullptr, *Step1 = nullptr;
    BinaryOperator *BO2 = nullptr;
    Value *Start2 = nullptr, *Step2 = nullptr;
    if (PN1->getParent() != PN2->getParent() ||
        !matchSimpleRecurrence(PN1, BO1, Start1, Step1) ||
        !matchSimpleRecurrence(PN2, BO2, Start2, Step2))
      break;

    auto Values =
        getInvertibleOperands(cast<Operator>(BO1), cast<Operator>(BO2));
    if (!Values)
      break;

    // We have to be careful of mutually defined recurrences here.  Ex:
    // * X_i = X_(i-1) OP Y_(i-1), and Y_i = X_(i-1) OP V
    // * X_i = Y_i = X_(i-1) OP Y_(i-1)
    // The invertibility of these is complicated, and not worth reasoning
    // about (yet?).
    if (Values->first != PN1 || Values->second != PN2)
      break;

    return std::make_pair(Start1, Start2);
  }
  }
  return std::nullopt;
}
} // namespace llvm::vtutils
