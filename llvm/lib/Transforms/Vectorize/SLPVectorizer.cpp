//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Vectorize.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;
using namespace slpvectorizer;

#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE "SLP"

STATISTIC(NumVectorInstructions, "Number of vector instructions generated");

cl::opt<bool> RunSLPVectorization("vectorize-slp", cl::init(true), cl::Hidden,
                                  cl::desc("Run the SLP vectorization passes"));

static cl::opt<int>
    SLPCostThreshold("slp-threshold", cl::init(0), cl::Hidden,
                     cl::desc("Only vectorize if you gain more than this "
                              "number "));

static cl::opt<bool>
ShouldVectorizeHor("slp-vectorize-hor", cl::init(true), cl::Hidden,
                   cl::desc("Attempt to vectorize horizontal reductions"));

static cl::opt<bool> ShouldStartVectorizeHorAtStore(
    "slp-vectorize-hor-store", cl::init(false), cl::Hidden,
    cl::desc(
        "Attempt to vectorize horizontal reductions feeding into a store"));

static cl::opt<int>
MaxVectorRegSizeOption("slp-max-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

static cl::opt<unsigned>
MaxVFOption("slp-max-vf", cl::init(0), cl::Hidden,
    cl::desc("Maximum SLP vectorization factor (0=unlimited)"));

static cl::opt<int>
MaxStoreLookup("slp-max-store-lookup", cl::init(32), cl::Hidden,
    cl::desc("Maximum depth of the lookup for consecutive stores."));

/// Limits the size of scheduling regions in a block.
/// It avoid long compile times for _very_ large blocks where vector
/// instructions are spread over a wide range.
/// This limit is way higher than needed by real-world functions.
static cl::opt<int>
ScheduleRegionSizeBudget("slp-schedule-budget", cl::init(100000), cl::Hidden,
    cl::desc("Limit the size of the SLP scheduling region per block"));

static cl::opt<int> MinVectorRegSizeOption(
    "slp-min-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

static cl::opt<unsigned> RecursionMaxDepth(
    "slp-recursion-max-depth", cl::init(12), cl::Hidden,
    cl::desc("Limit the recursion depth when building a vectorizable tree"));

static cl::opt<unsigned> MinTreeSize(
    "slp-min-tree-size", cl::init(3), cl::Hidden,
    cl::desc("Only vectorize small trees if they are fully vectorizable"));

// The maximum depth that the look-ahead score heuristic will explore.
// The higher this value, the higher the compilation time overhead.
static cl::opt<int> LookAheadMaxDepth(
    "slp-max-look-ahead-depth", cl::init(2), cl::Hidden,
    cl::desc("The maximum look-ahead depth for operand reordering scores"));

static cl::opt<bool>
    ViewSLPTree("view-slp-tree", cl::Hidden,
                cl::desc("Display the SLP trees with Graphviz"));

// Limit the number of alias checks. The limit is chosen so that
// it has no negative effect on the llvm benchmarks.
static const unsigned AliasedCheckLimit = 10;

// Another limit for the alias checks: The maximum distance between load/store
// instructions where alias checks are done.
// This limit is useful for very large basic blocks.
static const unsigned MaxMemDepDistance = 160;

/// If the ScheduleRegionSizeBudget is exhausted, we allow small scheduling
/// regions to be handled.
static const int MinScheduleRegionSize = 16;

/// Predicate for the element types that the SLP vectorizer supports.
///
/// The most important thing to filter here are types which are invalid in LLVM
/// vectors. We also filter target specific types which have absolutely no
/// meaningful vectorization path such as x86_fp80 and ppc_f128. This just
/// avoids spending time checking the cost model and realizing that they will
/// be inevitably scalarized.
static bool isValidElementType(Type *Ty) {
  return VectorType::isValidElementType(Ty) && !Ty->isX86_FP80Ty() &&
         !Ty->isPPC_FP128Ty();
}

/// \returns True if the value is a constant (but not globals/constant
/// expressions).
static bool isConstant(Value *V) {
  return isa<Constant>(V) && !isa<ConstantExpr>(V) && !isa<GlobalValue>(V);
}

/// Checks if \p V is one of vector-like instructions, i.e. undef,
/// insertelement/extractelement with constant indices for fixed vector type or
/// extractvalue instruction.
static bool isVectorLikeInstWithConstOps(Value *V) {
  if (!isa<InsertElementInst, ExtractElementInst>(V) &&
      !isa<ExtractValueInst, UndefValue>(V))
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || isa<ExtractValueInst>(I))
    return true;
  if (!isa<FixedVectorType>(I->getOperand(0)->getType()))
    return false;
  if (isa<ExtractElementInst>(I))
    return isConstant(I->getOperand(1));
  assert(isa<InsertElementInst>(V) && "Expected only insertelement.");
  return isConstant(I->getOperand(2));
}

/// \returns true if all of the instructions in \p VL are in the same block or
/// false otherwise.
static bool allSameBlock(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return false;
  if (all_of(VL, isVectorLikeInstWithConstOps))
    return true;

  BasicBlock *BB = I0->getParent();
  for (int I = 1, E = VL.size(); I < E; I++) {
    auto *II = dyn_cast<Instruction>(VL[I]);
    if (!II)
      return false;

    if (BB != II->getParent())
      return false;
  }
  return true;
}

/// \returns True if all of the values in \p VL are constants (but not
/// globals/constant expressions).
static bool allConstant(ArrayRef<Value *> VL) {
  // Constant expressions and globals can't be vectorized like normal integer/FP
  // constants.
  return all_of(VL, isConstant);
}

/// \returns True if all of the values in \p VL are identical or some of them
/// are UndefValue.
static bool isSplat(ArrayRef<Value *> VL) {
  Value *FirstNonUndef = nullptr;
  for (Value *V : VL) {
    if (isa<UndefValue>(V))
      continue;
    if (!FirstNonUndef) {
      FirstNonUndef = V;
      continue;
    }
    if (V != FirstNonUndef)
      return false;
  }
  return FirstNonUndef != nullptr;
}

/// \returns True if \p I is commutative, handles CmpInst and BinaryOperator.
static bool isCommutative(Instruction *I) {
  if (auto *Cmp = dyn_cast<CmpInst>(I))
    return Cmp->isCommutative();
  if (auto *BO = dyn_cast<BinaryOperator>(I))
    return BO->isCommutative();
  // TODO: This should check for generic Instruction::isCommutative(), but
  //       we need to confirm that the caller code correctly handles Intrinsics
  //       for example (does not have 2 operands).
  return false;
}

/// Checks if the given value is actually an undefined constant vector.
static bool isUndefVector(const Value *V) {
  if (isa<UndefValue>(V))
    return true;
  auto *C = dyn_cast<Constant>(V);
  if (!C)
    return false;
  if (!C->containsUndefOrPoisonElement())
    return false;
  auto *VecTy = dyn_cast<FixedVectorType>(C->getType());
  if (!VecTy)
    return false;
  for (unsigned I = 0, E = VecTy->getNumElements(); I != E; ++I) {
    if (Constant *Elem = C->getAggregateElement(I))
      if (!isa<UndefValue>(Elem))
        return false;
  }
  return true;
}

/// Checks if the vector of instructions can be represented as a shuffle, like:
/// %x0 = extractelement <4 x i8> %x, i32 0
/// %x3 = extractelement <4 x i8> %x, i32 3
/// %y1 = extractelement <4 x i8> %y, i32 1
/// %y2 = extractelement <4 x i8> %y, i32 2
/// %x0x0 = mul i8 %x0, %x0
/// %x3x3 = mul i8 %x3, %x3
/// %y1y1 = mul i8 %y1, %y1
/// %y2y2 = mul i8 %y2, %y2
/// %ins1 = insertelement <4 x i8> poison, i8 %x0x0, i32 0
/// %ins2 = insertelement <4 x i8> %ins1, i8 %x3x3, i32 1
/// %ins3 = insertelement <4 x i8> %ins2, i8 %y1y1, i32 2
/// %ins4 = insertelement <4 x i8> %ins3, i8 %y2y2, i32 3
/// ret <4 x i8> %ins4
/// can be transformed into:
/// %1 = shufflevector <4 x i8> %x, <4 x i8> %y, <4 x i32> <i32 0, i32 3, i32 5,
///                                                         i32 6>
/// %2 = mul <4 x i8> %1, %1
/// ret <4 x i8> %2
/// We convert this initially to something like:
/// %x0 = extractelement <4 x i8> %x, i32 0
/// %x3 = extractelement <4 x i8> %x, i32 3
/// %y1 = extractelement <4 x i8> %y, i32 1
/// %y2 = extractelement <4 x i8> %y, i32 2
/// %1 = insertelement <4 x i8> poison, i8 %x0, i32 0
/// %2 = insertelement <4 x i8> %1, i8 %x3, i32 1
/// %3 = insertelement <4 x i8> %2, i8 %y1, i32 2
/// %4 = insertelement <4 x i8> %3, i8 %y2, i32 3
/// %5 = mul <4 x i8> %4, %4
/// %6 = extractelement <4 x i8> %5, i32 0
/// %ins1 = insertelement <4 x i8> poison, i8 %6, i32 0
/// %7 = extractelement <4 x i8> %5, i32 1
/// %ins2 = insertelement <4 x i8> %ins1, i8 %7, i32 1
/// %8 = extractelement <4 x i8> %5, i32 2
/// %ins3 = insertelement <4 x i8> %ins2, i8 %8, i32 2
/// %9 = extractelement <4 x i8> %5, i32 3
/// %ins4 = insertelement <4 x i8> %ins3, i8 %9, i32 3
/// ret <4 x i8> %ins4
/// InstCombiner transforms this into a shuffle and vector mul
/// Mask will return the Shuffle Mask equivalent to the extracted elements.
/// TODO: Can we split off and reuse the shuffle mask detection from
/// TargetTransformInfo::getInstructionThroughput?
static Optional<TargetTransformInfo::ShuffleKind>
isFixedVectorShuffle(ArrayRef<Value *> VL, SmallVectorImpl<int> &Mask) {
  const auto *It =
      find_if(VL, [](Value *V) { return isa<ExtractElementInst>(V); });
  if (It == VL.end())
    return None;
  auto *EI0 = cast<ExtractElementInst>(*It);
  if (isa<ScalableVectorType>(EI0->getVectorOperandType()))
    return None;
  unsigned Size =
      cast<FixedVectorType>(EI0->getVectorOperandType())->getNumElements();
  Value *Vec1 = nullptr;
  Value *Vec2 = nullptr;
  enum ShuffleMode { Unknown, Select, Permute };
  ShuffleMode CommonShuffleMode = Unknown;
  Mask.assign(VL.size(), UndefMaskElem);
  for (unsigned I = 0, E = VL.size(); I < E; ++I) {
    // Undef can be represented as an undef element in a vector.
    if (isa<UndefValue>(VL[I]))
      continue;
    auto *EI = cast<ExtractElementInst>(VL[I]);
    if (isa<ScalableVectorType>(EI->getVectorOperandType()))
      return None;
    auto *Vec = EI->getVectorOperand();
    // We can extractelement from undef or poison vector.
    if (isUndefVector(Vec))
      continue;
    // All vector operands must have the same number of vector elements.
    if (cast<FixedVectorType>(Vec->getType())->getNumElements() != Size)
      return None;
    if (isa<UndefValue>(EI->getIndexOperand()))
      continue;
    auto *Idx = dyn_cast<ConstantInt>(EI->getIndexOperand());
    if (!Idx)
      return None;
    // Undefined behavior if Idx is negative or >= Size.
    if (Idx->getValue().uge(Size))
      continue;
    unsigned IntIdx = Idx->getValue().getZExtValue();
    Mask[I] = IntIdx;
    // For correct shuffling we have to have at most 2 different vector operands
    // in all extractelement instructions.
    if (!Vec1 || Vec1 == Vec) {
      Vec1 = Vec;
    } else if (!Vec2 || Vec2 == Vec) {
      Vec2 = Vec;
      Mask[I] += Size;
    } else {
      return None;
    }
    if (CommonShuffleMode == Permute)
      continue;
    // If the extract index is not the same as the operation number, it is a
    // permutation.
    if (IntIdx != I) {
      CommonShuffleMode = Permute;
      continue;
    }
    CommonShuffleMode = Select;
  }
  // If we're not crossing lanes in different vectors, consider it as blending.
  if (CommonShuffleMode == Select && Vec2)
    return TargetTransformInfo::SK_Select;
  // If Vec2 was never used, we have a permutation of a single vector, otherwise
  // we have permutation of 2 vectors.
  return Vec2 ? TargetTransformInfo::SK_PermuteTwoSrc
              : TargetTransformInfo::SK_PermuteSingleSrc;
}

namespace {

/// Main data required for vectorization of instructions.
struct InstructionsState {
  /// The very first instruction in the list with the main opcode.
  Value *OpValue = nullptr;

  /// The main/alternate instruction.
  Instruction *MainOp = nullptr;
  Instruction *AltOp = nullptr;

  /// The main/alternate opcodes for the list of instructions.
  unsigned getOpcode() const {
    return MainOp ? MainOp->getOpcode() : 0;
  }

  unsigned getAltOpcode() const {
    return AltOp ? AltOp->getOpcode() : 0;
  }

  /// Some of the instructions in the list have alternate opcodes.
  bool isAltShuffle() const { return AltOp != MainOp; }

  bool isOpcodeOrAlt(Instruction *I) const {
    unsigned CheckedOpcode = I->getOpcode();
    return getOpcode() == CheckedOpcode || getAltOpcode() == CheckedOpcode;
  }

  InstructionsState() = delete;
  InstructionsState(Value *OpValue, Instruction *MainOp, Instruction *AltOp)
      : OpValue(OpValue), MainOp(MainOp), AltOp(AltOp) {}
};

} // end anonymous namespace

/// Chooses the correct key for scheduling data. If \p Op has the same (or
/// alternate) opcode as \p OpValue, the key is \p Op. Otherwise the key is \p
/// OpValue.
static Value *isOneOf(const InstructionsState &S, Value *Op) {
  auto *I = dyn_cast<Instruction>(Op);
  if (I && S.isOpcodeOrAlt(I))
    return Op;
  return S.OpValue;
}

/// \returns true if \p Opcode is allowed as part of of the main/alternate
/// instruction for SLP vectorization.
///
/// Example of unsupported opcode is SDIV that can potentially cause UB if the
/// "shuffled out" lane would result in division by zero.
static bool isValidForAlternation(unsigned Opcode) {
  if (Instruction::isIntDivRem(Opcode))
    return false;

  return true;
}

static InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                       unsigned BaseIndex = 0);

/// Checks if the provided operands of 2 cmp instructions are compatible, i.e.
/// compatible instructions or constants, or just some other regular values.
static bool areCompatibleCmpOps(Value *BaseOp0, Value *BaseOp1, Value *Op0,
                                Value *Op1) {
  return (isConstant(BaseOp0) && isConstant(Op0)) ||
         (isConstant(BaseOp1) && isConstant(Op1)) ||
         (!isa<Instruction>(BaseOp0) && !isa<Instruction>(Op0) &&
          !isa<Instruction>(BaseOp1) && !isa<Instruction>(Op1)) ||
         getSameOpcode({BaseOp0, Op0}).getOpcode() ||
         getSameOpcode({BaseOp1, Op1}).getOpcode();
}

/// \returns analysis of the Instructions in \p VL described in
/// InstructionsState, the Opcode that we suppose the whole list
/// could be vectorized even if its structure is diverse.
static InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                       unsigned BaseIndex) {
  // Make sure these are all Instructions.
  if (llvm::any_of(VL, [](Value *V) { return !isa<Instruction>(V); }))
    return InstructionsState(VL[BaseIndex], nullptr, nullptr);

  bool IsCastOp = isa<CastInst>(VL[BaseIndex]);
  bool IsBinOp = isa<BinaryOperator>(VL[BaseIndex]);
  bool IsCmpOp = isa<CmpInst>(VL[BaseIndex]);
  CmpInst::Predicate BasePred =
      IsCmpOp ? cast<CmpInst>(VL[BaseIndex])->getPredicate()
              : CmpInst::BAD_ICMP_PREDICATE;
  unsigned Opcode = cast<Instruction>(VL[BaseIndex])->getOpcode();
  unsigned AltOpcode = Opcode;
  unsigned AltIndex = BaseIndex;

  // Check for one alternate opcode from another BinaryOperator.
  // TODO - generalize to support all operators (types, calls etc.).
  for (int Cnt = 0, E = VL.size(); Cnt < E; Cnt++) {
    unsigned InstOpcode = cast<Instruction>(VL[Cnt])->getOpcode();
    if (IsBinOp && isa<BinaryOperator>(VL[Cnt])) {
      if (InstOpcode == Opcode || InstOpcode == AltOpcode)
        continue;
      if (Opcode == AltOpcode && isValidForAlternation(InstOpcode) &&
          isValidForAlternation(Opcode)) {
        AltOpcode = InstOpcode;
        AltIndex = Cnt;
        continue;
      }
    } else if (IsCastOp && isa<CastInst>(VL[Cnt])) {
      Type *Ty0 = cast<Instruction>(VL[BaseIndex])->getOperand(0)->getType();
      Type *Ty1 = cast<Instruction>(VL[Cnt])->getOperand(0)->getType();
      if (Ty0 == Ty1) {
        if (InstOpcode == Opcode || InstOpcode == AltOpcode)
          continue;
        if (Opcode == AltOpcode) {
          assert(isValidForAlternation(Opcode) &&
                 isValidForAlternation(InstOpcode) &&
                 "Cast isn't safe for alternation, logic needs to be updated!");
          AltOpcode = InstOpcode;
          AltIndex = Cnt;
          continue;
        }
      }
    } else if (IsCmpOp && isa<CmpInst>(VL[Cnt])) {
      auto *BaseInst = cast<Instruction>(VL[BaseIndex]);
      auto *Inst = cast<Instruction>(VL[Cnt]);
      Type *Ty0 = BaseInst->getOperand(0)->getType();
      Type *Ty1 = Inst->getOperand(0)->getType();
      if (Ty0 == Ty1) {
        Value *BaseOp0 = BaseInst->getOperand(0);
        Value *BaseOp1 = BaseInst->getOperand(1);
        Value *Op0 = Inst->getOperand(0);
        Value *Op1 = Inst->getOperand(1);
        CmpInst::Predicate CurrentPred =
            cast<CmpInst>(VL[Cnt])->getPredicate();
        CmpInst::Predicate SwappedCurrentPred =
            CmpInst::getSwappedPredicate(CurrentPred);
        // Check for compatible operands. If the corresponding operands are not
        // compatible - need to perform alternate vectorization.
        if (InstOpcode == Opcode) {
          if (BasePred == CurrentPred &&
              areCompatibleCmpOps(BaseOp0, BaseOp1, Op0, Op1))
            continue;
          if (BasePred == SwappedCurrentPred &&
              areCompatibleCmpOps(BaseOp0, BaseOp1, Op1, Op0))
            continue;
          if (E == 2 &&
              (BasePred == CurrentPred || BasePred == SwappedCurrentPred))
            continue;
          auto *AltInst = cast<CmpInst>(VL[AltIndex]);
          CmpInst::Predicate AltPred = AltInst->getPredicate();
          Value *AltOp0 = AltInst->getOperand(0);
          Value *AltOp1 = AltInst->getOperand(1);
          // Check if operands are compatible with alternate operands.
          if (AltPred == CurrentPred &&
              areCompatibleCmpOps(AltOp0, AltOp1, Op0, Op1))
            continue;
          if (AltPred == SwappedCurrentPred &&
              areCompatibleCmpOps(AltOp0, AltOp1, Op1, Op0))
            continue;
        }
        if (BaseIndex == AltIndex && BasePred != CurrentPred) {
          assert(isValidForAlternation(Opcode) &&
                 isValidForAlternation(InstOpcode) &&
                 "Cast isn't safe for alternation, logic needs to be updated!");
          AltIndex = Cnt;
          continue;
        }
        auto *AltInst = cast<CmpInst>(VL[AltIndex]);
        CmpInst::Predicate AltPred = AltInst->getPredicate();
        if (BasePred == CurrentPred || BasePred == SwappedCurrentPred ||
            AltPred == CurrentPred || AltPred == SwappedCurrentPred)
          continue;
      }
    } else if (InstOpcode == Opcode || InstOpcode == AltOpcode)
      continue;
    return InstructionsState(VL[BaseIndex], nullptr, nullptr);
  }

  return InstructionsState(VL[BaseIndex], cast<Instruction>(VL[BaseIndex]),
                           cast<Instruction>(VL[AltIndex]));
}

/// \returns true if all of the values in \p VL have the same type or false
/// otherwise.
static bool allSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL[0]->getType();
  for (int i = 1, e = VL.size(); i < e; i++)
    if (VL[i]->getType() != Ty)
      return false;

  return true;
}

/// \returns True if Extract{Value,Element} instruction extracts element Idx.
static Optional<unsigned> getExtractIndex(Instruction *E) {
  unsigned Opcode = E->getOpcode();
  assert((Opcode == Instruction::ExtractElement ||
          Opcode == Instruction::ExtractValue) &&
         "Expected extractelement or extractvalue instruction.");
  if (Opcode == Instruction::ExtractElement) {
    auto *CI = dyn_cast<ConstantInt>(E->getOperand(1));
    if (!CI)
      return None;
    return CI->getZExtValue();
  }
  ExtractValueInst *EI = cast<ExtractValueInst>(E);
  if (EI->getNumIndices() != 1)
    return None;
  return *EI->idx_begin();
}

/// \returns True if in-tree use also needs extract. This refers to
/// possible scalar operand in vectorized instruction.
static bool InTreeUserNeedToExtract(Value *Scalar, Instruction *UserInst,
                                    TargetLibraryInfo *TLI) {
  unsigned Opcode = UserInst->getOpcode();
  switch (Opcode) {
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(UserInst);
    return (LI->getPointerOperand() == Scalar);
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(UserInst);
    return (SI->getPointerOperand() == Scalar);
  }
  case Instruction::Call: {
    CallInst *CI = cast<CallInst>(UserInst);
    Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
    for (unsigned i = 0, e = CI->arg_size(); i != e; ++i) {
      if (hasVectorInstrinsicScalarOpd(ID, i))
        return (CI->getArgOperand(i) == Scalar);
    }
    LLVM_FALLTHROUGH;
  }
  default:
    return false;
  }
}

/// \returns the AA location that is being access by the instruction.
static MemoryLocation getLocation(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return MemoryLocation::get(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return MemoryLocation::get(LI);
  return MemoryLocation();
}

/// \returns True if the instruction is not a volatile or atomic load/store.
static bool isSimple(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isSimple();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return true;
}

/// Shuffles \p Mask in accordance with the given \p SubMask.
static void addMask(SmallVectorImpl<int> &Mask, ArrayRef<int> SubMask) {
  if (SubMask.empty())
    return;
  if (Mask.empty()) {
    Mask.append(SubMask.begin(), SubMask.end());
    return;
  }
  SmallVector<int> NewMask(SubMask.size(), UndefMaskElem);
  int TermValue = std::min(Mask.size(), SubMask.size());
  for (int I = 0, E = SubMask.size(); I < E; ++I) {
    if (SubMask[I] >= TermValue || SubMask[I] == UndefMaskElem ||
        Mask[SubMask[I]] >= TermValue)
      continue;
    NewMask[I] = Mask[SubMask[I]];
  }
  Mask.swap(NewMask);
}

/// Order may have elements assigned special value (size) which is out of
/// bounds. Such indices only appear on places which correspond to undef values
/// (see canReuseExtract for details) and used in order to avoid undef values
/// have effect on operands ordering.
/// The first loop below simply finds all unused indices and then the next loop
/// nest assigns these indices for undef values positions.
/// As an example below Order has two undef positions and they have assigned
/// values 3 and 7 respectively:
/// before:  6 9 5 4 9 2 1 0
/// after:   6 3 5 4 7 2 1 0
static void fixupOrderingIndices(SmallVectorImpl<unsigned> &Order) {
  const unsigned Sz = Order.size();
  SmallBitVector UnusedIndices(Sz, /*t=*/true);
  SmallBitVector MaskedIndices(Sz);
  for (unsigned I = 0; I < Sz; ++I) {
    if (Order[I] < Sz)
      UnusedIndices.reset(Order[I]);
    else
      MaskedIndices.set(I);
  }
  if (MaskedIndices.none())
    return;
  assert(UnusedIndices.count() == MaskedIndices.count() &&
         "Non-synced masked/available indices.");
  int Idx = UnusedIndices.find_first();
  int MIdx = MaskedIndices.find_first();
  while (MIdx >= 0) {
    assert(Idx >= 0 && "Indices must be synced.");
    Order[MIdx] = Idx;
    Idx = UnusedIndices.find_next(Idx);
    MIdx = MaskedIndices.find_next(MIdx);
  }
}

namespace llvm {

static void inversePermutation(ArrayRef<unsigned> Indices,
                               SmallVectorImpl<int> &Mask) {
  Mask.clear();
  const unsigned E = Indices.size();
  Mask.resize(E, UndefMaskElem);
  for (unsigned I = 0; I < E; ++I)
    Mask[Indices[I]] = I;
}

/// \returns inserting index of InsertElement or InsertValue instruction,
/// using Offset as base offset for index.
static Optional<unsigned> getInsertIndex(Value *InsertInst,
                                         unsigned Offset = 0) {
  int Index = Offset;
  if (auto *IE = dyn_cast<InsertElementInst>(InsertInst)) {
    if (auto *CI = dyn_cast<ConstantInt>(IE->getOperand(2))) {
      auto *VT = cast<FixedVectorType>(IE->getType());
      if (CI->getValue().uge(VT->getNumElements()))
        return None;
      Index *= VT->getNumElements();
      Index += CI->getZExtValue();
      return Index;
    }
    return None;
  }

  auto *IV = cast<InsertValueInst>(InsertInst);
  Type *CurrentType = IV->getType();
  for (unsigned I : IV->indices()) {
    if (auto *ST = dyn_cast<StructType>(CurrentType)) {
      Index *= ST->getNumElements();
      CurrentType = ST->getElementType(I);
    } else if (auto *AT = dyn_cast<ArrayType>(CurrentType)) {
      Index *= AT->getNumElements();
      CurrentType = AT->getElementType();
    } else {
      return None;
    }
    Index += I;
  }
  return Index;
}

/// Reorders the list of scalars in accordance with the given \p Mask.
static void reorderScalars(SmallVectorImpl<Value *> &Scalars,
                           ArrayRef<int> Mask) {
  assert(!Mask.empty() && "Expected non-empty mask.");
  SmallVector<Value *> Prev(Scalars.size(),
                            UndefValue::get(Scalars.front()->getType()));
  Prev.swap(Scalars);
  for (unsigned I = 0, E = Prev.size(); I < E; ++I)
    if (Mask[I] != UndefMaskElem)
      Scalars[Mask[I]] = Prev[I];
}

namespace slpvectorizer {

/// Bottom Up SLP Vectorizer.
class BoUpSLP {
  struct TreeEntry;
  struct ScheduleData;

public:
  using ValueList = SmallVector<Value *, 8>;
  using InstrList = SmallVector<Instruction *, 16>;
  using ValueSet = SmallPtrSet<Value *, 16>;
  using StoreList = SmallVector<StoreInst *, 8>;
  using ExtraValueToDebugLocsMap =
      MapVector<Value *, SmallVector<Instruction *, 2>>;
  using OrdersType = SmallVector<unsigned, 4>;

  BoUpSLP(Function *Func, ScalarEvolution *Se, TargetTransformInfo *Tti,
          TargetLibraryInfo *TLi, AAResults *Aa, LoopInfo *Li,
          DominatorTree *Dt, AssumptionCache *AC, DemandedBits *DB,
          const DataLayout *DL, OptimizationRemarkEmitter *ORE)
      : BatchAA(*Aa), F(Func), SE(Se), TTI(Tti), TLI(TLi), LI(Li),
        DT(Dt), AC(AC), DB(DB), DL(DL), ORE(ORE), Builder(Se->getContext()) {
    CodeMetrics::collectEphemeralValues(F, AC, EphValues);
    // Use the vector register size specified by the target unless overridden
    // by a command-line option.
    // TODO: It would be better to limit the vectorization factor based on
    //       data type rather than just register size. For example, x86 AVX has
    //       256-bit registers, but it does not support integer operations
    //       at that width (that requires AVX2).
    if (MaxVectorRegSizeOption.getNumOccurrences())
      MaxVecRegSize = MaxVectorRegSizeOption;
    else
      MaxVecRegSize =
          TTI->getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
              .getFixedSize();

    if (MinVectorRegSizeOption.getNumOccurrences())
      MinVecRegSize = MinVectorRegSizeOption;
    else
      MinVecRegSize = TTI->getMinVectorRegisterBitWidth();
  }

  /// Vectorize the tree that starts with the elements in \p VL.
  /// Returns the vectorized root.
  Value *vectorizeTree();

  /// Vectorize the tree but with the list of externally used values \p
  /// ExternallyUsedValues. Values in this MapVector can be replaced but the
  /// generated extractvalue instructions.
  Value *vectorizeTree(ExtraValueToDebugLocsMap &ExternallyUsedValues);

  /// \returns the cost incurred by unwanted spills and fills, caused by
  /// holding live values over call sites.
  InstructionCost getSpillCost() const;

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  InstructionCost getTreeCost(ArrayRef<Value *> VectorizedVals = None);

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst.
  void buildTree(ArrayRef<Value *> Roots,
                 ArrayRef<Value *> UserIgnoreLst = None);

  /// Builds external uses of the vectorized scalars, i.e. the list of
  /// vectorized scalars to be extracted, their lanes and their scalar users. \p
  /// ExternallyUsedValues contains additional list of external uses to handle
  /// vectorization of reductions.
  void
  buildExternalUses(const ExtraValueToDebugLocsMap &ExternallyUsedValues = {});

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    VectorizableTree.clear();
    ScalarToTreeEntry.clear();
    MustGather.clear();
    ExternalUses.clear();
    for (auto &Iter : BlocksSchedules) {
      BlockScheduling *BS = Iter.second.get();
      BS->clear();
    }
    MinBWs.clear();
    InstrElementSize.clear();
  }

  unsigned getTreeSize() const { return VectorizableTree.size(); }

  /// Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  /// Checks if the specified gather tree entry \p TE can be represented as a
  /// shuffled vector entry + (possibly) permutation with other gathers. It
  /// implements the checks only for possibly ordered scalars (Loads,
  /// ExtractElement, ExtractValue), which can be part of the graph.
  Optional<OrdersType> findReusedOrderedScalars(const TreeEntry &TE);

  /// Gets reordering data for the given tree entry. If the entry is vectorized
  /// - just return ReorderIndices, otherwise check if the scalars can be
  /// reordered and return the most optimal order.
  /// \param TopToBottom If true, include the order of vectorized stores and
  /// insertelement nodes, otherwise skip them.
  Optional<OrdersType> getReorderingData(const TreeEntry &TE, bool TopToBottom);

  /// Reorders the current graph to the most profitable order starting from the
  /// root node to the leaf nodes. The best order is chosen only from the nodes
  /// of the same size (vectorization factor). Smaller nodes are considered
  /// parts of subgraph with smaller VF and they are reordered independently. We
  /// can make it because we still need to extend smaller nodes to the wider VF
  /// and we can merge reordering shuffles with the widening shuffles.
  void reorderTopToBottom();

  /// Reorders the current graph to the most profitable order starting from
  /// leaves to the root. It allows to rotate small subgraphs and reduce the
  /// number of reshuffles if the leaf nodes use the same order. In this case we
  /// can merge the orders and just shuffle user node instead of shuffling its
  /// operands. Plus, even the leaf nodes have different orders, it allows to
  /// sink reordering in the graph closer to the root node and merge it later
  /// during analysis.
  void reorderBottomToTop(bool IgnoreReorder = false);

  /// \return The vector element size in bits to use when vectorizing the
  /// expression tree ending at \p V. If V is a store, the size is the width of
  /// the stored value. Otherwise, the size is the width of the largest loaded
  /// value reaching V. This method is used by the vectorizer to calculate
  /// vectorization factors.
  unsigned getVectorElementSize(Value *V);

  /// Compute the minimum type sizes required to represent the entries in a
  /// vectorizable tree.
  void computeMinimumValueSizes();

  // \returns maximum vector register size as set by TTI or overridden by cl::opt.
  unsigned getMaxVecRegSize() const {
    return MaxVecRegSize;
  }

  // \returns minimum vector register size as set by cl::opt.
  unsigned getMinVecRegSize() const {
    return MinVecRegSize;
  }

  unsigned getMinVF(unsigned Sz) const {
    return std::max(2U, getMinVecRegSize() / Sz);
  }

  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const {
    unsigned MaxVF = MaxVFOption.getNumOccurrences() ?
      MaxVFOption : TTI->getMaximumVF(ElemWidth, Opcode);
    return MaxVF ? MaxVF : UINT_MAX;
  }

  /// Check if homogeneous aggregate is isomorphic to some VectorType.
  /// Accepts homogeneous multidimensional aggregate of scalars/vectors like
  /// {[4 x i16], [4 x i16]}, { <2 x float>, <2 x float> },
  /// {{{i16, i16}, {i16, i16}}, {{i16, i16}, {i16, i16}}} and so on.
  ///
  /// \returns number of elements in vector if isomorphism exists, 0 otherwise.
  unsigned canMapToVector(Type *T, const DataLayout &DL) const;

  /// \returns True if the VectorizableTree is both tiny and not fully
  /// vectorizable. We do not vectorize such trees.
  bool isTreeTinyAndNotFullyVectorizable(bool ForReduction = false) const;

  /// Assume that a legal-sized 'or'-reduction of shifted/zexted loaded values
  /// can be load combined in the backend. Load combining may not be allowed in
  /// the IR optimizer, so we do not want to alter the pattern. For example,
  /// partially transforming a scalar bswap() pattern into vector code is
  /// effectively impossible for the backend to undo.
  /// TODO: If load combining is allowed in the IR optimizer, this analysis
  ///       may not be necessary.
  bool isLoadCombineReductionCandidate(RecurKind RdxKind) const;

  /// Assume that a vector of stores of bitwise-or/shifted/zexted loaded values
  /// can be load combined in the backend. Load combining may not be allowed in
  /// the IR optimizer, so we do not want to alter the pattern. For example,
  /// partially transforming a scalar bswap() pattern into vector code is
  /// effectively impossible for the backend to undo.
  /// TODO: If load combining is allowed in the IR optimizer, this analysis
  ///       may not be necessary.
  bool isLoadCombineCandidate() const;

  OptimizationRemarkEmitter *getORE() { return ORE; }

  /// This structure holds any data we need about the edges being traversed
  /// during buildTree_rec(). We keep track of:
  /// (i) the user TreeEntry index, and
  /// (ii) the index of the edge.
  struct EdgeInfo {
    EdgeInfo() = default;
    EdgeInfo(TreeEntry *UserTE, unsigned EdgeIdx)
        : UserTE(UserTE), EdgeIdx(EdgeIdx) {}
    /// The user TreeEntry.
    TreeEntry *UserTE = nullptr;
    /// The operand index of the use.
    unsigned EdgeIdx = UINT_MAX;
#ifndef NDEBUG
    friend inline raw_ostream &operator<<(raw_ostream &OS,
                                          const BoUpSLP::EdgeInfo &EI) {
      EI.dump(OS);
      return OS;
    }
    /// Debug print.
    void dump(raw_ostream &OS) const {
      OS << "{User:" << (UserTE ? std::to_string(UserTE->Idx) : "null")
         << " EdgeIdx:" << EdgeIdx << "}";
    }
    LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
#endif
  };

  /// A helper data structure to hold the operands of a vector of instructions.
  /// This supports a fixed vector length for all operand vectors.
  class VLOperands {
    /// For each operand we need (i) the value, and (ii) the opcode that it
    /// would be attached to if the expression was in a left-linearized form.
    /// This is required to avoid illegal operand reordering.
    /// For example:
    /// \verbatim
    ///                         0 Op1
    ///                         |/
    /// Op1 Op2   Linearized    + Op2
    ///   \ /     ---------->   |/
    ///    -                    -
    ///
    /// Op1 - Op2            (0 + Op1) - Op2
    /// \endverbatim
    ///
    /// Value Op1 is attached to a '+' operation, and Op2 to a '-'.
    ///
    /// Another way to think of this is to track all the operations across the
    /// path from the operand all the way to the root of the tree and to
    /// calculate the operation that corresponds to this path. For example, the
    /// path from Op2 to the root crosses the RHS of the '-', therefore the
    /// corresponding operation is a '-' (which matches the one in the
    /// linearized tree, as shown above).
    ///
    /// For lack of a better term, we refer to this operation as Accumulated
    /// Path Operation (APO).
    struct OperandData {
      OperandData() = default;
      OperandData(Value *V, bool APO, bool IsUsed)
          : V(V), APO(APO), IsUsed(IsUsed) {}
      /// The operand value.
      Value *V = nullptr;
      /// TreeEntries only allow a single opcode, or an alternate sequence of
      /// them (e.g, +, -). Therefore, we can safely use a boolean value for the
      /// APO. It is set to 'true' if 'V' is attached to an inverse operation
      /// in the left-linearized form (e.g., Sub/Div), and 'false' otherwise
      /// (e.g., Add/Mul)
      bool APO = false;
      /// Helper data for the reordering function.
      bool IsUsed = false;
    };

    /// During operand reordering, we are trying to select the operand at lane
    /// that matches best with the operand at the neighboring lane. Our
    /// selection is based on the type of value we are looking for. For example,
    /// if the neighboring lane has a load, we need to look for a load that is
    /// accessing a consecutive address. These strategies are summarized in the
    /// 'ReorderingMode' enumerator.
    enum class ReorderingMode {
      Load,     ///< Matching loads to consecutive memory addresses
      Opcode,   ///< Matching instructions based on opcode (same or alternate)
      Constant, ///< Matching constants
      Splat,    ///< Matching the same instruction multiple times (broadcast)
      Failed,   ///< We failed to create a vectorizable group
    };

    using OperandDataVec = SmallVector<OperandData, 2>;

    /// A vector of operand vectors.
    SmallVector<OperandDataVec, 4> OpsVec;

    const DataLayout &DL;
    ScalarEvolution &SE;
    const BoUpSLP &R;

    /// \returns the operand data at \p OpIdx and \p Lane.
    OperandData &getData(unsigned OpIdx, unsigned Lane) {
      return OpsVec[OpIdx][Lane];
    }

    /// \returns the operand data at \p OpIdx and \p Lane. Const version.
    const OperandData &getData(unsigned OpIdx, unsigned Lane) const {
      return OpsVec[OpIdx][Lane];
    }

    /// Clears the used flag for all entries.
    void clearUsed() {
      for (unsigned OpIdx = 0, NumOperands = getNumOperands();
           OpIdx != NumOperands; ++OpIdx)
        for (unsigned Lane = 0, NumLanes = getNumLanes(); Lane != NumLanes;
             ++Lane)
          OpsVec[OpIdx][Lane].IsUsed = false;
    }

    /// Swap the operand at \p OpIdx1 with that one at \p OpIdx2.
    void swap(unsigned OpIdx1, unsigned OpIdx2, unsigned Lane) {
      std::swap(OpsVec[OpIdx1][Lane], OpsVec[OpIdx2][Lane]);
    }

    // The hard-coded scores listed here are not very important, though it shall
    // be higher for better matches to improve the resulting cost. When
    // computing the scores of matching one sub-tree with another, we are
    // basically counting the number of values that are matching. So even if all
    // scores are set to 1, we would still get a decent matching result.
    // However, sometimes we have to break ties. For example we may have to
    // choose between matching loads vs matching opcodes. This is what these
    // scores are helping us with: they provide the order of preference. Also,
    // this is important if the scalar is externally used or used in another
    // tree entry node in the different lane.

    /// Loads from consecutive memory addresses, e.g. load(A[i]), load(A[i+1]).
    static const int ScoreConsecutiveLoads = 4;
    /// Loads from reversed memory addresses, e.g. load(A[i+1]), load(A[i]).
    static const int ScoreReversedLoads = 3;
    /// ExtractElementInst from same vector and consecutive indexes.
    static const int ScoreConsecutiveExtracts = 4;
    /// ExtractElementInst from same vector and reversed indices.
    static const int ScoreReversedExtracts = 3;
    /// Constants.
    static const int ScoreConstants = 2;
    /// Instructions with the same opcode.
    static const int ScoreSameOpcode = 2;
    /// Instructions with alt opcodes (e.g, add + sub).
    static const int ScoreAltOpcodes = 1;
    /// Identical instructions (a.k.a. splat or broadcast).
    static const int ScoreSplat = 1;
    /// Matching with an undef is preferable to failing.
    static const int ScoreUndef = 1;
    /// Score for failing to find a decent match.
    static const int ScoreFail = 0;
    /// Score if all users are vectorized.
    static const int ScoreAllUserVectorized = 1;

    /// \returns the score of placing \p V1 and \p V2 in consecutive lanes.
    /// Also, checks if \p V1 and \p V2 are compatible with instructions in \p
    /// MainAltOps.
    static int getShallowScore(Value *V1, Value *V2, const DataLayout &DL,
                               ScalarEvolution &SE, int NumLanes,
                               ArrayRef<Value *> MainAltOps) {
      if (V1 == V2)
        return VLOperands::ScoreSplat;

      auto *LI1 = dyn_cast<LoadInst>(V1);
      auto *LI2 = dyn_cast<LoadInst>(V2);
      if (LI1 && LI2) {
        if (LI1->getParent() != LI2->getParent())
          return VLOperands::ScoreFail;

        Optional<int> Dist = getPointersDiff(
            LI1->getType(), LI1->getPointerOperand(), LI2->getType(),
            LI2->getPointerOperand(), DL, SE, /*StrictCheck=*/true);
        if (!Dist || *Dist == 0)
          return VLOperands::ScoreFail;
        // The distance is too large - still may be profitable to use masked
        // loads/gathers.
        if (std::abs(*Dist) > NumLanes / 2)
          return VLOperands::ScoreAltOpcodes;
        // This still will detect consecutive loads, but we might have "holes"
        // in some cases. It is ok for non-power-2 vectorization and may produce
        // better results. It should not affect current vectorization.
        return (*Dist > 0) ? VLOperands::ScoreConsecutiveLoads
                           : VLOperands::ScoreReversedLoads;
      }

      auto *C1 = dyn_cast<Constant>(V1);
      auto *C2 = dyn_cast<Constant>(V2);
      if (C1 && C2)
        return VLOperands::ScoreConstants;

      // Extracts from consecutive indexes of the same vector better score as
      // the extracts could be optimized away.
      Value *EV1;
      ConstantInt *Ex1Idx;
      if (match(V1, m_ExtractElt(m_Value(EV1), m_ConstantInt(Ex1Idx)))) {
        // Undefs are always profitable for extractelements.
        if (isa<UndefValue>(V2))
          return VLOperands::ScoreConsecutiveExtracts;
        Value *EV2 = nullptr;
        ConstantInt *Ex2Idx = nullptr;
        if (match(V2,
                  m_ExtractElt(m_Value(EV2), m_CombineOr(m_ConstantInt(Ex2Idx),
                                                         m_Undef())))) {
          // Undefs are always profitable for extractelements.
          if (!Ex2Idx)
            return VLOperands::ScoreConsecutiveExtracts;
          if (isUndefVector(EV2) && EV2->getType() == EV1->getType())
            return VLOperands::ScoreConsecutiveExtracts;
          if (EV2 == EV1) {
            int Idx1 = Ex1Idx->getZExtValue();
            int Idx2 = Ex2Idx->getZExtValue();
            int Dist = Idx2 - Idx1;
            // The distance is too large - still may be profitable to use
            // shuffles.
            if (std::abs(Dist) == 0)
              return VLOperands::ScoreSplat;
            if (std::abs(Dist) > NumLanes / 2)
              return VLOperands::ScoreSameOpcode;
            return (Dist > 0) ? VLOperands::ScoreConsecutiveExtracts
                              : VLOperands::ScoreReversedExtracts;
          }
          return VLOperands::ScoreAltOpcodes;
        }
        return VLOperands::ScoreFail;
      }

      auto *I1 = dyn_cast<Instruction>(V1);
      auto *I2 = dyn_cast<Instruction>(V2);
      if (I1 && I2) {
        if (I1->getParent() != I2->getParent())
          return VLOperands::ScoreFail;
        SmallVector<Value *, 4> Ops(MainAltOps.begin(), MainAltOps.end());
        Ops.push_back(I1);
        Ops.push_back(I2);
        InstructionsState S = getSameOpcode(Ops);
        // Note: Only consider instructions with <= 2 operands to avoid
        // complexity explosion.
        if (S.getOpcode() &&
            (S.MainOp->getNumOperands() <= 2 || !MainAltOps.empty() ||
             !S.isAltShuffle()) &&
            all_of(Ops, [&S](Value *V) {
              return cast<Instruction>(V)->getNumOperands() ==
                     S.MainOp->getNumOperands();
            }))
          return S.isAltShuffle() ? VLOperands::ScoreAltOpcodes
                                  : VLOperands::ScoreSameOpcode;
      }

      if (isa<UndefValue>(V2))
        return VLOperands::ScoreUndef;

      return VLOperands::ScoreFail;
    }

    /// \param Lane lane of the operands under analysis.
    /// \param OpIdx operand index in \p Lane lane we're looking the best
    /// candidate for.
    /// \param Idx operand index of the current candidate value.
    /// \returns The additional score due to possible broadcasting of the
    /// elements in the lane. It is more profitable to have power-of-2 unique
    /// elements in the lane, it will be vectorized with higher probability
    /// after removing duplicates. Currently the SLP vectorizer supports only
    /// vectorization of the power-of-2 number of unique scalars.
    int getSplatScore(unsigned Lane, unsigned OpIdx, unsigned Idx) const {
      Value *IdxLaneV = getData(Idx, Lane).V;
      if (!isa<Instruction>(IdxLaneV) || IdxLaneV == getData(OpIdx, Lane).V)
        return 0;
      SmallPtrSet<Value *, 4> Uniques;
      for (unsigned Ln = 0, E = getNumLanes(); Ln < E; ++Ln) {
        if (Ln == Lane)
          continue;
        Value *OpIdxLnV = getData(OpIdx, Ln).V;
        if (!isa<Instruction>(OpIdxLnV))
          return 0;
        Uniques.insert(OpIdxLnV);
      }
      int UniquesCount = Uniques.size();
      int UniquesCntWithIdxLaneV =
          Uniques.contains(IdxLaneV) ? UniquesCount : UniquesCount + 1;
      Value *OpIdxLaneV = getData(OpIdx, Lane).V;
      int UniquesCntWithOpIdxLaneV =
          Uniques.contains(OpIdxLaneV) ? UniquesCount : UniquesCount + 1;
      if (UniquesCntWithIdxLaneV == UniquesCntWithOpIdxLaneV)
        return 0;
      return (PowerOf2Ceil(UniquesCntWithOpIdxLaneV) -
              UniquesCntWithOpIdxLaneV) -
             (PowerOf2Ceil(UniquesCntWithIdxLaneV) - UniquesCntWithIdxLaneV);
    }

    /// \param Lane lane of the operands under analysis.
    /// \param OpIdx operand index in \p Lane lane we're looking the best
    /// candidate for.
    /// \param Idx operand index of the current candidate value.
    /// \returns The additional score for the scalar which users are all
    /// vectorized.
    int getExternalUseScore(unsigned Lane, unsigned OpIdx, unsigned Idx) const {
      Value *IdxLaneV = getData(Idx, Lane).V;
      Value *OpIdxLaneV = getData(OpIdx, Lane).V;
      // Do not care about number of uses for vector-like instructions
      // (extractelement/extractvalue with constant indices), they are extracts
      // themselves and already externally used. Vectorization of such
      // instructions does not add extra extractelement instruction, just may
      // remove it.
      if (isVectorLikeInstWithConstOps(IdxLaneV) &&
          isVectorLikeInstWithConstOps(OpIdxLaneV))
        return VLOperands::ScoreAllUserVectorized;
      auto *IdxLaneI = dyn_cast<Instruction>(IdxLaneV);
      if (!IdxLaneI || !isa<Instruction>(OpIdxLaneV))
        return 0;
      return R.areAllUsersVectorized(IdxLaneI, None)
                 ? VLOperands::ScoreAllUserVectorized
                 : 0;
    }

    /// Go through the operands of \p LHS and \p RHS recursively until \p
    /// MaxLevel, and return the cummulative score. For example:
    /// \verbatim
    ///  A[0]  B[0]  A[1]  B[1]  C[0] D[0]  B[1] A[1]
    ///     \ /         \ /         \ /        \ /
    ///      +           +           +          +
    ///     G1          G2          G3         G4
    /// \endverbatim
    /// The getScoreAtLevelRec(G1, G2) function will try to match the nodes at
    /// each level recursively, accumulating the score. It starts from matching
    /// the additions at level 0, then moves on to the loads (level 1). The
    /// score of G1 and G2 is higher than G1 and G3, because {A[0],A[1]} and
    /// {B[0],B[1]} match with VLOperands::ScoreConsecutiveLoads, while
    /// {A[0],C[0]} has a score of VLOperands::ScoreFail.
    /// Please note that the order of the operands does not matter, as we
    /// evaluate the score of all profitable combinations of operands. In
    /// other words the score of G1 and G4 is the same as G1 and G2. This
    /// heuristic is based on ideas described in:
    ///   Look-ahead SLP: Auto-vectorization in the presence of commutative
    ///   operations, CGO 2018 by Vasileios Porpodas, Rodrigo C. O. Rocha,
    ///   Luís F. W. Góes
    int getScoreAtLevelRec(Value *LHS, Value *RHS, int CurrLevel, int MaxLevel,
                           ArrayRef<Value *> MainAltOps) {

      // Get the shallow score of V1 and V2.
      int ShallowScoreAtThisLevel =
          getShallowScore(LHS, RHS, DL, SE, getNumLanes(), MainAltOps);

      // If reached MaxLevel,
      //  or if V1 and V2 are not instructions,
      //  or if they are SPLAT,
      //  or if they are not consecutive,
      //  or if profitable to vectorize loads or extractelements, early return
      //  the current cost.
      auto *I1 = dyn_cast<Instruction>(LHS);
      auto *I2 = dyn_cast<Instruction>(RHS);
      if (CurrLevel == MaxLevel || !(I1 && I2) || I1 == I2 ||
          ShallowScoreAtThisLevel == VLOperands::ScoreFail ||
          (((isa<LoadInst>(I1) && isa<LoadInst>(I2)) ||
            (I1->getNumOperands() > 2 && I2->getNumOperands() > 2) ||
            (isa<ExtractElementInst>(I1) && isa<ExtractElementInst>(I2))) &&
           ShallowScoreAtThisLevel))
        return ShallowScoreAtThisLevel;
      assert(I1 && I2 && "Should have early exited.");

      // Contains the I2 operand indexes that got matched with I1 operands.
      SmallSet<unsigned, 4> Op2Used;

      // Recursion towards the operands of I1 and I2. We are trying all possible
      // operand pairs, and keeping track of the best score.
      for (unsigned OpIdx1 = 0, NumOperands1 = I1->getNumOperands();
           OpIdx1 != NumOperands1; ++OpIdx1) {
        // Try to pair op1I with the best operand of I2.
        int MaxTmpScore = 0;
        unsigned MaxOpIdx2 = 0;
        bool FoundBest = false;
        // If I2 is commutative try all combinations.
        unsigned FromIdx = isCommutative(I2) ? 0 : OpIdx1;
        unsigned ToIdx = isCommutative(I2)
                             ? I2->getNumOperands()
                             : std::min(I2->getNumOperands(), OpIdx1 + 1);
        assert(FromIdx <= ToIdx && "Bad index");
        for (unsigned OpIdx2 = FromIdx; OpIdx2 != ToIdx; ++OpIdx2) {
          // Skip operands already paired with OpIdx1.
          if (Op2Used.count(OpIdx2))
            continue;
          // Recursively calculate the cost at each level
          int TmpScore =
              getScoreAtLevelRec(I1->getOperand(OpIdx1), I2->getOperand(OpIdx2),
                                 CurrLevel + 1, MaxLevel, None);
          // Look for the best score.
          if (TmpScore > VLOperands::ScoreFail && TmpScore > MaxTmpScore) {
            MaxTmpScore = TmpScore;
            MaxOpIdx2 = OpIdx2;
            FoundBest = true;
          }
        }
        if (FoundBest) {
          // Pair {OpIdx1, MaxOpIdx2} was found to be best. Never revisit it.
          Op2Used.insert(MaxOpIdx2);
          ShallowScoreAtThisLevel += MaxTmpScore;
        }
      }
      return ShallowScoreAtThisLevel;
    }

    /// Score scaling factor for fully compatible instructions but with
    /// different number of external uses. Allows better selection of the
    /// instructions with less external uses.
    static const int ScoreScaleFactor = 10;

    /// \Returns the look-ahead score, which tells us how much the sub-trees
    /// rooted at \p LHS and \p RHS match, the more they match the higher the
    /// score. This helps break ties in an informed way when we cannot decide on
    /// the order of the operands by just considering the immediate
    /// predecessors.
    int getLookAheadScore(Value *LHS, Value *RHS, ArrayRef<Value *> MainAltOps,
                          int Lane, unsigned OpIdx, unsigned Idx,
                          bool &IsUsed) {
      int Score =
          getScoreAtLevelRec(LHS, RHS, 1, LookAheadMaxDepth, MainAltOps);
      if (Score) {
        int SplatScore = getSplatScore(Lane, OpIdx, Idx);
        if (Score <= -SplatScore) {
          // Set the minimum score for splat-like sequence to avoid setting
          // failed state.
          Score = 1;
        } else {
          Score += SplatScore;
          // Scale score to see the difference between different operands
          // and similar operands but all vectorized/not all vectorized
          // uses. It does not affect actual selection of the best
          // compatible operand in general, just allows to select the
          // operand with all vectorized uses.
          Score *= ScoreScaleFactor;
          Score += getExternalUseScore(Lane, OpIdx, Idx);
          IsUsed = true;
        }
      }
      return Score;
    }

    /// Best defined scores per lanes between the passes. Used to choose the
    /// best operand (with the highest score) between the passes.
    /// The key - {Operand Index, Lane}.
    /// The value - the best score between the passes for the lane and the
    /// operand.
    SmallDenseMap<std::pair<unsigned, unsigned>, unsigned, 8>
        BestScoresPerLanes;

    // Search all operands in Ops[*][Lane] for the one that matches best
    // Ops[OpIdx][LastLane] and return its opreand index.
    // If no good match can be found, return None.
    Optional<unsigned> getBestOperand(unsigned OpIdx, int Lane, int LastLane,
                                      ArrayRef<ReorderingMode> ReorderingModes,
                                      ArrayRef<Value *> MainAltOps) {
      unsigned NumOperands = getNumOperands();

      // The operand of the previous lane at OpIdx.
      Value *OpLastLane = getData(OpIdx, LastLane).V;

      // Our strategy mode for OpIdx.
      ReorderingMode RMode = ReorderingModes[OpIdx];
      if (RMode == ReorderingMode::Failed)
        return None;

      // The linearized opcode of the operand at OpIdx, Lane.
      bool OpIdxAPO = getData(OpIdx, Lane).APO;

      // The best operand index and its score.
      // Sometimes we have more than one option (e.g., Opcode and Undefs), so we
      // are using the score to differentiate between the two.
      struct BestOpData {
        Optional<unsigned> Idx = None;
        unsigned Score = 0;
      } BestOp;
      BestOp.Score =
          BestScoresPerLanes.try_emplace(std::make_pair(OpIdx, Lane), 0)
              .first->second;

      // Track if the operand must be marked as used. If the operand is set to
      // Score 1 explicitly (because of non power-of-2 unique scalars, we may
      // want to reestimate the operands again on the following iterations).
      bool IsUsed =
          RMode == ReorderingMode::Splat || RMode == ReorderingMode::Constant;
      // Iterate through all unused operands and look for the best.
      for (unsigned Idx = 0; Idx != NumOperands; ++Idx) {
        // Get the operand at Idx and Lane.
        OperandData &OpData = getData(Idx, Lane);
        Value *Op = OpData.V;
        bool OpAPO = OpData.APO;

        // Skip already selected operands.
        if (OpData.IsUsed)
          continue;

        // Skip if we are trying to move the operand to a position with a
        // different opcode in the linearized tree form. This would break the
        // semantics.
        if (OpAPO != OpIdxAPO)
          continue;

        // Look for an operand that matches the current mode.
        switch (RMode) {
        case ReorderingMode::Load:
        case ReorderingMode::Constant:
        case ReorderingMode::Opcode: {
          bool LeftToRight = Lane > LastLane;
          Value *OpLeft = (LeftToRight) ? OpLastLane : Op;
          Value *OpRight = (LeftToRight) ? Op : OpLastLane;
          int Score = getLookAheadScore(OpLeft, OpRight, MainAltOps, Lane,
                                        OpIdx, Idx, IsUsed);
          if (Score > static_cast<int>(BestOp.Score)) {
            BestOp.Idx = Idx;
            BestOp.Score = Score;
            BestScoresPerLanes[std::make_pair(OpIdx, Lane)] = Score;
          }
          break;
        }
        case ReorderingMode::Splat:
          if (Op == OpLastLane)
            BestOp.Idx = Idx;
          break;
        case ReorderingMode::Failed:
          llvm_unreachable("Not expected Failed reordering mode.");
        }
      }

      if (BestOp.Idx) {
        getData(BestOp.Idx.getValue(), Lane).IsUsed = IsUsed;
        return BestOp.Idx;
      }
      // If we could not find a good match return None.
      return None;
    }

    /// Helper for reorderOperandVecs.
    /// \returns the lane that we should start reordering from. This is the one
    /// which has the least number of operands that can freely move about or
    /// less profitable because it already has the most optimal set of operands.
    unsigned getBestLaneToStartReordering() const {
      unsigned Min = UINT_MAX;
      unsigned SameOpNumber = 0;
      // std::pair<unsigned, unsigned> is used to implement a simple voting
      // algorithm and choose the lane with the least number of operands that
      // can freely move about or less profitable because it already has the
      // most optimal set of operands. The first unsigned is a counter for
      // voting, the second unsigned is the counter of lanes with instructions
      // with same/alternate opcodes and same parent basic block.
      MapVector<unsigned, std::pair<unsigned, unsigned>> HashMap;
      // Try to be closer to the original results, if we have multiple lanes
      // with same cost. If 2 lanes have the same cost, use the one with the
      // lowest index.
      for (int I = getNumLanes(); I > 0; --I) {
        unsigned Lane = I - 1;
        OperandsOrderData NumFreeOpsHash =
            getMaxNumOperandsThatCanBeReordered(Lane);
        // Compare the number of operands that can move and choose the one with
        // the least number.
        if (NumFreeOpsHash.NumOfAPOs < Min) {
          Min = NumFreeOpsHash.NumOfAPOs;
          SameOpNumber = NumFreeOpsHash.NumOpsWithSameOpcodeParent;
          HashMap.clear();
          HashMap[NumFreeOpsHash.Hash] = std::make_pair(1, Lane);
        } else if (NumFreeOpsHash.NumOfAPOs == Min &&
                   NumFreeOpsHash.NumOpsWithSameOpcodeParent < SameOpNumber) {
          // Select the most optimal lane in terms of number of operands that
          // should be moved around.
          SameOpNumber = NumFreeOpsHash.NumOpsWithSameOpcodeParent;
          HashMap[NumFreeOpsHash.Hash] = std::make_pair(1, Lane);
        } else if (NumFreeOpsHash.NumOfAPOs == Min &&
                   NumFreeOpsHash.NumOpsWithSameOpcodeParent == SameOpNumber) {
          auto It = HashMap.find(NumFreeOpsHash.Hash);
          if (It == HashMap.end())
            HashMap[NumFreeOpsHash.Hash] = std::make_pair(1, Lane);
          else
            ++It->second.first;
        }
      }
      // Select the lane with the minimum counter.
      unsigned BestLane = 0;
      unsigned CntMin = UINT_MAX;
      for (const auto &Data : reverse(HashMap)) {
        if (Data.second.first < CntMin) {
          CntMin = Data.second.first;
          BestLane = Data.second.second;
        }
      }
      return BestLane;
    }

    /// Data structure that helps to reorder operands.
    struct OperandsOrderData {
      /// The best number of operands with the same APOs, which can be
      /// reordered.
      unsigned NumOfAPOs = UINT_MAX;
      /// Number of operands with the same/alternate instruction opcode and
      /// parent.
      unsigned NumOpsWithSameOpcodeParent = 0;
      /// Hash for the actual operands ordering.
      /// Used to count operands, actually their position id and opcode
      /// value. It is used in the voting mechanism to find the lane with the
      /// least number of operands that can freely move about or less profitable
      /// because it already has the most optimal set of operands. Can be
      /// replaced with SmallVector<unsigned> instead but hash code is faster
      /// and requires less memory.
      unsigned Hash = 0;
    };
    /// \returns the maximum number of operands that are allowed to be reordered
    /// for \p Lane and the number of compatible instructions(with the same
    /// parent/opcode). This is used as a heuristic for selecting the first lane
    /// to start operand reordering.
    OperandsOrderData getMaxNumOperandsThatCanBeReordered(unsigned Lane) const {
      unsigned CntTrue = 0;
      unsigned NumOperands = getNumOperands();
      // Operands with the same APO can be reordered. We therefore need to count
      // how many of them we have for each APO, like this: Cnt[APO] = x.
      // Since we only have two APOs, namely true and false, we can avoid using
      // a map. Instead we can simply count the number of operands that
      // correspond to one of them (in this case the 'true' APO), and calculate
      // the other by subtracting it from the total number of operands.
      // Operands with the same instruction opcode and parent are more
      // profitable since we don't need to move them in many cases, with a high
      // probability such lane already can be vectorized effectively.
      bool AllUndefs = true;
      unsigned NumOpsWithSameOpcodeParent = 0;
      Instruction *OpcodeI = nullptr;
      BasicBlock *Parent = nullptr;
      unsigned Hash = 0;
      for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
        const OperandData &OpData = getData(OpIdx, Lane);
        if (OpData.APO)
          ++CntTrue;
        // Use Boyer-Moore majority voting for finding the majority opcode and
        // the number of times it occurs.
        if (auto *I = dyn_cast<Instruction>(OpData.V)) {
          if (!OpcodeI || !getSameOpcode({OpcodeI, I}).getOpcode() ||
              I->getParent() != Parent) {
            if (NumOpsWithSameOpcodeParent == 0) {
              NumOpsWithSameOpcodeParent = 1;
              OpcodeI = I;
              Parent = I->getParent();
            } else {
              --NumOpsWithSameOpcodeParent;
            }
          } else {
            ++NumOpsWithSameOpcodeParent;
          }
        }
        Hash = hash_combine(
            Hash, hash_value((OpIdx + 1) * (OpData.V->getValueID() + 1)));
        AllUndefs = AllUndefs && isa<UndefValue>(OpData.V);
      }
      if (AllUndefs)
        return {};
      OperandsOrderData Data;
      Data.NumOfAPOs = std::max(CntTrue, NumOperands - CntTrue);
      Data.NumOpsWithSameOpcodeParent = NumOpsWithSameOpcodeParent;
      Data.Hash = Hash;
      return Data;
    }

    /// Go through the instructions in VL and append their operands.
    void appendOperandsOfVL(ArrayRef<Value *> VL) {
      assert(!VL.empty() && "Bad VL");
      assert((empty() || VL.size() == getNumLanes()) &&
             "Expected same number of lanes");
      assert(isa<Instruction>(VL[0]) && "Expected instruction");
      unsigned NumOperands = cast<Instruction>(VL[0])->getNumOperands();
      OpsVec.resize(NumOperands);
      unsigned NumLanes = VL.size();
      for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
        OpsVec[OpIdx].resize(NumLanes);
        for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
          assert(isa<Instruction>(VL[Lane]) && "Expected instruction");
          // Our tree has just 3 nodes: the root and two operands.
          // It is therefore trivial to get the APO. We only need to check the
          // opcode of VL[Lane] and whether the operand at OpIdx is the LHS or
          // RHS operand. The LHS operand of both add and sub is never attached
          // to an inversese operation in the linearized form, therefore its APO
          // is false. The RHS is true only if VL[Lane] is an inverse operation.

          // Since operand reordering is performed on groups of commutative
          // operations or alternating sequences (e.g., +, -), we can safely
          // tell the inverse operations by checking commutativity.
          bool IsInverseOperation = !isCommutative(cast<Instruction>(VL[Lane]));
          bool APO = (OpIdx == 0) ? false : IsInverseOperation;
          OpsVec[OpIdx][Lane] = {cast<Instruction>(VL[Lane])->getOperand(OpIdx),
                                 APO, false};
        }
      }
    }

    /// \returns the number of operands.
    unsigned getNumOperands() const { return OpsVec.size(); }

    /// \returns the number of lanes.
    unsigned getNumLanes() const { return OpsVec[0].size(); }

    /// \returns the operand value at \p OpIdx and \p Lane.
    Value *getValue(unsigned OpIdx, unsigned Lane) const {
      return getData(OpIdx, Lane).V;
    }

    /// \returns true if the data structure is empty.
    bool empty() const { return OpsVec.empty(); }

    /// Clears the data.
    void clear() { OpsVec.clear(); }

    /// \Returns true if there are enough operands identical to \p Op to fill
    /// the whole vector.
    /// Note: This modifies the 'IsUsed' flag, so a cleanUsed() must follow.
    bool shouldBroadcast(Value *Op, unsigned OpIdx, unsigned Lane) {
      bool OpAPO = getData(OpIdx, Lane).APO;
      for (unsigned Ln = 0, Lns = getNumLanes(); Ln != Lns; ++Ln) {
        if (Ln == Lane)
          continue;
        // This is set to true if we found a candidate for broadcast at Lane.
        bool FoundCandidate = false;
        for (unsigned OpI = 0, OpE = getNumOperands(); OpI != OpE; ++OpI) {
          OperandData &Data = getData(OpI, Ln);
          if (Data.APO != OpAPO || Data.IsUsed)
            continue;
          if (Data.V == Op) {
            FoundCandidate = true;
            Data.IsUsed = true;
            break;
          }
        }
        if (!FoundCandidate)
          return false;
      }
      return true;
    }

  public:
    /// Initialize with all the operands of the instruction vector \p RootVL.
    VLOperands(ArrayRef<Value *> RootVL, const DataLayout &DL,
               ScalarEvolution &SE, const BoUpSLP &R)
        : DL(DL), SE(SE), R(R) {
      // Append all the operands of RootVL.
      appendOperandsOfVL(RootVL);
    }

    /// \Returns a value vector with the operands across all lanes for the
    /// opearnd at \p OpIdx.
    ValueList getVL(unsigned OpIdx) const {
      ValueList OpVL(OpsVec[OpIdx].size());
      assert(OpsVec[OpIdx].size() == getNumLanes() &&
             "Expected same num of lanes across all operands");
      for (unsigned Lane = 0, Lanes = getNumLanes(); Lane != Lanes; ++Lane)
        OpVL[Lane] = OpsVec[OpIdx][Lane].V;
      return OpVL;
    }

    // Performs operand reordering for 2 or more operands.
    // The original operands are in OrigOps[OpIdx][Lane].
    // The reordered operands are returned in 'SortedOps[OpIdx][Lane]'.
    void reorder() {
      unsigned NumOperands = getNumOperands();
      unsigned NumLanes = getNumLanes();
      // Each operand has its own mode. We are using this mode to help us select
      // the instructions for each lane, so that they match best with the ones
      // we have selected so far.
      SmallVector<ReorderingMode, 2> ReorderingModes(NumOperands);

      // This is a greedy single-pass algorithm. We are going over each lane
      // once and deciding on the best order right away with no back-tracking.
      // However, in order to increase its effectiveness, we start with the lane
      // that has operands that can move the least. For example, given the
      // following lanes:
      //  Lane 0 : A[0] = B[0] + C[0]   // Visited 3rd
      //  Lane 1 : A[1] = C[1] - B[1]   // Visited 1st
      //  Lane 2 : A[2] = B[2] + C[2]   // Visited 2nd
      //  Lane 3 : A[3] = C[3] - B[3]   // Visited 4th
      // we will start at Lane 1, since the operands of the subtraction cannot
      // be reordered. Then we will visit the rest of the lanes in a circular
      // fashion. That is, Lanes 2, then Lane 0, and finally Lane 3.

      // Find the first lane that we will start our search from.
      unsigned FirstLane = getBestLaneToStartReordering();

      // Initialize the modes.
      for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
        Value *OpLane0 = getValue(OpIdx, FirstLane);
        // Keep track if we have instructions with all the same opcode on one
        // side.
        if (isa<LoadInst>(OpLane0))
          ReorderingModes[OpIdx] = ReorderingMode::Load;
        else if (isa<Instruction>(OpLane0)) {
          // Check if OpLane0 should be broadcast.
          if (shouldBroadcast(OpLane0, OpIdx, FirstLane))
            ReorderingModes[OpIdx] = ReorderingMode::Splat;
          else
            ReorderingModes[OpIdx] = ReorderingMode::Opcode;
        }
        else if (isa<Constant>(OpLane0))
          ReorderingModes[OpIdx] = ReorderingMode::Constant;
        else if (isa<Argument>(OpLane0))
          // Our best hope is a Splat. It may save some cost in some cases.
          ReorderingModes[OpIdx] = ReorderingMode::Splat;
        else
          // NOTE: This should be unreachable.
          ReorderingModes[OpIdx] = ReorderingMode::Failed;
      }

      // Check that we don't have same operands. No need to reorder if operands
      // are just perfect diamond or shuffled diamond match. Do not do it only
      // for possible broadcasts or non-power of 2 number of scalars (just for
      // now).
      auto &&SkipReordering = [this]() {
        SmallPtrSet<Value *, 4> UniqueValues;
        ArrayRef<OperandData> Op0 = OpsVec.front();
        for (const OperandData &Data : Op0)
          UniqueValues.insert(Data.V);
        for (ArrayRef<OperandData> Op : drop_begin(OpsVec, 1)) {
          if (any_of(Op, [&UniqueValues](const OperandData &Data) {
                return !UniqueValues.contains(Data.V);
              }))
            return false;
        }
        // TODO: Check if we can remove a check for non-power-2 number of
        // scalars after full support of non-power-2 vectorization.
        return UniqueValues.size() != 2 && isPowerOf2_32(UniqueValues.size());
      };

      // If the initial strategy fails for any of the operand indexes, then we
      // perform reordering again in a second pass. This helps avoid assigning
      // high priority to the failed strategy, and should improve reordering for
      // the non-failed operand indexes.
      for (int Pass = 0; Pass != 2; ++Pass) {
        // Check if no need to reorder operands since they're are perfect or
        // shuffled diamond match.
        // Need to to do it to avoid extra external use cost counting for
        // shuffled matches, which may cause regressions.
        if (SkipReordering())
          break;
        // Skip the second pass if the first pass did not fail.
        bool StrategyFailed = false;
        // Mark all operand data as free to use.
        clearUsed();
        // We keep the original operand order for the FirstLane, so reorder the
        // rest of the lanes. We are visiting the nodes in a circular fashion,
        // using FirstLane as the center point and increasing the radius
        // distance.
        SmallVector<SmallVector<Value *, 2>> MainAltOps(NumOperands);
        for (unsigned I = 0; I < NumOperands; ++I)
          MainAltOps[I].push_back(getData(I, FirstLane).V);

        for (unsigned Distance = 1; Distance != NumLanes; ++Distance) {
          // Visit the lane on the right and then the lane on the left.
          for (int Direction : {+1, -1}) {
            int Lane = FirstLane + Direction * Distance;
            if (Lane < 0 || Lane >= (int)NumLanes)
              continue;
            int LastLane = Lane - Direction;
            assert(LastLane >= 0 && LastLane < (int)NumLanes &&
                   "Out of bounds");
            // Look for a good match for each operand.
            for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
              // Search for the operand that matches SortedOps[OpIdx][Lane-1].
              Optional<unsigned> BestIdx = getBestOperand(
                  OpIdx, Lane, LastLane, ReorderingModes, MainAltOps[OpIdx]);
              // By not selecting a value, we allow the operands that follow to
              // select a better matching value. We will get a non-null value in
              // the next run of getBestOperand().
              if (BestIdx) {
                // Swap the current operand with the one returned by
                // getBestOperand().
                swap(OpIdx, BestIdx.getValue(), Lane);
              } else {
                // We failed to find a best operand, set mode to 'Failed'.
                ReorderingModes[OpIdx] = ReorderingMode::Failed;
                // Enable the second pass.
                StrategyFailed = true;
              }
              // Try to get the alternate opcode and follow it during analysis.
              if (MainAltOps[OpIdx].size() != 2) {
                OperandData &AltOp = getData(OpIdx, Lane);
                InstructionsState OpS =
                    getSameOpcode({MainAltOps[OpIdx].front(), AltOp.V});
                if (OpS.getOpcode() && OpS.isAltShuffle())
                  MainAltOps[OpIdx].push_back(AltOp.V);
              }
            }
          }
        }
        // Skip second pass if the strategy did not fail.
        if (!StrategyFailed)
          break;
      }
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    LLVM_DUMP_METHOD static StringRef getModeStr(ReorderingMode RMode) {
      switch (RMode) {
      case ReorderingMode::Load:
        return "Load";
      case ReorderingMode::Opcode:
        return "Opcode";
      case ReorderingMode::Constant:
        return "Constant";
      case ReorderingMode::Splat:
        return "Splat";
      case ReorderingMode::Failed:
        return "Failed";
      }
      llvm_unreachable("Unimplemented Reordering Type");
    }

    LLVM_DUMP_METHOD static raw_ostream &printMode(ReorderingMode RMode,
                                                   raw_ostream &OS) {
      return OS << getModeStr(RMode);
    }

    /// Debug print.
    LLVM_DUMP_METHOD static void dumpMode(ReorderingMode RMode) {
      printMode(RMode, dbgs());
    }

    friend raw_ostream &operator<<(raw_ostream &OS, ReorderingMode RMode) {
      return printMode(RMode, OS);
    }

    LLVM_DUMP_METHOD raw_ostream &print(raw_ostream &OS) const {
      const unsigned Indent = 2;
      unsigned Cnt = 0;
      for (const OperandDataVec &OpDataVec : OpsVec) {
        OS << "Operand " << Cnt++ << "\n";
        for (const OperandData &OpData : OpDataVec) {
          OS.indent(Indent) << "{";
          if (Value *V = OpData.V)
            OS << *V;
          else
            OS << "null";
          OS << ", APO:" << OpData.APO << "}\n";
        }
        OS << "\n";
      }
      return OS;
    }

    /// Debug print.
    LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif
  };

  /// Checks if the instruction is marked for deletion.
  bool isDeleted(Instruction *I) const { return DeletedInstructions.count(I); }

  /// Marks values operands for later deletion by replacing them with Undefs.
  void eraseInstructions(ArrayRef<Value *> AV);

  ~BoUpSLP();

private:
  /// Check if the operands on the edges \p Edges of the \p UserTE allows
  /// reordering (i.e. the operands can be reordered because they have only one
  /// user and reordarable).
  /// \param NonVectorized List of all gather nodes that require reordering
  /// (e.g., gather of extractlements or partially vectorizable loads).
  /// \param GatherOps List of gather operand nodes for \p UserTE that require
  /// reordering, subset of \p NonVectorized.
  bool
  canReorderOperands(TreeEntry *UserTE,
                     SmallVectorImpl<std::pair<unsigned, TreeEntry *>> &Edges,
                     ArrayRef<TreeEntry *> ReorderableGathers,
                     SmallVectorImpl<TreeEntry *> &GatherOps);

  /// Returns vectorized operand \p OpIdx of the node \p UserTE from the graph,
  /// if any. If it is not vectorized (gather node), returns nullptr.
  TreeEntry *getVectorizedOperand(TreeEntry *UserTE, unsigned OpIdx) {
    ArrayRef<Value *> VL = UserTE->getOperand(OpIdx);
    TreeEntry *TE = nullptr;
    const auto *It = find_if(VL, [this, &TE](Value *V) {
      TE = getTreeEntry(V);
      return TE;
    });
    if (It != VL.end() && TE->isSame(VL))
      return TE;
    return nullptr;
  }

  /// Returns vectorized operand \p OpIdx of the node \p UserTE from the graph,
  /// if any. If it is not vectorized (gather node), returns nullptr.
  const TreeEntry *getVectorizedOperand(const TreeEntry *UserTE,
                                        unsigned OpIdx) const {
    return const_cast<BoUpSLP *>(this)->getVectorizedOperand(
        const_cast<TreeEntry *>(UserTE), OpIdx);
  }

  /// Checks if all users of \p I are the part of the vectorization tree.
  bool areAllUsersVectorized(Instruction *I,
                             ArrayRef<Value *> VectorizedVals) const;

  /// \returns the cost of the vectorizable entry.
  InstructionCost getEntryCost(const TreeEntry *E,
                               ArrayRef<Value *> VectorizedVals);

  /// This is the recursive part of buildTree.
  void buildTree_rec(ArrayRef<Value *> Roots, unsigned Depth,
                     const EdgeInfo &EI);

  /// \returns true if the ExtractElement/ExtractValue instructions in \p VL can
  /// be vectorized to use the original vector (or aggregate "bitcast" to a
  /// vector) and sets \p CurrentOrder to the identity permutation; otherwise
  /// returns false, setting \p CurrentOrder to either an empty vector or a
  /// non-identity permutation that allows to reuse extract instructions.
  bool canReuseExtract(ArrayRef<Value *> VL, Value *OpValue,
                       SmallVectorImpl<unsigned> &CurrentOrder) const;

  /// Vectorize a single entry in the tree.
  Value *vectorizeTree(TreeEntry *E);

  /// Vectorize a single entry in the tree, starting in \p VL.
  Value *vectorizeTree(ArrayRef<Value *> VL);

  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars. If \p
  /// NeedToShuffle is true, need to add a cost of reshuffling some of the
  /// vector elements.
  InstructionCost getGatherCost(FixedVectorType *Ty,
                                const APInt &ShuffledIndices,
                                bool NeedToShuffle) const;

  /// Checks if the gathered \p VL can be represented as shuffle(s) of previous
  /// tree entries.
  /// \returns ShuffleKind, if gathered values can be represented as shuffles of
  /// previous tree entries. \p Mask is filled with the shuffle mask.
  Optional<TargetTransformInfo::ShuffleKind>
  isGatherShuffledEntry(const TreeEntry *TE, SmallVectorImpl<int> &Mask,
                        SmallVectorImpl<const TreeEntry *> &Entries);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  InstructionCost getGatherCost(ArrayRef<Value *> VL) const;

  /// Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(const TreeEntry *E);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *gather(ArrayRef<Value *> VL);

  /// \returns whether the VectorizableTree is fully vectorizable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree(bool ForReduction) const;

  /// Reorder commutative or alt operands to get better probability of
  /// generating vectorized code.
  static void reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                             SmallVectorImpl<Value *> &Left,
                                             SmallVectorImpl<Value *> &Right,
                                             const DataLayout &DL,
                                             ScalarEvolution &SE,
                                             const BoUpSLP &R);
  struct TreeEntry {
    using VecTreeTy = SmallVector<std::unique_ptr<TreeEntry>, 8>;
    TreeEntry(VecTreeTy &Container) : Container(Container) {}

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) const {
      auto &&IsSame = [VL](ArrayRef<Value *> Scalars, ArrayRef<int> Mask) {
        if (Mask.size() != VL.size() && VL.size() == Scalars.size())
          return std::equal(VL.begin(), VL.end(), Scalars.begin());
        return VL.size() == Mask.size() &&
               std::equal(VL.begin(), VL.end(), Mask.begin(),
                          [Scalars](Value *V, int Idx) {
                            return (isa<UndefValue>(V) &&
                                    Idx == UndefMaskElem) ||
                                   (Idx != UndefMaskElem && V == Scalars[Idx]);
                          });
      };
      if (!ReorderIndices.empty()) {
        // TODO: implement matching if the nodes are just reordered, still can
        // treat the vector as the same if the list of scalars matches VL
        // directly, without reordering.
        SmallVector<int> Mask;
        inversePermutation(ReorderIndices, Mask);
        if (VL.size() == Scalars.size())
          return IsSame(Scalars, Mask);
        if (VL.size() == ReuseShuffleIndices.size()) {
          ::addMask(Mask, ReuseShuffleIndices);
          return IsSame(Scalars, Mask);
        }
        return false;
      }
      return IsSame(Scalars, ReuseShuffleIndices);
    }

    /// \returns true if current entry has same operands as \p TE.
    bool hasEqualOperands(const TreeEntry &TE) const {
      if (TE.getNumOperands() != getNumOperands())
        return false;
      SmallBitVector Used(getNumOperands());
      for (unsigned I = 0, E = getNumOperands(); I < E; ++I) {
        unsigned PrevCount = Used.count();
        for (unsigned K = 0; K < E; ++K) {
          if (Used.test(K))
            continue;
          if (getOperand(K) == TE.getOperand(I)) {
            Used.set(K);
            break;
          }
        }
        // Check if we actually found the matching operand.
        if (PrevCount == Used.count())
          return false;
      }
      return true;
    }

    /// \return Final vectorization factor for the node. Defined by the total
    /// number of vectorized scalars, including those, used several times in the
    /// entry and counted in the \a ReuseShuffleIndices, if any.
    unsigned getVectorFactor() const {
      if (!ReuseShuffleIndices.empty())
        return ReuseShuffleIndices.size();
      return Scalars.size();
    };

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    Value *VectorizedValue = nullptr;

    /// Do we need to gather this sequence or vectorize it
    /// (either with vector instruction or with scatter/gather
    /// intrinsics for store/load)?
    enum EntryState { Vectorize, ScatterVectorize, NeedToGather };
    EntryState State;

    /// Does this sequence require some shuffling?
    SmallVector<int, 4> ReuseShuffleIndices;

    /// Does this entry require reordering?
    SmallVector<unsigned, 4> ReorderIndices;

    /// Points back to the VectorizableTree.
    ///
    /// Only used for Graphviz right now.  Unfortunately GraphTrait::NodeRef has
    /// to be a pointer and needs to be able to initialize the child iterator.
    /// Thus we need a reference back to the container to translate the indices
    /// to entries.
    VecTreeTy &Container;

    /// The TreeEntry index containing the user of this entry.  We can actually
    /// have multiple users so the data structure is not truly a tree.
    SmallVector<EdgeInfo, 1> UserTreeIndices;

    /// The index of this treeEntry in VectorizableTree.
    int Idx = -1;

  private:
    /// The operands of each instruction in each lane Operands[op_index][lane].
    /// Note: This helps avoid the replication of the code that performs the
    /// reordering of operands during buildTree_rec() and vectorizeTree().
    SmallVector<ValueList, 2> Operands;

    /// The main/alternate instruction.
    Instruction *MainOp = nullptr;
    Instruction *AltOp = nullptr;

  public:
    /// Set this bundle's \p OpIdx'th operand to \p OpVL.
    void setOperand(unsigned OpIdx, ArrayRef<Value *> OpVL) {
      if (Operands.size() < OpIdx + 1)
        Operands.resize(OpIdx + 1);
      assert(Operands[OpIdx].empty() && "Already resized?");
      assert(OpVL.size() <= Scalars.size() &&
             "Number of operands is greater than the number of scalars.");
      Operands[OpIdx].resize(OpVL.size());
      copy(OpVL, Operands[OpIdx].begin());
    }

    /// Set the operands of this bundle in their original order.
    void setOperandsInOrder() {
      assert(Operands.empty() && "Already initialized?");
      auto *I0 = cast<Instruction>(Scalars[0]);
      Operands.resize(I0->getNumOperands());
      unsigned NumLanes = Scalars.size();
      for (unsigned OpIdx = 0, NumOperands = I0->getNumOperands();
           OpIdx != NumOperands; ++OpIdx) {
        Operands[OpIdx].resize(NumLanes);
        for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
          auto *I = cast<Instruction>(Scalars[Lane]);
          assert(I->getNumOperands() == NumOperands &&
                 "Expected same number of operands");
          Operands[OpIdx][Lane] = I->getOperand(OpIdx);
        }
      }
    }

    /// Reorders operands of the node to the given mask \p Mask.
    void reorderOperands(ArrayRef<int> Mask) {
      for (ValueList &Operand : Operands)
        reorderScalars(Operand, Mask);
    }

    /// \returns the \p OpIdx operand of this TreeEntry.
    ValueList &getOperand(unsigned OpIdx) {
      assert(OpIdx < Operands.size() && "Off bounds");
      return Operands[OpIdx];
    }

    /// \returns the \p OpIdx operand of this TreeEntry.
    ArrayRef<Value *> getOperand(unsigned OpIdx) const {
      assert(OpIdx < Operands.size() && "Off bounds");
      return Operands[OpIdx];
    }

    /// \returns the number of operands.
    unsigned getNumOperands() const { return Operands.size(); }

    /// \return the single \p OpIdx operand.
    Value *getSingleOperand(unsigned OpIdx) const {
      assert(OpIdx < Operands.size() && "Off bounds");
      assert(!Operands[OpIdx].empty() && "No operand available");
      return Operands[OpIdx][0];
    }

    /// Some of the instructions in the list have alternate opcodes.
    bool isAltShuffle() const { return MainOp != AltOp; }

    bool isOpcodeOrAlt(Instruction *I) const {
      unsigned CheckedOpcode = I->getOpcode();
      return (getOpcode() == CheckedOpcode ||
              getAltOpcode() == CheckedOpcode);
    }

    /// Chooses the correct key for scheduling data. If \p Op has the same (or
    /// alternate) opcode as \p OpValue, the key is \p Op. Otherwise the key is
    /// \p OpValue.
    Value *isOneOf(Value *Op) const {
      auto *I = dyn_cast<Instruction>(Op);
      if (I && isOpcodeOrAlt(I))
        return Op;
      return MainOp;
    }

    void setOperations(const InstructionsState &S) {
      MainOp = S.MainOp;
      AltOp = S.AltOp;
    }

    Instruction *getMainOp() const {
      return MainOp;
    }

    Instruction *getAltOp() const {
      return AltOp;
    }

    /// The main/alternate opcodes for the list of instructions.
    unsigned getOpcode() const {
      return MainOp ? MainOp->getOpcode() : 0;
    }

    unsigned getAltOpcode() const {
      return AltOp ? AltOp->getOpcode() : 0;
    }

    /// When ReuseReorderShuffleIndices is empty it just returns position of \p
    /// V within vector of Scalars. Otherwise, try to remap on its reuse index.
    int findLaneForValue(Value *V) const {
      unsigned FoundLane = std::distance(Scalars.begin(), find(Scalars, V));
      assert(FoundLane < Scalars.size() && "Couldn't find extract lane");
      if (!ReorderIndices.empty())
        FoundLane = ReorderIndices[FoundLane];
      assert(FoundLane < Scalars.size() && "Couldn't find extract lane");
      if (!ReuseShuffleIndices.empty()) {
        FoundLane = std::distance(ReuseShuffleIndices.begin(),
                                  find(ReuseShuffleIndices, FoundLane));
      }
      return FoundLane;
    }

#ifndef NDEBUG
    /// Debug printer.
    LLVM_DUMP_METHOD void dump() const {
      dbgs() << Idx << ".\n";
      for (unsigned OpI = 0, OpE = Operands.size(); OpI != OpE; ++OpI) {
        dbgs() << "Operand " << OpI << ":\n";
        for (const Value *V : Operands[OpI])
          dbgs().indent(2) << *V << "\n";
      }
      dbgs() << "Scalars: \n";
      for (Value *V : Scalars)
        dbgs().indent(2) << *V << "\n";
      dbgs() << "State: ";
      switch (State) {
      case Vectorize:
        dbgs() << "Vectorize\n";
        break;
      case ScatterVectorize:
        dbgs() << "ScatterVectorize\n";
        break;
      case NeedToGather:
        dbgs() << "NeedToGather\n";
        break;
      }
      dbgs() << "MainOp: ";
      if (MainOp)
        dbgs() << *MainOp << "\n";
      else
        dbgs() << "NULL\n";
      dbgs() << "AltOp: ";
      if (AltOp)
        dbgs() << *AltOp << "\n";
      else
        dbgs() << "NULL\n";
      dbgs() << "VectorizedValue: ";
      if (VectorizedValue)
        dbgs() << *VectorizedValue << "\n";
      else
        dbgs() << "NULL\n";
      dbgs() << "ReuseShuffleIndices: ";
      if (ReuseShuffleIndices.empty())
        dbgs() << "Empty";
      else
        for (int ReuseIdx : ReuseShuffleIndices)
          dbgs() << ReuseIdx << ", ";
      dbgs() << "\n";
      dbgs() << "ReorderIndices: ";
      for (unsigned ReorderIdx : ReorderIndices)
        dbgs() << ReorderIdx << ", ";
      dbgs() << "\n";
      dbgs() << "UserTreeIndices: ";
      for (const auto &EInfo : UserTreeIndices)
        dbgs() << EInfo << ", ";
      dbgs() << "\n";
    }
#endif
  };

#ifndef NDEBUG
  void dumpTreeCosts(const TreeEntry *E, InstructionCost ReuseShuffleCost,
                     InstructionCost VecCost,
                     InstructionCost ScalarCost) const {
    dbgs() << "SLP: Calculated costs for Tree:\n"; E->dump();
    dbgs() << "SLP: Costs:\n";
    dbgs() << "SLP:     ReuseShuffleCost = " << ReuseShuffleCost << "\n";
    dbgs() << "SLP:     VectorCost = " << VecCost << "\n";
    dbgs() << "SLP:     ScalarCost = " << ScalarCost << "\n";
    dbgs() << "SLP:     ReuseShuffleCost + VecCost - ScalarCost = " <<
               ReuseShuffleCost + VecCost - ScalarCost << "\n";
  }
#endif

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, Optional<ScheduleData *> Bundle,
                          const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<int> ReuseShuffleIndices = None,
                          ArrayRef<unsigned> ReorderIndices = None) {
    TreeEntry::EntryState EntryState =
        Bundle ? TreeEntry::Vectorize : TreeEntry::NeedToGather;
    return newTreeEntry(VL, EntryState, Bundle, S, UserTreeIdx,
                        ReuseShuffleIndices, ReorderIndices);
  }

  TreeEntry *newTreeEntry(ArrayRef<Value *> VL,
                          TreeEntry::EntryState EntryState,
                          Optional<ScheduleData *> Bundle,
                          const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<int> ReuseShuffleIndices = None,
                          ArrayRef<unsigned> ReorderIndices = None) {
    assert(((!Bundle && EntryState == TreeEntry::NeedToGather) ||
            (Bundle && EntryState != TreeEntry::NeedToGather)) &&
           "Need to vectorize gather entry?");
    VectorizableTree.push_back(std::make_unique<TreeEntry>(VectorizableTree));
    TreeEntry *Last = VectorizableTree.back().get();
    Last->Idx = VectorizableTree.size() - 1;
    Last->State = EntryState;
    Last->ReuseShuffleIndices.append(ReuseShuffleIndices.begin(),
                                     ReuseShuffleIndices.end());
    if (ReorderIndices.empty()) {
      Last->Scalars.assign(VL.begin(), VL.end());
      Last->setOperations(S);
    } else {
      // Reorder scalars and build final mask.
      Last->Scalars.assign(VL.size(), nullptr);
      transform(ReorderIndices, Last->Scalars.begin(),
                [VL](unsigned Idx) -> Value * {
                  if (Idx >= VL.size())
                    return UndefValue::get(VL.front()->getType());
                  return VL[Idx];
                });
      InstructionsState S = getSameOpcode(Last->Scalars);
      Last->setOperations(S);
      Last->ReorderIndices.append(ReorderIndices.begin(), ReorderIndices.end());
    }
    if (Last->State != TreeEntry::NeedToGather) {
      for (Value *V : VL) {
        assert(!getTreeEntry(V) && "Scalar already in tree!");
        ScalarToTreeEntry[V] = Last;
      }
      // Update the scheduler bundle to point to this TreeEntry.
      unsigned Lane = 0;
      for (ScheduleData *BundleMember = Bundle.getValue(); BundleMember;
           BundleMember = BundleMember->NextInBundle) {
        BundleMember->TE = Last;
        BundleMember->Lane = Lane;
        ++Lane;
      }
      assert((!Bundle.getValue() || Lane == VL.size()) &&
             "Bundle and VL out of sync");
    } else {
      MustGather.insert(VL.begin(), VL.end());
    }

    if (UserTreeIdx.UserTE)
      Last->UserTreeIndices.push_back(UserTreeIdx);

    return Last;
  }

  /// -- Vectorization State --
  /// Holds all of the tree entries.
  TreeEntry::VecTreeTy VectorizableTree;

#ifndef NDEBUG
  /// Debug printer.
  LLVM_DUMP_METHOD void dumpVectorizableTree() const {
    for (unsigned Id = 0, IdE = VectorizableTree.size(); Id != IdE; ++Id) {
      VectorizableTree[Id]->dump();
      dbgs() << "\n";
    }
  }
#endif

  TreeEntry *getTreeEntry(Value *V) { return ScalarToTreeEntry.lookup(V); }

  const TreeEntry *getTreeEntry(Value *V) const {
    return ScalarToTreeEntry.lookup(V);
  }

  /// Maps a specific scalar to its tree entry.
  SmallDenseMap<Value*, TreeEntry *> ScalarToTreeEntry;

  /// Maps a value to the proposed vectorizable size.
  SmallDenseMap<Value *, unsigned> InstrElementSize;

  /// A list of scalars that we found that we need to keep as scalars.
  ValueSet MustGather;

  /// This POD struct describes one external user in the vectorized tree.
  struct ExternalUser {
    ExternalUser(Value *S, llvm::User *U, int L)
        : Scalar(S), User(U), Lane(L) {}

    // Which scalar in our function.
    Value *Scalar;

    // Which user that uses the scalar.
    llvm::User *User;

    // Which lane does the scalar belong to.
    int Lane;
  };
  using UserList = SmallVector<ExternalUser, 16>;

  /// Checks if two instructions may access the same memory.
  ///
  /// \p Loc1 is the location of \p Inst1. It is passed explicitly because it
  /// is invariant in the calling loop.
  bool isAliased(const MemoryLocation &Loc1, Instruction *Inst1,
                 Instruction *Inst2) {
    // First check if the result is already in the cache.
    AliasCacheKey key = std::make_pair(Inst1, Inst2);
    Optional<bool> &result = AliasCache[key];
    if (result.hasValue()) {
      return result.getValue();
    }
    bool aliased = true;
    if (Loc1.Ptr && isSimple(Inst1))
      aliased = isModOrRefSet(BatchAA.getModRefInfo(Inst2, Loc1));
    // Store the result in the cache.
    result = aliased;
    return aliased;
  }

  using AliasCacheKey = std::pair<Instruction *, Instruction *>;

  /// Cache for alias results.
  /// TODO: consider moving this to the AliasAnalysis itself.
  DenseMap<AliasCacheKey, Optional<bool>> AliasCache;

  // Cache for pointerMayBeCaptured calls inside AA.  This is preserved
  // globally through SLP because we don't perform any action which
  // invalidates capture results.
  BatchAAResults BatchAA;

  /// Removes an instruction from its block and eventually deletes it.
  /// It's like Instruction::eraseFromParent() except that the actual deletion
  /// is delayed until BoUpSLP is destructed.
  /// This is required to ensure that there are no incorrect collisions in the
  /// AliasCache, which can happen if a new instruction is allocated at the
  /// same address as a previously deleted instruction.
  void eraseInstruction(Instruction *I, bool ReplaceOpsWithUndef = false) {
    auto It = DeletedInstructions.try_emplace(I, ReplaceOpsWithUndef).first;
    It->getSecond() = It->getSecond() && ReplaceOpsWithUndef;
  }

  /// Temporary store for deleted instructions. Instructions will be deleted
  /// eventually when the BoUpSLP is destructed.
  DenseMap<Instruction *, bool> DeletedInstructions;

  /// A list of values that need to extracted out of the tree.
  /// This list holds pairs of (Internal Scalar : External User). External User
  /// can be nullptr, it means that this Internal Scalar will be used later,
  /// after vectorization.
  UserList ExternalUses;

  /// Values used only by @llvm.assume calls.
  SmallPtrSet<const Value *, 32> EphValues;

  /// Holds all of the instructions that we gathered.
  SetVector<Instruction *> GatherShuffleSeq;

  /// A list of blocks that we are going to CSE.
  SetVector<BasicBlock *> CSEBlocks;

  /// Contains all scheduling relevant data for an instruction.
  /// A ScheduleData either represents a single instruction or a member of an
  /// instruction bundle (= a group of instructions which is combined into a
  /// vector instruction).
  struct ScheduleData {
    // The initial value for the dependency counters. It means that the
    // dependencies are not calculated yet.
    enum { InvalidDeps = -1 };

    ScheduleData() = default;

    void init(int BlockSchedulingRegionID, Value *OpVal) {
      FirstInBundle = this;
      NextInBundle = nullptr;
      NextLoadStore = nullptr;
      IsScheduled = false;
      SchedulingRegionID = BlockSchedulingRegionID;
      clearDependencies();
      OpValue = OpVal;
      TE = nullptr;
      Lane = -1;
    }

    /// Verify basic self consistency properties
    void verify() {
      if (hasValidDependencies()) {
        assert(UnscheduledDeps <= Dependencies && "invariant");
      } else {
        assert(UnscheduledDeps == Dependencies && "invariant");
      }

      if (IsScheduled) {
        assert(isSchedulingEntity() &&
                "unexpected scheduled state");
        for (const ScheduleData *BundleMember = this; BundleMember;
             BundleMember = BundleMember->NextInBundle) {
          assert(BundleMember->hasValidDependencies() &&
                 BundleMember->UnscheduledDeps == 0 &&
                 "unexpected scheduled state");
          assert((BundleMember == this || !BundleMember->IsScheduled) &&
                 "only bundle is marked scheduled");
        }
      }

      assert(Inst->getParent() == FirstInBundle->Inst->getParent() &&
             "all bundle members must be in same basic block");
    }

    /// Returns true if the dependency information has been calculated.
    /// Note that depenendency validity can vary between instructions within
    /// a single bundle.
    bool hasValidDependencies() const { return Dependencies != InvalidDeps; }

    /// Returns true for single instructions and for bundle representatives
    /// (= the head of a bundle).
    bool isSchedulingEntity() const { return FirstInBundle == this; }

    /// Returns true if it represents an instruction bundle and not only a
    /// single instruction.
    bool isPartOfBundle() const {
      return NextInBundle != nullptr || FirstInBundle != this;
    }

    /// Returns true if it is ready for scheduling, i.e. it has no more
    /// unscheduled depending instructions/bundles.
    bool isReady() const {
      assert(isSchedulingEntity() &&
             "can't consider non-scheduling entity for ready list");
      return unscheduledDepsInBundle() == 0 && !IsScheduled;
    }

    /// Modifies the number of unscheduled dependencies for this instruction,
    /// and returns the number of remaining dependencies for the containing
    /// bundle.
    int incrementUnscheduledDeps(int Incr) {
      assert(hasValidDependencies() &&
             "increment of unscheduled deps would be meaningless");
      UnscheduledDeps += Incr;
      return FirstInBundle->unscheduledDepsInBundle();
    }

    /// Sets the number of unscheduled dependencies to the number of
    /// dependencies.
    void resetUnscheduledDeps() {
      UnscheduledDeps = Dependencies;
    }

    /// Clears all dependency information.
    void clearDependencies() {
      Dependencies = InvalidDeps;
      resetUnscheduledDeps();
      MemoryDependencies.clear();
    }

    int unscheduledDepsInBundle() const {
      assert(isSchedulingEntity() && "only meaningful on the bundle");
      int Sum = 0;
      for (const ScheduleData *BundleMember = this; BundleMember;
           BundleMember = BundleMember->NextInBundle) {
        if (BundleMember->UnscheduledDeps == InvalidDeps)
          return InvalidDeps;
        Sum += BundleMember->UnscheduledDeps;
      }
      return Sum;
    }

    void dump(raw_ostream &os) const {
      if (!isSchedulingEntity()) {
        os << "/ " << *Inst;
      } else if (NextInBundle) {
        os << '[' << *Inst;
        ScheduleData *SD = NextInBundle;
        while (SD) {
          os << ';' << *SD->Inst;
          SD = SD->NextInBundle;
        }
        os << ']';
      } else {
        os << *Inst;
      }
    }

    Instruction *Inst = nullptr;

    /// Opcode of the current instruction in the schedule data.
    Value *OpValue = nullptr;

    /// The TreeEntry that this instruction corresponds to.
    TreeEntry *TE = nullptr;

    /// Points to the head in an instruction bundle (and always to this for
    /// single instructions).
    ScheduleData *FirstInBundle = nullptr;

    /// Single linked list of all instructions in a bundle. Null if it is a
    /// single instruction.
    ScheduleData *NextInBundle = nullptr;

    /// Single linked list of all memory instructions (e.g. load, store, call)
    /// in the block - until the end of the scheduling region.
    ScheduleData *NextLoadStore = nullptr;

    /// The dependent memory instructions.
    /// This list is derived on demand in calculateDependencies().
    SmallVector<ScheduleData *, 4> MemoryDependencies;

    /// This ScheduleData is in the current scheduling region if this matches
    /// the current SchedulingRegionID of BlockScheduling.
    int SchedulingRegionID = 0;

    /// Used for getting a "good" final ordering of instructions.
    int SchedulingPriority = 0;

    /// The number of dependencies. Constitutes of the number of users of the
    /// instruction plus the number of dependent memory instructions (if any).
    /// This value is calculated on demand.
    /// If InvalidDeps, the number of dependencies is not calculated yet.
    int Dependencies = InvalidDeps;

    /// The number of dependencies minus the number of dependencies of scheduled
    /// instructions. As soon as this is zero, the instruction/bundle gets ready
    /// for scheduling.
    /// Note that this is negative as long as Dependencies is not calculated.
    int UnscheduledDeps = InvalidDeps;

    /// The lane of this node in the TreeEntry.
    int Lane = -1;

    /// True if this instruction is scheduled (or considered as scheduled in the
    /// dry-run).
    bool IsScheduled = false;
  };

#ifndef NDEBUG
  friend inline raw_ostream &operator<<(raw_ostream &os,
                                        const BoUpSLP::ScheduleData &SD) {
    SD.dump(os);
    return os;
  }
#endif

  friend struct GraphTraits<BoUpSLP *>;
  friend struct DOTGraphTraits<BoUpSLP *>;

  /// Contains all scheduling data for a basic block.
  struct BlockScheduling {
    BlockScheduling(BasicBlock *BB)
        : BB(BB), ChunkSize(BB->size()), ChunkPos(ChunkSize) {}

    void clear() {
      ReadyInsts.clear();
      ScheduleStart = nullptr;
      ScheduleEnd = nullptr;
      FirstLoadStoreInRegion = nullptr;
      LastLoadStoreInRegion = nullptr;

      // Reduce the maximum schedule region size by the size of the
      // previous scheduling run.
      ScheduleRegionSizeLimit -= ScheduleRegionSize;
      if (ScheduleRegionSizeLimit < MinScheduleRegionSize)
        ScheduleRegionSizeLimit = MinScheduleRegionSize;
      ScheduleRegionSize = 0;

      // Make a new scheduling region, i.e. all existing ScheduleData is not
      // in the new region yet.
      ++SchedulingRegionID;
    }

    ScheduleData *getScheduleData(Instruction *I) {
      if (BB != I->getParent())
        // Avoid lookup if can't possibly be in map.
        return nullptr;
      ScheduleData *SD = ScheduleDataMap[I];
      if (SD && isInSchedulingRegion(SD))
        return SD;
      return nullptr;
    }

    ScheduleData *getScheduleData(Value *V) {
      if (auto *I = dyn_cast<Instruction>(V))
        return getScheduleData(I);
      return nullptr;
    }

    ScheduleData *getScheduleData(Value *V, Value *Key) {
      if (V == Key)
        return getScheduleData(V);
      auto I = ExtraScheduleDataMap.find(V);
      if (I != ExtraScheduleDataMap.end()) {
        ScheduleData *SD = I->second[Key];
        if (SD && isInSchedulingRegion(SD))
          return SD;
      }
      return nullptr;
    }

    bool isInSchedulingRegion(ScheduleData *SD) const {
      return SD->SchedulingRegionID == SchedulingRegionID;
    }

    /// Marks an instruction as scheduled and puts all dependent ready
    /// instructions into the ready-list.
    template <typename ReadyListType>
    void schedule(ScheduleData *SD, ReadyListType &ReadyList) {
      SD->IsScheduled = true;
      LLVM_DEBUG(dbgs() << "SLP:   schedule " << *SD << "\n");

      for (ScheduleData *BundleMember = SD; BundleMember;
           BundleMember = BundleMember->NextInBundle) {
        if (BundleMember->Inst != BundleMember->OpValue)
          continue;
        
        // Handle the def-use chain dependencies.

        // Decrement the unscheduled counter and insert to ready list if ready.
        auto &&DecrUnsched = [this, &ReadyList](Instruction *I) {
          doForAllOpcodes(I, [&ReadyList](ScheduleData *OpDef) {
            if (OpDef && OpDef->hasValidDependencies() &&
                OpDef->incrementUnscheduledDeps(-1) == 0) {
              // There are no more unscheduled dependencies after
              // decrementing, so we can put the dependent instruction
              // into the ready list.
              ScheduleData *DepBundle = OpDef->FirstInBundle;
              assert(!DepBundle->IsScheduled &&
                     "already scheduled bundle gets ready");
              ReadyList.insert(DepBundle);
              LLVM_DEBUG(dbgs()
                         << "SLP:    gets ready (def): " << *DepBundle << "\n");
            }
          });
        };

        // If BundleMember is a vector bundle, its operands may have been
        // reordered during buildTree(). We therefore need to get its operands
        // through the TreeEntry.
        if (TreeEntry *TE = BundleMember->TE) {
          int Lane = BundleMember->Lane;
          assert(Lane >= 0 && "Lane not set");

          // Since vectorization tree is being built recursively this assertion
          // ensures that the tree entry has all operands set before reaching
          // this code. Couple of exceptions known at the moment are extracts
          // where their second (immediate) operand is not added. Since
          // immediates do not affect scheduler behavior this is considered
          // okay.
          auto *In = TE->getMainOp();
          assert(In &&
                 (isa<ExtractValueInst>(In) || isa<ExtractElementInst>(In) ||
                  In->getNumOperands() == TE->getNumOperands()) &&
                 "Missed TreeEntry operands?");
          (void)In; // fake use to avoid build failure when assertions disabled

          for (unsigned OpIdx = 0, NumOperands = TE->getNumOperands();
               OpIdx != NumOperands; ++OpIdx)
            if (auto *I = dyn_cast<Instruction>(TE->getOperand(OpIdx)[Lane]))
              DecrUnsched(I);
        } else {
          // If BundleMember is a stand-alone instruction, no operand reordering
          // has taken place, so we directly access its operands.
          for (Use &U : BundleMember->Inst->operands())
            if (auto *I = dyn_cast<Instruction>(U.get()))
              DecrUnsched(I);
        }
        // Handle the memory dependencies.
        for (ScheduleData *MemoryDepSD : BundleMember->MemoryDependencies) {
          if (MemoryDepSD->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after decrementing,
            // so we can put the dependent instruction into the ready list.
            ScheduleData *DepBundle = MemoryDepSD->FirstInBundle;
            assert(!DepBundle->IsScheduled &&
                   "already scheduled bundle gets ready");
            ReadyList.insert(DepBundle);
            LLVM_DEBUG(dbgs()
                       << "SLP:    gets ready (mem): " << *DepBundle << "\n");
          }
        }
      }
    }

    /// Verify basic self consistency properties of the data structure.
    void verify() {
      if (!ScheduleStart)
        return;

      assert(ScheduleStart->getParent() == ScheduleEnd->getParent() &&
             ScheduleStart->comesBefore(ScheduleEnd) &&
             "Not a valid scheduling region?");

      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
        auto *SD = getScheduleData(I);
        assert(SD && "primary scheduledata must exist in window");
        assert(isInSchedulingRegion(SD) &&
               "primary schedule data not in window?");
        assert(isInSchedulingRegion(SD->FirstInBundle) &&
               "entire bundle in window!");
        (void)SD;
        doForAllOpcodes(I, [](ScheduleData *SD) { SD->verify(); });
      }

      for (auto *SD : ReadyInsts) {
        assert(SD->isSchedulingEntity() && SD->isReady() &&
               "item in ready list not ready?");
        (void)SD;
      }
    }

    void doForAllOpcodes(Value *V,
                         function_ref<void(ScheduleData *SD)> Action) {
      if (ScheduleData *SD = getScheduleData(V))
        Action(SD);
      auto I = ExtraScheduleDataMap.find(V);
      if (I != ExtraScheduleDataMap.end())
        for (auto &P : I->second)
          if (isInSchedulingRegion(P.second))
            Action(P.second);
    }

    /// Put all instructions into the ReadyList which are ready for scheduling.
    template <typename ReadyListType>
    void initialFillReadyList(ReadyListType &ReadyList) {
      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
        doForAllOpcodes(I, [&](ScheduleData *SD) {
          if (SD->isSchedulingEntity() && SD->isReady()) {
            ReadyList.insert(SD);
            LLVM_DEBUG(dbgs()
                       << "SLP:    initially in ready list: " << *SD << "\n");
          }
        });
      }
    }

    /// Build a bundle from the ScheduleData nodes corresponding to the
    /// scalar instruction for each lane.
    ScheduleData *buildBundle(ArrayRef<Value *> VL);

    /// Checks if a bundle of instructions can be scheduled, i.e. has no
    /// cyclic dependencies. This is only a dry-run, no instructions are
    /// actually moved at this stage.
    /// \returns the scheduling bundle. The returned Optional value is non-None
    /// if \p VL is allowed to be scheduled.
    Optional<ScheduleData *>
    tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP,
                      const InstructionsState &S);

    /// Un-bundles a group of instructions.
    void cancelScheduling(ArrayRef<Value *> VL, Value *OpValue);

    /// Allocates schedule data chunk.
    ScheduleData *allocateScheduleDataChunks();

    /// Extends the scheduling region so that V is inside the region.
    /// \returns true if the region size is within the limit.
    bool extendSchedulingRegion(Value *V, const InstructionsState &S);

    /// Initialize the ScheduleData structures for new instructions in the
    /// scheduling region.
    void initScheduleData(Instruction *FromI, Instruction *ToI,
                          ScheduleData *PrevLoadStore,
                          ScheduleData *NextLoadStore);

    /// Updates the dependency information of a bundle and of all instructions/
    /// bundles which depend on the original bundle.
    void calculateDependencies(ScheduleData *SD, bool InsertInReadyList,
                               BoUpSLP *SLP);

    /// Sets all instruction in the scheduling region to un-scheduled.
    void resetSchedule();

    BasicBlock *BB;

    /// Simple memory allocation for ScheduleData.
    std::vector<std::unique_ptr<ScheduleData[]>> ScheduleDataChunks;

    /// The size of a ScheduleData array in ScheduleDataChunks.
    int ChunkSize;

    /// The allocator position in the current chunk, which is the last entry
    /// of ScheduleDataChunks.
    int ChunkPos;

    /// Attaches ScheduleData to Instruction.
    /// Note that the mapping survives during all vectorization iterations, i.e.
    /// ScheduleData structures are recycled.
    DenseMap<Instruction *, ScheduleData *> ScheduleDataMap;

    /// Attaches ScheduleData to Instruction with the leading key.
    DenseMap<Value *, SmallDenseMap<Value *, ScheduleData *>>
        ExtraScheduleDataMap;

    /// The ready-list for scheduling (only used for the dry-run).
    SetVector<ScheduleData *> ReadyInsts;

    /// The first instruction of the scheduling region.
    Instruction *ScheduleStart = nullptr;

    /// The first instruction _after_ the scheduling region.
    Instruction *ScheduleEnd = nullptr;

    /// The first memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *FirstLoadStoreInRegion = nullptr;

    /// The last memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *LastLoadStoreInRegion = nullptr;

    /// The current size of the scheduling region.
    int ScheduleRegionSize = 0;

    /// The maximum size allowed for the scheduling region.
    int ScheduleRegionSizeLimit = ScheduleRegionSizeBudget;

    /// The ID of the scheduling region. For a new vectorization iteration this
    /// is incremented which "removes" all ScheduleData from the region.
    /// Make sure that the initial SchedulingRegionID is greater than the
    /// initial SchedulingRegionID in ScheduleData (which is 0).
    int SchedulingRegionID = 1;
  };

  /// Attaches the BlockScheduling structures to basic blocks.
  MapVector<BasicBlock *, std::unique_ptr<BlockScheduling>> BlocksSchedules;

  /// Performs the "real" scheduling. Done before vectorization is actually
  /// performed in a basic block.
  void scheduleBlock(BlockScheduling *BS);

  /// List of users to ignore during scheduling and that don't need extracting.
  ArrayRef<Value *> UserIgnoreList;

  /// A DenseMapInfo implementation for holding DenseMaps and DenseSets of
  /// sorted SmallVectors of unsigned.
  struct OrdersTypeDenseMapInfo {
    static OrdersType getEmptyKey() {
      OrdersType V;
      V.push_back(~1U);
      return V;
    }

    static OrdersType getTombstoneKey() {
      OrdersType V;
      V.push_back(~2U);
      return V;
    }

    static unsigned getHashValue(const OrdersType &V) {
      return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
    }

    static bool isEqual(const OrdersType &LHS, const OrdersType &RHS) {
      return LHS == RHS;
    }
  };

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  LoopInfo *LI;
  DominatorTree *DT;
  AssumptionCache *AC;
  DemandedBits *DB;
  const DataLayout *DL;
  OptimizationRemarkEmitter *ORE;

  unsigned MaxVecRegSize; // This is set by TTI or overridden by cl::opt.
  unsigned MinVecRegSize; // Set by cl::opt (default: 128).

  /// Instruction builder to construct the vectorized tree.
  IRBuilder<> Builder;

  /// A map of scalar integer values to the smallest bit width with which they
  /// can legally be represented. The values map to (width, signed) pairs,
  /// where "width" indicates the minimum bit width and "signed" is True if the
  /// value must be signed-extended, rather than zero-extended, back to its
  /// original width.
  MapVector<Value *, std::pair<uint64_t, bool>> MinBWs;
};

} // end namespace slpvectorizer

template <> struct GraphTraits<BoUpSLP *> {
  using TreeEntry = BoUpSLP::TreeEntry;

  /// NodeRef has to be a pointer per the GraphWriter.
  using NodeRef = TreeEntry *;

  using ContainerTy = BoUpSLP::TreeEntry::VecTreeTy;

  /// Add the VectorizableTree to the index iterator to be able to return
  /// TreeEntry pointers.
  struct ChildIteratorType
      : public iterator_adaptor_base<
            ChildIteratorType, SmallVector<BoUpSLP::EdgeInfo, 1>::iterator> {
    ContainerTy &VectorizableTree;

    ChildIteratorType(SmallVector<BoUpSLP::EdgeInfo, 1>::iterator W,
                      ContainerTy &VT)
        : ChildIteratorType::iterator_adaptor_base(W), VectorizableTree(VT) {}

    NodeRef operator*() { return I->UserTE; }
  };

  static NodeRef getEntryNode(BoUpSLP &R) {
    return R.VectorizableTree[0].get();
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return {N->UserTreeIndices.begin(), N->Container};
  }

  static ChildIteratorType child_end(NodeRef N) {
    return {N->UserTreeIndices.end(), N->Container};
  }

  /// For the node iterator we just need to turn the TreeEntry iterator into a
  /// TreeEntry* iterator so that it dereferences to NodeRef.
  class nodes_iterator {
    using ItTy = ContainerTy::iterator;
    ItTy It;

  public:
    nodes_iterator(const ItTy &It2) : It(It2) {}
    NodeRef operator*() { return It->get(); }
    nodes_iterator operator++() {
      ++It;
      return *this;
    }
    bool operator!=(const nodes_iterator &N2) const { return N2.It != It; }
  };

  static nodes_iterator nodes_begin(BoUpSLP *R) {
    return nodes_iterator(R->VectorizableTree.begin());
  }

  static nodes_iterator nodes_end(BoUpSLP *R) {
    return nodes_iterator(R->VectorizableTree.end());
  }

  static unsigned size(BoUpSLP *R) { return R->VectorizableTree.size(); }
};

template <> struct DOTGraphTraits<BoUpSLP *> : public DefaultDOTGraphTraits {
  using TreeEntry = BoUpSLP::TreeEntry;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(const TreeEntry *Entry, const BoUpSLP *R) {
    std::string Str;
    raw_string_ostream OS(Str);
    if (isSplat(Entry->Scalars))
      OS << "<splat> ";
    for (auto V : Entry->Scalars) {
      OS << *V;
      if (llvm::any_of(R->ExternalUses, [&](const BoUpSLP::ExternalUser &EU) {
            return EU.Scalar == V;
          }))
        OS << " <extract>";
      OS << "\n";
    }
    return Str;
  }

  static std::string getNodeAttributes(const TreeEntry *Entry,
                                       const BoUpSLP *) {
    if (Entry->State == TreeEntry::NeedToGather)
      return "color=red";
    return "";
  }
};

} // end namespace llvm

BoUpSLP::~BoUpSLP() {
  for (const auto &Pair : DeletedInstructions) {
    // Replace operands of ignored instructions with Undefs in case if they were
    // marked for deletion.
    if (Pair.getSecond()) {
      Value *Undef = UndefValue::get(Pair.getFirst()->getType());
      Pair.getFirst()->replaceAllUsesWith(Undef);
    }
    Pair.getFirst()->dropAllReferences();
  }
  for (const auto &Pair : DeletedInstructions) {
    assert(Pair.getFirst()->use_empty() &&
           "trying to erase instruction with users.");
    Pair.getFirst()->eraseFromParent();
  }
#ifdef EXPENSIVE_CHECKS
  // If we could guarantee that this call is not extremely slow, we could
  // remove the ifdef limitation (see PR47712).
  assert(!verifyFunction(*F, &dbgs()));
#endif
}

void BoUpSLP::eraseInstructions(ArrayRef<Value *> AV) {
  for (auto *V : AV) {
    if (auto *I = dyn_cast<Instruction>(V))
      eraseInstruction(I, /*ReplaceOpsWithUndef=*/true);
  };
}

/// Reorders the given \p Reuses mask according to the given \p Mask. \p Reuses
/// contains original mask for the scalars reused in the node. Procedure
/// transform this mask in accordance with the given \p Mask.
static void reorderReuses(SmallVectorImpl<int> &Reuses, ArrayRef<int> Mask) {
  assert(!Mask.empty() && Reuses.size() == Mask.size() &&
         "Expected non-empty mask.");
  SmallVector<int> Prev(Reuses.begin(), Reuses.end());
  Prev.swap(Reuses);
  for (unsigned I = 0, E = Prev.size(); I < E; ++I)
    if (Mask[I] != UndefMaskElem)
      Reuses[Mask[I]] = Prev[I];
}

/// Reorders the given \p Order according to the given \p Mask. \p Order - is
/// the original order of the scalars. Procedure transforms the provided order
/// in accordance with the given \p Mask. If the resulting \p Order is just an
/// identity order, \p Order is cleared.
static void reorderOrder(SmallVectorImpl<unsigned> &Order, ArrayRef<int> Mask) {
  assert(!Mask.empty() && "Expected non-empty mask.");
  SmallVector<int> MaskOrder;
  if (Order.empty()) {
    MaskOrder.resize(Mask.size());
    std::iota(MaskOrder.begin(), MaskOrder.end(), 0);
  } else {
    inversePermutation(Order, MaskOrder);
  }
  reorderReuses(MaskOrder, Mask);
  if (ShuffleVectorInst::isIdentityMask(MaskOrder)) {
    Order.clear();
    return;
  }
  Order.assign(Mask.size(), Mask.size());
  for (unsigned I = 0, E = Mask.size(); I < E; ++I)
    if (MaskOrder[I] != UndefMaskElem)
      Order[MaskOrder[I]] = I;
  fixupOrderingIndices(Order);
}

Optional<BoUpSLP::OrdersType>
BoUpSLP::findReusedOrderedScalars(const BoUpSLP::TreeEntry &TE) {
  assert(TE.State == TreeEntry::NeedToGather && "Expected gather node only.");
  unsigned NumScalars = TE.Scalars.size();
  OrdersType CurrentOrder(NumScalars, NumScalars);
  SmallVector<int> Positions;
  SmallBitVector UsedPositions(NumScalars);
  const TreeEntry *STE = nullptr;
  // Try to find all gathered scalars that are gets vectorized in other
  // vectorize node. Here we can have only one single tree vector node to
  // correctly identify order of the gathered scalars.
  for (unsigned I = 0; I < NumScalars; ++I) {
    Value *V = TE.Scalars[I];
    if (!isa<LoadInst, ExtractElementInst, ExtractValueInst>(V))
      continue;
    if (const auto *LocalSTE = getTreeEntry(V)) {
      if (!STE)
        STE = LocalSTE;
      else if (STE != LocalSTE)
        // Take the order only from the single vector node.
        return None;
      unsigned Lane =
          std::distance(STE->Scalars.begin(), find(STE->Scalars, V));
      if (Lane >= NumScalars)
        return None;
      if (CurrentOrder[Lane] != NumScalars) {
        if (Lane != I)
          continue;
        UsedPositions.reset(CurrentOrder[Lane]);
      }
      // The partial identity (where only some elements of the gather node are
      // in the identity order) is good.
      CurrentOrder[Lane] = I;
      UsedPositions.set(I);
    }
  }
  // Need to keep the order if we have a vector entry and at least 2 scalars or
  // the vectorized entry has just 2 scalars.
  if (STE && (UsedPositions.count() > 1 || STE->Scalars.size() == 2)) {
    auto &&IsIdentityOrder = [NumScalars](ArrayRef<unsigned> CurrentOrder) {
      for (unsigned I = 0; I < NumScalars; ++I)
        if (CurrentOrder[I] != I && CurrentOrder[I] != NumScalars)
          return false;
      return true;
    };
    if (IsIdentityOrder(CurrentOrder)) {
      CurrentOrder.clear();
      return CurrentOrder;
    }
    auto *It = CurrentOrder.begin();
    for (unsigned I = 0; I < NumScalars;) {
      if (UsedPositions.test(I)) {
        ++I;
        continue;
      }
      if (*It == NumScalars) {
        *It = I;
        ++I;
      }
      ++It;
    }
    return CurrentOrder;
  }
  return None;
}

Optional<BoUpSLP::OrdersType> BoUpSLP::getReorderingData(const TreeEntry &TE,
                                                         bool TopToBottom) {
  // No need to reorder if need to shuffle reuses, still need to shuffle the
  // node.
  if (!TE.ReuseShuffleIndices.empty())
    return None;
  if (TE.State == TreeEntry::Vectorize &&
      (isa<LoadInst, ExtractElementInst, ExtractValueInst>(TE.getMainOp()) ||
       (TopToBottom && isa<StoreInst, InsertElementInst>(TE.getMainOp()))) &&
      !TE.isAltShuffle())
    return TE.ReorderIndices;
  if (TE.State == TreeEntry::NeedToGather) {
    // TODO: add analysis of other gather nodes with extractelement
    // instructions and other values/instructions, not only undefs.
    if (((TE.getOpcode() == Instruction::ExtractElement &&
          !TE.isAltShuffle()) ||
         (all_of(TE.Scalars,
                 [](Value *V) {
                   return isa<UndefValue, ExtractElementInst>(V);
                 }) &&
          any_of(TE.Scalars,
                 [](Value *V) { return isa<ExtractElementInst>(V); }))) &&
        all_of(TE.Scalars,
               [](Value *V) {
                 auto *EE = dyn_cast<ExtractElementInst>(V);
                 return !EE || isa<FixedVectorType>(EE->getVectorOperandType());
               }) &&
        allSameType(TE.Scalars)) {
      // Check that gather of extractelements can be represented as
      // just a shuffle of a single vector.
      OrdersType CurrentOrder;
      bool Reuse = canReuseExtract(TE.Scalars, TE.getMainOp(), CurrentOrder);
      if (Reuse || !CurrentOrder.empty()) {
        if (!CurrentOrder.empty())
          fixupOrderingIndices(CurrentOrder);
        return CurrentOrder;
      }
    }
    if (Optional<OrdersType> CurrentOrder = findReusedOrderedScalars(TE))
      return CurrentOrder;
  }
  return None;
}

void BoUpSLP::reorderTopToBottom() {
  // Maps VF to the graph nodes.
  DenseMap<unsigned, SetVector<TreeEntry *>> VFToOrderedEntries;
  // ExtractElement gather nodes which can be vectorized and need to handle
  // their ordering.
  DenseMap<const TreeEntry *, OrdersType> GathersToOrders;
  // Find all reorderable nodes with the given VF.
  // Currently the are vectorized stores,loads,extracts + some gathering of
  // extracts.
  for_each(VectorizableTree, [this, &VFToOrderedEntries, &GathersToOrders](
                                 const std::unique_ptr<TreeEntry> &TE) {
    if (Optional<OrdersType> CurrentOrder =
            getReorderingData(*TE.get(), /*TopToBottom=*/true)) {
      // Do not include ordering for nodes used in the alt opcode vectorization,
      // better to reorder them during bottom-to-top stage. If follow the order
      // here, it causes reordering of the whole graph though actually it is
      // profitable just to reorder the subgraph that starts from the alternate
      // opcode vectorization node. Such nodes already end-up with the shuffle
      // instruction and it is just enough to change this shuffle rather than
      // rotate the scalars for the whole graph.
      unsigned Cnt = 0;
      const TreeEntry *UserTE = TE.get();
      while (UserTE && Cnt < RecursionMaxDepth) {
        if (UserTE->UserTreeIndices.size() != 1)
          break;
        if (all_of(UserTE->UserTreeIndices, [](const EdgeInfo &EI) {
              return EI.UserTE->State == TreeEntry::Vectorize &&
                     EI.UserTE->isAltShuffle() && EI.UserTE->Idx != 0;
            }))
          return;
        if (UserTE->UserTreeIndices.empty())
          UserTE = nullptr;
        else
          UserTE = UserTE->UserTreeIndices.back().UserTE;
        ++Cnt;
      }
      VFToOrderedEntries[TE->Scalars.size()].insert(TE.get());
      if (TE->State != TreeEntry::Vectorize)
        GathersToOrders.try_emplace(TE.get(), *CurrentOrder);
    }
  });

  // Reorder the graph nodes according to their vectorization factor.
  for (unsigned VF = VectorizableTree.front()->Scalars.size(); VF > 1;
       VF /= 2) {
    auto It = VFToOrderedEntries.find(VF);
    if (It == VFToOrderedEntries.end())
      continue;
    // Try to find the most profitable order. We just are looking for the most
    // used order and reorder scalar elements in the nodes according to this
    // mostly used order.
    ArrayRef<TreeEntry *> OrderedEntries = It->second.getArrayRef();
    // All operands are reordered and used only in this node - propagate the
    // most used order to the user node.
    MapVector<OrdersType, unsigned,
              DenseMap<OrdersType, unsigned, OrdersTypeDenseMapInfo>>
        OrdersUses;
    SmallPtrSet<const TreeEntry *, 4> VisitedOps;
    for (const TreeEntry *OpTE : OrderedEntries) {
      // No need to reorder this nodes, still need to extend and to use shuffle,
      // just need to merge reordering shuffle and the reuse shuffle.
      if (!OpTE->ReuseShuffleIndices.empty())
        continue;
      // Count number of orders uses.
      const auto &Order = [OpTE, &GathersToOrders]() -> const OrdersType & {
        if (OpTE->State == TreeEntry::NeedToGather)
          return GathersToOrders.find(OpTE)->second;
        return OpTE->ReorderIndices;
      }();
      // Stores actually store the mask, not the order, need to invert.
      if (OpTE->State == TreeEntry::Vectorize && !OpTE->isAltShuffle() &&
          OpTE->getOpcode() == Instruction::Store && !Order.empty()) {
        SmallVector<int> Mask;
        inversePermutation(Order, Mask);
        unsigned E = Order.size();
        OrdersType CurrentOrder(E, E);
        transform(Mask, CurrentOrder.begin(), [E](int Idx) {
          return Idx == UndefMaskElem ? E : static_cast<unsigned>(Idx);
        });
        fixupOrderingIndices(CurrentOrder);
        ++OrdersUses.insert(std::make_pair(CurrentOrder, 0)).first->second;
      } else {
        ++OrdersUses.insert(std::make_pair(Order, 0)).first->second;
      }
    }
    // Set order of the user node.
    if (OrdersUses.empty())
      continue;
    // Choose the most used order.
    ArrayRef<unsigned> BestOrder = OrdersUses.front().first;
    unsigned Cnt = OrdersUses.front().second;
    for (const auto &Pair : drop_begin(OrdersUses)) {
      if (Cnt < Pair.second || (Cnt == Pair.second && Pair.first.empty())) {
        BestOrder = Pair.first;
        Cnt = Pair.second;
      }
    }
    // Set order of the user node.
    if (BestOrder.empty())
      continue;
    SmallVector<int> Mask;
    inversePermutation(BestOrder, Mask);
    SmallVector<int> MaskOrder(BestOrder.size(), UndefMaskElem);
    unsigned E = BestOrder.size();
    transform(BestOrder, MaskOrder.begin(), [E](unsigned I) {
      return I < E ? static_cast<int>(I) : UndefMaskElem;
    });
    // Do an actual reordering, if profitable.
    for (std::unique_ptr<TreeEntry> &TE : VectorizableTree) {
      // Just do the reordering for the nodes with the given VF.
      if (TE->Scalars.size() != VF) {
        if (TE->ReuseShuffleIndices.size() == VF) {
          // Need to reorder the reuses masks of the operands with smaller VF to
          // be able to find the match between the graph nodes and scalar
          // operands of the given node during vectorization/cost estimation.
          assert(all_of(TE->UserTreeIndices,
                        [VF, &TE](const EdgeInfo &EI) {
                          return EI.UserTE->Scalars.size() == VF ||
                                 EI.UserTE->Scalars.size() ==
                                     TE->Scalars.size();
                        }) &&
                 "All users must be of VF size.");
          // Update ordering of the operands with the smaller VF than the given
          // one.
          reorderReuses(TE->ReuseShuffleIndices, Mask);
        }
        continue;
      }
      if (TE->State == TreeEntry::Vectorize &&
          isa<ExtractElementInst, ExtractValueInst, LoadInst, StoreInst,
              InsertElementInst>(TE->getMainOp()) &&
          !TE->isAltShuffle()) {
        // Build correct orders for extract{element,value}, loads and
        // stores.
        reorderOrder(TE->ReorderIndices, Mask);
        if (isa<InsertElementInst, StoreInst>(TE->getMainOp()))
          TE->reorderOperands(Mask);
      } else {
        // Reorder the node and its operands.
        TE->reorderOperands(Mask);
        assert(TE->ReorderIndices.empty() &&
               "Expected empty reorder sequence.");
        reorderScalars(TE->Scalars, Mask);
      }
      if (!TE->ReuseShuffleIndices.empty()) {
        // Apply reversed order to keep the original ordering of the reused
        // elements to avoid extra reorder indices shuffling.
        OrdersType CurrentOrder;
        reorderOrder(CurrentOrder, MaskOrder);
        SmallVector<int> NewReuses;
        inversePermutation(CurrentOrder, NewReuses);
        addMask(NewReuses, TE->ReuseShuffleIndices);
        TE->ReuseShuffleIndices.swap(NewReuses);
      }
    }
  }
}

bool BoUpSLP::canReorderOperands(
    TreeEntry *UserTE, SmallVectorImpl<std::pair<unsigned, TreeEntry *>> &Edges,
    ArrayRef<TreeEntry *> ReorderableGathers,
    SmallVectorImpl<TreeEntry *> &GatherOps) {
  for (unsigned I = 0, E = UserTE->getNumOperands(); I < E; ++I) {
    if (any_of(Edges, [I](const std::pair<unsigned, TreeEntry *> &OpData) {
          return OpData.first == I &&
                 OpData.second->State == TreeEntry::Vectorize;
        }))
      continue;
    if (TreeEntry *TE = getVectorizedOperand(UserTE, I)) {
      // Do not reorder if operand node is used by many user nodes.
      if (any_of(TE->UserTreeIndices,
                 [UserTE](const EdgeInfo &EI) { return EI.UserTE != UserTE; }))
        return false;
      // Add the node to the list of the ordered nodes with the identity
      // order.
      Edges.emplace_back(I, TE);
      continue;
    }
    ArrayRef<Value *> VL = UserTE->getOperand(I);
    TreeEntry *Gather = nullptr;
    if (count_if(ReorderableGathers, [VL, &Gather](TreeEntry *TE) {
          assert(TE->State != TreeEntry::Vectorize &&
                 "Only non-vectorized nodes are expected.");
          if (TE->isSame(VL)) {
            Gather = TE;
            return true;
          }
          return false;
        }) > 1)
      return false;
    if (Gather)
      GatherOps.push_back(Gather);
  }
  return true;
}

void BoUpSLP::reorderBottomToTop(bool IgnoreReorder) {
  SetVector<TreeEntry *> OrderedEntries;
  DenseMap<const TreeEntry *, OrdersType> GathersToOrders;
  // Find all reorderable leaf nodes with the given VF.
  // Currently the are vectorized loads,extracts without alternate operands +
  // some gathering of extracts.
  SmallVector<TreeEntry *> NonVectorized;
  for_each(VectorizableTree, [this, &OrderedEntries, &GathersToOrders,
                              &NonVectorized](
                                 const std::unique_ptr<TreeEntry> &TE) {
    if (TE->State != TreeEntry::Vectorize)
      NonVectorized.push_back(TE.get());
    if (Optional<OrdersType> CurrentOrder =
            getReorderingData(*TE.get(), /*TopToBottom=*/false)) {
      OrderedEntries.insert(TE.get());
      if (TE->State != TreeEntry::Vectorize)
        GathersToOrders.try_emplace(TE.get(), *CurrentOrder);
    }
  });

  // 1. Propagate order to the graph nodes, which use only reordered nodes.
  // I.e., if the node has operands, that are reordered, try to make at least
  // one operand order in the natural order and reorder others + reorder the
  // user node itself.
  SmallPtrSet<const TreeEntry *, 4> Visited;
  while (!OrderedEntries.empty()) {
    // 1. Filter out only reordered nodes.
    // 2. If the entry has multiple uses - skip it and jump to the next node.
    MapVector<TreeEntry *, SmallVector<std::pair<unsigned, TreeEntry *>>> Users;
    SmallVector<TreeEntry *> Filtered;
    for (TreeEntry *TE : OrderedEntries) {
      if (!(TE->State == TreeEntry::Vectorize ||
            (TE->State == TreeEntry::NeedToGather &&
             GathersToOrders.count(TE))) ||
          TE->UserTreeIndices.empty() || !TE->ReuseShuffleIndices.empty() ||
          !all_of(drop_begin(TE->UserTreeIndices),
                  [TE](const EdgeInfo &EI) {
                    return EI.UserTE == TE->UserTreeIndices.front().UserTE;
                  }) ||
          !Visited.insert(TE).second) {
        Filtered.push_back(TE);
        continue;
      }
      // Build a map between user nodes and their operands order to speedup
      // search. The graph currently does not provide this dependency directly.
      for (EdgeInfo &EI : TE->UserTreeIndices) {
        TreeEntry *UserTE = EI.UserTE;
        auto It = Users.find(UserTE);
        if (It == Users.end())
          It = Users.insert({UserTE, {}}).first;
        It->second.emplace_back(EI.EdgeIdx, TE);
      }
    }
    // Erase filtered entries.
    for_each(Filtered,
             [&OrderedEntries](TreeEntry *TE) { OrderedEntries.remove(TE); });
    for (auto &Data : Users) {
      // Check that operands are used only in the User node.
      SmallVector<TreeEntry *> GatherOps;
      if (!canReorderOperands(Data.first, Data.second, NonVectorized,
                              GatherOps)) {
        for_each(Data.second,
                 [&OrderedEntries](const std::pair<unsigned, TreeEntry *> &Op) {
                   OrderedEntries.remove(Op.second);
                 });
        continue;
      }
      // All operands are reordered and used only in this node - propagate the
      // most used order to the user node.
      MapVector<OrdersType, unsigned,
                DenseMap<OrdersType, unsigned, OrdersTypeDenseMapInfo>>
          OrdersUses;
      // Do the analysis for each tree entry only once, otherwise the order of
      // the same node my be considered several times, though might be not
      // profitable.
      SmallPtrSet<const TreeEntry *, 4> VisitedOps;
      SmallPtrSet<const TreeEntry *, 4> VisitedUsers;
      for (const auto &Op : Data.second) {
        TreeEntry *OpTE = Op.second;
        if (!VisitedOps.insert(OpTE).second)
          continue;
        if (!OpTE->ReuseShuffleIndices.empty() ||
            (IgnoreReorder && OpTE == VectorizableTree.front().get()))
          continue;
        const auto &Order = [OpTE, &GathersToOrders]() -> const OrdersType & {
          if (OpTE->State == TreeEntry::NeedToGather)
            return GathersToOrders.find(OpTE)->second;
          return OpTE->ReorderIndices;
        }();
        unsigned NumOps = count_if(
            Data.second, [OpTE](const std::pair<unsigned, TreeEntry *> &P) {
              return P.second == OpTE;
            });
        // Stores actually store the mask, not the order, need to invert.
        if (OpTE->State == TreeEntry::Vectorize && !OpTE->isAltShuffle() &&
            OpTE->getOpcode() == Instruction::Store && !Order.empty()) {
          SmallVector<int> Mask;
          inversePermutation(Order, Mask);
          unsigned E = Order.size();
          OrdersType CurrentOrder(E, E);
          transform(Mask, CurrentOrder.begin(), [E](int Idx) {
            return Idx == UndefMaskElem ? E : static_cast<unsigned>(Idx);
          });
          fixupOrderingIndices(CurrentOrder);
          OrdersUses.insert(std::make_pair(CurrentOrder, 0)).first->second +=
              NumOps;
        } else {
          OrdersUses.insert(std::make_pair(Order, 0)).first->second += NumOps;
        }
        auto Res = OrdersUses.insert(std::make_pair(OrdersType(), 0));
        const auto &&AllowsReordering = [IgnoreReorder, &GathersToOrders](
                                            const TreeEntry *TE) {
          if (!TE->ReorderIndices.empty() || !TE->ReuseShuffleIndices.empty() ||
              (TE->State == TreeEntry::Vectorize && TE->isAltShuffle()) ||
              (IgnoreReorder && TE->Idx == 0))
            return true;
          if (TE->State == TreeEntry::NeedToGather) {
            auto It = GathersToOrders.find(TE);
            if (It != GathersToOrders.end())
              return !It->second.empty();
            return true;
          }
          return false;
        };
        for (const EdgeInfo &EI : OpTE->UserTreeIndices) {
          TreeEntry *UserTE = EI.UserTE;
          if (!VisitedUsers.insert(UserTE).second)
            continue;
          // May reorder user node if it requires reordering, has reused
          // scalars, is an alternate op vectorize node or its op nodes require
          // reordering.
          if (AllowsReordering(UserTE))
            continue;
          // Check if users allow reordering.
          // Currently look up just 1 level of operands to avoid increase of
          // the compile time.
          // Profitable to reorder if definitely more operands allow
          // reordering rather than those with natural order.
          ArrayRef<std::pair<unsigned, TreeEntry *>> Ops = Users[UserTE];
          if (static_cast<unsigned>(count_if(
                  Ops, [UserTE, &AllowsReordering](
                           const std::pair<unsigned, TreeEntry *> &Op) {
                    return AllowsReordering(Op.second) &&
                           all_of(Op.second->UserTreeIndices,
                                  [UserTE](const EdgeInfo &EI) {
                                    return EI.UserTE == UserTE;
                                  });
                  })) <= Ops.size() / 2)
            ++Res.first->second;
        }
      }
      // If no orders - skip current nodes and jump to the next one, if any.
      if (OrdersUses.empty()) {
        for_each(Data.second,
                 [&OrderedEntries](const std::pair<unsigned, TreeEntry *> &Op) {
                   OrderedEntries.remove(Op.second);
                 });
        continue;
      }
      // Choose the best order.
      ArrayRef<unsigned> BestOrder = OrdersUses.front().first;
      unsigned Cnt = OrdersUses.front().second;
      for (const auto &Pair : drop_begin(OrdersUses)) {
        if (Cnt < Pair.second || (Cnt == Pair.second && Pair.first.empty())) {
          BestOrder = Pair.first;
          Cnt = Pair.second;
        }
      }
      // Set order of the user node (reordering of operands and user nodes).
      if (BestOrder.empty()) {
        for_each(Data.second,
                 [&OrderedEntries](const std::pair<unsigned, TreeEntry *> &Op) {
                   OrderedEntries.remove(Op.second);
                 });
        continue;
      }
      // Erase operands from OrderedEntries list and adjust their orders.
      VisitedOps.clear();
      SmallVector<int> Mask;
      inversePermutation(BestOrder, Mask);
      SmallVector<int> MaskOrder(BestOrder.size(), UndefMaskElem);
      unsigned E = BestOrder.size();
      transform(BestOrder, MaskOrder.begin(), [E](unsigned I) {
        return I < E ? static_cast<int>(I) : UndefMaskElem;
      });
      for (const std::pair<unsigned, TreeEntry *> &Op : Data.second) {
        TreeEntry *TE = Op.second;
        OrderedEntries.remove(TE);
        if (!VisitedOps.insert(TE).second)
          continue;
        if (TE->ReuseShuffleIndices.size() == BestOrder.size()) {
          // Just reorder reuses indices.
          reorderReuses(TE->ReuseShuffleIndices, Mask);
          continue;
        }
        // Gathers are processed separately.
        if (TE->State != TreeEntry::Vectorize)
          continue;
        assert((BestOrder.size() == TE->ReorderIndices.size() ||
                TE->ReorderIndices.empty()) &&
               "Non-matching sizes of user/operand entries.");
        reorderOrder(TE->ReorderIndices, Mask);
      }
      // For gathers just need to reorder its scalars.
      for (TreeEntry *Gather : GatherOps) {
        assert(Gather->ReorderIndices.empty() &&
               "Unexpected reordering of gathers.");
        if (!Gather->ReuseShuffleIndices.empty()) {
          // Just reorder reuses indices.
          reorderReuses(Gather->ReuseShuffleIndices, Mask);
          continue;
        }
        reorderScalars(Gather->Scalars, Mask);
        OrderedEntries.remove(Gather);
      }
      // Reorder operands of the user node and set the ordering for the user
      // node itself.
      if (Data.first->State != TreeEntry::Vectorize ||
          !isa<ExtractElementInst, ExtractValueInst, LoadInst>(
              Data.first->getMainOp()) ||
          Data.first->isAltShuffle())
        Data.first->reorderOperands(Mask);
      if (!isa<InsertElementInst, StoreInst>(Data.first->getMainOp()) ||
          Data.first->isAltShuffle()) {
        reorderScalars(Data.first->Scalars, Mask);
        reorderOrder(Data.first->ReorderIndices, MaskOrder);
        if (Data.first->ReuseShuffleIndices.empty() &&
            !Data.first->ReorderIndices.empty() &&
            !Data.first->isAltShuffle()) {
          // Insert user node to the list to try to sink reordering deeper in
          // the graph.
          OrderedEntries.insert(Data.first);
        }
      } else {
        reorderOrder(Data.first->ReorderIndices, Mask);
      }
    }
  }
  // If the reordering is unnecessary, just remove the reorder.
  if (IgnoreReorder && !VectorizableTree.front()->ReorderIndices.empty() &&
      VectorizableTree.front()->ReuseShuffleIndices.empty())
    VectorizableTree.front()->ReorderIndices.clear();
}

void BoUpSLP::buildExternalUses(
    const ExtraValueToDebugLocsMap &ExternallyUsedValues) {
  // Collect the values that we need to extract from the tree.
  for (auto &TEPtr : VectorizableTree) {
    TreeEntry *Entry = TEPtr.get();

    // No need to handle users of gathered values.
    if (Entry->State == TreeEntry::NeedToGather)
      continue;

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];
      int FoundLane = Entry->findLaneForValue(Scalar);

      // Check if the scalar is externally used as an extra arg.
      auto ExtI = ExternallyUsedValues.find(Scalar);
      if (ExtI != ExternallyUsedValues.end()) {
        LLVM_DEBUG(dbgs() << "SLP: Need to extract: Extra arg from lane "
                          << Lane << " from " << *Scalar << ".\n");
        ExternalUses.emplace_back(Scalar, nullptr, FoundLane);
      }
      for (User *U : Scalar->users()) {
        LLVM_DEBUG(dbgs() << "SLP: Checking user:" << *U << ".\n");

        Instruction *UserInst = dyn_cast<Instruction>(U);
        if (!UserInst)
          continue;

        if (isDeleted(UserInst))
          continue;

        // Skip in-tree scalars that become vectors
        if (TreeEntry *UseEntry = getTreeEntry(U)) {
          Value *UseScalar = UseEntry->Scalars[0];
          // Some in-tree scalars will remain as scalar in vectorized
          // instructions. If that is the case, the one in Lane 0 will
          // be used.
          if (UseScalar != U ||
              UseEntry->State == TreeEntry::ScatterVectorize ||
              !InTreeUserNeedToExtract(Scalar, UserInst, TLI)) {
            LLVM_DEBUG(dbgs() << "SLP: \tInternal user will be removed:" << *U
                              << ".\n");
            assert(UseEntry->State != TreeEntry::NeedToGather && "Bad state");
            continue;
          }
        }

        // Ignore users in the user ignore list.
        if (is_contained(UserIgnoreList, UserInst))
          continue;

        LLVM_DEBUG(dbgs() << "SLP: Need to extract:" << *U << " from lane "
                          << Lane << " from " << *Scalar << ".\n");
        ExternalUses.push_back(ExternalUser(Scalar, U, FoundLane));
      }
    }
  }
}

void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        ArrayRef<Value *> UserIgnoreLst) {
  deleteTree();
  UserIgnoreList = UserIgnoreLst;
  if (!allSameType(Roots))
    return;
  buildTree_rec(Roots, 0, EdgeInfo());
}

namespace {
/// Tracks the state we can represent the loads in the given sequence.
enum class LoadsState { Gather, Vectorize, ScatterVectorize };
} // anonymous namespace

/// Checks if the given array of loads can be represented as a vectorized,
/// scatter or just simple gather.
static LoadsState canVectorizeLoads(ArrayRef<Value *> VL, const Value *VL0,
                                    const TargetTransformInfo &TTI,
                                    const DataLayout &DL, ScalarEvolution &SE,
                                    SmallVectorImpl<unsigned> &Order,
                                    SmallVectorImpl<Value *> &PointerOps) {
  // Check that a vectorized load would load the same memory as a scalar
  // load. For example, we don't want to vectorize loads that are smaller
  // than 8-bit. Even though we have a packed struct {<i2, i2, i2, i2>} LLVM
  // treats loading/storing it as an i8 struct. If we vectorize loads/stores
  // from such a struct, we read/write packed bits disagreeing with the
  // unvectorized version.
  Type *ScalarTy = VL0->getType();

  if (DL.getTypeSizeInBits(ScalarTy) != DL.getTypeAllocSizeInBits(ScalarTy))
    return LoadsState::Gather;

  // Make sure all loads in the bundle are simple - we can't vectorize
  // atomic or volatile loads.
  PointerOps.clear();
  PointerOps.resize(VL.size());
  auto *POIter = PointerOps.begin();
  for (Value *V : VL) {
    auto *L = cast<LoadInst>(V);
    if (!L->isSimple())
      return LoadsState::Gather;
    *POIter = L->getPointerOperand();
    ++POIter;
  }

  Order.clear();
  // Check the order of pointer operands.
  if (llvm::sortPtrAccesses(PointerOps, ScalarTy, DL, SE, Order)) {
    Value *Ptr0;
    Value *PtrN;
    if (Order.empty()) {
      Ptr0 = PointerOps.front();
      PtrN = PointerOps.back();
    } else {
      Ptr0 = PointerOps[Order.front()];
      PtrN = PointerOps[Order.back()];
    }
    Optional<int> Diff =
        getPointersDiff(ScalarTy, Ptr0, ScalarTy, PtrN, DL, SE);
    // Check that the sorted loads are consecutive.
    if (static_cast<unsigned>(*Diff) == VL.size() - 1)
      return LoadsState::Vectorize;
    Align CommonAlignment = cast<LoadInst>(VL0)->getAlign();
    for (Value *V : VL)
      CommonAlignment =
          commonAlignment(CommonAlignment, cast<LoadInst>(V)->getAlign());
    if (TTI.isLegalMaskedGather(FixedVectorType::get(ScalarTy, VL.size()),
                                CommonAlignment))
      return LoadsState::ScatterVectorize;
  }

  return LoadsState::Gather;
}

void BoUpSLP::buildTree_rec(ArrayRef<Value *> VL, unsigned Depth,
                            const EdgeInfo &UserTreeIdx) {
  assert((allConstant(VL) || allSameType(VL)) && "Invalid types!");

  SmallVector<int> ReuseShuffleIndicies;
  SmallVector<Value *> UniqueValues;
  auto &&TryToFindDuplicates = [&VL, &ReuseShuffleIndicies, &UniqueValues,
                                &UserTreeIdx,
                                this](const InstructionsState &S) {
    // Check that every instruction appears once in this bundle.
    DenseMap<Value *, unsigned> UniquePositions;
    for (Value *V : VL) {
      if (isConstant(V)) {
        ReuseShuffleIndicies.emplace_back(
            isa<UndefValue>(V) ? UndefMaskElem : UniqueValues.size());
        UniqueValues.emplace_back(V);
        continue;
      }
      auto Res = UniquePositions.try_emplace(V, UniqueValues.size());
      ReuseShuffleIndicies.emplace_back(Res.first->second);
      if (Res.second)
        UniqueValues.emplace_back(V);
    }
    size_t NumUniqueScalarValues = UniqueValues.size();
    if (NumUniqueScalarValues == VL.size()) {
      ReuseShuffleIndicies.clear();
    } else {
      LLVM_DEBUG(dbgs() << "SLP: Shuffle for reused scalars.\n");
      if (NumUniqueScalarValues <= 1 ||
          (UniquePositions.size() == 1 && all_of(UniqueValues,
                                                 [](Value *V) {
                                                   return isa<UndefValue>(V) ||
                                                          !isConstant(V);
                                                 })) ||
          !llvm::isPowerOf2_32(NumUniqueScalarValues)) {
        LLVM_DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
        return false;
      }
      VL = UniqueValues;
    }
    return true;
  };

  InstructionsState S = getSameOpcode(VL);
  if (Depth == RecursionMaxDepth) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    if (TryToFindDuplicates(S))
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
    return;
  }

  // Don't handle scalable vectors
  if (S.getOpcode() == Instruction::ExtractElement &&
      isa<ScalableVectorType>(
          cast<ExtractElementInst>(S.OpValue)->getVectorOperandType())) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to scalable vector type.\n");
    if (TryToFindDuplicates(S))
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
    return;
  }

  // Don't handle vectors.
  if (S.OpValue->getType()->isVectorTy() &&
      !isa<InsertElementInst>(S.OpValue)) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  // Avoid attempting to schedule allocas; there are unmodeled dependencies
  // for "static" alloca status and for reordering with stacksave calls.
  for (Value *V : VL) {
    if (isa<AllocaInst>(V)) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to alloca.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
      return;
    }
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(S.OpValue))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
      return;
    }

  // If all of the operands are identical or constant we have a simple solution.
  // If we deal with insert/extract instructions, they all must have constant
  // indices, otherwise we should gather them, not try to vectorize.
  if (allConstant(VL) || isSplat(VL) || !allSameBlock(VL) || !S.getOpcode() ||
      (isa<InsertElementInst, ExtractValueInst, ExtractElementInst>(S.MainOp) &&
       !all_of(VL, isVectorLikeInstWithConstOps))) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O. \n");
    if (TryToFindDuplicates(S))
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
    return;
  }

  // We now know that this is a vector of instructions of the same type from
  // the same block.

  // Don't vectorize ephemeral values.
  for (Value *V : VL) {
    if (EphValues.count(V)) {
      LLVM_DEBUG(dbgs() << "SLP: The instruction (" << *V
                        << ") is ephemeral.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
      return;
    }
  }

  // Check if this is a duplicate of another entry.
  if (TreeEntry *E = getTreeEntry(S.OpValue)) {
    LLVM_DEBUG(dbgs() << "SLP: \tChecking bundle: " << *S.OpValue << ".\n");
    if (!E->isSame(VL)) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to partial overlap.\n");
      if (TryToFindDuplicates(S))
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
      return;
    }
    // Record the reuse of the tree node.  FIXME, currently this is only used to
    // properly draw the graph rather than for the actual vectorization.
    E->UserTreeIndices.push_back(UserTreeIdx);
    LLVM_DEBUG(dbgs() << "SLP: Perfect diamond merge at " << *S.OpValue
                      << ".\n");
    return;
  }

  // Check that none of the instructions in the bundle are already in the tree.
  for (Value *V : VL) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      continue;
    if (getTreeEntry(I)) {
      LLVM_DEBUG(dbgs() << "SLP: The instruction (" << *V
                        << ") is already in tree.\n");
      if (TryToFindDuplicates(S))
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
      return;
    }
  }

  // The reduction nodes (stored in UserIgnoreList) also should stay scalar.
  for (Value *V : VL) {
    if (is_contained(UserIgnoreList, V)) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to gathered scalar.\n");
      if (TryToFindDuplicates(S))
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
      return;
    }
  }

  // Check that all of the users of the scalars that we want to vectorize are
  // schedulable.
  auto *VL0 = cast<Instruction>(S.OpValue);
  BasicBlock *BB = VL0->getParent();

  if (!DT->isReachableFromEntry(BB)) {
    // Don't go into unreachable blocks. They may contain instructions with
    // dependency cycles which confuse the final scheduling.
    LLVM_DEBUG(dbgs() << "SLP: bundle in unreachable block.\n");
    newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  // Check that every instruction appears once in this bundle.
  if (!TryToFindDuplicates(S))
    return;

  auto &BSRef = BlocksSchedules[BB];
  if (!BSRef)
    BSRef = std::make_unique<BlockScheduling>(BB);

  BlockScheduling &BS = *BSRef.get();

  Optional<ScheduleData *> Bundle = BS.tryScheduleBundle(VL, this, S);
#ifdef EXPENSIVE_CHECKS
  // Make sure we didn't break any internal invariants
  BS.verify();
#endif
  if (!Bundle) {
    LLVM_DEBUG(dbgs() << "SLP: We are not able to schedule this bundle!\n");
    assert((!BS.getScheduleData(VL0) ||
            !BS.getScheduleData(VL0)->isPartOfBundle()) &&
           "tryScheduleBundle should cancelScheduling on failure");
    newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                 ReuseShuffleIndicies);
    return;
  }
  LLVM_DEBUG(dbgs() << "SLP: We are able to schedule this bundle.\n");

  unsigned ShuffleOrOp = S.isAltShuffle() ?
                (unsigned) Instruction::ShuffleVector : S.getOpcode();
  switch (ShuffleOrOp) {
    case Instruction::PHI: {
      auto *PH = cast<PHINode>(VL0);

      // Check for terminator values (e.g. invoke).
      for (Value *V : VL)
        for (unsigned I = 0, E = PH->getNumIncomingValues(); I < E; ++I) {
          Instruction *Term = dyn_cast<Instruction>(
              cast<PHINode>(V)->getIncomingValueForBlock(
                  PH->getIncomingBlock(I)));
          if (Term && Term->isTerminator()) {
            LLVM_DEBUG(dbgs()
                       << "SLP: Need to swizzle PHINodes (terminator use).\n");
            BS.cancelScheduling(VL, VL0);
            newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                         ReuseShuffleIndicies);
            return;
          }
        }

      TreeEntry *TE =
          newTreeEntry(VL, Bundle, S, UserTreeIdx, ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of PHINodes.\n");

      // Keeps the reordered operands to avoid code duplication.
      SmallVector<ValueList, 2> OperandsVec;
      for (unsigned I = 0, E = PH->getNumIncomingValues(); I < E; ++I) {
        if (!DT->isReachableFromEntry(PH->getIncomingBlock(I))) {
          ValueList Operands(VL.size(), PoisonValue::get(PH->getType()));
          TE->setOperand(I, Operands);
          OperandsVec.push_back(Operands);
          continue;
        }
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *V : VL)
          Operands.push_back(cast<PHINode>(V)->getIncomingValueForBlock(
              PH->getIncomingBlock(I)));
        TE->setOperand(I, Operands);
        OperandsVec.push_back(Operands);
      }
      for (unsigned OpIdx = 0, OpE = OperandsVec.size(); OpIdx != OpE; ++OpIdx)
        buildTree_rec(OperandsVec[OpIdx], Depth + 1, {TE, OpIdx});
      return;
    }
    case Instruction::ExtractValue:
    case Instruction::ExtractElement: {
      OrdersType CurrentOrder;
      bool Reuse = canReuseExtract(VL, VL0, CurrentOrder);
      if (Reuse) {
        LLVM_DEBUG(dbgs() << "SLP: Reusing or shuffling extract sequence.\n");
        newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
        // This is a special case, as it does not gather, but at the same time
        // we are not extending buildTree_rec() towards the operands.
        ValueList Op0;
        Op0.assign(VL.size(), VL0->getOperand(0));
        VectorizableTree.back()->setOperand(0, Op0);
        return;
      }
      if (!CurrentOrder.empty()) {
        LLVM_DEBUG({
          dbgs() << "SLP: Reusing or shuffling of reordered extract sequence "
                    "with order";
          for (unsigned Idx : CurrentOrder)
            dbgs() << " " << Idx;
          dbgs() << "\n";
        });
        fixupOrderingIndices(CurrentOrder);
        // Insert new order with initial value 0, if it does not exist,
        // otherwise return the iterator to the existing one.
        newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies, CurrentOrder);
        // This is a special case, as it does not gather, but at the same time
        // we are not extending buildTree_rec() towards the operands.
        ValueList Op0;
        Op0.assign(VL.size(), VL0->getOperand(0));
        VectorizableTree.back()->setOperand(0, Op0);
        return;
      }
      LLVM_DEBUG(dbgs() << "SLP: Gather extract sequence.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
      BS.cancelScheduling(VL, VL0);
      return;
    }
    case Instruction::InsertElement: {
      assert(ReuseShuffleIndicies.empty() && "All inserts should be unique");

      // Check that we have a buildvector and not a shuffle of 2 or more
      // different vectors.
      ValueSet SourceVectors;
      for (Value *V : VL) {
        SourceVectors.insert(cast<Instruction>(V)->getOperand(0));
        assert(getInsertIndex(V) != None && "Non-constant or undef index?");
      }

      if (count_if(VL, [&SourceVectors](Value *V) {
            return !SourceVectors.contains(V);
          }) >= 2) {
        // Found 2nd source vector - cancel.
        LLVM_DEBUG(dbgs() << "SLP: Gather of insertelement vectors with "
                             "different source vectors.\n");
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
        BS.cancelScheduling(VL, VL0);
        return;
      }

      auto OrdCompare = [](const std::pair<int, int> &P1,
                           const std::pair<int, int> &P2) {
        return P1.first > P2.first;
      };
      PriorityQueue<std::pair<int, int>, SmallVector<std::pair<int, int>>,
                    decltype(OrdCompare)>
          Indices(OrdCompare);
      for (int I = 0, E = VL.size(); I < E; ++I) {
        unsigned Idx = *getInsertIndex(VL[I]);
        Indices.emplace(Idx, I);
      }
      OrdersType CurrentOrder(VL.size(), VL.size());
      bool IsIdentity = true;
      for (int I = 0, E = VL.size(); I < E; ++I) {
        CurrentOrder[Indices.top().second] = I;
        IsIdentity &= Indices.top().second == I;
        Indices.pop();
      }
      if (IsIdentity)
        CurrentOrder.clear();
      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   None, CurrentOrder);
      LLVM_DEBUG(dbgs() << "SLP: added inserts bundle.\n");

      constexpr int NumOps = 2;
      ValueList VectorOperands[NumOps];
      for (int I = 0; I < NumOps; ++I) {
        for (Value *V : VL)
          VectorOperands[I].push_back(cast<Instruction>(V)->getOperand(I));

        TE->setOperand(I, VectorOperands[I]);
      }
      buildTree_rec(VectorOperands[NumOps - 1], Depth + 1, {TE, NumOps - 1});
      return;
    }
    case Instruction::Load: {
      // Check that a vectorized load would load the same memory as a scalar
      // load. For example, we don't want to vectorize loads that are smaller
      // than 8-bit. Even though we have a packed struct {<i2, i2, i2, i2>} LLVM
      // treats loading/storing it as an i8 struct. If we vectorize loads/stores
      // from such a struct, we read/write packed bits disagreeing with the
      // unvectorized version.
      SmallVector<Value *> PointerOps;
      OrdersType CurrentOrder;
      TreeEntry *TE = nullptr;
      switch (canVectorizeLoads(VL, VL0, *TTI, *DL, *SE, CurrentOrder,
                                PointerOps)) {
      case LoadsState::Vectorize:
        if (CurrentOrder.empty()) {
          // Original loads are consecutive and does not require reordering.
          TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                            ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs() << "SLP: added a vector of loads.\n");
        } else {
          fixupOrderingIndices(CurrentOrder);
          // Need to reorder.
          TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                            ReuseShuffleIndicies, CurrentOrder);
          LLVM_DEBUG(dbgs() << "SLP: added a vector of jumbled loads.\n");
        }
        TE->setOperandsInOrder();
        break;
      case LoadsState::ScatterVectorize:
        // Vectorizing non-consecutive loads with `llvm.masked.gather`.
        TE = newTreeEntry(VL, TreeEntry::ScatterVectorize, Bundle, S,
                          UserTreeIdx, ReuseShuffleIndicies);
        TE->setOperandsInOrder();
        buildTree_rec(PointerOps, Depth + 1, {TE, 0});
        LLVM_DEBUG(dbgs() << "SLP: added a vector of non-consecutive loads.\n");
        break;
      case LoadsState::Gather:
        BS.cancelScheduling(VL, VL0);
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
#ifndef NDEBUG
        Type *ScalarTy = VL0->getType();
        if (DL->getTypeSizeInBits(ScalarTy) !=
            DL->getTypeAllocSizeInBits(ScalarTy))
          LLVM_DEBUG(dbgs() << "SLP: Gathering loads of non-packed type.\n");
        else if (any_of(VL, [](Value *V) {
                   return !cast<LoadInst>(V)->isSimple();
                 }))
          LLVM_DEBUG(dbgs() << "SLP: Gathering non-simple loads.\n");
        else
          LLVM_DEBUG(dbgs() << "SLP: Gathering non-consecutive loads.\n");
#endif // NDEBUG
        break;
      }
      return;
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = VL0->getOperand(0)->getType();
      for (Value *V : VL) {
        Type *Ty = cast<Instruction>(V)->getOperand(0)->getType();
        if (Ty != SrcTy || !isValidElementType(Ty)) {
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs()
                     << "SLP: Gathering casts with different src types.\n");
          return;
        }
      }
      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of casts.\n");

      TE->setOperandsInOrder();
      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *V : VL)
          Operands.push_back(cast<Instruction>(V)->getOperand(i));

        buildTree_rec(Operands, Depth + 1, {TE, i});
      }
      return;
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Check that all of the compares have the same predicate.
      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      CmpInst::Predicate SwapP0 = CmpInst::getSwappedPredicate(P0);
      Type *ComparedTy = VL0->getOperand(0)->getType();
      for (Value *V : VL) {
        CmpInst *Cmp = cast<CmpInst>(V);
        if ((Cmp->getPredicate() != P0 && Cmp->getPredicate() != SwapP0) ||
            Cmp->getOperand(0)->getType() != ComparedTy) {
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs()
                     << "SLP: Gathering cmp with different predicate.\n");
          return;
        }
      }

      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of compares.\n");

      ValueList Left, Right;
      if (cast<CmpInst>(VL0)->isCommutative()) {
        // Commutative predicate - collect + sort operands of the instructions
        // so that each side is more likely to have the same opcode.
        assert(P0 == SwapP0 && "Commutative Predicate mismatch");
        reorderInputsAccordingToOpcode(VL, Left, Right, *DL, *SE, *this);
      } else {
        // Collect operands - commute if it uses the swapped predicate.
        for (Value *V : VL) {
          auto *Cmp = cast<CmpInst>(V);
          Value *LHS = Cmp->getOperand(0);
          Value *RHS = Cmp->getOperand(1);
          if (Cmp->getPredicate() != P0)
            std::swap(LHS, RHS);
          Left.push_back(LHS);
          Right.push_back(RHS);
        }
      }
      TE->setOperand(0, Left);
      TE->setOperand(1, Right);
      buildTree_rec(Left, Depth + 1, {TE, 0});
      buildTree_rec(Right, Depth + 1, {TE, 1});
      return;
    }
    case Instruction::Select:
    case Instruction::FNeg:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of un/bin op.\n");

      // Sort operands of the instructions so that each side is more likely to
      // have the same opcode.
      if (isa<BinaryOperator>(VL0) && VL0->isCommutative()) {
        ValueList Left, Right;
        reorderInputsAccordingToOpcode(VL, Left, Right, *DL, *SE, *this);
        TE->setOperand(0, Left);
        TE->setOperand(1, Right);
        buildTree_rec(Left, Depth + 1, {TE, 0});
        buildTree_rec(Right, Depth + 1, {TE, 1});
        return;
      }

      TE->setOperandsInOrder();
      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *V : VL)
          Operands.push_back(cast<Instruction>(V)->getOperand(i));

        buildTree_rec(Operands, Depth + 1, {TE, i});
      }
      return;
    }
    case Instruction::GetElementPtr: {
      // We don't combine GEPs with complicated (nested) indexing.
      for (Value *V : VL) {
        if (cast<Instruction>(V)->getNumOperands() != 2) {
          LLVM_DEBUG(dbgs() << "SLP: not-vectorizable GEP (nested indexes).\n");
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          return;
        }
      }

      // We can't combine several GEPs into one vector if they operate on
      // different types.
      Type *Ty0 = cast<GEPOperator>(VL0)->getSourceElementType();
      for (Value *V : VL) {
        Type *CurTy = cast<GEPOperator>(V)->getSourceElementType();
        if (Ty0 != CurTy) {
          LLVM_DEBUG(dbgs()
                     << "SLP: not-vectorizable GEP (different types).\n");
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          return;
        }
      }

      // We don't combine GEPs with non-constant indexes.
      Type *Ty1 = VL0->getOperand(1)->getType();
      for (Value *V : VL) {
        auto Op = cast<Instruction>(V)->getOperand(1);
        if (!isa<ConstantInt>(Op) ||
            (Op->getType() != Ty1 &&
             Op->getType()->getScalarSizeInBits() >
                 DL->getIndexSizeInBits(
                     V->getType()->getPointerAddressSpace()))) {
          LLVM_DEBUG(dbgs()
                     << "SLP: not-vectorizable GEP (non-constant indexes).\n");
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          return;
        }
      }

      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of GEPs.\n");
      SmallVector<ValueList, 2> Operands(2);
      // Prepare the operand vector for pointer operands.
      for (Value *V : VL)
        Operands.front().push_back(
            cast<GetElementPtrInst>(V)->getPointerOperand());
      TE->setOperand(0, Operands.front());
      // Need to cast all indices to the same type before vectorization to
      // avoid crash.
      // Required to be able to find correct matches between different gather
      // nodes and reuse the vectorized values rather than trying to gather them
      // again.
      int IndexIdx = 1;
      Type *VL0Ty = VL0->getOperand(IndexIdx)->getType();
      Type *Ty = all_of(VL,
                        [VL0Ty, IndexIdx](Value *V) {
                          return VL0Ty == cast<GetElementPtrInst>(V)
                                              ->getOperand(IndexIdx)
                                              ->getType();
                        })
                     ? VL0Ty
                     : DL->getIndexType(cast<GetElementPtrInst>(VL0)
                                            ->getPointerOperandType()
                                            ->getScalarType());
      // Prepare the operand vector.
      for (Value *V : VL) {
        auto *Op = cast<Instruction>(V)->getOperand(IndexIdx);
        auto *CI = cast<ConstantInt>(Op);
        Operands.back().push_back(ConstantExpr::getIntegerCast(
            CI, Ty, CI->getValue().isSignBitSet()));
      }
      TE->setOperand(IndexIdx, Operands.back());

      for (unsigned I = 0, Ops = Operands.size(); I < Ops; ++I)
        buildTree_rec(Operands[I], Depth + 1, {TE, I});
      return;
    }
    case Instruction::Store: {
      // Check if the stores are consecutive or if we need to swizzle them.
      llvm::Type *ScalarTy = cast<StoreInst>(VL0)->getValueOperand()->getType();
      // Avoid types that are padded when being allocated as scalars, while
      // being packed together in a vector (such as i1).
      if (DL->getTypeSizeInBits(ScalarTy) !=
          DL->getTypeAllocSizeInBits(ScalarTy)) {
        BS.cancelScheduling(VL, VL0);
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
        LLVM_DEBUG(dbgs() << "SLP: Gathering stores of non-packed type.\n");
        return;
      }
      // Make sure all stores in the bundle are simple - we can't vectorize
      // atomic or volatile stores.
      SmallVector<Value *, 4> PointerOps(VL.size());
      ValueList Operands(VL.size());
      auto POIter = PointerOps.begin();
      auto OIter = Operands.begin();
      for (Value *V : VL) {
        auto *SI = cast<StoreInst>(V);
        if (!SI->isSimple()) {
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs() << "SLP: Gathering non-simple stores.\n");
          return;
        }
        *POIter = SI->getPointerOperand();
        *OIter = SI->getValueOperand();
        ++POIter;
        ++OIter;
      }

      OrdersType CurrentOrder;
      // Check the order of pointer operands.
      if (llvm::sortPtrAccesses(PointerOps, ScalarTy, *DL, *SE, CurrentOrder)) {
        Value *Ptr0;
        Value *PtrN;
        if (CurrentOrder.empty()) {
          Ptr0 = PointerOps.front();
          PtrN = PointerOps.back();
        } else {
          Ptr0 = PointerOps[CurrentOrder.front()];
          PtrN = PointerOps[CurrentOrder.back()];
        }
        Optional<int> Dist =
            getPointersDiff(ScalarTy, Ptr0, ScalarTy, PtrN, *DL, *SE);
        // Check that the sorted pointer operands are consecutive.
        if (static_cast<unsigned>(*Dist) == VL.size() - 1) {
          if (CurrentOrder.empty()) {
            // Original stores are consecutive and does not require reordering.
            TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S,
                                         UserTreeIdx, ReuseShuffleIndicies);
            TE->setOperandsInOrder();
            buildTree_rec(Operands, Depth + 1, {TE, 0});
            LLVM_DEBUG(dbgs() << "SLP: added a vector of stores.\n");
          } else {
            fixupOrderingIndices(CurrentOrder);
            TreeEntry *TE =
                newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                             ReuseShuffleIndicies, CurrentOrder);
            TE->setOperandsInOrder();
            buildTree_rec(Operands, Depth + 1, {TE, 0});
            LLVM_DEBUG(dbgs() << "SLP: added a vector of jumbled stores.\n");
          }
          return;
        }
      }

      BS.cancelScheduling(VL, VL0);
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: Non-consecutive store.\n");
      return;
    }
    case Instruction::Call: {
      // Check if the calls are all to the same vectorizable intrinsic or
      // library function.
      CallInst *CI = cast<CallInst>(VL0);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      VFShape Shape = VFShape::get(
          *CI, ElementCount::getFixed(static_cast<unsigned int>(VL.size())),
          false /*HasGlobalPred*/);
      Function *VecFunc = VFDatabase(*CI).getVectorizedFunction(Shape);

      if (!VecFunc && !isTriviallyVectorizable(ID)) {
        BS.cancelScheduling(VL, VL0);
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
        LLVM_DEBUG(dbgs() << "SLP: Non-vectorizable call.\n");
        return;
      }
      Function *F = CI->getCalledFunction();
      unsigned NumArgs = CI->arg_size();
      SmallVector<Value*, 4> ScalarArgs(NumArgs, nullptr);
      for (unsigned j = 0; j != NumArgs; ++j)
        if (hasVectorInstrinsicScalarOpd(ID, j))
          ScalarArgs[j] = CI->getArgOperand(j);
      for (Value *V : VL) {
        CallInst *CI2 = dyn_cast<CallInst>(V);
        if (!CI2 || CI2->getCalledFunction() != F ||
            getVectorIntrinsicIDForCall(CI2, TLI) != ID ||
            (VecFunc &&
             VecFunc != VFDatabase(*CI2).getVectorizedFunction(Shape)) ||
            !CI->hasIdenticalOperandBundleSchema(*CI2)) {
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs() << "SLP: mismatched calls:" << *CI << "!=" << *V
                            << "\n");
          return;
        }
        // Some intrinsics have scalar arguments and should be same in order for
        // them to be vectorized.
        for (unsigned j = 0; j != NumArgs; ++j) {
          if (hasVectorInstrinsicScalarOpd(ID, j)) {
            Value *A1J = CI2->getArgOperand(j);
            if (ScalarArgs[j] != A1J) {
              BS.cancelScheduling(VL, VL0);
              newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                           ReuseShuffleIndicies);
              LLVM_DEBUG(dbgs() << "SLP: mismatched arguments in call:" << *CI
                                << " argument " << ScalarArgs[j] << "!=" << A1J
                                << "\n");
              return;
            }
          }
        }
        // Verify that the bundle operands are identical between the two calls.
        if (CI->hasOperandBundles() &&
            !std::equal(CI->op_begin() + CI->getBundleOperandsStartIndex(),
                        CI->op_begin() + CI->getBundleOperandsEndIndex(),
                        CI2->op_begin() + CI2->getBundleOperandsStartIndex())) {
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs() << "SLP: mismatched bundle operands in calls:"
                            << *CI << "!=" << *V << '\n');
          return;
        }
      }

      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      TE->setOperandsInOrder();
      for (unsigned i = 0, e = CI->arg_size(); i != e; ++i) {
        // For scalar operands no need to to create an entry since no need to
        // vectorize it.
        if (hasVectorInstrinsicScalarOpd(ID, i))
          continue;
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *V : VL) {
          auto *CI2 = cast<CallInst>(V);
          Operands.push_back(CI2->getArgOperand(i));
        }
        buildTree_rec(Operands, Depth + 1, {TE, i});
      }
      return;
    }
    case Instruction::ShuffleVector: {
      // If this is not an alternate sequence of opcode like add-sub
      // then do not vectorize this instruction.
      if (!S.isAltShuffle()) {
        BS.cancelScheduling(VL, VL0);
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
        LLVM_DEBUG(dbgs() << "SLP: ShuffleVector are not vectorized.\n");
        return;
      }
      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a ShuffleVector op.\n");

      // Reorder operands if reordering would enable vectorization.
      auto *CI = dyn_cast<CmpInst>(VL0);
      if (isa<BinaryOperator>(VL0) || CI) {
        ValueList Left, Right;
        if (!CI || all_of(VL, [](Value *V) {
              return cast<CmpInst>(V)->isCommutative();
            })) {
          reorderInputsAccordingToOpcode(VL, Left, Right, *DL, *SE, *this);
        } else {
          CmpInst::Predicate P0 = CI->getPredicate();
          CmpInst::Predicate AltP0 = cast<CmpInst>(S.AltOp)->getPredicate();
          assert(P0 != AltP0 &&
                 "Expected different main/alternate predicates.");
          CmpInst::Predicate AltP0Swapped = CmpInst::getSwappedPredicate(AltP0);
          Value *BaseOp0 = VL0->getOperand(0);
          Value *BaseOp1 = VL0->getOperand(1);
          // Collect operands - commute if it uses the swapped predicate or
          // alternate operation.
          for (Value *V : VL) {
            auto *Cmp = cast<CmpInst>(V);
            Value *LHS = Cmp->getOperand(0);
            Value *RHS = Cmp->getOperand(1);
            CmpInst::Predicate CurrentPred = Cmp->getPredicate();
            if (P0 == AltP0Swapped) {
              if (CI != Cmp && S.AltOp != Cmp &&
                  ((P0 == CurrentPred &&
                    !areCompatibleCmpOps(BaseOp0, BaseOp1, LHS, RHS)) ||
                   (AltP0 == CurrentPred &&
                    areCompatibleCmpOps(BaseOp0, BaseOp1, LHS, RHS))))
                std::swap(LHS, RHS);
            } else if (P0 != CurrentPred && AltP0 != CurrentPred) {
              std::swap(LHS, RHS);
            }
            Left.push_back(LHS);
            Right.push_back(RHS);
          }
        }
        TE->setOperand(0, Left);
        TE->setOperand(1, Right);
        buildTree_rec(Left, Depth + 1, {TE, 0});
        buildTree_rec(Right, Depth + 1, {TE, 1});
        return;
      }

      TE->setOperandsInOrder();
      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *V : VL)
          Operands.push_back(cast<Instruction>(V)->getOperand(i));

        buildTree_rec(Operands, Depth + 1, {TE, i});
      }
      return;
    }
    default:
      BS.cancelScheduling(VL, VL0);
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: Gathering unknown instruction.\n");
      return;
  }
}

unsigned BoUpSLP::canMapToVector(Type *T, const DataLayout &DL) const {
  unsigned N = 1;
  Type *EltTy = T;

  while (isa<StructType>(EltTy) || isa<ArrayType>(EltTy) ||
         isa<VectorType>(EltTy)) {
    if (auto *ST = dyn_cast<StructType>(EltTy)) {
      // Check that struct is homogeneous.
      for (const auto *Ty : ST->elements())
        if (Ty != *ST->element_begin())
          return 0;
      N *= ST->getNumElements();
      EltTy = *ST->element_begin();
    } else if (auto *AT = dyn_cast<ArrayType>(EltTy)) {
      N *= AT->getNumElements();
      EltTy = AT->getElementType();
    } else {
      auto *VT = cast<FixedVectorType>(EltTy);
      N *= VT->getNumElements();
      EltTy = VT->getElementType();
    }
  }

  if (!isValidElementType(EltTy))
    return 0;
  uint64_t VTSize = DL.getTypeStoreSizeInBits(FixedVectorType::get(EltTy, N));
  if (VTSize < MinVecRegSize || VTSize > MaxVecRegSize || VTSize != DL.getTypeStoreSizeInBits(T))
    return 0;
  return N;
}

bool BoUpSLP::canReuseExtract(ArrayRef<Value *> VL, Value *OpValue,
                              SmallVectorImpl<unsigned> &CurrentOrder) const {
  const auto *It = find_if(VL, [](Value *V) {
    return isa<ExtractElementInst, ExtractValueInst>(V);
  });
  assert(It != VL.end() && "Expected at least one extract instruction.");
  auto *E0 = cast<Instruction>(*It);
  assert(all_of(VL,
                [](Value *V) {
                  return isa<UndefValue, ExtractElementInst, ExtractValueInst>(
                      V);
                }) &&
         "Invalid opcode");
  // Check if all of the extracts come from the same vector and from the
  // correct offset.
  Value *Vec = E0->getOperand(0);

  CurrentOrder.clear();

  // We have to extract from a vector/aggregate with the same number of elements.
  unsigned NElts;
  if (E0->getOpcode() == Instruction::ExtractValue) {
    const DataLayout &DL = E0->getModule()->getDataLayout();
    NElts = canMapToVector(Vec->getType(), DL);
    if (!NElts)
      return false;
    // Check if load can be rewritten as load of vector.
    LoadInst *LI = dyn_cast<LoadInst>(Vec);
    if (!LI || !LI->isSimple() || !LI->hasNUses(VL.size()))
      return false;
  } else {
    NElts = cast<FixedVectorType>(Vec->getType())->getNumElements();
  }

  if (NElts != VL.size())
    return false;

  // Check that all of the indices extract from the correct offset.
  bool ShouldKeepOrder = true;
  unsigned E = VL.size();
  // Assign to all items the initial value E + 1 so we can check if the extract
  // instruction index was used already.
  // Also, later we can check that all the indices are used and we have a
  // consecutive access in the extract instructions, by checking that no
  // element of CurrentOrder still has value E + 1.
  CurrentOrder.assign(E, E);
  unsigned I = 0;
  for (; I < E; ++I) {
    auto *Inst = dyn_cast<Instruction>(VL[I]);
    if (!Inst)
      continue;
    if (Inst->getOperand(0) != Vec)
      break;
    if (auto *EE = dyn_cast<ExtractElementInst>(Inst))
      if (isa<UndefValue>(EE->getIndexOperand()))
        continue;
    Optional<unsigned> Idx = getExtractIndex(Inst);
    if (!Idx)
      break;
    const unsigned ExtIdx = *Idx;
    if (ExtIdx != I) {
      if (ExtIdx >= E || CurrentOrder[ExtIdx] != E)
        break;
      ShouldKeepOrder = false;
      CurrentOrder[ExtIdx] = I;
    } else {
      if (CurrentOrder[I] != E)
        break;
      CurrentOrder[I] = I;
    }
  }
  if (I < E) {
    CurrentOrder.clear();
    return false;
  }
  if (ShouldKeepOrder)
    CurrentOrder.clear();

  return ShouldKeepOrder;
}

bool BoUpSLP::areAllUsersVectorized(Instruction *I,
                                    ArrayRef<Value *> VectorizedVals) const {
  return (I->hasOneUse() && is_contained(VectorizedVals, I)) ||
         all_of(I->users(), [this](User *U) {
           return ScalarToTreeEntry.count(U) > 0 ||
                  isVectorLikeInstWithConstOps(U) ||
                  (isa<ExtractElementInst>(U) && MustGather.contains(U));
         });
}

static std::pair<InstructionCost, InstructionCost>
getVectorCallCosts(CallInst *CI, FixedVectorType *VecTy,
                   TargetTransformInfo *TTI, TargetLibraryInfo *TLI) {
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

  // Calculate the cost of the scalar and vector calls.
  SmallVector<Type *, 4> VecTys;
  for (Use &Arg : CI->args())
    VecTys.push_back(
        FixedVectorType::get(Arg->getType(), VecTy->getNumElements()));
  FastMathFlags FMF;
  if (auto *FPCI = dyn_cast<FPMathOperator>(CI))
    FMF = FPCI->getFastMathFlags();
  SmallVector<const Value *> Arguments(CI->args());
  IntrinsicCostAttributes CostAttrs(ID, VecTy, Arguments, VecTys, FMF,
                                    dyn_cast<IntrinsicInst>(CI));
  auto IntrinsicCost =
    TTI->getIntrinsicInstrCost(CostAttrs, TTI::TCK_RecipThroughput);

  auto Shape = VFShape::get(*CI, ElementCount::getFixed(static_cast<unsigned>(
                                     VecTy->getNumElements())),
                            false /*HasGlobalPred*/);
  Function *VecFunc = VFDatabase(*CI).getVectorizedFunction(Shape);
  auto LibCost = IntrinsicCost;
  if (!CI->isNoBuiltin() && VecFunc) {
    // Calculate the cost of the vector library call.
    // If the corresponding vector call is cheaper, return its cost.
    LibCost = TTI->getCallInstrCost(nullptr, VecTy, VecTys,
                                    TTI::TCK_RecipThroughput);
  }
  return {IntrinsicCost, LibCost};
}

/// Compute the cost of creating a vector of type \p VecTy containing the
/// extracted values from \p VL.
static InstructionCost
computeExtractCost(ArrayRef<Value *> VL, FixedVectorType *VecTy,
                   TargetTransformInfo::ShuffleKind ShuffleKind,
                   ArrayRef<int> Mask, TargetTransformInfo &TTI) {
  unsigned NumOfParts = TTI.getNumberOfParts(VecTy);

  if (ShuffleKind != TargetTransformInfo::SK_PermuteSingleSrc || !NumOfParts ||
      VecTy->getNumElements() < NumOfParts)
    return TTI.getShuffleCost(ShuffleKind, VecTy, Mask);

  bool AllConsecutive = true;
  unsigned EltsPerVector = VecTy->getNumElements() / NumOfParts;
  unsigned Idx = -1;
  InstructionCost Cost = 0;

  // Process extracts in blocks of EltsPerVector to check if the source vector
  // operand can be re-used directly. If not, add the cost of creating a shuffle
  // to extract the values into a vector register.
  for (auto *V : VL) {
    ++Idx;

    // Need to exclude undefs from analysis.
    if (isa<UndefValue>(V) || Mask[Idx] == UndefMaskElem)
      continue;

    // Reached the start of a new vector registers.
    if (Idx % EltsPerVector == 0) {
      AllConsecutive = true;
      continue;
    }

    // Check all extracts for a vector register on the target directly
    // extract values in order.
    unsigned CurrentIdx = *getExtractIndex(cast<Instruction>(V));
    if (!isa<UndefValue>(VL[Idx - 1]) && Mask[Idx - 1] != UndefMaskElem) {
      unsigned PrevIdx = *getExtractIndex(cast<Instruction>(VL[Idx - 1]));
      AllConsecutive &= PrevIdx + 1 == CurrentIdx &&
                        CurrentIdx % EltsPerVector == Idx % EltsPerVector;
    }

    if (AllConsecutive)
      continue;

    // Skip all indices, except for the last index per vector block.
    if ((Idx + 1) % EltsPerVector != 0 && Idx + 1 != VL.size())
      continue;

    // If we have a series of extracts which are not consecutive and hence
    // cannot re-use the source vector register directly, compute the shuffle
    // cost to extract the a vector with EltsPerVector elements.
    Cost += TTI.getShuffleCost(
        TargetTransformInfo::SK_PermuteSingleSrc,
        FixedVectorType::get(VecTy->getElementType(), EltsPerVector));
  }
  return Cost;
}

/// Build shuffle mask for shuffle graph entries and lists of main and alternate
/// operations operands.
static void
buildShuffleEntryMask(ArrayRef<Value *> VL, ArrayRef<unsigned> ReorderIndices,
                      ArrayRef<int> ReusesIndices,
                      const function_ref<bool(Instruction *)> IsAltOp,
                      SmallVectorImpl<int> &Mask,
                      SmallVectorImpl<Value *> *OpScalars = nullptr,
                      SmallVectorImpl<Value *> *AltScalars = nullptr) {
  unsigned Sz = VL.size();
  Mask.assign(Sz, UndefMaskElem);
  SmallVector<int> OrderMask;
  if (!ReorderIndices.empty())
    inversePermutation(ReorderIndices, OrderMask);
  for (unsigned I = 0; I < Sz; ++I) {
    unsigned Idx = I;
    if (!ReorderIndices.empty())
      Idx = OrderMask[I];
    auto *OpInst = cast<Instruction>(VL[Idx]);
    if (IsAltOp(OpInst)) {
      Mask[I] = Sz + Idx;
      if (AltScalars)
        AltScalars->push_back(OpInst);
    } else {
      Mask[I] = Idx;
      if (OpScalars)
        OpScalars->push_back(OpInst);
    }
  }
  if (!ReusesIndices.empty()) {
    SmallVector<int> NewMask(ReusesIndices.size(), UndefMaskElem);
    transform(ReusesIndices, NewMask.begin(), [&Mask](int Idx) {
      return Idx != UndefMaskElem ? Mask[Idx] : UndefMaskElem;
    });
    Mask.swap(NewMask);
  }
}

/// Checks if the specified instruction \p I is an alternate operation for the
/// given \p MainOp and \p AltOp instructions.
static bool isAlternateInstruction(const Instruction *I,
                                   const Instruction *MainOp,
                                   const Instruction *AltOp) {
  if (auto *CI0 = dyn_cast<CmpInst>(MainOp)) {
    auto *AltCI0 = cast<CmpInst>(AltOp);
    auto *CI = cast<CmpInst>(I);
    CmpInst::Predicate P0 = CI0->getPredicate();
    CmpInst::Predicate AltP0 = AltCI0->getPredicate();
    assert(P0 != AltP0 && "Expected different main/alternate predicates.");
    CmpInst::Predicate AltP0Swapped = CmpInst::getSwappedPredicate(AltP0);
    CmpInst::Predicate CurrentPred = CI->getPredicate();
    if (P0 == AltP0Swapped)
      return I == AltCI0 ||
             (I != MainOp &&
              !areCompatibleCmpOps(CI0->getOperand(0), CI0->getOperand(1),
                                   CI->getOperand(0), CI->getOperand(1)));
    return AltP0 == CurrentPred || AltP0Swapped == CurrentPred;
  }
  return I->getOpcode() == AltOp->getOpcode();
}

InstructionCost BoUpSLP::getEntryCost(const TreeEntry *E,
                                      ArrayRef<Value *> VectorizedVals) {
  ArrayRef<Value*> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  else if (CmpInst *CI = dyn_cast<CmpInst>(VL[0]))
    ScalarTy = CI->getOperand(0)->getType();
  else if (auto *IE = dyn_cast<InsertElementInst>(VL[0]))
    ScalarTy = IE->getOperand(1)->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  // If we have computed a smaller type for the expression, update VecTy so
  // that the costs will be accurate.
  if (MinBWs.count(VL[0]))
    VecTy = FixedVectorType::get(
        IntegerType::get(F->getContext(), MinBWs[VL[0]].first), VL.size());
  unsigned EntryVF = E->getVectorFactor();
  auto *FinalVecTy = FixedVectorType::get(VecTy->getElementType(), EntryVF);

  bool NeedToShuffleReuses = !E->ReuseShuffleIndices.empty();
  // FIXME: it tries to fix a problem with MSVC buildbots.
  TargetTransformInfo &TTIRef = *TTI;
  auto &&AdjustExtractsCost = [this, &TTIRef, CostKind, VL, VecTy,
                               VectorizedVals, E](InstructionCost &Cost) {
    DenseMap<Value *, int> ExtractVectorsTys;
    SmallPtrSet<Value *, 4> CheckedExtracts;
    for (auto *V : VL) {
      if (isa<UndefValue>(V))
        continue;
      // If all users of instruction are going to be vectorized and this
      // instruction itself is not going to be vectorized, consider this
      // instruction as dead and remove its cost from the final cost of the
      // vectorized tree.
      // Also, avoid adjusting the cost for extractelements with multiple uses
      // in different graph entries.
      const TreeEntry *VE = getTreeEntry(V);
      if (!CheckedExtracts.insert(V).second ||
          !areAllUsersVectorized(cast<Instruction>(V), VectorizedVals) ||
          (VE && VE != E))
        continue;
      auto *EE = cast<ExtractElementInst>(V);
      Optional<unsigned> EEIdx = getExtractIndex(EE);
      if (!EEIdx)
        continue;
      unsigned Idx = *EEIdx;
      if (TTIRef.getNumberOfParts(VecTy) !=
          TTIRef.getNumberOfParts(EE->getVectorOperandType())) {
        auto It =
            ExtractVectorsTys.try_emplace(EE->getVectorOperand(), Idx).first;
        It->getSecond() = std::min<int>(It->second, Idx);
      }
      // Take credit for instruction that will become dead.
      if (EE->hasOneUse()) {
        Instruction *Ext = EE->user_back();
        if ((isa<SExtInst>(Ext) || isa<ZExtInst>(Ext)) &&
            all_of(Ext->users(),
                   [](User *U) { return isa<GetElementPtrInst>(U); })) {
          // Use getExtractWithExtendCost() to calculate the cost of
          // extractelement/ext pair.
          Cost -=
              TTIRef.getExtractWithExtendCost(Ext->getOpcode(), Ext->getType(),
                                              EE->getVectorOperandType(), Idx);
          // Add back the cost of s|zext which is subtracted separately.
          Cost += TTIRef.getCastInstrCost(
              Ext->getOpcode(), Ext->getType(), EE->getType(),
              TTI::getCastContextHint(Ext), CostKind, Ext);
          continue;
        }
      }
      Cost -= TTIRef.getVectorInstrCost(Instruction::ExtractElement,
                                        EE->getVectorOperandType(), Idx);
    }
    // Add a cost for subvector extracts/inserts if required.
    for (const auto &Data : ExtractVectorsTys) {
      auto *EEVTy = cast<FixedVectorType>(Data.first->getType());
      unsigned NumElts = VecTy->getNumElements();
      if (Data.second % NumElts == 0)
        continue;
      if (TTIRef.getNumberOfParts(EEVTy) > TTIRef.getNumberOfParts(VecTy)) {
        unsigned Idx = (Data.second / NumElts) * NumElts;
        unsigned EENumElts = EEVTy->getNumElements();
        if (Idx + NumElts <= EENumElts) {
          Cost +=
              TTIRef.getShuffleCost(TargetTransformInfo::SK_ExtractSubvector,
                                    EEVTy, None, Idx, VecTy);
        } else {
          // Need to round up the subvector type vectorization factor to avoid a
          // crash in cost model functions. Make SubVT so that Idx + VF of SubVT
          // <= EENumElts.
          auto *SubVT =
              FixedVectorType::get(VecTy->getElementType(), EENumElts - Idx);
          Cost +=
              TTIRef.getShuffleCost(TargetTransformInfo::SK_ExtractSubvector,
                                    EEVTy, None, Idx, SubVT);
        }
      } else {
        Cost += TTIRef.getShuffleCost(TargetTransformInfo::SK_InsertSubvector,
                                      VecTy, None, 0, EEVTy);
      }
    }
  };
  if (E->State == TreeEntry::NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isa<InsertElementInst>(VL[0]))
      return InstructionCost::getInvalid();
    SmallVector<int> Mask;
    SmallVector<const TreeEntry *> Entries;
    Optional<TargetTransformInfo::ShuffleKind> Shuffle =
        isGatherShuffledEntry(E, Mask, Entries);
    if (Shuffle.hasValue()) {
      InstructionCost GatherCost = 0;
      if (ShuffleVectorInst::isIdentityMask(Mask)) {
        // Perfect match in the graph, will reuse the previously vectorized
        // node. Cost is 0.
        LLVM_DEBUG(
            dbgs()
            << "SLP: perfect diamond match for gather bundle that starts with "
            << *VL.front() << ".\n");
        if (NeedToShuffleReuses)
          GatherCost =
              TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                  FinalVecTy, E->ReuseShuffleIndices);
      } else {
        LLVM_DEBUG(dbgs() << "SLP: shuffled " << Entries.size()
                          << " entries for bundle that starts with "
                          << *VL.front() << ".\n");
        // Detected that instead of gather we can emit a shuffle of single/two
        // previously vectorized nodes. Add the cost of the permutation rather
        // than gather.
        ::addMask(Mask, E->ReuseShuffleIndices);
        GatherCost = TTI->getShuffleCost(*Shuffle, FinalVecTy, Mask);
      }
      return GatherCost;
    }
    if ((E->getOpcode() == Instruction::ExtractElement ||
         all_of(E->Scalars,
                [](Value *V) {
                  return isa<ExtractElementInst, UndefValue>(V);
                })) &&
        allSameType(VL)) {
      // Check that gather of extractelements can be represented as just a
      // shuffle of a single/two vectors the scalars are extracted from.
      SmallVector<int> Mask;
      Optional<TargetTransformInfo::ShuffleKind> ShuffleKind =
          isFixedVectorShuffle(VL, Mask);
      if (ShuffleKind.hasValue()) {
        // Found the bunch of extractelement instructions that must be gathered
        // into a vector and can be represented as a permutation elements in a
        // single input vector or of 2 input vectors.
        InstructionCost Cost =
            computeExtractCost(VL, VecTy, *ShuffleKind, Mask, *TTI);
        AdjustExtractsCost(Cost);
        if (NeedToShuffleReuses)
          Cost += TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                      FinalVecTy, E->ReuseShuffleIndices);
        return Cost;
      }
    }
    if (isSplat(VL)) {
      // Found the broadcasting of the single scalar, calculate the cost as the
      // broadcast.
      assert(VecTy == FinalVecTy &&
             "No reused scalars expected for broadcast.");
      return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy);
    }
    InstructionCost ReuseShuffleCost = 0;
    if (NeedToShuffleReuses)
      ReuseShuffleCost = TTI->getShuffleCost(
          TTI::SK_PermuteSingleSrc, FinalVecTy, E->ReuseShuffleIndices);
    // Improve gather cost for gather of loads, if we can group some of the
    // loads into vector loads.
    if (VL.size() > 2 && E->getOpcode() == Instruction::Load &&
        !E->isAltShuffle()) {
      BoUpSLP::ValueSet VectorizedLoads;
      unsigned StartIdx = 0;
      unsigned VF = VL.size() / 2;
      unsigned VectorizedCnt = 0;
      unsigned ScatterVectorizeCnt = 0;
      const unsigned Sz = DL->getTypeSizeInBits(E->getMainOp()->getType());
      for (unsigned MinVF = getMinVF(2 * Sz); VF >= MinVF; VF /= 2) {
        for (unsigned Cnt = StartIdx, End = VL.size(); Cnt + VF <= End;
             Cnt += VF) {
          ArrayRef<Value *> Slice = VL.slice(Cnt, VF);
          if (!VectorizedLoads.count(Slice.front()) &&
              !VectorizedLoads.count(Slice.back()) && allSameBlock(Slice)) {
            SmallVector<Value *> PointerOps;
            OrdersType CurrentOrder;
            LoadsState LS = canVectorizeLoads(Slice, Slice.front(), *TTI, *DL,
                                              *SE, CurrentOrder, PointerOps);
            switch (LS) {
            case LoadsState::Vectorize:
            case LoadsState::ScatterVectorize:
              // Mark the vectorized loads so that we don't vectorize them
              // again.
              if (LS == LoadsState::Vectorize)
                ++VectorizedCnt;
              else
                ++ScatterVectorizeCnt;
              VectorizedLoads.insert(Slice.begin(), Slice.end());
              // If we vectorized initial block, no need to try to vectorize it
              // again.
              if (Cnt == StartIdx)
                StartIdx += VF;
              break;
            case LoadsState::Gather:
              break;
            }
          }
        }
        // Check if the whole array was vectorized already - exit.
        if (StartIdx >= VL.size())
          break;
        // Found vectorizable parts - exit.
        if (!VectorizedLoads.empty())
          break;
      }
      if (!VectorizedLoads.empty()) {
        InstructionCost GatherCost = 0;
        unsigned NumParts = TTI->getNumberOfParts(VecTy);
        bool NeedInsertSubvectorAnalysis =
            !NumParts || (VL.size() / VF) > NumParts;
        // Get the cost for gathered loads.
        for (unsigned I = 0, End = VL.size(); I < End; I += VF) {
          if (VectorizedLoads.contains(VL[I]))
            continue;
          GatherCost += getGatherCost(VL.slice(I, VF));
        }
        // The cost for vectorized loads.
        InstructionCost ScalarsCost = 0;
        for (Value *V : VectorizedLoads) {
          auto *LI = cast<LoadInst>(V);
          ScalarsCost += TTI->getMemoryOpCost(
              Instruction::Load, LI->getType(), LI->getAlign(),
              LI->getPointerAddressSpace(), CostKind, LI);
        }
        auto *LI = cast<LoadInst>(E->getMainOp());
        auto *LoadTy = FixedVectorType::get(LI->getType(), VF);
        Align Alignment = LI->getAlign();
        GatherCost +=
            VectorizedCnt *
            TTI->getMemoryOpCost(Instruction::Load, LoadTy, Alignment,
                                 LI->getPointerAddressSpace(), CostKind, LI);
        GatherCost += ScatterVectorizeCnt *
                      TTI->getGatherScatterOpCost(
                          Instruction::Load, LoadTy, LI->getPointerOperand(),
                          /*VariableMask=*/false, Alignment, CostKind, LI);
        if (NeedInsertSubvectorAnalysis) {
          // Add the cost for the subvectors insert.
          for (int I = VF, E = VL.size(); I < E; I += VF)
            GatherCost += TTI->getShuffleCost(TTI::SK_InsertSubvector, VecTy,
                                              None, I, LoadTy);
        }
        return ReuseShuffleCost + GatherCost - ScalarsCost;
      }
    }
    return ReuseShuffleCost + getGatherCost(VL);
  }
  InstructionCost CommonCost = 0;
  SmallVector<int> Mask;
  if (!E->ReorderIndices.empty()) {
    SmallVector<int> NewMask;
    if (E->getOpcode() == Instruction::Store) {
      // For stores the order is actually a mask.
      NewMask.resize(E->ReorderIndices.size());
      copy(E->ReorderIndices, NewMask.begin());
    } else {
      inversePermutation(E->ReorderIndices, NewMask);
    }
    ::addMask(Mask, NewMask);
  }
  if (NeedToShuffleReuses)
    ::addMask(Mask, E->ReuseShuffleIndices);
  if (!Mask.empty() && !ShuffleVectorInst::isIdentityMask(Mask))
    CommonCost =
        TTI->getShuffleCost(TTI::SK_PermuteSingleSrc, FinalVecTy, Mask);
  assert((E->State == TreeEntry::Vectorize ||
          E->State == TreeEntry::ScatterVectorize) &&
         "Unhandled state");
  assert(E->getOpcode() && allSameType(VL) && allSameBlock(VL) && "Invalid VL");
  Instruction *VL0 = E->getMainOp();
  unsigned ShuffleOrOp =
      E->isAltShuffle() ? (unsigned)Instruction::ShuffleVector : E->getOpcode();
  switch (ShuffleOrOp) {
    case Instruction::PHI:
      return 0;

    case Instruction::ExtractValue:
    case Instruction::ExtractElement: {
      // The common cost of removal ExtractElement/ExtractValue instructions +
      // the cost of shuffles, if required to resuffle the original vector.
      if (NeedToShuffleReuses) {
        unsigned Idx = 0;
        for (unsigned I : E->ReuseShuffleIndices) {
          if (ShuffleOrOp == Instruction::ExtractElement) {
            auto *EE = cast<ExtractElementInst>(VL[I]);
            CommonCost -= TTI->getVectorInstrCost(Instruction::ExtractElement,
                                                  EE->getVectorOperandType(),
                                                  *getExtractIndex(EE));
          } else {
            CommonCost -= TTI->getVectorInstrCost(Instruction::ExtractElement,
                                                  VecTy, Idx);
            ++Idx;
          }
        }
        Idx = EntryVF;
        for (Value *V : VL) {
          if (ShuffleOrOp == Instruction::ExtractElement) {
            auto *EE = cast<ExtractElementInst>(V);
            CommonCost += TTI->getVectorInstrCost(Instruction::ExtractElement,
                                                  EE->getVectorOperandType(),
                                                  *getExtractIndex(EE));
          } else {
            --Idx;
            CommonCost += TTI->getVectorInstrCost(Instruction::ExtractElement,
                                                  VecTy, Idx);
          }
        }
      }
      if (ShuffleOrOp == Instruction::ExtractValue) {
        for (unsigned I = 0, E = VL.size(); I < E; ++I) {
          auto *EI = cast<Instruction>(VL[I]);
          // Take credit for instruction that will become dead.
          if (EI->hasOneUse()) {
            Instruction *Ext = EI->user_back();
            if ((isa<SExtInst>(Ext) || isa<ZExtInst>(Ext)) &&
                all_of(Ext->users(),
                       [](User *U) { return isa<GetElementPtrInst>(U); })) {
              // Use getExtractWithExtendCost() to calculate the cost of
              // extractelement/ext pair.
              CommonCost -= TTI->getExtractWithExtendCost(
                  Ext->getOpcode(), Ext->getType(), VecTy, I);
              // Add back the cost of s|zext which is subtracted separately.
              CommonCost += TTI->getCastInstrCost(
                  Ext->getOpcode(), Ext->getType(), EI->getType(),
                  TTI::getCastContextHint(Ext), CostKind, Ext);
              continue;
            }
          }
          CommonCost -=
              TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, I);
        }
      } else {
        AdjustExtractsCost(CommonCost);
      }
      return CommonCost;
    }
    case Instruction::InsertElement: {
      assert(E->ReuseShuffleIndices.empty() &&
             "Unique insertelements only are expected.");
      auto *SrcVecTy = cast<FixedVectorType>(VL0->getType());

      unsigned const NumElts = SrcVecTy->getNumElements();
      unsigned const NumScalars = VL.size();
      APInt DemandedElts = APInt::getZero(NumElts);
      // TODO: Add support for Instruction::InsertValue.
      SmallVector<int> Mask;
      if (!E->ReorderIndices.empty()) {
        inversePermutation(E->ReorderIndices, Mask);
        Mask.append(NumElts - NumScalars, UndefMaskElem);
      } else {
        Mask.assign(NumElts, UndefMaskElem);
        std::iota(Mask.begin(), std::next(Mask.begin(), NumScalars), 0);
      }
      unsigned Offset = *getInsertIndex(VL0);
      bool IsIdentity = true;
      SmallVector<int> PrevMask(NumElts, UndefMaskElem);
      Mask.swap(PrevMask);
      for (unsigned I = 0; I < NumScalars; ++I) {
        unsigned InsertIdx = *getInsertIndex(VL[PrevMask[I]]);
        DemandedElts.setBit(InsertIdx);
        IsIdentity &= InsertIdx - Offset == I;
        Mask[InsertIdx - Offset] = I;
      }
      assert(Offset < NumElts && "Failed to find vector index offset");

      InstructionCost Cost = 0;
      Cost -= TTI->getScalarizationOverhead(SrcVecTy, DemandedElts,
                                            /*Insert*/ true, /*Extract*/ false);

      if (IsIdentity && NumElts != NumScalars && Offset % NumScalars != 0) {
        // FIXME: Replace with SK_InsertSubvector once it is properly supported.
        unsigned Sz = PowerOf2Ceil(Offset + NumScalars);
        Cost += TTI->getShuffleCost(
            TargetTransformInfo::SK_PermuteSingleSrc,
            FixedVectorType::get(SrcVecTy->getElementType(), Sz));
      } else if (!IsIdentity) {
        auto *FirstInsert =
            cast<Instruction>(*find_if(E->Scalars, [E](Value *V) {
              return !is_contained(E->Scalars,
                                   cast<Instruction>(V)->getOperand(0));
            }));
        if (isUndefVector(FirstInsert->getOperand(0))) {
          Cost += TTI->getShuffleCost(TTI::SK_PermuteSingleSrc, SrcVecTy, Mask);
        } else {
          SmallVector<int> InsertMask(NumElts);
          std::iota(InsertMask.begin(), InsertMask.end(), 0);
          for (unsigned I = 0; I < NumElts; I++) {
            if (Mask[I] != UndefMaskElem)
              InsertMask[Offset + I] = NumElts + I;
          }
          Cost +=
              TTI->getShuffleCost(TTI::SK_PermuteTwoSrc, SrcVecTy, InsertMask);
        }
      }

      return Cost;
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = VL0->getOperand(0)->getType();
      InstructionCost ScalarEltCost =
          TTI->getCastInstrCost(E->getOpcode(), ScalarTy, SrcTy,
                                TTI::getCastContextHint(VL0), CostKind, VL0);
      if (NeedToShuffleReuses) {
        CommonCost -= (EntryVF - VL.size()) * ScalarEltCost;
      }

      // Calculate the cost of this instruction.
      InstructionCost ScalarCost = VL.size() * ScalarEltCost;

      auto *SrcVecTy = FixedVectorType::get(SrcTy, VL.size());
      InstructionCost VecCost = 0;
      // Check if the values are candidates to demote.
      if (!MinBWs.count(VL0) || VecTy != SrcVecTy) {
        VecCost = CommonCost + TTI->getCastInstrCost(
                                   E->getOpcode(), VecTy, SrcVecTy,
                                   TTI::getCastContextHint(VL0), CostKind, VL0);
      }
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecCost, ScalarCost));
      return VecCost - ScalarCost;
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select: {
      // Calculate the cost of this instruction.
      InstructionCost ScalarEltCost =
          TTI->getCmpSelInstrCost(E->getOpcode(), ScalarTy, Builder.getInt1Ty(),
                                  CmpInst::BAD_ICMP_PREDICATE, CostKind, VL0);
      if (NeedToShuffleReuses) {
        CommonCost -= (EntryVF - VL.size()) * ScalarEltCost;
      }
      auto *MaskTy = FixedVectorType::get(Builder.getInt1Ty(), VL.size());
      InstructionCost ScalarCost = VecTy->getNumElements() * ScalarEltCost;

      // Check if all entries in VL are either compares or selects with compares
      // as condition that have the same predicates.
      CmpInst::Predicate VecPred = CmpInst::BAD_ICMP_PREDICATE;
      bool First = true;
      for (auto *V : VL) {
        CmpInst::Predicate CurrentPred;
        auto MatchCmp = m_Cmp(CurrentPred, m_Value(), m_Value());
        if ((!match(V, m_Select(MatchCmp, m_Value(), m_Value())) &&
             !match(V, MatchCmp)) ||
            (!First && VecPred != CurrentPred)) {
          VecPred = CmpInst::BAD_ICMP_PREDICATE;
          break;
        }
        First = false;
        VecPred = CurrentPred;
      }

      InstructionCost VecCost = TTI->getCmpSelInstrCost(
          E->getOpcode(), VecTy, MaskTy, VecPred, CostKind, VL0);
      // Check if it is possible and profitable to use min/max for selects in
      // VL.
      //
      auto IntrinsicAndUse = canConvertToMinOrMaxIntrinsic(VL);
      if (IntrinsicAndUse.first != Intrinsic::not_intrinsic) {
        IntrinsicCostAttributes CostAttrs(IntrinsicAndUse.first, VecTy,
                                          {VecTy, VecTy});
        InstructionCost IntrinsicCost =
            TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
        // If the selects are the only uses of the compares, they will be dead
        // and we can adjust the cost by removing their cost.
        if (IntrinsicAndUse.second)
          IntrinsicCost -= TTI->getCmpSelInstrCost(Instruction::ICmp, VecTy,
                                                   MaskTy, VecPred, CostKind);
        VecCost = std::min(VecCost, IntrinsicCost);
      }
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecCost, ScalarCost));
      return CommonCost + VecCost - ScalarCost;
    }
    case Instruction::FNeg:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      // Certain instructions can be cheaper to vectorize if they have a
      // constant second vector operand.
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;
      TargetTransformInfo::OperandValueProperties Op1VP =
          TargetTransformInfo::OP_None;
      TargetTransformInfo::OperandValueProperties Op2VP =
          TargetTransformInfo::OP_PowerOf2;

      // If all operands are exactly the same ConstantInt then set the
      // operand kind to OK_UniformConstantValue.
      // If instead not all operands are constants, then set the operand kind
      // to OK_AnyValue. If all operands are constants but not the same,
      // then set the operand kind to OK_NonUniformConstantValue.
      ConstantInt *CInt0 = nullptr;
      for (unsigned i = 0, e = VL.size(); i < e; ++i) {
        const Instruction *I = cast<Instruction>(VL[i]);
        unsigned OpIdx = isa<BinaryOperator>(I) ? 1 : 0;
        ConstantInt *CInt = dyn_cast<ConstantInt>(I->getOperand(OpIdx));
        if (!CInt) {
          Op2VK = TargetTransformInfo::OK_AnyValue;
          Op2VP = TargetTransformInfo::OP_None;
          break;
        }
        if (Op2VP == TargetTransformInfo::OP_PowerOf2 &&
            !CInt->getValue().isPowerOf2())
          Op2VP = TargetTransformInfo::OP_None;
        if (i == 0) {
          CInt0 = CInt;
          continue;
        }
        if (CInt0 != CInt)
          Op2VK = TargetTransformInfo::OK_NonUniformConstantValue;
      }

      SmallVector<const Value *, 4> Operands(VL0->operand_values());
      InstructionCost ScalarEltCost =
          TTI->getArithmeticInstrCost(E->getOpcode(), ScalarTy, CostKind, Op1VK,
                                      Op2VK, Op1VP, Op2VP, Operands, VL0);
      if (NeedToShuffleReuses) {
        CommonCost -= (EntryVF - VL.size()) * ScalarEltCost;
      }
      InstructionCost ScalarCost = VecTy->getNumElements() * ScalarEltCost;
      InstructionCost VecCost =
          TTI->getArithmeticInstrCost(E->getOpcode(), VecTy, CostKind, Op1VK,
                                      Op2VK, Op1VP, Op2VP, Operands, VL0);
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecCost, ScalarCost));
      return CommonCost + VecCost - ScalarCost;
    }
    case Instruction::GetElementPtr: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;

      InstructionCost ScalarEltCost = TTI->getArithmeticInstrCost(
          Instruction::Add, ScalarTy, CostKind, Op1VK, Op2VK);
      if (NeedToShuffleReuses) {
        CommonCost -= (EntryVF - VL.size()) * ScalarEltCost;
      }
      InstructionCost ScalarCost = VecTy->getNumElements() * ScalarEltCost;
      InstructionCost VecCost = TTI->getArithmeticInstrCost(
          Instruction::Add, VecTy, CostKind, Op1VK, Op2VK);
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecCost, ScalarCost));
      return CommonCost + VecCost - ScalarCost;
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      Align Alignment = cast<LoadInst>(VL0)->getAlign();
      InstructionCost ScalarEltCost = TTI->getMemoryOpCost(
          Instruction::Load, ScalarTy, Alignment, 0, CostKind, VL0);
      if (NeedToShuffleReuses) {
        CommonCost -= (EntryVF - VL.size()) * ScalarEltCost;
      }
      InstructionCost ScalarLdCost = VecTy->getNumElements() * ScalarEltCost;
      InstructionCost VecLdCost;
      if (E->State == TreeEntry::Vectorize) {
        VecLdCost = TTI->getMemoryOpCost(Instruction::Load, VecTy, Alignment, 0,
                                         CostKind, VL0);
      } else {
        assert(E->State == TreeEntry::ScatterVectorize && "Unknown EntryState");
        Align CommonAlignment = Alignment;
        for (Value *V : VL)
          CommonAlignment =
              commonAlignment(CommonAlignment, cast<LoadInst>(V)->getAlign());
        VecLdCost = TTI->getGatherScatterOpCost(
            Instruction::Load, VecTy, cast<LoadInst>(VL0)->getPointerOperand(),
            /*VariableMask=*/false, CommonAlignment, CostKind, VL0);
      }
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecLdCost, ScalarLdCost));
      return CommonCost + VecLdCost - ScalarLdCost;
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      bool IsReorder = !E->ReorderIndices.empty();
      auto *SI =
          cast<StoreInst>(IsReorder ? VL[E->ReorderIndices.front()] : VL0);
      Align Alignment = SI->getAlign();
      InstructionCost ScalarEltCost = TTI->getMemoryOpCost(
          Instruction::Store, ScalarTy, Alignment, 0, CostKind, VL0);
      InstructionCost ScalarStCost = VecTy->getNumElements() * ScalarEltCost;
      InstructionCost VecStCost = TTI->getMemoryOpCost(
          Instruction::Store, VecTy, Alignment, 0, CostKind, VL0);
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecStCost, ScalarStCost));
      return CommonCost + VecStCost - ScalarStCost;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      // Calculate the cost of the scalar and vector calls.
      IntrinsicCostAttributes CostAttrs(ID, *CI, 1);
      InstructionCost ScalarEltCost =
          TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
      if (NeedToShuffleReuses) {
        CommonCost -= (EntryVF - VL.size()) * ScalarEltCost;
      }
      InstructionCost ScalarCallCost = VecTy->getNumElements() * ScalarEltCost;

      auto VecCallCosts = getVectorCallCosts(CI, VecTy, TTI, TLI);
      InstructionCost VecCallCost =
          std::min(VecCallCosts.first, VecCallCosts.second);

      LLVM_DEBUG(dbgs() << "SLP: Call cost " << VecCallCost - ScalarCallCost
                        << " (" << VecCallCost << "-" << ScalarCallCost << ")"
                        << " for " << *CI << "\n");

      return CommonCost + VecCallCost - ScalarCallCost;
    }
    case Instruction::ShuffleVector: {
      assert(E->isAltShuffle() &&
             ((Instruction::isBinaryOp(E->getOpcode()) &&
               Instruction::isBinaryOp(E->getAltOpcode())) ||
              (Instruction::isCast(E->getOpcode()) &&
               Instruction::isCast(E->getAltOpcode())) ||
              (isa<CmpInst>(VL0) && isa<CmpInst>(E->getAltOp()))) &&
             "Invalid Shuffle Vector Operand");
      InstructionCost ScalarCost = 0;
      if (NeedToShuffleReuses) {
        for (unsigned Idx : E->ReuseShuffleIndices) {
          Instruction *I = cast<Instruction>(VL[Idx]);
          CommonCost -= TTI->getInstructionCost(I, CostKind);
        }
        for (Value *V : VL) {
          Instruction *I = cast<Instruction>(V);
          CommonCost += TTI->getInstructionCost(I, CostKind);
        }
      }
      for (Value *V : VL) {
        Instruction *I = cast<Instruction>(V);
        assert(E->isOpcodeOrAlt(I) && "Unexpected main/alternate opcode");
        ScalarCost += TTI->getInstructionCost(I, CostKind);
      }
      // VecCost is equal to sum of the cost of creating 2 vectors
      // and the cost of creating shuffle.
      InstructionCost VecCost = 0;
      // Try to find the previous shuffle node with the same operands and same
      // main/alternate ops.
      auto &&TryFindNodeWithEqualOperands = [this, E]() {
        for (const std::unique_ptr<TreeEntry> &TE : VectorizableTree) {
          if (TE.get() == E)
            break;
          if (TE->isAltShuffle() &&
              ((TE->getOpcode() == E->getOpcode() &&
                TE->getAltOpcode() == E->getAltOpcode()) ||
               (TE->getOpcode() == E->getAltOpcode() &&
                TE->getAltOpcode() == E->getOpcode())) &&
              TE->hasEqualOperands(*E))
            return true;
        }
        return false;
      };
      if (TryFindNodeWithEqualOperands()) {
        LLVM_DEBUG({
          dbgs() << "SLP: diamond match for alternate node found.\n";
          E->dump();
        });
        // No need to add new vector costs here since we're going to reuse
        // same main/alternate vector ops, just do different shuffling.
      } else if (Instruction::isBinaryOp(E->getOpcode())) {
        VecCost = TTI->getArithmeticInstrCost(E->getOpcode(), VecTy, CostKind);
        VecCost += TTI->getArithmeticInstrCost(E->getAltOpcode(), VecTy,
                                               CostKind);
      } else if (auto *CI0 = dyn_cast<CmpInst>(VL0)) {
        VecCost = TTI->getCmpSelInstrCost(E->getOpcode(), ScalarTy,
                                          Builder.getInt1Ty(),
                                          CI0->getPredicate(), CostKind, VL0);
        VecCost += TTI->getCmpSelInstrCost(
            E->getOpcode(), ScalarTy, Builder.getInt1Ty(),
            cast<CmpInst>(E->getAltOp())->getPredicate(), CostKind,
            E->getAltOp());
      } else {
        Type *Src0SclTy = E->getMainOp()->getOperand(0)->getType();
        Type *Src1SclTy = E->getAltOp()->getOperand(0)->getType();
        auto *Src0Ty = FixedVectorType::get(Src0SclTy, VL.size());
        auto *Src1Ty = FixedVectorType::get(Src1SclTy, VL.size());
        VecCost = TTI->getCastInstrCost(E->getOpcode(), VecTy, Src0Ty,
                                        TTI::CastContextHint::None, CostKind);
        VecCost += TTI->getCastInstrCost(E->getAltOpcode(), VecTy, Src1Ty,
                                         TTI::CastContextHint::None, CostKind);
      }

      SmallVector<int> Mask;
      buildShuffleEntryMask(
          E->Scalars, E->ReorderIndices, E->ReuseShuffleIndices,
          [E](Instruction *I) {
            assert(E->isOpcodeOrAlt(I) && "Unexpected main/alternate opcode");
            return isAlternateInstruction(I, E->getMainOp(), E->getAltOp());
          },
          Mask);
      CommonCost =
          TTI->getShuffleCost(TargetTransformInfo::SK_Select, FinalVecTy, Mask);
      LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecCost, ScalarCost));
      return CommonCost + VecCost - ScalarCost;
    }
    default:
      llvm_unreachable("Unknown instruction");
  }
}

bool BoUpSLP::isFullyVectorizableTinyTree(bool ForReduction) const {
  LLVM_DEBUG(dbgs() << "SLP: Check whether the tree with height "
                    << VectorizableTree.size() << " is fully vectorizable .\n");

  auto &&AreVectorizableGathers = [this](const TreeEntry *TE, unsigned Limit) {
    SmallVector<int> Mask;
    return TE->State == TreeEntry::NeedToGather &&
           !any_of(TE->Scalars,
                   [this](Value *V) { return EphValues.contains(V); }) &&
           (allConstant(TE->Scalars) || isSplat(TE->Scalars) ||
            TE->Scalars.size() < Limit ||
            ((TE->getOpcode() == Instruction::ExtractElement ||
              all_of(TE->Scalars,
                     [](Value *V) {
                       return isa<ExtractElementInst, UndefValue>(V);
                     })) &&
             isFixedVectorShuffle(TE->Scalars, Mask)) ||
            (TE->State == TreeEntry::NeedToGather &&
             TE->getOpcode() == Instruction::Load && !TE->isAltShuffle()));
  };

  // We only handle trees of heights 1 and 2.
  if (VectorizableTree.size() == 1 &&
      (VectorizableTree[0]->State == TreeEntry::Vectorize ||
       (ForReduction &&
        AreVectorizableGathers(VectorizableTree[0].get(),
                               VectorizableTree[0]->Scalars.size()) &&
        VectorizableTree[0]->getVectorFactor() > 2)))
    return true;

  if (VectorizableTree.size() != 2)
    return false;

  // Handle splat and all-constants stores. Also try to vectorize tiny trees
  // with the second gather nodes if they have less scalar operands rather than
  // the initial tree element (may be profitable to shuffle the second gather)
  // or they are extractelements, which form shuffle.
  SmallVector<int> Mask;
  if (VectorizableTree[0]->State == TreeEntry::Vectorize &&
      AreVectorizableGathers(VectorizableTree[1].get(),
                             VectorizableTree[0]->Scalars.size()))
    return true;

  // Gathering cost would be too much for tiny trees.
  if (VectorizableTree[0]->State == TreeEntry::NeedToGather ||
      (VectorizableTree[1]->State == TreeEntry::NeedToGather &&
       VectorizableTree[0]->State != TreeEntry::ScatterVectorize))
    return false;

  return true;
}

static bool isLoadCombineCandidateImpl(Value *Root, unsigned NumElts,
                                       TargetTransformInfo *TTI,
                                       bool MustMatchOrInst) {
  // Look past the root to find a source value. Arbitrarily follow the
  // path through operand 0 of any 'or'. Also, peek through optional
  // shift-left-by-multiple-of-8-bits.
  Value *ZextLoad = Root;
  const APInt *ShAmtC;
  bool FoundOr = false;
  while (!isa<ConstantExpr>(ZextLoad) &&
         (match(ZextLoad, m_Or(m_Value(), m_Value())) ||
          (match(ZextLoad, m_Shl(m_Value(), m_APInt(ShAmtC))) &&
           ShAmtC->urem(8) == 0))) {
    auto *BinOp = cast<BinaryOperator>(ZextLoad);
    ZextLoad = BinOp->getOperand(0);
    if (BinOp->getOpcode() == Instruction::Or)
      FoundOr = true;
  }
  // Check if the input is an extended load of the required or/shift expression.
  Value *Load;
  if ((MustMatchOrInst && !FoundOr) || ZextLoad == Root ||
      !match(ZextLoad, m_ZExt(m_Value(Load))) || !isa<LoadInst>(Load))
    return false;

  // Require that the total load bit width is a legal integer type.
  // For example, <8 x i8> --> i64 is a legal integer on a 64-bit target.
  // But <16 x i8> --> i128 is not, so the backend probably can't reduce it.
  Type *SrcTy = Load->getType();
  unsigned LoadBitWidth = SrcTy->getIntegerBitWidth() * NumElts;
  if (!TTI->isTypeLegal(IntegerType::get(Root->getContext(), LoadBitWidth)))
    return false;

  // Everything matched - assume that we can fold the whole sequence using
  // load combining.
  LLVM_DEBUG(dbgs() << "SLP: Assume load combining for tree starting at "
             << *(cast<Instruction>(Root)) << "\n");

  return true;
}

bool BoUpSLP::isLoadCombineReductionCandidate(RecurKind RdxKind) const {
  if (RdxKind != RecurKind::Or)
    return false;

  unsigned NumElts = VectorizableTree[0]->Scalars.size();
  Value *FirstReduced = VectorizableTree[0]->Scalars[0];
  return isLoadCombineCandidateImpl(FirstReduced, NumElts, TTI,
                                    /* MatchOr */ false);
}

bool BoUpSLP::isLoadCombineCandidate() const {
  // Peek through a final sequence of stores and check if all operations are
  // likely to be load-combined.
  unsigned NumElts = VectorizableTree[0]->Scalars.size();
  for (Value *Scalar : VectorizableTree[0]->Scalars) {
    Value *X;
    if (!match(Scalar, m_Store(m_Value(X), m_Value())) ||
        !isLoadCombineCandidateImpl(X, NumElts, TTI, /* MatchOr */ true))
      return false;
  }
  return true;
}

bool BoUpSLP::isTreeTinyAndNotFullyVectorizable(bool ForReduction) const {
  // No need to vectorize inserts of gathered values.
  if (VectorizableTree.size() == 2 &&
      isa<InsertElementInst>(VectorizableTree[0]->Scalars[0]) &&
      VectorizableTree[1]->State == TreeEntry::NeedToGather)
    return true;

  // We can vectorize the tree if its size is greater than or equal to the
  // minimum size specified by the MinTreeSize command line option.
  if (VectorizableTree.size() >= MinTreeSize)
    return false;

  // If we have a tiny tree (a tree whose size is less than MinTreeSize), we
  // can vectorize it if we can prove it fully vectorizable.
  if (isFullyVectorizableTinyTree(ForReduction))
    return false;

  assert(VectorizableTree.empty()
             ? ExternalUses.empty()
             : true && "We shouldn't have any external users");

  // Otherwise, we can't vectorize the tree. It is both tiny and not fully
  // vectorizable.
  return true;
}

InstructionCost BoUpSLP::getSpillCost() const {
  // Walk from the bottom of the tree to the top, tracking which values are
  // live. When we see a call instruction that is not part of our tree,
  // query TTI to see if there is a cost to keeping values live over it
  // (for example, if spills and fills are required).
  unsigned BundleWidth = VectorizableTree.front()->Scalars.size();
  InstructionCost Cost = 0;

  SmallPtrSet<Instruction*, 4> LiveValues;
  Instruction *PrevInst = nullptr;

  // The entries in VectorizableTree are not necessarily ordered by their
  // position in basic blocks. Collect them and order them by dominance so later
  // instructions are guaranteed to be visited first. For instructions in
  // different basic blocks, we only scan to the beginning of the block, so
  // their order does not matter, as long as all instructions in a basic block
  // are grouped together. Using dominance ensures a deterministic order.
  SmallVector<Instruction *, 16> OrderedScalars;
  for (const auto &TEPtr : VectorizableTree) {
    Instruction *Inst = dyn_cast<Instruction>(TEPtr->Scalars[0]);
    if (!Inst)
      continue;
    OrderedScalars.push_back(Inst);
  }
  llvm::sort(OrderedScalars, [&](Instruction *A, Instruction *B) {
    auto *NodeA = DT->getNode(A->getParent());
    auto *NodeB = DT->getNode(B->getParent());
    assert(NodeA && "Should only process reachable instructions");
    assert(NodeB && "Should only process reachable instructions");
    assert((NodeA == NodeB) == (NodeA->getDFSNumIn() == NodeB->getDFSNumIn()) &&
           "Different nodes should have different DFS numbers");
    if (NodeA != NodeB)
      return NodeA->getDFSNumIn() < NodeB->getDFSNumIn();
    return B->comesBefore(A);
  });

  for (Instruction *Inst : OrderedScalars) {
    if (!PrevInst) {
      PrevInst = Inst;
      continue;
    }

    // Update LiveValues.
    LiveValues.erase(PrevInst);
    for (auto &J : PrevInst->operands()) {
      if (isa<Instruction>(&*J) && getTreeEntry(&*J))
        LiveValues.insert(cast<Instruction>(&*J));
    }

    LLVM_DEBUG({
      dbgs() << "SLP: #LV: " << LiveValues.size();
      for (auto *X : LiveValues)
        dbgs() << " " << X->getName();
      dbgs() << ", Looking at ";
      Inst->dump();
    });

    // Now find the sequence of instructions between PrevInst and Inst.
    unsigned NumCalls = 0;
    BasicBlock::reverse_iterator InstIt = ++Inst->getIterator().getReverse(),
                                 PrevInstIt =
                                     PrevInst->getIterator().getReverse();
    while (InstIt != PrevInstIt) {
      if (PrevInstIt == PrevInst->getParent()->rend()) {
        PrevInstIt = Inst->getParent()->rbegin();
        continue;
      }

      // Debug information does not impact spill cost.
      if ((isa<CallInst>(&*PrevInstIt) &&
           !isa<DbgInfoIntrinsic>(&*PrevInstIt)) &&
          &*PrevInstIt != PrevInst)
        NumCalls++;

      ++PrevInstIt;
    }

    if (NumCalls) {
      SmallVector<Type*, 4> V;
      for (auto *II : LiveValues) {
        auto *ScalarTy = II->getType();
        if (auto *VectorTy = dyn_cast<FixedVectorType>(ScalarTy))
          ScalarTy = VectorTy->getElementType();
        V.push_back(FixedVectorType::get(ScalarTy, BundleWidth));
      }
      Cost += NumCalls * TTI->getCostOfKeepingLiveOverCall(V);
    }

    PrevInst = Inst;
  }

  return Cost;
}

/// Check if two insertelement instructions are from the same buildvector.
static bool areTwoInsertFromSameBuildVector(InsertElementInst *VU,
                                            InsertElementInst *V) {
  // Instructions must be from the same basic blocks.
  if (VU->getParent() != V->getParent())
    return false;
  // Checks if 2 insertelements are from the same buildvector.
  if (VU->getType() != V->getType())
    return false;
  // Multiple used inserts are separate nodes.
  if (!VU->hasOneUse() && !V->hasOneUse())
    return false;
  auto *IE1 = VU;
  auto *IE2 = V;
  // Go through the vector operand of insertelement instructions trying to find
  // either VU as the original vector for IE2 or V as the original vector for
  // IE1.
  do {
    if (IE2 == VU || IE1 == V)
      return true;
    if (IE1) {
      if (IE1 != VU && !IE1->hasOneUse())
        IE1 = nullptr;
      else
        IE1 = dyn_cast<InsertElementInst>(IE1->getOperand(0));
    }
    if (IE2) {
      if (IE2 != V && !IE2->hasOneUse())
        IE2 = nullptr;
      else
        IE2 = dyn_cast<InsertElementInst>(IE2->getOperand(0));
    }
  } while (IE1 || IE2);
  return false;
}

InstructionCost BoUpSLP::getTreeCost(ArrayRef<Value *> VectorizedVals) {
  InstructionCost Cost = 0;
  LLVM_DEBUG(dbgs() << "SLP: Calculating cost for tree of size "
                    << VectorizableTree.size() << ".\n");

  unsigned BundleWidth = VectorizableTree[0]->Scalars.size();

  for (unsigned I = 0, E = VectorizableTree.size(); I < E; ++I) {
    TreeEntry &TE = *VectorizableTree[I].get();

    InstructionCost C = getEntryCost(&TE, VectorizedVals);
    Cost += C;
    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                      << " for bundle that starts with " << *TE.Scalars[0]
                      << ".\n"
                      << "SLP: Current total cost = " << Cost << "\n");
  }

  SmallPtrSet<Value *, 16> ExtractCostCalculated;
  InstructionCost ExtractCost = 0;
  SmallVector<unsigned> VF;
  SmallVector<SmallVector<int>> ShuffleMask;
  SmallVector<Value *> FirstUsers;
  SmallVector<APInt> DemandedElts;
  for (ExternalUser &EU : ExternalUses) {
    // We only add extract cost once for the same scalar.
    if (!isa_and_nonnull<InsertElementInst>(EU.User) &&
        !ExtractCostCalculated.insert(EU.Scalar).second)
      continue;

    // Uses by ephemeral values are free (because the ephemeral value will be
    // removed prior to code generation, and so the extraction will be
    // removed as well).
    if (EphValues.count(EU.User))
      continue;

    // No extract cost for vector "scalar"
    if (isa<FixedVectorType>(EU.Scalar->getType()))
      continue;

    // Already counted the cost for external uses when tried to adjust the cost
    // for extractelements, no need to add it again.
    if (isa<ExtractElementInst>(EU.Scalar))
      continue;

    // If found user is an insertelement, do not calculate extract cost but try
    // to detect it as a final shuffled/identity match.
    if (auto *VU = dyn_cast_or_null<InsertElementInst>(EU.User)) {
      if (auto *FTy = dyn_cast<FixedVectorType>(VU->getType())) {
        Optional<unsigned> InsertIdx = getInsertIndex(VU);
        if (InsertIdx) {
          auto *It = find_if(FirstUsers, [VU](Value *V) {
            return areTwoInsertFromSameBuildVector(VU,
                                                   cast<InsertElementInst>(V));
          });
          int VecId = -1;
          if (It == FirstUsers.end()) {
            VF.push_back(FTy->getNumElements());
            ShuffleMask.emplace_back(VF.back(), UndefMaskElem);
            // Find the insertvector, vectorized in tree, if any.
            Value *Base = VU;
            while (isa<InsertElementInst>(Base)) {
              // Build the mask for the vectorized insertelement instructions.
              if (const TreeEntry *E = getTreeEntry(Base)) {
                VU = cast<InsertElementInst>(Base);
                do {
                  int Idx = E->findLaneForValue(Base);
                  ShuffleMask.back()[Idx] = Idx;
                  Base = cast<InsertElementInst>(Base)->getOperand(0);
                } while (E == getTreeEntry(Base));
                break;
              }
              Base = cast<InsertElementInst>(Base)->getOperand(0);
            }
            FirstUsers.push_back(VU);
            DemandedElts.push_back(APInt::getZero(VF.back()));
            VecId = FirstUsers.size() - 1;
          } else {
            VecId = std::distance(FirstUsers.begin(), It);
          }
          ShuffleMask[VecId][*InsertIdx] = EU.Lane;
          DemandedElts[VecId].setBit(*InsertIdx);
          continue;
        }
      }
    }

    // If we plan to rewrite the tree in a smaller type, we will need to sign
    // extend the extracted value back to the original type. Here, we account
    // for the extract and the added cost of the sign extend if needed.
    auto *VecTy = FixedVectorType::get(EU.Scalar->getType(), BundleWidth);
    auto *ScalarRoot = VectorizableTree[0]->Scalars[0];
    if (MinBWs.count(ScalarRoot)) {
      auto *MinTy = IntegerType::get(F->getContext(), MinBWs[ScalarRoot].first);
      auto Extend =
          MinBWs[ScalarRoot].second ? Instruction::SExt : Instruction::ZExt;
      VecTy = FixedVectorType::get(MinTy, BundleWidth);
      ExtractCost += TTI->getExtractWithExtendCost(Extend, EU.Scalar->getType(),
                                                   VecTy, EU.Lane);
    } else {
      ExtractCost +=
          TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, EU.Lane);
    }
  }

  InstructionCost SpillCost = getSpillCost();
  Cost += SpillCost + ExtractCost;
  if (FirstUsers.size() == 1) {
    int Limit = ShuffleMask.front().size() * 2;
    if (all_of(ShuffleMask.front(), [Limit](int Idx) { return Idx < Limit; }) &&
        !ShuffleVectorInst::isIdentityMask(ShuffleMask.front())) {
      InstructionCost C = TTI->getShuffleCost(
          TTI::SK_PermuteSingleSrc,
          cast<FixedVectorType>(FirstUsers.front()->getType()),
          ShuffleMask.front());
      LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                        << " for final shuffle of insertelement external users "
                        << *VectorizableTree.front()->Scalars.front() << ".\n"
                        << "SLP: Current total cost = " << Cost << "\n");
      Cost += C;
    }
    InstructionCost InsertCost = TTI->getScalarizationOverhead(
        cast<FixedVectorType>(FirstUsers.front()->getType()),
        DemandedElts.front(), /*Insert*/ true, /*Extract*/ false);
    LLVM_DEBUG(dbgs() << "SLP: subtracting the cost " << InsertCost
                      << " for insertelements gather.\n"
                      << "SLP: Current total cost = " << Cost << "\n");
    Cost -= InsertCost;
  } else if (FirstUsers.size() >= 2) {
    unsigned MaxVF = *std::max_element(VF.begin(), VF.end());
    // Combined masks of the first 2 vectors.
    SmallVector<int> CombinedMask(MaxVF, UndefMaskElem);
    copy(ShuffleMask.front(), CombinedMask.begin());
    APInt CombinedDemandedElts = DemandedElts.front().zextOrSelf(MaxVF);
    auto *VecTy = FixedVectorType::get(
        cast<VectorType>(FirstUsers.front()->getType())->getElementType(),
        MaxVF);
    for (int I = 0, E = ShuffleMask[1].size(); I < E; ++I) {
      if (ShuffleMask[1][I] != UndefMaskElem) {
        CombinedMask[I] = ShuffleMask[1][I] + MaxVF;
        CombinedDemandedElts.setBit(I);
      }
    }
    InstructionCost C =
        TTI->getShuffleCost(TTI::SK_PermuteTwoSrc, VecTy, CombinedMask);
    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                      << " for final shuffle of vector node and external "
                         "insertelement users "
                      << *VectorizableTree.front()->Scalars.front() << ".\n"
                      << "SLP: Current total cost = " << Cost << "\n");
    Cost += C;
    InstructionCost InsertCost = TTI->getScalarizationOverhead(
        VecTy, CombinedDemandedElts, /*Insert*/ true, /*Extract*/ false);
    LLVM_DEBUG(dbgs() << "SLP: subtracting the cost " << InsertCost
                      << " for insertelements gather.\n"
                      << "SLP: Current total cost = " << Cost << "\n");
    Cost -= InsertCost;
    for (int I = 2, E = FirstUsers.size(); I < E; ++I) {
      // Other elements - permutation of 2 vectors (the initial one and the
      // next Ith incoming vector).
      unsigned VF = ShuffleMask[I].size();
      for (unsigned Idx = 0; Idx < VF; ++Idx) {
        int Mask = ShuffleMask[I][Idx];
        if (Mask != UndefMaskElem)
          CombinedMask[Idx] = MaxVF + Mask;
        else if (CombinedMask[Idx] != UndefMaskElem)
          CombinedMask[Idx] = Idx;
      }
      for (unsigned Idx = VF; Idx < MaxVF; ++Idx)
        if (CombinedMask[Idx] != UndefMaskElem)
          CombinedMask[Idx] = Idx;
      InstructionCost C =
          TTI->getShuffleCost(TTI::SK_PermuteTwoSrc, VecTy, CombinedMask);
      LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                        << " for final shuffle of vector node and external "
                           "insertelement users "
                        << *VectorizableTree.front()->Scalars.front() << ".\n"
                        << "SLP: Current total cost = " << Cost << "\n");
      Cost += C;
      InstructionCost InsertCost = TTI->getScalarizationOverhead(
          cast<FixedVectorType>(FirstUsers[I]->getType()), DemandedElts[I],
          /*Insert*/ true, /*Extract*/ false);
      LLVM_DEBUG(dbgs() << "SLP: subtracting the cost " << InsertCost
                        << " for insertelements gather.\n"
                        << "SLP: Current total cost = " << Cost << "\n");
      Cost -= InsertCost;
    }
  }

#ifndef NDEBUG
  SmallString<256> Str;
  {
    raw_svector_ostream OS(Str);
    OS << "SLP: Spill Cost = " << SpillCost << ".\n"
       << "SLP: Extract Cost = " << ExtractCost << ".\n"
       << "SLP: Total Cost = " << Cost << ".\n";
  }
  LLVM_DEBUG(dbgs() << Str);
  if (ViewSLPTree)
    ViewGraph(this, "SLP" + F->getName(), false, Str);
#endif

  return Cost;
}

Optional<TargetTransformInfo::ShuffleKind>
BoUpSLP::isGatherShuffledEntry(const TreeEntry *TE, SmallVectorImpl<int> &Mask,
                               SmallVectorImpl<const TreeEntry *> &Entries) {
  // TODO: currently checking only for Scalars in the tree entry, need to count
  // reused elements too for better cost estimation.
  Mask.assign(TE->Scalars.size(), UndefMaskElem);
  Entries.clear();
  // Build a lists of values to tree entries.
  DenseMap<Value *, SmallPtrSet<const TreeEntry *, 4>> ValueToTEs;
  for (const std::unique_ptr<TreeEntry> &EntryPtr : VectorizableTree) {
    if (EntryPtr.get() == TE)
      break;
    if (EntryPtr->State != TreeEntry::NeedToGather)
      continue;
    for (Value *V : EntryPtr->Scalars)
      ValueToTEs.try_emplace(V).first->getSecond().insert(EntryPtr.get());
  }
  // Find all tree entries used by the gathered values. If no common entries
  // found - not a shuffle.
  // Here we build a set of tree nodes for each gathered value and trying to
  // find the intersection between these sets. If we have at least one common
  // tree node for each gathered value - we have just a permutation of the
  // single vector. If we have 2 different sets, we're in situation where we
  // have a permutation of 2 input vectors.
  SmallVector<SmallPtrSet<const TreeEntry *, 4>> UsedTEs;
  DenseMap<Value *, int> UsedValuesEntry;
  for (Value *V : TE->Scalars) {
    if (isa<UndefValue>(V))
      continue;
    // Build a list of tree entries where V is used.
    SmallPtrSet<const TreeEntry *, 4> VToTEs;
    auto It = ValueToTEs.find(V);
    if (It != ValueToTEs.end())
      VToTEs = It->second;
    if (const TreeEntry *VTE = getTreeEntry(V))
      VToTEs.insert(VTE);
    if (VToTEs.empty())
      return None;
    if (UsedTEs.empty()) {
      // The first iteration, just insert the list of nodes to vector.
      UsedTEs.push_back(VToTEs);
    } else {
      // Need to check if there are any previously used tree nodes which use V.
      // If there are no such nodes, consider that we have another one input
      // vector.
      SmallPtrSet<const TreeEntry *, 4> SavedVToTEs(VToTEs);
      unsigned Idx = 0;
      for (SmallPtrSet<const TreeEntry *, 4> &Set : UsedTEs) {
        // Do we have a non-empty intersection of previously listed tree entries
        // and tree entries using current V?
        set_intersect(VToTEs, Set);
        if (!VToTEs.empty()) {
          // Yes, write the new subset and continue analysis for the next
          // scalar.
          Set.swap(VToTEs);
          break;
        }
        VToTEs = SavedVToTEs;
        ++Idx;
      }
      // No non-empty intersection found - need to add a second set of possible
      // source vectors.
      if (Idx == UsedTEs.size()) {
        // If the number of input vectors is greater than 2 - not a permutation,
        // fallback to the regular gather.
        if (UsedTEs.size() == 2)
          return None;
        UsedTEs.push_back(SavedVToTEs);
        Idx = UsedTEs.size() - 1;
      }
      UsedValuesEntry.try_emplace(V, Idx);
    }
  }

  unsigned VF = 0;
  if (UsedTEs.size() == 1) {
    // Try to find the perfect match in another gather node at first.
    auto It = find_if(UsedTEs.front(), [TE](const TreeEntry *EntryPtr) {
      return EntryPtr->isSame(TE->Scalars);
    });
    if (It != UsedTEs.front().end()) {
      Entries.push_back(*It);
      std::iota(Mask.begin(), Mask.end(), 0);
      return TargetTransformInfo::SK_PermuteSingleSrc;
    }
    // No perfect match, just shuffle, so choose the first tree node.
    Entries.push_back(*UsedTEs.front().begin());
  } else {
    // Try to find nodes with the same vector factor.
    assert(UsedTEs.size() == 2 && "Expected at max 2 permuted entries.");
    DenseMap<int, const TreeEntry *> VFToTE;
    for (const TreeEntry *TE : UsedTEs.front())
      VFToTE.try_emplace(TE->getVectorFactor(), TE);
    for (const TreeEntry *TE : UsedTEs.back()) {
      auto It = VFToTE.find(TE->getVectorFactor());
      if (It != VFToTE.end()) {
        VF = It->first;
        Entries.push_back(It->second);
        Entries.push_back(TE);
        break;
      }
    }
    // No 2 source vectors with the same vector factor - give up and do regular
    // gather.
    if (Entries.empty())
      return None;
  }

  // Build a shuffle mask for better cost estimation and vector emission.
  for (int I = 0, E = TE->Scalars.size(); I < E; ++I) {
    Value *V = TE->Scalars[I];
    if (isa<UndefValue>(V))
      continue;
    unsigned Idx = UsedValuesEntry.lookup(V);
    const TreeEntry *VTE = Entries[Idx];
    int FoundLane = VTE->findLaneForValue(V);
    Mask[I] = Idx * VF + FoundLane;
    // Extra check required by isSingleSourceMaskImpl function (called by
    // ShuffleVectorInst::isSingleSourceMask).
    if (Mask[I] >= 2 * E)
      return None;
  }
  switch (Entries.size()) {
  case 1:
    return TargetTransformInfo::SK_PermuteSingleSrc;
  case 2:
    return TargetTransformInfo::SK_PermuteTwoSrc;
  default:
    break;
  }
  return None;
}

InstructionCost BoUpSLP::getGatherCost(FixedVectorType *Ty,
                                       const APInt &ShuffledIndices,
                                       bool NeedToShuffle) const {
  InstructionCost Cost =
      TTI->getScalarizationOverhead(Ty, ~ShuffledIndices, /*Insert*/ true,
                                    /*Extract*/ false);
  if (NeedToShuffle)
    Cost += TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, Ty);
  return Cost;
}

InstructionCost BoUpSLP::getGatherCost(ArrayRef<Value *> VL) const {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());
  bool DuplicateNonConst = false;
  // Find the cost of inserting/extracting values from the vector.
  // Check if the same elements are inserted several times and count them as
  // shuffle candidates.
  APInt ShuffledElements = APInt::getZero(VL.size());
  DenseSet<Value *> UniqueElements;
  // Iterate in reverse order to consider insert elements with the high cost.
  for (unsigned I = VL.size(); I > 0; --I) {
    unsigned Idx = I - 1;
    // No need to shuffle duplicates for constants.
    if (isConstant(VL[Idx])) {
      ShuffledElements.setBit(Idx);
      continue;
    }
    if (!UniqueElements.insert(VL[Idx]).second) {
      DuplicateNonConst = true;
      ShuffledElements.setBit(Idx);
    }
  }
  return getGatherCost(VecTy, ShuffledElements, DuplicateNonConst);
}

// Perform operand reordering on the instructions in VL and return the reordered
// operands in Left and Right.
void BoUpSLP::reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                             SmallVectorImpl<Value *> &Left,
                                             SmallVectorImpl<Value *> &Right,
                                             const DataLayout &DL,
                                             ScalarEvolution &SE,
                                             const BoUpSLP &R) {
  if (VL.empty())
    return;
  VLOperands Ops(VL, DL, SE, R);
  // Reorder the operands in place.
  Ops.reorder();
  Left = Ops.getVL(0);
  Right = Ops.getVL(1);
}

void BoUpSLP::setInsertPointAfterBundle(const TreeEntry *E) {
  // Get the basic block this bundle is in. All instructions in the bundle
  // should be in this block.
  auto *Front = E->getMainOp();
  auto *BB = Front->getParent();
  assert(llvm::all_of(E->Scalars, [=](Value *V) -> bool {
    auto *I = cast<Instruction>(V);
    return !E->isOpcodeOrAlt(I) || I->getParent() == BB;
  }));

  // The last instruction in the bundle in program order.
  Instruction *LastInst = nullptr;

  // Find the last instruction. The common case should be that BB has been
  // scheduled, and the last instruction is VL.back(). So we start with
  // VL.back() and iterate over schedule data until we reach the end of the
  // bundle. The end of the bundle is marked by null ScheduleData.
  if (BlocksSchedules.count(BB)) {
    auto *Bundle =
        BlocksSchedules[BB]->getScheduleData(E->isOneOf(E->Scalars.back()));
    if (Bundle && Bundle->isPartOfBundle())
      for (; Bundle; Bundle = Bundle->NextInBundle)
        if (Bundle->OpValue == Bundle->Inst)
          LastInst = Bundle->Inst;
  }

  // LastInst can still be null at this point if there's either not an entry
  // for BB in BlocksSchedules or there's no ScheduleData available for
  // VL.back(). This can be the case if buildTree_rec aborts for various
  // reasons (e.g., the maximum recursion depth is reached, the maximum region
  // size is reached, etc.). ScheduleData is initialized in the scheduling
  // "dry-run".
  //
  // If this happens, we can still find the last instruction by brute force. We
  // iterate forwards from Front (inclusive) until we either see all
  // instructions in the bundle or reach the end of the block. If Front is the
  // last instruction in program order, LastInst will be set to Front, and we
  // will visit all the remaining instructions in the block.
  //
  // One of the reasons we exit early from buildTree_rec is to place an upper
  // bound on compile-time. Thus, taking an additional compile-time hit here is
  // not ideal. However, this should be exceedingly rare since it requires that
  // we both exit early from buildTree_rec and that the bundle be out-of-order
  // (causing us to iterate all the way to the end of the block).
  if (!LastInst) {
    SmallPtrSet<Value *, 16> Bundle(E->Scalars.begin(), E->Scalars.end());
    for (auto &I : make_range(BasicBlock::iterator(Front), BB->end())) {
      if (Bundle.erase(&I) && E->isOpcodeOrAlt(&I))
        LastInst = &I;
      if (Bundle.empty())
        break;
    }
  }
  assert(LastInst && "Failed to find last instruction in bundle");

  // Set the insertion point after the last instruction in the bundle. Set the
  // debug location to Front.
  Builder.SetInsertPoint(BB, ++LastInst->getIterator());
  Builder.SetCurrentDebugLocation(Front->getDebugLoc());
}

Value *BoUpSLP::gather(ArrayRef<Value *> VL) {
  // List of instructions/lanes from current block and/or the blocks which are
  // part of the current loop. These instructions will be inserted at the end to
  // make it possible to optimize loops and hoist invariant instructions out of
  // the loops body with better chances for success.
  SmallVector<std::pair<Value *, unsigned>, 4> PostponedInsts;
  SmallSet<int, 4> PostponedIndices;
  Loop *L = LI->getLoopFor(Builder.GetInsertBlock());
  auto &&CheckPredecessor = [](BasicBlock *InstBB, BasicBlock *InsertBB) {
    SmallPtrSet<BasicBlock *, 4> Visited;
    while (InsertBB && InsertBB != InstBB && Visited.insert(InsertBB).second)
      InsertBB = InsertBB->getSinglePredecessor();
    return InsertBB && InsertBB == InstBB;
  };
  for (int I = 0, E = VL.size(); I < E; ++I) {
    if (auto *Inst = dyn_cast<Instruction>(VL[I]))
      if ((CheckPredecessor(Inst->getParent(), Builder.GetInsertBlock()) ||
           getTreeEntry(Inst) || (L && (L->contains(Inst)))) &&
          PostponedIndices.insert(I).second)
        PostponedInsts.emplace_back(Inst, I);
  }

  auto &&CreateInsertElement = [this](Value *Vec, Value *V, unsigned Pos) {
    Vec = Builder.CreateInsertElement(Vec, V, Builder.getInt32(Pos));
    auto *InsElt = dyn_cast<InsertElementInst>(Vec);
    if (!InsElt)
      return Vec;
    GatherShuffleSeq.insert(InsElt);
    CSEBlocks.insert(InsElt->getParent());
    // Add to our 'need-to-extract' list.
    if (TreeEntry *Entry = getTreeEntry(V)) {
      // Find which lane we need to extract.
      unsigned FoundLane = Entry->findLaneForValue(V);
      ExternalUses.emplace_back(V, InsElt, FoundLane);
    }
    return Vec;
  };
  Value *Val0 =
      isa<StoreInst>(VL[0]) ? cast<StoreInst>(VL[0])->getValueOperand() : VL[0];
  FixedVectorType *VecTy = FixedVectorType::get(Val0->getType(), VL.size());
  Value *Vec = PoisonValue::get(VecTy);
  SmallVector<int> NonConsts;
  // Insert constant values at first.
  for (int I = 0, E = VL.size(); I < E; ++I) {
    if (PostponedIndices.contains(I))
      continue;
    if (!isConstant(VL[I])) {
      NonConsts.push_back(I);
      continue;
    }
    Vec = CreateInsertElement(Vec, VL[I], I);
  }
  // Insert non-constant values.
  for (int I : NonConsts)
    Vec = CreateInsertElement(Vec, VL[I], I);
  // Append instructions, which are/may be part of the loop, in the end to make
  // it possible to hoist non-loop-based instructions.
  for (const std::pair<Value *, unsigned> &Pair : PostponedInsts)
    Vec = CreateInsertElement(Vec, Pair.first, Pair.second);

  return Vec;
}

namespace {
/// Merges shuffle masks and emits final shuffle instruction, if required.
class ShuffleInstructionBuilder {
  IRBuilderBase &Builder;
  const unsigned VF = 0;
  bool IsFinalized = false;
  SmallVector<int, 4> Mask;
  /// Holds all of the instructions that we gathered.
  SetVector<Instruction *> &GatherShuffleSeq;
  /// A list of blocks that we are going to CSE.
  SetVector<BasicBlock *> &CSEBlocks;

public:
  ShuffleInstructionBuilder(IRBuilderBase &Builder, unsigned VF,
                            SetVector<Instruction *> &GatherShuffleSeq,
                            SetVector<BasicBlock *> &CSEBlocks)
      : Builder(Builder), VF(VF), GatherShuffleSeq(GatherShuffleSeq),
        CSEBlocks(CSEBlocks) {}

  /// Adds a mask, inverting it before applying.
  void addInversedMask(ArrayRef<unsigned> SubMask) {
    if (SubMask.empty())
      return;
    SmallVector<int, 4> NewMask;
    inversePermutation(SubMask, NewMask);
    addMask(NewMask);
  }

  /// Functions adds masks, merging them into  single one.
  void addMask(ArrayRef<unsigned> SubMask) {
    SmallVector<int, 4> NewMask(SubMask.begin(), SubMask.end());
    addMask(NewMask);
  }

  void addMask(ArrayRef<int> SubMask) { ::addMask(Mask, SubMask); }

  Value *finalize(Value *V) {
    IsFinalized = true;
    unsigned ValueVF = cast<FixedVectorType>(V->getType())->getNumElements();
    if (VF == ValueVF && Mask.empty())
      return V;
    SmallVector<int, 4> NormalizedMask(VF, UndefMaskElem);
    std::iota(NormalizedMask.begin(), NormalizedMask.end(), 0);
    addMask(NormalizedMask);

    if (VF == ValueVF && ShuffleVectorInst::isIdentityMask(Mask))
      return V;
    Value *Vec = Builder.CreateShuffleVector(V, Mask, "shuffle");
    if (auto *I = dyn_cast<Instruction>(Vec)) {
      GatherShuffleSeq.insert(I);
      CSEBlocks.insert(I->getParent());
    }
    return Vec;
  }

  ~ShuffleInstructionBuilder() {
    assert((IsFinalized || Mask.empty()) &&
           "Shuffle construction must be finalized.");
  }
};
} // namespace

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL) {
  unsigned VF = VL.size();
  InstructionsState S = getSameOpcode(VL);
  if (S.getOpcode()) {
    if (TreeEntry *E = getTreeEntry(S.OpValue))
      if (E->isSame(VL)) {
        Value *V = vectorizeTree(E);
        if (VF != cast<FixedVectorType>(V->getType())->getNumElements()) {
          if (!E->ReuseShuffleIndices.empty()) {
            // Reshuffle to get only unique values.
            // If some of the scalars are duplicated in the vectorization tree
            // entry, we do not vectorize them but instead generate a mask for
            // the reuses. But if there are several users of the same entry,
            // they may have different vectorization factors. This is especially
            // important for PHI nodes. In this case, we need to adapt the
            // resulting instruction for the user vectorization factor and have
            // to reshuffle it again to take only unique elements of the vector.
            // Without this code the function incorrectly returns reduced vector
            // instruction with the same elements, not with the unique ones.

            // block:
            // %phi = phi <2 x > { .., %entry} {%shuffle, %block}
            // %2 = shuffle <2 x > %phi, poison, <4 x > <1, 1, 0, 0>
            // ... (use %2)
            // %shuffle = shuffle <2 x> %2, poison, <2 x> {2, 0}
            // br %block
            SmallVector<int> UniqueIdxs(VF, UndefMaskElem);
            SmallSet<int, 4> UsedIdxs;
            int Pos = 0;
            int Sz = VL.size();
            for (int Idx : E->ReuseShuffleIndices) {
              if (Idx != Sz && Idx != UndefMaskElem &&
                  UsedIdxs.insert(Idx).second)
                UniqueIdxs[Idx] = Pos;
              ++Pos;
            }
            assert(VF >= UsedIdxs.size() && "Expected vectorization factor "
                                            "less than original vector size.");
            UniqueIdxs.append(VF - UsedIdxs.size(), UndefMaskElem);
            V = Builder.CreateShuffleVector(V, UniqueIdxs, "shrink.shuffle");
          } else {
            assert(VF < cast<FixedVectorType>(V->getType())->getNumElements() &&
                   "Expected vectorization factor less "
                   "than original vector size.");
            SmallVector<int> UniformMask(VF, 0);
            std::iota(UniformMask.begin(), UniformMask.end(), 0);
            V = Builder.CreateShuffleVector(V, UniformMask, "shrink.shuffle");
          }
          if (auto *I = dyn_cast<Instruction>(V)) {
            GatherShuffleSeq.insert(I);
            CSEBlocks.insert(I->getParent());
          }
        }
        return V;
      }
  }

  // Check that every instruction appears once in this bundle.
  SmallVector<int> ReuseShuffleIndicies;
  SmallVector<Value *> UniqueValues;
  if (VL.size() > 2) {
    DenseMap<Value *, unsigned> UniquePositions;
    unsigned NumValues =
        std::distance(VL.begin(), find_if(reverse(VL), [](Value *V) {
                                    return !isa<UndefValue>(V);
                                  }).base());
    VF = std::max<unsigned>(VF, PowerOf2Ceil(NumValues));
    int UniqueVals = 0;
    for (Value *V : VL.drop_back(VL.size() - VF)) {
      if (isa<UndefValue>(V)) {
        ReuseShuffleIndicies.emplace_back(UndefMaskElem);
        continue;
      }
      if (isConstant(V)) {
        ReuseShuffleIndicies.emplace_back(UniqueValues.size());
        UniqueValues.emplace_back(V);
        continue;
      }
      auto Res = UniquePositions.try_emplace(V, UniqueValues.size());
      ReuseShuffleIndicies.emplace_back(Res.first->second);
      if (Res.second) {
        UniqueValues.emplace_back(V);
        ++UniqueVals;
      }
    }
    if (UniqueVals == 1 && UniqueValues.size() == 1) {
      // Emit pure splat vector.
      ReuseShuffleIndicies.append(VF - ReuseShuffleIndicies.size(),
                                  UndefMaskElem);
    } else if (UniqueValues.size() >= VF - 1 || UniqueValues.size() <= 1) {
      ReuseShuffleIndicies.clear();
      UniqueValues.clear();
      UniqueValues.append(VL.begin(), std::next(VL.begin(), NumValues));
    }
    UniqueValues.append(VF - UniqueValues.size(),
                        PoisonValue::get(VL[0]->getType()));
    VL = UniqueValues;
  }

  ShuffleInstructionBuilder ShuffleBuilder(Builder, VF, GatherShuffleSeq,
                                           CSEBlocks);
  Value *Vec = gather(VL);
  if (!ReuseShuffleIndicies.empty()) {
    ShuffleBuilder.addMask(ReuseShuffleIndicies);
    Vec = ShuffleBuilder.finalize(Vec);
  }
  return Vec;
}

Value *BoUpSLP::vectorizeTree(TreeEntry *E) {
  IRBuilder<>::InsertPointGuard Guard(Builder);

  if (E->VectorizedValue) {
    LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *E->Scalars[0] << ".\n");
    return E->VectorizedValue;
  }

  bool NeedToShuffleReuses = !E->ReuseShuffleIndices.empty();
  unsigned VF = E->getVectorFactor();
  ShuffleInstructionBuilder ShuffleBuilder(Builder, VF, GatherShuffleSeq,
                                           CSEBlocks);
  if (E->State == TreeEntry::NeedToGather) {
    if (E->getMainOp())
      setInsertPointAfterBundle(E);
    Value *Vec;
    SmallVector<int> Mask;
    SmallVector<const TreeEntry *> Entries;
    Optional<TargetTransformInfo::ShuffleKind> Shuffle =
        isGatherShuffledEntry(E, Mask, Entries);
    if (Shuffle.hasValue()) {
      assert((Entries.size() == 1 || Entries.size() == 2) &&
             "Expected shuffle of 1 or 2 entries.");
      Vec = Builder.CreateShuffleVector(Entries.front()->VectorizedValue,
                                        Entries.back()->VectorizedValue, Mask);
      if (auto *I = dyn_cast<Instruction>(Vec)) {
        GatherShuffleSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }
    } else {
      Vec = gather(E->Scalars);
    }
    if (NeedToShuffleReuses) {
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      Vec = ShuffleBuilder.finalize(Vec);
    }
    E->VectorizedValue = Vec;
    return Vec;
  }

  assert((E->State == TreeEntry::Vectorize ||
          E->State == TreeEntry::ScatterVectorize) &&
         "Unhandled state");
  unsigned ShuffleOrOp =
      E->isAltShuffle() ? (unsigned)Instruction::ShuffleVector : E->getOpcode();
  Instruction *VL0 = E->getMainOp();
  Type *ScalarTy = VL0->getType();
  if (auto *Store = dyn_cast<StoreInst>(VL0))
    ScalarTy = Store->getValueOperand()->getType();
  else if (auto *IE = dyn_cast<InsertElementInst>(VL0))
    ScalarTy = IE->getOperand(1)->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, E->Scalars.size());
  switch (ShuffleOrOp) {
    case Instruction::PHI: {
      assert(
          (E->ReorderIndices.empty() || E != VectorizableTree.front().get()) &&
          "PHI reordering is free.");
      auto *PH = cast<PHINode>(VL0);
      Builder.SetInsertPoint(PH->getParent()->getFirstNonPHI());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());
      PHINode *NewPhi = Builder.CreatePHI(VecTy, PH->getNumIncomingValues());
      Value *V = NewPhi;

      // Adjust insertion point once all PHI's have been generated.
      Builder.SetInsertPoint(&*PH->getParent()->getFirstInsertionPt());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());

      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;

      // PHINodes may have multiple entries from the same block. We want to
      // visit every block once.
      SmallPtrSet<BasicBlock*, 4> VisitedBBs;

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        BasicBlock *IBB = PH->getIncomingBlock(i);

        if (!VisitedBBs.insert(IBB).second) {
          NewPhi->addIncoming(NewPhi->getIncomingValueForBlock(IBB), IBB);
          continue;
        }

        Builder.SetInsertPoint(IBB->getTerminator());
        Builder.SetCurrentDebugLocation(PH->getDebugLoc());
        Value *Vec = vectorizeTree(E->getOperand(i));
        NewPhi->addIncoming(Vec, IBB);
      }

      assert(NewPhi->getNumIncomingValues() == PH->getNumIncomingValues() &&
             "Invalid number of incoming values");
      return V;
    }

    case Instruction::ExtractElement: {
      Value *V = E->getSingleOperand(0);
      Builder.SetInsertPoint(VL0);
      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::ExtractValue: {
      auto *LI = cast<LoadInst>(E->getSingleOperand(0));
      Builder.SetInsertPoint(LI);
      auto *PtrTy = PointerType::get(VecTy, LI->getPointerAddressSpace());
      Value *Ptr = Builder.CreateBitCast(LI->getOperand(0), PtrTy);
      LoadInst *V = Builder.CreateAlignedLoad(VecTy, Ptr, LI->getAlign());
      Value *NewV = propagateMetadata(V, E->Scalars);
      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      NewV = ShuffleBuilder.finalize(NewV);
      E->VectorizedValue = NewV;
      return NewV;
    }
    case Instruction::InsertElement: {
      assert(E->ReuseShuffleIndices.empty() && "All inserts should be unique");
      Builder.SetInsertPoint(cast<Instruction>(E->Scalars.back()));
      Value *V = vectorizeTree(E->getOperand(1));

      // Create InsertVector shuffle if necessary
      auto *FirstInsert = cast<Instruction>(*find_if(E->Scalars, [E](Value *V) {
        return !is_contained(E->Scalars, cast<Instruction>(V)->getOperand(0));
      }));
      const unsigned NumElts =
          cast<FixedVectorType>(FirstInsert->getType())->getNumElements();
      const unsigned NumScalars = E->Scalars.size();

      unsigned Offset = *getInsertIndex(VL0);
      assert(Offset < NumElts && "Failed to find vector index offset");

      // Create shuffle to resize vector
      SmallVector<int> Mask;
      if (!E->ReorderIndices.empty()) {
        inversePermutation(E->ReorderIndices, Mask);
        Mask.append(NumElts - NumScalars, UndefMaskElem);
      } else {
        Mask.assign(NumElts, UndefMaskElem);
        std::iota(Mask.begin(), std::next(Mask.begin(), NumScalars), 0);
      }
      // Create InsertVector shuffle if necessary
      bool IsIdentity = true;
      SmallVector<int> PrevMask(NumElts, UndefMaskElem);
      Mask.swap(PrevMask);
      for (unsigned I = 0; I < NumScalars; ++I) {
        Value *Scalar = E->Scalars[PrevMask[I]];
        unsigned InsertIdx = *getInsertIndex(Scalar);
        IsIdentity &= InsertIdx - Offset == I;
        Mask[InsertIdx - Offset] = I;
      }
      if (!IsIdentity || NumElts != NumScalars) {
        V = Builder.CreateShuffleVector(V, Mask);
        if (auto *I = dyn_cast<Instruction>(V)) {
          GatherShuffleSeq.insert(I);
          CSEBlocks.insert(I->getParent());
        }
      }

      if ((!IsIdentity || Offset != 0 ||
           !isUndefVector(FirstInsert->getOperand(0))) &&
          NumElts != NumScalars) {
        SmallVector<int> InsertMask(NumElts);
        std::iota(InsertMask.begin(), InsertMask.end(), 0);
        for (unsigned I = 0; I < NumElts; I++) {
          if (Mask[I] != UndefMaskElem)
            InsertMask[Offset + I] = NumElts + I;
        }

        V = Builder.CreateShuffleVector(
            FirstInsert->getOperand(0), V, InsertMask,
            cast<Instruction>(E->Scalars.back())->getName());
        if (auto *I = dyn_cast<Instruction>(V)) {
          GatherShuffleSeq.insert(I);
          CSEBlocks.insert(I->getParent());
        }
      }

      ++NumVectorInstructions;
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      setInsertPointAfterBundle(E);

      Value *InVec = vectorizeTree(E->getOperand(0));

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      auto *CI = cast<CastInst>(VL0);
      Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::FCmp:
    case Instruction::ICmp: {
      setInsertPointAfterBundle(E);

      Value *L = vectorizeTree(E->getOperand(0));
      Value *R = vectorizeTree(E->getOperand(1));

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      Value *V = Builder.CreateCmp(P0, L, R);
      propagateIRFlags(V, E->Scalars, VL0);
      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Select: {
      setInsertPointAfterBundle(E);

      Value *Cond = vectorizeTree(E->getOperand(0));
      Value *True = vectorizeTree(E->getOperand(1));
      Value *False = vectorizeTree(E->getOperand(2));

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      Value *V = Builder.CreateSelect(Cond, True, False);
      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::FNeg: {
      setInsertPointAfterBundle(E);

      Value *Op = vectorizeTree(E->getOperand(0));

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      Value *V = Builder.CreateUnOp(
          static_cast<Instruction::UnaryOps>(E->getOpcode()), Op);
      propagateIRFlags(V, E->Scalars, VL0);
      if (auto *I = dyn_cast<Instruction>(V))
        V = propagateMetadata(I, E->Scalars);

      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;

      return V;
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      setInsertPointAfterBundle(E);

      Value *LHS = vectorizeTree(E->getOperand(0));
      Value *RHS = vectorizeTree(E->getOperand(1));

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      Value *V = Builder.CreateBinOp(
          static_cast<Instruction::BinaryOps>(E->getOpcode()), LHS,
          RHS);
      propagateIRFlags(V, E->Scalars, VL0);
      if (auto *I = dyn_cast<Instruction>(V))
        V = propagateMetadata(I, E->Scalars);

      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;

      return V;
    }
    case Instruction::Load: {
      // Loads are inserted at the head of the tree because we don't want to
      // sink them all the way down past store instructions.
      setInsertPointAfterBundle(E);

      LoadInst *LI = cast<LoadInst>(VL0);
      Instruction *NewLI;
      unsigned AS = LI->getPointerAddressSpace();
      Value *PO = LI->getPointerOperand();
      if (E->State == TreeEntry::Vectorize) {

        Value *VecPtr = Builder.CreateBitCast(PO, VecTy->getPointerTo(AS));

        // The pointer operand uses an in-tree scalar so we add the new BitCast
        // to ExternalUses list to make sure that an extract will be generated
        // in the future.
        if (TreeEntry *Entry = getTreeEntry(PO)) {
          // Find which lane we need to extract.
          unsigned FoundLane = Entry->findLaneForValue(PO);
          ExternalUses.emplace_back(PO, cast<User>(VecPtr), FoundLane);
        }

        NewLI = Builder.CreateAlignedLoad(VecTy, VecPtr, LI->getAlign());
      } else {
        assert(E->State == TreeEntry::ScatterVectorize && "Unhandled state");
        Value *VecPtr = vectorizeTree(E->getOperand(0));
        // Use the minimum alignment of the gathered loads.
        Align CommonAlignment = LI->getAlign();
        for (Value *V : E->Scalars)
          CommonAlignment =
              commonAlignment(CommonAlignment, cast<LoadInst>(V)->getAlign());
        NewLI = Builder.CreateMaskedGather(VecTy, VecPtr, CommonAlignment);
      }
      Value *V = propagateMetadata(NewLI, E->Scalars);

      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Store: {
      auto *SI = cast<StoreInst>(VL0);
      unsigned AS = SI->getPointerAddressSpace();

      setInsertPointAfterBundle(E);

      Value *VecValue = vectorizeTree(E->getOperand(0));
      ShuffleBuilder.addMask(E->ReorderIndices);
      VecValue = ShuffleBuilder.finalize(VecValue);

      Value *ScalarPtr = SI->getPointerOperand();
      Value *VecPtr = Builder.CreateBitCast(
          ScalarPtr, VecValue->getType()->getPointerTo(AS));
      StoreInst *ST = Builder.CreateAlignedStore(VecValue, VecPtr,
                                                 SI->getAlign());

      // The pointer operand uses an in-tree scalar, so add the new BitCast to
      // ExternalUses to make sure that an extract will be generated in the
      // future.
      if (TreeEntry *Entry = getTreeEntry(ScalarPtr)) {
        // Find which lane we need to extract.
        unsigned FoundLane = Entry->findLaneForValue(ScalarPtr);
        ExternalUses.push_back(
            ExternalUser(ScalarPtr, cast<User>(VecPtr), FoundLane));
      }

      Value *V = propagateMetadata(ST, E->Scalars);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::GetElementPtr: {
      auto *GEP0 = cast<GetElementPtrInst>(VL0);
      setInsertPointAfterBundle(E);

      Value *Op0 = vectorizeTree(E->getOperand(0));

      SmallVector<Value *> OpVecs;
      for (int J = 1, N = GEP0->getNumOperands(); J < N; ++J) {
        Value *OpVec = vectorizeTree(E->getOperand(J));
        OpVecs.push_back(OpVec);
      }

      Value *V = Builder.CreateGEP(GEP0->getSourceElementType(), Op0, OpVecs);
      if (Instruction *I = dyn_cast<Instruction>(V))
        V = propagateMetadata(I, E->Scalars);

      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;

      return V;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      setInsertPointAfterBundle(E);

      Intrinsic::ID IID  = Intrinsic::not_intrinsic;
      if (Function *FI = CI->getCalledFunction())
        IID = FI->getIntrinsicID();

      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      auto VecCallCosts = getVectorCallCosts(CI, VecTy, TTI, TLI);
      bool UseIntrinsic = ID != Intrinsic::not_intrinsic &&
                          VecCallCosts.first <= VecCallCosts.second;

      Value *ScalarArg = nullptr;
      std::vector<Value *> OpVecs;
      SmallVector<Type *, 2> TysForDecl =
          {FixedVectorType::get(CI->getType(), E->Scalars.size())};
      for (int j = 0, e = CI->arg_size(); j < e; ++j) {
        ValueList OpVL;
        // Some intrinsics have scalar arguments. This argument should not be
        // vectorized.
        if (UseIntrinsic && hasVectorInstrinsicScalarOpd(IID, j)) {
          CallInst *CEI = cast<CallInst>(VL0);
          ScalarArg = CEI->getArgOperand(j);
          OpVecs.push_back(CEI->getArgOperand(j));
          if (hasVectorInstrinsicOverloadedScalarOpd(IID, j))
            TysForDecl.push_back(ScalarArg->getType());
          continue;
        }

        Value *OpVec = vectorizeTree(E->getOperand(j));
        LLVM_DEBUG(dbgs() << "SLP: OpVec[" << j << "]: " << *OpVec << "\n");
        OpVecs.push_back(OpVec);
      }

      Function *CF;
      if (!UseIntrinsic) {
        VFShape Shape =
            VFShape::get(*CI, ElementCount::getFixed(static_cast<unsigned>(
                                  VecTy->getNumElements())),
                         false /*HasGlobalPred*/);
        CF = VFDatabase(*CI).getVectorizedFunction(Shape);
      } else {
        CF = Intrinsic::getDeclaration(F->getParent(), ID, TysForDecl);
      }

      SmallVector<OperandBundleDef, 1> OpBundles;
      CI->getOperandBundlesAsDefs(OpBundles);
      Value *V = Builder.CreateCall(CF, OpVecs, OpBundles);

      // The scalar argument uses an in-tree scalar so we add the new vectorized
      // call to ExternalUses list to make sure that an extract will be
      // generated in the future.
      if (ScalarArg) {
        if (TreeEntry *Entry = getTreeEntry(ScalarArg)) {
          // Find which lane we need to extract.
          unsigned FoundLane = Entry->findLaneForValue(ScalarArg);
          ExternalUses.push_back(
              ExternalUser(ScalarArg, cast<User>(V), FoundLane));
        }
      }

      propagateIRFlags(V, E->Scalars, VL0);
      ShuffleBuilder.addInversedMask(E->ReorderIndices);
      ShuffleBuilder.addMask(E->ReuseShuffleIndices);
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::ShuffleVector: {
      assert(E->isAltShuffle() &&
             ((Instruction::isBinaryOp(E->getOpcode()) &&
               Instruction::isBinaryOp(E->getAltOpcode())) ||
              (Instruction::isCast(E->getOpcode()) &&
               Instruction::isCast(E->getAltOpcode())) ||
              (isa<CmpInst>(VL0) && isa<CmpInst>(E->getAltOp()))) &&
             "Invalid Shuffle Vector Operand");

      Value *LHS = nullptr, *RHS = nullptr;
      if (Instruction::isBinaryOp(E->getOpcode()) || isa<CmpInst>(VL0)) {
        setInsertPointAfterBundle(E);
        LHS = vectorizeTree(E->getOperand(0));
        RHS = vectorizeTree(E->getOperand(1));
      } else {
        setInsertPointAfterBundle(E);
        LHS = vectorizeTree(E->getOperand(0));
      }

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      Value *V0, *V1;
      if (Instruction::isBinaryOp(E->getOpcode())) {
        V0 = Builder.CreateBinOp(
            static_cast<Instruction::BinaryOps>(E->getOpcode()), LHS, RHS);
        V1 = Builder.CreateBinOp(
            static_cast<Instruction::BinaryOps>(E->getAltOpcode()), LHS, RHS);
      } else if (auto *CI0 = dyn_cast<CmpInst>(VL0)) {
        V0 = Builder.CreateCmp(CI0->getPredicate(), LHS, RHS);
        auto *AltCI = cast<CmpInst>(E->getAltOp());
        CmpInst::Predicate AltPred = AltCI->getPredicate();
        V1 = Builder.CreateCmp(AltPred, LHS, RHS);
      } else {
        V0 = Builder.CreateCast(
            static_cast<Instruction::CastOps>(E->getOpcode()), LHS, VecTy);
        V1 = Builder.CreateCast(
            static_cast<Instruction::CastOps>(E->getAltOpcode()), LHS, VecTy);
      }
      // Add V0 and V1 to later analysis to try to find and remove matching
      // instruction, if any.
      for (Value *V : {V0, V1}) {
        if (auto *I = dyn_cast<Instruction>(V)) {
          GatherShuffleSeq.insert(I);
          CSEBlocks.insert(I->getParent());
        }
      }

      // Create shuffle to take alternate operations from the vector.
      // Also, gather up main and alt scalar ops to propagate IR flags to
      // each vector operation.
      ValueList OpScalars, AltScalars;
      SmallVector<int> Mask;
      buildShuffleEntryMask(
          E->Scalars, E->ReorderIndices, E->ReuseShuffleIndices,
          [E](Instruction *I) {
            assert(E->isOpcodeOrAlt(I) && "Unexpected main/alternate opcode");
            return isAlternateInstruction(I, E->getMainOp(), E->getAltOp());
          },
          Mask, &OpScalars, &AltScalars);

      propagateIRFlags(V0, OpScalars);
      propagateIRFlags(V1, AltScalars);

      Value *V = Builder.CreateShuffleVector(V0, V1, Mask);
      if (auto *I = dyn_cast<Instruction>(V)) {
        V = propagateMetadata(I, E->Scalars);
        GatherShuffleSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }
      V = ShuffleBuilder.finalize(V);

      E->VectorizedValue = V;
      ++NumVectorInstructions;

      return V;
    }
    default:
    llvm_unreachable("unknown inst");
  }
  return nullptr;
}

Value *BoUpSLP::vectorizeTree() {
  ExtraValueToDebugLocsMap ExternallyUsedValues;
  return vectorizeTree(ExternallyUsedValues);
}

Value *
BoUpSLP::vectorizeTree(ExtraValueToDebugLocsMap &ExternallyUsedValues) {
  // All blocks must be scheduled before any instructions are inserted.
  for (auto &BSIter : BlocksSchedules) {
    scheduleBlock(BSIter.second.get());
  }

  Builder.SetInsertPoint(&F->getEntryBlock().front());
  auto *VectorRoot = vectorizeTree(VectorizableTree[0].get());

  // If the vectorized tree can be rewritten in a smaller type, we truncate the
  // vectorized root. InstCombine will then rewrite the entire expression. We
  // sign extend the extracted values below.
  auto *ScalarRoot = VectorizableTree[0]->Scalars[0];
  if (MinBWs.count(ScalarRoot)) {
    if (auto *I = dyn_cast<Instruction>(VectorRoot)) {
      // If current instr is a phi and not the last phi, insert it after the
      // last phi node.
      if (isa<PHINode>(I))
        Builder.SetInsertPoint(&*I->getParent()->getFirstInsertionPt());
      else
        Builder.SetInsertPoint(&*++BasicBlock::iterator(I));
    }
    auto BundleWidth = VectorizableTree[0]->Scalars.size();
    auto *MinTy = IntegerType::get(F->getContext(), MinBWs[ScalarRoot].first);
    auto *VecTy = FixedVectorType::get(MinTy, BundleWidth);
    auto *Trunc = Builder.CreateTrunc(VectorRoot, VecTy);
    VectorizableTree[0]->VectorizedValue = Trunc;
  }

  LLVM_DEBUG(dbgs() << "SLP: Extracting " << ExternalUses.size()
                    << " values .\n");

  // Extract all of the elements with the external uses.
  for (const auto &ExternalUse : ExternalUses) {
    Value *Scalar = ExternalUse.Scalar;
    llvm::User *User = ExternalUse.User;

    // Skip users that we already RAUW. This happens when one instruction
    // has multiple uses of the same value.
    if (User && !is_contained(Scalar->users(), User))
      continue;
    TreeEntry *E = getTreeEntry(Scalar);
    assert(E && "Invalid scalar");
    assert(E->State != TreeEntry::NeedToGather &&
           "Extracting from a gather list");

    Value *Vec = E->VectorizedValue;
    assert(Vec && "Can't find vectorizable value");

    Value *Lane = Builder.getInt32(ExternalUse.Lane);
    auto ExtractAndExtendIfNeeded = [&](Value *Vec) {
      if (Scalar->getType() != Vec->getType()) {
        Value *Ex;
        // "Reuse" the existing extract to improve final codegen.
        if (auto *ES = dyn_cast<ExtractElementInst>(Scalar)) {
          Ex = Builder.CreateExtractElement(ES->getOperand(0),
                                            ES->getOperand(1));
        } else {
          Ex = Builder.CreateExtractElement(Vec, Lane);
        }
        // If necessary, sign-extend or zero-extend ScalarRoot
        // to the larger type.
        if (!MinBWs.count(ScalarRoot))
          return Ex;
        if (MinBWs[ScalarRoot].second)
          return Builder.CreateSExt(Ex, Scalar->getType());
        return Builder.CreateZExt(Ex, Scalar->getType());
      }
      assert(isa<FixedVectorType>(Scalar->getType()) &&
             isa<InsertElementInst>(Scalar) &&
             "In-tree scalar of vector type is not insertelement?");
      return Vec;
    };
    // If User == nullptr, the Scalar is used as extra arg. Generate
    // ExtractElement instruction and update the record for this scalar in
    // ExternallyUsedValues.
    if (!User) {
      assert(ExternallyUsedValues.count(Scalar) &&
             "Scalar with nullptr as an external user must be registered in "
             "ExternallyUsedValues map");
      if (auto *VecI = dyn_cast<Instruction>(Vec)) {
        Builder.SetInsertPoint(VecI->getParent(),
                               std::next(VecI->getIterator()));
      } else {
        Builder.SetInsertPoint(&F->getEntryBlock().front());
      }
      Value *NewInst = ExtractAndExtendIfNeeded(Vec);
      CSEBlocks.insert(cast<Instruction>(Scalar)->getParent());
      auto &NewInstLocs = ExternallyUsedValues[NewInst];
      auto It = ExternallyUsedValues.find(Scalar);
      assert(It != ExternallyUsedValues.end() &&
             "Externally used scalar is not found in ExternallyUsedValues");
      NewInstLocs.append(It->second);
      ExternallyUsedValues.erase(Scalar);
      // Required to update internally referenced instructions.
      Scalar->replaceAllUsesWith(NewInst);
      continue;
    }

    // Generate extracts for out-of-tree users.
    // Find the insertion point for the extractelement lane.
    if (auto *VecI = dyn_cast<Instruction>(Vec)) {
      if (PHINode *PH = dyn_cast<PHINode>(User)) {
        for (int i = 0, e = PH->getNumIncomingValues(); i != e; ++i) {
          if (PH->getIncomingValue(i) == Scalar) {
            Instruction *IncomingTerminator =
                PH->getIncomingBlock(i)->getTerminator();
            if (isa<CatchSwitchInst>(IncomingTerminator)) {
              Builder.SetInsertPoint(VecI->getParent(),
                                     std::next(VecI->getIterator()));
            } else {
              Builder.SetInsertPoint(PH->getIncomingBlock(i)->getTerminator());
            }
            Value *NewInst = ExtractAndExtendIfNeeded(Vec);
            CSEBlocks.insert(PH->getIncomingBlock(i));
            PH->setOperand(i, NewInst);
          }
        }
      } else {
        Builder.SetInsertPoint(cast<Instruction>(User));
        Value *NewInst = ExtractAndExtendIfNeeded(Vec);
        CSEBlocks.insert(cast<Instruction>(User)->getParent());
        User->replaceUsesOfWith(Scalar, NewInst);
      }
    } else {
      Builder.SetInsertPoint(&F->getEntryBlock().front());
      Value *NewInst = ExtractAndExtendIfNeeded(Vec);
      CSEBlocks.insert(&F->getEntryBlock());
      User->replaceUsesOfWith(Scalar, NewInst);
    }

    LLVM_DEBUG(dbgs() << "SLP: Replaced:" << *User << ".\n");
  }

  // For each vectorized value:
  for (auto &TEPtr : VectorizableTree) {
    TreeEntry *Entry = TEPtr.get();

    // No need to handle users of gathered values.
    if (Entry->State == TreeEntry::NeedToGather)
      continue;

    assert(Entry->VectorizedValue && "Can't find vectorizable value");

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];

#ifndef NDEBUG
      Type *Ty = Scalar->getType();
      if (!Ty->isVoidTy()) {
        for (User *U : Scalar->users()) {
          LLVM_DEBUG(dbgs() << "SLP: \tvalidating user:" << *U << ".\n");

          // It is legal to delete users in the ignorelist.
          assert((getTreeEntry(U) || is_contained(UserIgnoreList, U) ||
                  (isa_and_nonnull<Instruction>(U) &&
                   isDeleted(cast<Instruction>(U)))) &&
                 "Deleting out-of-tree value");
        }
      }
#endif
      LLVM_DEBUG(dbgs() << "SLP: \tErasing scalar:" << *Scalar << ".\n");
      eraseInstruction(cast<Instruction>(Scalar));
    }
  }

  Builder.ClearInsertionPoint();
  InstrElementSize.clear();

  return VectorizableTree[0]->VectorizedValue;
}

void BoUpSLP::optimizeGatherSequence() {
  LLVM_DEBUG(dbgs() << "SLP: Optimizing " << GatherShuffleSeq.size()
                    << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (Instruction *I : GatherShuffleSeq) {
    if (isDeleted(I))
      continue;

    // Check if this block is inside a loop.
    Loop *L = LI->getLoopFor(I->getParent());
    if (!L)
      continue;

    // Check if it has a preheader.
    BasicBlock *PreHeader = L->getLoopPreheader();
    if (!PreHeader)
      continue;

    // If the vector or the element that we insert into it are
    // instructions that are defined in this basic block then we can't
    // hoist this instruction.
    if (any_of(I->operands(), [L](Value *V) {
          auto *OpI = dyn_cast<Instruction>(V);
          return OpI && L->contains(OpI);
        }))
      continue;

    // We can hoist this instruction. Move it to the pre-header.
    I->moveBefore(PreHeader->getTerminator());
  }

  // Make a list of all reachable blocks in our CSE queue.
  SmallVector<const DomTreeNode *, 8> CSEWorkList;
  CSEWorkList.reserve(CSEBlocks.size());
  for (BasicBlock *BB : CSEBlocks)
    if (DomTreeNode *N = DT->getNode(BB)) {
      assert(DT->isReachableFromEntry(N));
      CSEWorkList.push_back(N);
    }

  // Sort blocks by domination. This ensures we visit a block after all blocks
  // dominating it are visited.
  llvm::sort(CSEWorkList, [](const DomTreeNode *A, const DomTreeNode *B) {
    assert((A == B) == (A->getDFSNumIn() == B->getDFSNumIn()) &&
           "Different nodes should have different DFS numbers");
    return A->getDFSNumIn() < B->getDFSNumIn();
  });

  // Less defined shuffles can be replaced by the more defined copies.
  // Between two shuffles one is less defined if it has the same vector operands
  // and its mask indeces are the same as in the first one or undefs. E.g.
  // shuffle %0, poison, <0, 0, 0, undef> is less defined than shuffle %0,
  // poison, <0, 0, 0, 0>.
  auto &&IsIdenticalOrLessDefined = [this](Instruction *I1, Instruction *I2,
                                           SmallVectorImpl<int> &NewMask) {
    if (I1->getType() != I2->getType())
      return false;
    auto *SI1 = dyn_cast<ShuffleVectorInst>(I1);
    auto *SI2 = dyn_cast<ShuffleVectorInst>(I2);
    if (!SI1 || !SI2)
      return I1->isIdenticalTo(I2);
    if (SI1->isIdenticalTo(SI2))
      return true;
    for (int I = 0, E = SI1->getNumOperands(); I < E; ++I)
      if (SI1->getOperand(I) != SI2->getOperand(I))
        return false;
    // Check if the second instruction is more defined than the first one.
    NewMask.assign(SI2->getShuffleMask().begin(), SI2->getShuffleMask().end());
    ArrayRef<int> SM1 = SI1->getShuffleMask();
    // Count trailing undefs in the mask to check the final number of used
    // registers.
    unsigned LastUndefsCnt = 0;
    for (int I = 0, E = NewMask.size(); I < E; ++I) {
      if (SM1[I] == UndefMaskElem)
        ++LastUndefsCnt;
      else
        LastUndefsCnt = 0;
      if (NewMask[I] != UndefMaskElem && SM1[I] != UndefMaskElem &&
          NewMask[I] != SM1[I])
        return false;
      if (NewMask[I] == UndefMaskElem)
        NewMask[I] = SM1[I];
    }
    // Check if the last undefs actually change the final number of used vector
    // registers.
    return SM1.size() - LastUndefsCnt > 1 &&
           TTI->getNumberOfParts(SI1->getType()) ==
               TTI->getNumberOfParts(
                   FixedVectorType::get(SI1->getType()->getElementType(),
                                        SM1.size() - LastUndefsCnt));
  };
  // Perform O(N^2) search over the gather/shuffle sequences and merge identical
  // instructions. TODO: We can further optimize this scan if we split the
  // instructions into different buckets based on the insert lane.
  SmallVector<Instruction *, 16> Visited;
  for (auto I = CSEWorkList.begin(), E = CSEWorkList.end(); I != E; ++I) {
    assert(*I &&
           (I == CSEWorkList.begin() || !DT->dominates(*I, *std::prev(I))) &&
           "Worklist not sorted properly!");
    BasicBlock *BB = (*I)->getBlock();
    // For all instructions in blocks containing gather sequences:
    for (Instruction &In : llvm::make_early_inc_range(*BB)) {
      if (isDeleted(&In))
        continue;
      if (!isa<InsertElementInst>(&In) && !isa<ExtractElementInst>(&In) &&
          !isa<ShuffleVectorInst>(&In) && !GatherShuffleSeq.contains(&In))
        continue;

      // Check if we can replace this instruction with any of the
      // visited instructions.
      bool Replaced = false;
      for (Instruction *&V : Visited) {
        SmallVector<int> NewMask;
        if (IsIdenticalOrLessDefined(&In, V, NewMask) &&
            DT->dominates(V->getParent(), In.getParent())) {
          In.replaceAllUsesWith(V);
          eraseInstruction(&In);
          if (auto *SI = dyn_cast<ShuffleVectorInst>(V))
            if (!NewMask.empty())
              SI->setShuffleMask(NewMask);
          Replaced = true;
          break;
        }
        if (isa<ShuffleVectorInst>(In) && isa<ShuffleVectorInst>(V) &&
            GatherShuffleSeq.contains(V) &&
            IsIdenticalOrLessDefined(V, &In, NewMask) &&
            DT->dominates(In.getParent(), V->getParent())) {
          In.moveAfter(V);
          V->replaceAllUsesWith(&In);
          eraseInstruction(V);
          if (auto *SI = dyn_cast<ShuffleVectorInst>(&In))
            if (!NewMask.empty())
              SI->setShuffleMask(NewMask);
          V = &In;
          Replaced = true;
          break;
        }
      }
      if (!Replaced) {
        assert(!is_contained(Visited, &In));
        Visited.push_back(&In);
      }
    }
  }
  CSEBlocks.clear();
  GatherShuffleSeq.clear();
}

BoUpSLP::ScheduleData *
BoUpSLP::BlockScheduling::buildBundle(ArrayRef<Value *> VL) {
  ScheduleData *Bundle = nullptr;  
  ScheduleData *PrevInBundle = nullptr;
  for (Value *V : VL) {
    ScheduleData *BundleMember = getScheduleData(V);
    assert(BundleMember &&
           "no ScheduleData for bundle member "
           "(maybe not in same basic block)");
    assert(BundleMember->isSchedulingEntity() &&
           "bundle member already part of other bundle");
    if (PrevInBundle) {
      PrevInBundle->NextInBundle = BundleMember;
    } else {
      Bundle = BundleMember;
    }

    // Group the instructions to a bundle.
    BundleMember->FirstInBundle = Bundle;
    PrevInBundle = BundleMember;
  }
  assert(Bundle && "Failed to find schedule bundle");
  return Bundle;
}

// Groups the instructions to a bundle (which is then a single scheduling entity)
// and schedules instructions until the bundle gets ready.
Optional<BoUpSLP::ScheduleData *>
BoUpSLP::BlockScheduling::tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP,
                                            const InstructionsState &S) {
  // No need to schedule PHIs, insertelement, extractelement and extractvalue
  // instructions.
  if (isa<PHINode>(S.OpValue) || isVectorLikeInstWithConstOps(S.OpValue))
    return nullptr;

  // Initialize the instruction bundle.
  Instruction *OldScheduleEnd = ScheduleEnd;
  LLVM_DEBUG(dbgs() << "SLP:  bundle: " << *S.OpValue << "\n");

  auto TryScheduleBundleImpl = [this, OldScheduleEnd, SLP](bool ReSchedule,
                                                         ScheduleData *Bundle) {
    // The scheduling region got new instructions at the lower end (or it is a
    // new region for the first bundle). This makes it necessary to
    // recalculate all dependencies.
    // It is seldom that this needs to be done a second time after adding the
    // initial bundle to the region.
    if (ScheduleEnd != OldScheduleEnd) {
      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode())
        doForAllOpcodes(I, [](ScheduleData *SD) { SD->clearDependencies(); });
      ReSchedule = true;
    }
    if (Bundle) {
      LLVM_DEBUG(dbgs() << "SLP: try schedule bundle " << *Bundle
                        << " in block " << BB->getName() << "\n");
      calculateDependencies(Bundle, /*InsertInReadyList=*/true, SLP);
    }

    if (ReSchedule) {
      resetSchedule();
      initialFillReadyList(ReadyInsts);
    }

    // Now try to schedule the new bundle or (if no bundle) just calculate
    // dependencies. As soon as the bundle is "ready" it means that there are no
    // cyclic dependencies and we can schedule it. Note that's important that we
    // don't "schedule" the bundle yet (see cancelScheduling).
    while (((!Bundle && ReSchedule) || (Bundle && !Bundle->isReady())) &&
           !ReadyInsts.empty()) {
      ScheduleData *Picked = ReadyInsts.pop_back_val();
      assert(Picked->isSchedulingEntity() && Picked->isReady() &&
             "must be ready to schedule");
      schedule(Picked, ReadyInsts);
    }
  };

  // Make sure that the scheduling region contains all
  // instructions of the bundle.
  for (Value *V : VL) {
    if (!extendSchedulingRegion(V, S)) {
      // If the scheduling region got new instructions at the lower end (or it
      // is a new region for the first bundle). This makes it necessary to
      // recalculate all dependencies.
      // Otherwise the compiler may crash trying to incorrectly calculate
      // dependencies and emit instruction in the wrong order at the actual
      // scheduling.
      TryScheduleBundleImpl(/*ReSchedule=*/false, nullptr);
      return None;
    }
  }

  bool ReSchedule = false;
  for (Value *V : VL) {
    ScheduleData *BundleMember = getScheduleData(V);
    assert(BundleMember &&
           "no ScheduleData for bundle member (maybe not in same basic block)");

    // Make sure we don't leave the pieces of the bundle in the ready list when
    // whole bundle might not be ready.
    ReadyInsts.remove(BundleMember);

    if (!BundleMember->IsScheduled)
      continue;
    // A bundle member was scheduled as single instruction before and now
    // needs to be scheduled as part of the bundle. We just get rid of the
    // existing schedule.
    LLVM_DEBUG(dbgs() << "SLP:  reset schedule because " << *BundleMember
                      << " was already scheduled\n");
    ReSchedule = true;
  }

  auto *Bundle = buildBundle(VL);
  TryScheduleBundleImpl(ReSchedule, Bundle);
  if (!Bundle->isReady()) {
    cancelScheduling(VL, S.OpValue);
    return None;
  }
  return Bundle;
}

void BoUpSLP::BlockScheduling::cancelScheduling(ArrayRef<Value *> VL,
                                                Value *OpValue) {
  if (isa<PHINode>(OpValue) || isVectorLikeInstWithConstOps(OpValue))
    return;

  ScheduleData *Bundle = getScheduleData(OpValue);
  LLVM_DEBUG(dbgs() << "SLP:  cancel scheduling of " << *Bundle << "\n");
  assert(!Bundle->IsScheduled &&
         "Can't cancel bundle which is already scheduled");
  assert(Bundle->isSchedulingEntity() && Bundle->isPartOfBundle() &&
         "tried to unbundle something which is not a bundle");

  // Remove the bundle from the ready list.
  if (Bundle->isReady())
    ReadyInsts.remove(Bundle);

  // Un-bundle: make single instructions out of the bundle.
  ScheduleData *BundleMember = Bundle;
  while (BundleMember) {
    assert(BundleMember->FirstInBundle == Bundle && "corrupt bundle links");
    BundleMember->FirstInBundle = BundleMember;
    ScheduleData *Next = BundleMember->NextInBundle;
    BundleMember->NextInBundle = nullptr;
    if (BundleMember->unscheduledDepsInBundle() == 0) {
      ReadyInsts.insert(BundleMember);
    }
    BundleMember = Next;
  }
}

BoUpSLP::ScheduleData *BoUpSLP::BlockScheduling::allocateScheduleDataChunks() {
  // Allocate a new ScheduleData for the instruction.
  if (ChunkPos >= ChunkSize) {
    ScheduleDataChunks.push_back(std::make_unique<ScheduleData[]>(ChunkSize));
    ChunkPos = 0;
  }
  return &(ScheduleDataChunks.back()[ChunkPos++]);
}

bool BoUpSLP::BlockScheduling::extendSchedulingRegion(Value *V,
                                                      const InstructionsState &S) {
  if (getScheduleData(V, isOneOf(S, V)))
    return true;
  Instruction *I = dyn_cast<Instruction>(V);
  assert(I && "bundle member must be an instruction");
  assert(!isa<PHINode>(I) && !isVectorLikeInstWithConstOps(I) &&
         "phi nodes/insertelements/extractelements/extractvalues don't need to "
         "be scheduled");
  auto &&CheckSheduleForI = [this, &S](Instruction *I) -> bool {
    ScheduleData *ISD = getScheduleData(I);
    if (!ISD)
      return false;
    assert(isInSchedulingRegion(ISD) &&
           "ScheduleData not in scheduling region");
    ScheduleData *SD = allocateScheduleDataChunks();
    SD->Inst = I;
    SD->init(SchedulingRegionID, S.OpValue);
    ExtraScheduleDataMap[I][S.OpValue] = SD;
    return true;
  };
  if (CheckSheduleForI(I))
    return true;
  if (!ScheduleStart) {
    // It's the first instruction in the new region.
    initScheduleData(I, I->getNextNode(), nullptr, nullptr);
    ScheduleStart = I;
    ScheduleEnd = I->getNextNode();
    if (isOneOf(S, I) != I)
      CheckSheduleForI(I);
    assert(ScheduleEnd && "tried to vectorize a terminator?");
    LLVM_DEBUG(dbgs() << "SLP:  initialize schedule region to " << *I << "\n");
    return true;
  }
  // Search up and down at the same time, because we don't know if the new
  // instruction is above or below the existing scheduling region.
  BasicBlock::reverse_iterator UpIter =
      ++ScheduleStart->getIterator().getReverse();
  BasicBlock::reverse_iterator UpperEnd = BB->rend();
  BasicBlock::iterator DownIter = ScheduleEnd->getIterator();
  BasicBlock::iterator LowerEnd = BB->end();
  while (UpIter != UpperEnd && DownIter != LowerEnd && &*UpIter != I &&
         &*DownIter != I) {
    if (++ScheduleRegionSize > ScheduleRegionSizeLimit) {
      LLVM_DEBUG(dbgs() << "SLP:  exceeded schedule region size limit\n");
      return false;
    }

    ++UpIter;
    ++DownIter;
  }
  if (DownIter == LowerEnd || (UpIter != UpperEnd && &*UpIter == I)) {
    assert(I->getParent() == ScheduleStart->getParent() &&
           "Instruction is in wrong basic block.");
    initScheduleData(I, ScheduleStart, nullptr, FirstLoadStoreInRegion);
    ScheduleStart = I;
    if (isOneOf(S, I) != I)
      CheckSheduleForI(I);
    LLVM_DEBUG(dbgs() << "SLP:  extend schedule region start to " << *I
                      << "\n");
    return true;
  }
  assert((UpIter == UpperEnd || (DownIter != LowerEnd && &*DownIter == I)) &&
         "Expected to reach top of the basic block or instruction down the "
         "lower end.");
  assert(I->getParent() == ScheduleEnd->getParent() &&
         "Instruction is in wrong basic block.");
  initScheduleData(ScheduleEnd, I->getNextNode(), LastLoadStoreInRegion,
                   nullptr);
  ScheduleEnd = I->getNextNode();
  if (isOneOf(S, I) != I)
    CheckSheduleForI(I);
  assert(ScheduleEnd && "tried to vectorize a terminator?");
  LLVM_DEBUG(dbgs() << "SLP:  extend schedule region end to " << *I << "\n");
  return true;
}

void BoUpSLP::BlockScheduling::initScheduleData(Instruction *FromI,
                                                Instruction *ToI,
                                                ScheduleData *PrevLoadStore,
                                                ScheduleData *NextLoadStore) {
  ScheduleData *CurrentLoadStore = PrevLoadStore;
  for (Instruction *I = FromI; I != ToI; I = I->getNextNode()) {
    ScheduleData *SD = ScheduleDataMap[I];
    if (!SD) {
      SD = allocateScheduleDataChunks();
      ScheduleDataMap[I] = SD;
      SD->Inst = I;
    }
    assert(!isInSchedulingRegion(SD) &&
           "new ScheduleData already in scheduling region");
    SD->init(SchedulingRegionID, I);

    if (I->mayReadOrWriteMemory() &&
        (!isa<IntrinsicInst>(I) ||
         (cast<IntrinsicInst>(I)->getIntrinsicID() != Intrinsic::sideeffect &&
          cast<IntrinsicInst>(I)->getIntrinsicID() !=
              Intrinsic::pseudoprobe))) {
      // Update the linked list of memory accessing instructions.
      if (CurrentLoadStore) {
        CurrentLoadStore->NextLoadStore = SD;
      } else {
        FirstLoadStoreInRegion = SD;
      }
      CurrentLoadStore = SD;
    }
  }
  if (NextLoadStore) {
    if (CurrentLoadStore)
      CurrentLoadStore->NextLoadStore = NextLoadStore;
  } else {
    LastLoadStoreInRegion = CurrentLoadStore;
  }
}

void BoUpSLP::BlockScheduling::calculateDependencies(ScheduleData *SD,
                                                     bool InsertInReadyList,
                                                     BoUpSLP *SLP) {
  assert(SD->isSchedulingEntity());

  SmallVector<ScheduleData *, 10> WorkList;
  WorkList.push_back(SD);

  while (!WorkList.empty()) {
    ScheduleData *SD = WorkList.pop_back_val();
    for (ScheduleData *BundleMember = SD; BundleMember;
         BundleMember = BundleMember->NextInBundle) {
      assert(isInSchedulingRegion(BundleMember));
      if (BundleMember->hasValidDependencies())
        continue;

      LLVM_DEBUG(dbgs() << "SLP:       update deps of " << *BundleMember
                 << "\n");
      BundleMember->Dependencies = 0;
      BundleMember->resetUnscheduledDeps();

      // Handle def-use chain dependencies.
      if (BundleMember->OpValue != BundleMember->Inst) {
        if (ScheduleData *UseSD = getScheduleData(BundleMember->Inst)) {
          BundleMember->Dependencies++;
          ScheduleData *DestBundle = UseSD->FirstInBundle;
          if (!DestBundle->IsScheduled)
            BundleMember->incrementUnscheduledDeps(1);
          if (!DestBundle->hasValidDependencies())
            WorkList.push_back(DestBundle);
        }
      } else {
        for (User *U : BundleMember->Inst->users()) {
          if (ScheduleData *UseSD = getScheduleData(cast<Instruction>(U))) {
            BundleMember->Dependencies++;
            ScheduleData *DestBundle = UseSD->FirstInBundle;
            if (!DestBundle->IsScheduled)
              BundleMember->incrementUnscheduledDeps(1);
            if (!DestBundle->hasValidDependencies())
              WorkList.push_back(DestBundle);
          }
        }
      }

      // Handle the memory dependencies (if any).
      ScheduleData *DepDest = BundleMember->NextLoadStore;
      if (!DepDest)
        continue;
      Instruction *SrcInst = BundleMember->Inst;
      assert(SrcInst->mayReadOrWriteMemory() &&
             "NextLoadStore list for non memory effecting bundle?");
      MemoryLocation SrcLoc = getLocation(SrcInst);
      bool SrcMayWrite = BundleMember->Inst->mayWriteToMemory();
      unsigned numAliased = 0;
      unsigned DistToSrc = 1;

      for ( ; DepDest; DepDest = DepDest->NextLoadStore) {
        assert(isInSchedulingRegion(DepDest));

        // We have two limits to reduce the complexity:
        // 1) AliasedCheckLimit: It's a small limit to reduce calls to
        //    SLP->isAliased (which is the expensive part in this loop).
        // 2) MaxMemDepDistance: It's for very large blocks and it aborts
        //    the whole loop (even if the loop is fast, it's quadratic).
        //    It's important for the loop break condition (see below) to
        //    check this limit even between two read-only instructions.
        if (DistToSrc >= MaxMemDepDistance ||
            ((SrcMayWrite || DepDest->Inst->mayWriteToMemory()) &&
             (numAliased >= AliasedCheckLimit ||
              SLP->isAliased(SrcLoc, SrcInst, DepDest->Inst)))) {

          // We increment the counter only if the locations are aliased
          // (instead of counting all alias checks). This gives a better
          // balance between reduced runtime and accurate dependencies.
          numAliased++;

          DepDest->MemoryDependencies.push_back(BundleMember);
          BundleMember->Dependencies++;
          ScheduleData *DestBundle = DepDest->FirstInBundle;
          if (!DestBundle->IsScheduled) {
            BundleMember->incrementUnscheduledDeps(1);
          }
          if (!DestBundle->hasValidDependencies()) {
            WorkList.push_back(DestBundle);
          }
        }

        // Example, explaining the loop break condition: Let's assume our
        // starting instruction is i0 and MaxMemDepDistance = 3.
        //
        //                      +--------v--v--v
        //             i0,i1,i2,i3,i4,i5,i6,i7,i8
        //             +--------^--^--^
        //
        // MaxMemDepDistance let us stop alias-checking at i3 and we add
        // dependencies from i0 to i3,i4,.. (even if they are not aliased).
        // Previously we already added dependencies from i3 to i6,i7,i8
        // (because of MaxMemDepDistance). As we added a dependency from
        // i0 to i3, we have transitive dependencies from i0 to i6,i7,i8
        // and we can abort this loop at i6.
        if (DistToSrc >= 2 * MaxMemDepDistance)
          break;
        DistToSrc++;
      }
    }
    if (InsertInReadyList && SD->isReady()) {
      ReadyInsts.insert(SD);
      LLVM_DEBUG(dbgs() << "SLP:     gets ready on update: " << *SD->Inst
                        << "\n");
    }
  }
}

void BoUpSLP::BlockScheduling::resetSchedule() {
  assert(ScheduleStart &&
         "tried to reset schedule on block which has not been scheduled");
  for (Instruction *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
    doForAllOpcodes(I, [&](ScheduleData *SD) {
      assert(isInSchedulingRegion(SD) &&
             "ScheduleData not in scheduling region");
      SD->IsScheduled = false;
      SD->resetUnscheduledDeps();
    });
  }
  ReadyInsts.clear();
}

void BoUpSLP::scheduleBlock(BlockScheduling *BS) {
  if (!BS->ScheduleStart)
    return;

  LLVM_DEBUG(dbgs() << "SLP: schedule block " << BS->BB->getName() << "\n");

  BS->resetSchedule();

  // For the real scheduling we use a more sophisticated ready-list: it is
  // sorted by the original instruction location. This lets the final schedule
  // be as  close as possible to the original instruction order.
  struct ScheduleDataCompare {
    bool operator()(ScheduleData *SD1, ScheduleData *SD2) const {
      return SD2->SchedulingPriority < SD1->SchedulingPriority;
    }
  };
  std::set<ScheduleData *, ScheduleDataCompare> ReadyInsts;

  // Ensure that all dependency data is updated and fill the ready-list with
  // initial instructions.
  int Idx = 0;
  int NumToSchedule = 0;
  for (auto *I = BS->ScheduleStart; I != BS->ScheduleEnd;
       I = I->getNextNode()) {
    BS->doForAllOpcodes(I, [this, &Idx, &NumToSchedule, BS](ScheduleData *SD) {
      assert((isVectorLikeInstWithConstOps(SD->Inst) ||
              SD->isPartOfBundle() == (getTreeEntry(SD->Inst) != nullptr)) &&
             "scheduler and vectorizer bundle mismatch");
      SD->FirstInBundle->SchedulingPriority = Idx++;
      if (SD->isSchedulingEntity()) {
        BS->calculateDependencies(SD, false, this);
        NumToSchedule++;
      }
    });
  }
  BS->initialFillReadyList(ReadyInsts);

  Instruction *LastScheduledInst = BS->ScheduleEnd;

  // Do the "real" scheduling.
  while (!ReadyInsts.empty()) {
    ScheduleData *picked = *ReadyInsts.begin();
    ReadyInsts.erase(ReadyInsts.begin());

    // Move the scheduled instruction(s) to their dedicated places, if not
    // there yet.
    for (ScheduleData *BundleMember = picked; BundleMember;
         BundleMember = BundleMember->NextInBundle) {
      Instruction *pickedInst = BundleMember->Inst;
      if (pickedInst->getNextNode() != LastScheduledInst)
        pickedInst->moveBefore(LastScheduledInst);
      LastScheduledInst = pickedInst;
    }

    BS->schedule(picked, ReadyInsts);
    NumToSchedule--;
  }
  assert(NumToSchedule == 0 && "could not schedule all instructions");

  // Check that we didn't break any of our invariants.
#ifdef EXPENSIVE_CHECKS
  BS->verify();
#endif

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
  // Check that all schedulable entities got scheduled
  for (auto *I = BS->ScheduleStart; I != BS->ScheduleEnd; I = I->getNextNode()) {
    BS->doForAllOpcodes(I, [&](ScheduleData *SD) {
      if (SD->isSchedulingEntity() && SD->hasValidDependencies()) {
        assert(SD->IsScheduled && "must be scheduled at this point");
      }
    });
  }
#endif

  // Avoid duplicate scheduling of the block.
  BS->ScheduleStart = nullptr;
}

unsigned BoUpSLP::getVectorElementSize(Value *V) {
  // If V is a store, just return the width of the stored value (or value
  // truncated just before storing) without traversing the expression tree.
  // This is the common case.
  if (auto *Store = dyn_cast<StoreInst>(V)) {
    if (auto *Trunc = dyn_cast<TruncInst>(Store->getValueOperand()))
      return DL->getTypeSizeInBits(Trunc->getSrcTy());
    return DL->getTypeSizeInBits(Store->getValueOperand()->getType());
  }

  if (auto *IEI = dyn_cast<InsertElementInst>(V))
    return getVectorElementSize(IEI->getOperand(1));

  auto E = InstrElementSize.find(V);
  if (E != InstrElementSize.end())
    return E->second;

  // If V is not a store, we can traverse the expression tree to find loads
  // that feed it. The type of the loaded value may indicate a more suitable
  // width than V's type. We want to base the vector element size on the width
  // of memory operations where possible.
  SmallVector<std::pair<Instruction *, BasicBlock *>, 16> Worklist;
  SmallPtrSet<Instruction *, 16> Visited;
  if (auto *I = dyn_cast<Instruction>(V)) {
    Worklist.emplace_back(I, I->getParent());
    Visited.insert(I);
  }

  // Traverse the expression tree in bottom-up order looking for loads. If we
  // encounter an instruction we don't yet handle, we give up.
  auto Width = 0u;
  while (!Worklist.empty()) {
    Instruction *I;
    BasicBlock *Parent;
    std::tie(I, Parent) = Worklist.pop_back_val();

    // We should only be looking at scalar instructions here. If the current
    // instruction has a vector type, skip.
    auto *Ty = I->getType();
    if (isa<VectorType>(Ty))
      continue;

    // If the current instruction is a load, update MaxWidth to reflect the
    // width of the loaded value.
    if (isa<LoadInst>(I) || isa<ExtractElementInst>(I) ||
        isa<ExtractValueInst>(I))
      Width = std::max<unsigned>(Width, DL->getTypeSizeInBits(Ty));

    // Otherwise, we need to visit the operands of the instruction. We only
    // handle the interesting cases from buildTree here. If an operand is an
    // instruction we haven't yet visited and from the same basic block as the
    // user or the use is a PHI node, we add it to the worklist.
    else if (isa<PHINode>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I) ||
             isa<CmpInst>(I) || isa<SelectInst>(I) || isa<BinaryOperator>(I) ||
             isa<UnaryOperator>(I)) {
      for (Use &U : I->operands())
        if (auto *J = dyn_cast<Instruction>(U.get()))
          if (Visited.insert(J).second &&
              (isa<PHINode>(I) || J->getParent() == Parent))
            Worklist.emplace_back(J, J->getParent());
    } else {
      break;
    }
  }

  // If we didn't encounter a memory access in the expression tree, or if we
  // gave up for some reason, just return the width of V. Otherwise, return the
  // maximum width we found.
  if (!Width) {
    if (auto *CI = dyn_cast<CmpInst>(V))
      V = CI->getOperand(0);
    Width = DL->getTypeSizeInBits(V->getType());
  }

  for (Instruction *I : Visited)
    InstrElementSize[I] = Width;

  return Width;
}

// Determine if a value V in a vectorizable expression Expr can be demoted to a
// smaller type with a truncation. We collect the values that will be demoted
// in ToDemote and additional roots that require investigating in Roots.
static bool collectValuesToDemote(Value *V, SmallPtrSetImpl<Value *> &Expr,
                                  SmallVectorImpl<Value *> &ToDemote,
                                  SmallVectorImpl<Value *> &Roots) {
  // We can always demote constants.
  if (isa<Constant>(V)) {
    ToDemote.push_back(V);
    return true;
  }

  // If the value is not an instruction in the expression with only one use, it
  // cannot be demoted.
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !I->hasOneUse() || !Expr.count(I))
    return false;

  switch (I->getOpcode()) {

  // We can always demote truncations and extensions. Since truncations can
  // seed additional demotion, we save the truncated value.
  case Instruction::Trunc:
    Roots.push_back(I->getOperand(0));
    break;
  case Instruction::ZExt:
  case Instruction::SExt:
    if (isa<ExtractElementInst>(I->getOperand(0)) ||
        isa<InsertElementInst>(I->getOperand(0)))
      return false;
    break;

  // We can demote certain binary operations if we can demote both of their
  // operands.
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    if (!collectValuesToDemote(I->getOperand(0), Expr, ToDemote, Roots) ||
        !collectValuesToDemote(I->getOperand(1), Expr, ToDemote, Roots))
      return false;
    break;

  // We can demote selects if we can demote their true and false values.
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    if (!collectValuesToDemote(SI->getTrueValue(), Expr, ToDemote, Roots) ||
        !collectValuesToDemote(SI->getFalseValue(), Expr, ToDemote, Roots))
      return false;
    break;
  }

  // We can demote phis if we can demote all their incoming operands. Note that
  // we don't need to worry about cycles since we ensure single use above.
  case Instruction::PHI: {
    PHINode *PN = cast<PHINode>(I);
    for (Value *IncValue : PN->incoming_values())
      if (!collectValuesToDemote(IncValue, Expr, ToDemote, Roots))
        return false;
    break;
  }

  // Otherwise, conservatively give up.
  default:
    return false;
  }

  // Record the value that we can demote.
  ToDemote.push_back(V);
  return true;
}

void BoUpSLP::computeMinimumValueSizes() {
  // If there are no external uses, the expression tree must be rooted by a
  // store. We can't demote in-memory values, so there is nothing to do here.
  if (ExternalUses.empty())
    return;

  // We only attempt to truncate integer expressions.
  auto &TreeRoot = VectorizableTree[0]->Scalars;
  auto *TreeRootIT = dyn_cast<IntegerType>(TreeRoot[0]->getType());
  if (!TreeRootIT)
    return;

  // If the expression is not rooted by a store, these roots should have
  // external uses. We will rely on InstCombine to rewrite the expression in
  // the narrower type. However, InstCombine only rewrites single-use values.
  // This means that if a tree entry other than a root is used externally, it
  // must have multiple uses and InstCombine will not rewrite it. The code
  // below ensures that only the roots are used externally.
  SmallPtrSet<Value *, 32> Expr(TreeRoot.begin(), TreeRoot.end());
  for (auto &EU : ExternalUses)
    if (!Expr.erase(EU.Scalar))
      return;
  if (!Expr.empty())
    return;

  // Collect the scalar values of the vectorizable expression. We will use this
  // context to determine which values can be demoted. If we see a truncation,
  // we mark it as seeding another demotion.
  for (auto &EntryPtr : VectorizableTree)
    Expr.insert(EntryPtr->Scalars.begin(), EntryPtr->Scalars.end());

  // Ensure the roots of the vectorizable tree don't form a cycle. They must
  // have a single external user that is not in the vectorizable tree.
  for (auto *Root : TreeRoot)
    if (!Root->hasOneUse() || Expr.count(*Root->user_begin()))
      return;

  // Conservatively determine if we can actually truncate the roots of the
  // expression. Collect the values that can be demoted in ToDemote and
  // additional roots that require investigating in Roots.
  SmallVector<Value *, 32> ToDemote;
  SmallVector<Value *, 4> Roots;
  for (auto *Root : TreeRoot)
    if (!collectValuesToDemote(Root, Expr, ToDemote, Roots))
      return;

  // The maximum bit width required to represent all the values that can be
  // demoted without loss of precision. It would be safe to truncate the roots
  // of the expression to this width.
  auto MaxBitWidth = 8u;

  // We first check if all the bits of the roots are demanded. If they're not,
  // we can truncate the roots to this narrower type.
  for (auto *Root : TreeRoot) {
    auto Mask = DB->getDemandedBits(cast<Instruction>(Root));
    MaxBitWidth = std::max<unsigned>(
        Mask.getBitWidth() - Mask.countLeadingZeros(), MaxBitWidth);
  }

  // True if the roots can be zero-extended back to their original type, rather
  // than sign-extended. We know that if the leading bits are not demanded, we
  // can safely zero-extend. So we initialize IsKnownPositive to True.
  bool IsKnownPositive = true;

  // If all the bits of the roots are demanded, we can try a little harder to
  // compute a narrower type. This can happen, for example, if the roots are
  // getelementptr indices. InstCombine promotes these indices to the pointer
  // width. Thus, all their bits are technically demanded even though the
  // address computation might be vectorized in a smaller type.
  //
  // We start by looking at each entry that can be demoted. We compute the
  // maximum bit width required to store the scalar by using ValueTracking to
  // compute the number of high-order bits we can truncate.
  if (MaxBitWidth == DL->getTypeSizeInBits(TreeRoot[0]->getType()) &&
      llvm::all_of(TreeRoot, [](Value *R) {
        assert(R->hasOneUse() && "Root should have only one use!");
        return isa<GetElementPtrInst>(R->user_back());
      })) {
    MaxBitWidth = 8u;

    // Determine if the sign bit of all the roots is known to be zero. If not,
    // IsKnownPositive is set to False.
    IsKnownPositive = llvm::all_of(TreeRoot, [&](Value *R) {
      KnownBits Known = computeKnownBits(R, *DL);
      return Known.isNonNegative();
    });

    // Determine the maximum number of bits required to store the scalar
    // values.
    for (auto *Scalar : ToDemote) {
      auto NumSignBits = ComputeNumSignBits(Scalar, *DL, 0, AC, nullptr, DT);
      auto NumTypeBits = DL->getTypeSizeInBits(Scalar->getType());
      MaxBitWidth = std::max<unsigned>(NumTypeBits - NumSignBits, MaxBitWidth);
    }

    // If we can't prove that the sign bit is zero, we must add one to the
    // maximum bit width to account for the unknown sign bit. This preserves
    // the existing sign bit so we can safely sign-extend the root back to the
    // original type. Otherwise, if we know the sign bit is zero, we will
    // zero-extend the root instead.
    //
    // FIXME: This is somewhat suboptimal, as there will be cases where adding
    //        one to the maximum bit width will yield a larger-than-necessary
    //        type. In general, we need to add an extra bit only if we can't
    //        prove that the upper bit of the original type is equal to the
    //        upper bit of the proposed smaller type. If these two bits are the
    //        same (either zero or one) we know that sign-extending from the
    //        smaller type will result in the same value. Here, since we can't
    //        yet prove this, we are just making the proposed smaller type
    //        larger to ensure correctness.
    if (!IsKnownPositive)
      ++MaxBitWidth;
  }

  // Round MaxBitWidth up to the next power-of-two.
  if (!isPowerOf2_64(MaxBitWidth))
    MaxBitWidth = NextPowerOf2(MaxBitWidth);

  // If the maximum bit width we compute is less than the with of the roots'
  // type, we can proceed with the narrowing. Otherwise, do nothing.
  if (MaxBitWidth >= TreeRootIT->getBitWidth())
    return;

  // If we can truncate the root, we must collect additional values that might
  // be demoted as a result. That is, those seeded by truncations we will
  // modify.
  while (!Roots.empty())
    collectValuesToDemote(Roots.pop_back_val(), Expr, ToDemote, Roots);

  // Finally, map the values we can demote to the maximum bit with we computed.
  for (auto *Scalar : ToDemote)
    MinBWs[Scalar] = std::make_pair(MaxBitWidth, !IsKnownPositive);
}

namespace {

/// The SLPVectorizer Pass.
struct SLPVectorizer : public FunctionPass {
  SLPVectorizerPass Impl;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : FunctionPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override { return false; }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    auto *TLI = TLIP ? &TLIP->getTLI(F) : nullptr;
    auto *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *DB = &getAnalysis<DemandedBitsWrapperPass>().getDemandedBits();
    auto *ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    return Impl.runImpl(F, SE, TTI, TLI, AA, LI, DT, AC, DB, ORE);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<DemandedBitsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addRequired<InjectTLIMappingsLegacy>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

PreservedAnalyses SLPVectorizerPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  auto *TTI = &AM.getResult<TargetIRAnalysis>(F);
  auto *TLI = AM.getCachedResult<TargetLibraryAnalysis>(F);
  auto *AA = &AM.getResult<AAManager>(F);
  auto *LI = &AM.getResult<LoopAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = &AM.getResult<AssumptionAnalysis>(F);
  auto *DB = &AM.getResult<DemandedBitsAnalysis>(F);
  auto *ORE = &AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  bool Changed = runImpl(F, SE, TTI, TLI, AA, LI, DT, AC, DB, ORE);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SLPVectorizerPass::runImpl(Function &F, ScalarEvolution *SE_,
                                TargetTransformInfo *TTI_,
                                TargetLibraryInfo *TLI_, AAResults *AA_,
                                LoopInfo *LI_, DominatorTree *DT_,
                                AssumptionCache *AC_, DemandedBits *DB_,
                                OptimizationRemarkEmitter *ORE_) {
  if (!RunSLPVectorization)
    return false;
  SE = SE_;
  TTI = TTI_;
  TLI = TLI_;
  AA = AA_;
  LI = LI_;
  DT = DT_;
  AC = AC_;
  DB = DB_;
  DL = &F.getParent()->getDataLayout();

  Stores.clear();
  GEPs.clear();
  bool Changed = false;

  // If the target claims to have no vector registers don't attempt
  // vectorization.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true))) {
    LLVM_DEBUG(
        dbgs() << "SLP: Didn't find any vector registers for target, abort.\n");
    return false;
  }

  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  LLVM_DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

  // Use the bottom up slp vectorizer to construct chains that start with
  // store instructions.
  BoUpSLP R(&F, SE, TTI, TLI, AA, LI, DT, AC, DB, DL, ORE_);

  // A general note: the vectorizer must use BoUpSLP::eraseInstruction() to
  // delete instructions.

  // Update DFS numbers now so that we can use them for ordering.
  DT->updateDFSNumbers();

  // Scan the blocks in the function in post order.
  for (auto BB : post_order(&F.getEntryBlock())) {
    collectSeedInstructions(BB);

    // Vectorize trees that end at stores.
    if (!Stores.empty()) {
      LLVM_DEBUG(dbgs() << "SLP: Found stores for " << Stores.size()
                        << " underlying objects.\n");
      Changed |= vectorizeStoreChains(R);
    }

    // Vectorize trees that end at reductions.
    Changed |= vectorizeChainsInBlock(BB, R);

    // Vectorize the index computations of getelementptr instructions. This
    // is primarily intended to catch gather-like idioms ending at
    // non-consecutive loads.
    if (!GEPs.empty()) {
      LLVM_DEBUG(dbgs() << "SLP: Found GEPs for " << GEPs.size()
                        << " underlying objects.\n");
      Changed |= vectorizeGEPIndices(BB, R);
    }
  }

  if (Changed) {
    R.optimizeGatherSequence();
    LLVM_DEBUG(dbgs() << "SLP: vectorized \"" << F.getName() << "\"\n");
  }
  return Changed;
}

bool SLPVectorizerPass::vectorizeStoreChain(ArrayRef<Value *> Chain, BoUpSLP &R,
                                            unsigned Idx) {
  LLVM_DEBUG(dbgs() << "SLP: Analyzing a store chain of length " << Chain.size()
                    << "\n");
  const unsigned Sz = R.getVectorElementSize(Chain[0]);
  const unsigned MinVF = R.getMinVecRegSize() / Sz;
  unsigned VF = Chain.size();

  if (!isPowerOf2_32(Sz) || !isPowerOf2_32(VF) || VF < 2 || VF < MinVF)
    return false;

  LLVM_DEBUG(dbgs() << "SLP: Analyzing " << VF << " stores at offset " << Idx
                    << "\n");

  R.buildTree(Chain);
  if (R.isTreeTinyAndNotFullyVectorizable())
    return false;
  if (R.isLoadCombineCandidate())
    return false;
  R.reorderTopToBottom();
  R.reorderBottomToTop();
  R.buildExternalUses();

  R.computeMinimumValueSizes();

  InstructionCost Cost = R.getTreeCost();

  LLVM_DEBUG(dbgs() << "SLP: Found cost = " << Cost << " for VF =" << VF << "\n");
  if (Cost < -SLPCostThreshold) {
    LLVM_DEBUG(dbgs() << "SLP: Decided to vectorize cost = " << Cost << "\n");

    using namespace ore;

    R.getORE()->emit(OptimizationRemark(SV_NAME, "StoresVectorized",
                                        cast<StoreInst>(Chain[0]))
                     << "Stores SLP vectorized with cost " << NV("Cost", Cost)
                     << " and with tree size "
                     << NV("TreeSize", R.getTreeSize()));

    R.vectorizeTree();
    return true;
  }

  return false;
}

bool SLPVectorizerPass::vectorizeStores(ArrayRef<StoreInst *> Stores,
                                        BoUpSLP &R) {
  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  BoUpSLP::ValueSet VectorizedStores;
  bool Changed = false;

  int E = Stores.size();
  SmallBitVector Tails(E, false);
  int MaxIter = MaxStoreLookup.getValue();
  SmallVector<std::pair<int, int>, 16> ConsecutiveChain(
      E, std::make_pair(E, INT_MAX));
  SmallVector<SmallBitVector, 4> CheckedPairs(E, SmallBitVector(E, false));
  int IterCnt;
  auto &&FindConsecutiveAccess = [this, &Stores, &Tails, &IterCnt, MaxIter,
                                  &CheckedPairs,
                                  &ConsecutiveChain](int K, int Idx) {
    if (IterCnt >= MaxIter)
      return true;
    if (CheckedPairs[Idx].test(K))
      return ConsecutiveChain[K].second == 1 &&
             ConsecutiveChain[K].first == Idx;
    ++IterCnt;
    CheckedPairs[Idx].set(K);
    CheckedPairs[K].set(Idx);
    Optional<int> Diff = getPointersDiff(
        Stores[K]->getValueOperand()->getType(), Stores[K]->getPointerOperand(),
        Stores[Idx]->getValueOperand()->getType(),
        Stores[Idx]->getPointerOperand(), *DL, *SE, /*StrictCheck=*/true);
    if (!Diff || *Diff == 0)
      return false;
    int Val = *Diff;
    if (Val < 0) {
      if (ConsecutiveChain[Idx].second > -Val) {
        Tails.set(K);
        ConsecutiveChain[Idx] = std::make_pair(K, -Val);
      }
      return false;
    }
    if (ConsecutiveChain[K].second <= Val)
      return false;

    Tails.set(Idx);
    ConsecutiveChain[K] = std::make_pair(Idx, Val);
    return Val == 1;
  };
  // Do a quadratic search on all of the given stores in reverse order and find
  // all of the pairs of stores that follow each other.
  for (int Idx = E - 1; Idx >= 0; --Idx) {
    // If a store has multiple consecutive store candidates, search according
    // to the sequence: Idx-1, Idx+1, Idx-2, Idx+2, ...
    // This is because usually pairing with immediate succeeding or preceding
    // candidate create the best chance to find slp vectorization opportunity.
    const int MaxLookDepth = std::max(E - Idx, Idx + 1);
    IterCnt = 0;
    for (int Offset = 1, F = MaxLookDepth; Offset < F; ++Offset)
      if ((Idx >= Offset && FindConsecutiveAccess(Idx - Offset, Idx)) ||
          (Idx + Offset < E && FindConsecutiveAccess(Idx + Offset, Idx)))
        break;
  }

  // Tracks if we tried to vectorize stores starting from the given tail
  // already.
  SmallBitVector TriedTails(E, false);
  // For stores that start but don't end a link in the chain:
  for (int Cnt = E; Cnt > 0; --Cnt) {
    int I = Cnt - 1;
    if (ConsecutiveChain[I].first == E || Tails.test(I))
      continue;
    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    BoUpSLP::ValueList Operands;
    // Collect the chain into a list.
    while (I != E && !VectorizedStores.count(Stores[I])) {
      Operands.push_back(Stores[I]);
      Tails.set(I);
      if (ConsecutiveChain[I].second != 1) {
        // Mark the new end in the chain and go back, if required. It might be
        // required if the original stores come in reversed order, for example.
        if (ConsecutiveChain[I].first != E &&
            Tails.test(ConsecutiveChain[I].first) && !TriedTails.test(I) &&
            !VectorizedStores.count(Stores[ConsecutiveChain[I].first])) {
          TriedTails.set(I);
          Tails.reset(ConsecutiveChain[I].first);
          if (Cnt < ConsecutiveChain[I].first + 2)
            Cnt = ConsecutiveChain[I].first + 2;
        }
        break;
      }
      // Move to the next value in the chain.
      I = ConsecutiveChain[I].first;
    }
    assert(!Operands.empty() && "Expected non-empty list of stores.");

    unsigned MaxVecRegSize = R.getMaxVecRegSize();
    unsigned EltSize = R.getVectorElementSize(Operands[0]);
    unsigned MaxElts = llvm::PowerOf2Floor(MaxVecRegSize / EltSize);

    unsigned MinVF = R.getMinVF(EltSize);
    unsigned MaxVF = std::min(R.getMaximumVF(EltSize, Instruction::Store),
                              MaxElts);

    // FIXME: Is division-by-2 the correct step? Should we assert that the
    // register size is a power-of-2?
    unsigned StartIdx = 0;
    for (unsigned Size = MaxVF; Size >= MinVF; Size /= 2) {
      for (unsigned Cnt = StartIdx, E = Operands.size(); Cnt + Size <= E;) {
        ArrayRef<Value *> Slice = makeArrayRef(Operands).slice(Cnt, Size);
        if (!VectorizedStores.count(Slice.front()) &&
            !VectorizedStores.count(Slice.back()) &&
            vectorizeStoreChain(Slice, R, Cnt)) {
          // Mark the vectorized stores so that we don't vectorize them again.
          VectorizedStores.insert(Slice.begin(), Slice.end());
          Changed = true;
          // If we vectorized initial block, no need to try to vectorize it
          // again.
          if (Cnt == StartIdx)
            StartIdx += Size;
          Cnt += Size;
          continue;
        }
        ++Cnt;
      }
      // Check if the whole array was vectorized already - exit.
      if (StartIdx >= Operands.size())
        break;
    }
  }

  return Changed;
}

void SLPVectorizerPass::collectSeedInstructions(BasicBlock *BB) {
  // Initialize the collections. We will make a single pass over the block.
  Stores.clear();
  GEPs.clear();

  // Visit the store and getelementptr instructions in BB and organize them in
  // Stores and GEPs according to the underlying objects of their pointer
  // operands.
  for (Instruction &I : *BB) {
    // Ignore store instructions that are volatile or have a pointer operand
    // that doesn't point to a scalar type.
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple())
        continue;
      if (!isValidElementType(SI->getValueOperand()->getType()))
        continue;
      Stores[getUnderlyingObject(SI->getPointerOperand())].push_back(SI);
    }

    // Ignore getelementptr instructions that have more than one index, a
    // constant index, or a pointer operand that doesn't point to a scalar
    // type.
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      auto Idx = GEP->idx_begin()->get();
      if (GEP->getNumIndices() > 1 || isa<Constant>(Idx))
        continue;
      if (!isValidElementType(Idx->getType()))
        continue;
      if (GEP->getType()->isVectorTy())
        continue;
      GEPs[GEP->getPointerOperand()].push_back(GEP);
    }
  }
}

bool SLPVectorizerPass::tryToVectorizePair(Value *A, Value *B, BoUpSLP &R) {
  if (!A || !B)
    return false;
  if (isa<InsertElementInst>(A) || isa<InsertElementInst>(B))
    return false;
  Value *VL[] = {A, B};
  return tryToVectorizeList(VL, R);
}

bool SLPVectorizerPass::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                           bool LimitForRegisterSize) {
  if (VL.size() < 2)
    return false;

  LLVM_DEBUG(dbgs() << "SLP: Trying to vectorize a list of length = "
                    << VL.size() << ".\n");

  // Check that all of the parts are instructions of the same type,
  // we permit an alternate opcode via InstructionsState.
  InstructionsState S = getSameOpcode(VL);
  if (!S.getOpcode())
    return false;

  Instruction *I0 = cast<Instruction>(S.OpValue);
  // Make sure invalid types (including vector type) are rejected before
  // determining vectorization factor for scalar instructions.
  for (Value *V : VL) {
    Type *Ty = V->getType();
    if (!isa<InsertElementInst>(V) && !isValidElementType(Ty)) {
      // NOTE: the following will give user internal llvm type name, which may
      // not be useful.
      R.getORE()->emit([&]() {
        std::string type_str;
        llvm::raw_string_ostream rso(type_str);
        Ty->print(rso);
        return OptimizationRemarkMissed(SV_NAME, "UnsupportedType", I0)
               << "Cannot SLP vectorize list: type "
               << rso.str() + " is unsupported by vectorizer";
      });
      return false;
    }
  }

  unsigned Sz = R.getVectorElementSize(I0);
  unsigned MinVF = R.getMinVF(Sz);
  unsigned MaxVF = std::max<unsigned>(PowerOf2Floor(VL.size()), MinVF);
  MaxVF = std::min(R.getMaximumVF(Sz, S.getOpcode()), MaxVF);
  if (MaxVF < 2) {
    R.getORE()->emit([&]() {
      return OptimizationRemarkMissed(SV_NAME, "SmallVF", I0)
             << "Cannot SLP vectorize list: vectorization factor "
             << "less than 2 is not supported";
    });
    return false;
  }

  bool Changed = false;
  bool CandidateFound = false;
  InstructionCost MinCost = SLPCostThreshold.getValue();
  Type *ScalarTy = VL[0]->getType();
  if (auto *IE = dyn_cast<InsertElementInst>(VL[0]))
    ScalarTy = IE->getOperand(1)->getType();

  unsigned NextInst = 0, MaxInst = VL.size();
  for (unsigned VF = MaxVF; NextInst + 1 < MaxInst && VF >= MinVF; VF /= 2) {
    // No actual vectorization should happen, if number of parts is the same as
    // provided vectorization factor (i.e. the scalar type is used for vector
    // code during codegen).
    auto *VecTy = FixedVectorType::get(ScalarTy, VF);
    if (TTI->getNumberOfParts(VecTy) == VF)
      continue;
    for (unsigned I = NextInst; I < MaxInst; ++I) {
      unsigned OpsWidth = 0;

      if (I + VF > MaxInst)
        OpsWidth = MaxInst - I;
      else
        OpsWidth = VF;

      if (!isPowerOf2_32(OpsWidth))
        continue;

      if ((LimitForRegisterSize && OpsWidth < MaxVF) ||
          (VF > MinVF && OpsWidth <= VF / 2) || (VF == MinVF && OpsWidth < 2))
        break;

      ArrayRef<Value *> Ops = VL.slice(I, OpsWidth);
      // Check that a previous iteration of this loop did not delete the Value.
      if (llvm::any_of(Ops, [&R](Value *V) {
            auto *I = dyn_cast<Instruction>(V);
            return I && R.isDeleted(I);
          }))
        continue;

      LLVM_DEBUG(dbgs() << "SLP: Analyzing " << OpsWidth << " operations "
                        << "\n");

      R.buildTree(Ops);
      if (R.isTreeTinyAndNotFullyVectorizable())
        continue;
      R.reorderTopToBottom();
      R.reorderBottomToTop(!isa<InsertElementInst>(Ops.front()));
      R.buildExternalUses();

      R.computeMinimumValueSizes();
      InstructionCost Cost = R.getTreeCost();
      CandidateFound = true;
      MinCost = std::min(MinCost, Cost);

      if (Cost < -SLPCostThreshold) {
        LLVM_DEBUG(dbgs() << "SLP: Vectorizing list at cost:" << Cost << ".\n");
        R.getORE()->emit(OptimizationRemark(SV_NAME, "VectorizedList",
                                                    cast<Instruction>(Ops[0]))
                                 << "SLP vectorized with cost " << ore::NV("Cost", Cost)
                                 << " and with tree size "
                                 << ore::NV("TreeSize", R.getTreeSize()));

        R.vectorizeTree();
        // Move to the next bundle.
        I += VF - 1;
        NextInst = I + 1;
        Changed = true;
      }
    }
  }

  if (!Changed && CandidateFound) {
    R.getORE()->emit([&]() {
      return OptimizationRemarkMissed(SV_NAME, "NotBeneficial", I0)
             << "List vectorization was possible but not beneficial with cost "
             << ore::NV("Cost", MinCost) << " >= "
             << ore::NV("Treshold", -SLPCostThreshold);
    });
  } else if (!Changed) {
    R.getORE()->emit([&]() {
      return OptimizationRemarkMissed(SV_NAME, "NotPossible", I0)
             << "Cannot SLP vectorize list: vectorization was impossible"
             << " with available vectorization factors";
    });
  }
  return Changed;
}

bool SLPVectorizerPass::tryToVectorize(Instruction *I, BoUpSLP &R) {
  if (!I)
    return false;

  if (!isa<BinaryOperator>(I) && !isa<CmpInst>(I))
    return false;

  Value *P = I->getParent();

  // Vectorize in current basic block only.
  auto *Op0 = dyn_cast<Instruction>(I->getOperand(0));
  auto *Op1 = dyn_cast<Instruction>(I->getOperand(1));
  if (!Op0 || !Op1 || Op0->getParent() != P || Op1->getParent() != P)
    return false;

  // Try to vectorize V.
  if (tryToVectorizePair(Op0, Op1, R))
    return true;

  auto *A = dyn_cast<BinaryOperator>(Op0);
  auto *B = dyn_cast<BinaryOperator>(Op1);
  // Try to skip B.
  if (B && B->hasOneUse()) {
    auto *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    auto *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (B0 && B0->getParent() == P && tryToVectorizePair(A, B0, R))
      return true;
    if (B1 && B1->getParent() == P && tryToVectorizePair(A, B1, R))
      return true;
  }

  // Try to skip A.
  if (A && A->hasOneUse()) {
    auto *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    auto *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (A0 && A0->getParent() == P && tryToVectorizePair(A0, B, R))
      return true;
    if (A1 && A1->getParent() == P && tryToVectorizePair(A1, B, R))
      return true;
  }
  return false;
}

namespace {

/// Model horizontal reductions.
///
/// A horizontal reduction is a tree of reduction instructions that has values
/// that can be put into a vector as its leaves. For example:
///
/// mul mul mul mul
///  \  /    \  /
///   +       +
///    \     /
///       +
/// This tree has "mul" as its leaf values and "+" as its reduction
/// instructions. A reduction can feed into a store or a binary operation
/// feeding a phi.
///    ...
///    \  /
///     +
///     |
///  phi +=
///
///  Or:
///    ...
///    \  /
///     +
///     |
///   *p =
///
class HorizontalReduction {
  using ReductionOpsType = SmallVector<Value *, 16>;
  using ReductionOpsListType = SmallVector<ReductionOpsType, 2>;
  ReductionOpsListType ReductionOps;
  SmallVector<Value *, 32> ReducedVals;
  // Use map vector to make stable output.
  MapVector<Instruction *, Value *> ExtraArgs;
  WeakTrackingVH ReductionRoot;
  /// The type of reduction operation.
  RecurKind RdxKind;

  const unsigned INVALID_OPERAND_INDEX = std::numeric_limits<unsigned>::max();

  static bool isCmpSelMinMax(Instruction *I) {
    return match(I, m_Select(m_Cmp(), m_Value(), m_Value())) &&
           RecurrenceDescriptor::isMinMaxRecurrenceKind(getRdxKind(I));
  }

  // And/or are potentially poison-safe logical patterns like:
  // select x, y, false
  // select x, true, y
  static bool isBoolLogicOp(Instruction *I) {
    return match(I, m_LogicalAnd(m_Value(), m_Value())) ||
           match(I, m_LogicalOr(m_Value(), m_Value()));
  }

  /// Checks if instruction is associative and can be vectorized.
  static bool isVectorizable(RecurKind Kind, Instruction *I) {
    if (Kind == RecurKind::None)
      return false;

    // Integer ops that map to select instructions or intrinsics are fine.
    if (RecurrenceDescriptor::isIntMinMaxRecurrenceKind(Kind) ||
        isBoolLogicOp(I))
      return true;

    if (Kind == RecurKind::FMax || Kind == RecurKind::FMin) {
      // FP min/max are associative except for NaN and -0.0. We do not
      // have to rule out -0.0 here because the intrinsic semantics do not
      // specify a fixed result for it.
      return I->getFastMathFlags().noNaNs();
    }

    return I->isAssociative();
  }

  static Value *getRdxOperand(Instruction *I, unsigned Index) {
    // Poison-safe 'or' takes the form: select X, true, Y
    // To make that work with the normal operand processing, we skip the
    // true value operand.
    // TODO: Change the code and data structures to handle this without a hack.
    if (getRdxKind(I) == RecurKind::Or && isa<SelectInst>(I) && Index == 1)
      return I->getOperand(2);
    return I->getOperand(Index);
  }

  /// Checks if the ParentStackElem.first should be marked as a reduction
  /// operation with an extra argument or as extra argument itself.
  void markExtraArg(std::pair<Instruction *, unsigned> &ParentStackElem,
                    Value *ExtraArg) {
    if (ExtraArgs.count(ParentStackElem.first)) {
      ExtraArgs[ParentStackElem.first] = nullptr;
      // We ran into something like:
      // ParentStackElem.first = ExtraArgs[ParentStackElem.first] + ExtraArg.
      // The whole ParentStackElem.first should be considered as an extra value
      // in this case.
      // Do not perform analysis of remaining operands of ParentStackElem.first
      // instruction, this whole instruction is an extra argument.
      ParentStackElem.second = INVALID_OPERAND_INDEX;
    } else {
      // We ran into something like:
      // ParentStackElem.first += ... + ExtraArg + ...
      ExtraArgs[ParentStackElem.first] = ExtraArg;
    }
  }

  /// Creates reduction operation with the current opcode.
  static Value *createOp(IRBuilder<> &Builder, RecurKind Kind, Value *LHS,
                         Value *RHS, const Twine &Name, bool UseSelect) {
    unsigned RdxOpcode = RecurrenceDescriptor::getOpcode(Kind);
    switch (Kind) {
    case RecurKind::Or:
      if (UseSelect &&
          LHS->getType() == CmpInst::makeCmpResultType(LHS->getType()))
        return Builder.CreateSelect(LHS, Builder.getTrue(), RHS, Name);
      return Builder.CreateBinOp((Instruction::BinaryOps)RdxOpcode, LHS, RHS,
                                 Name);
    case RecurKind::And:
      if (UseSelect &&
          LHS->getType() == CmpInst::makeCmpResultType(LHS->getType()))
        return Builder.CreateSelect(LHS, RHS, Builder.getFalse(), Name);
      return Builder.CreateBinOp((Instruction::BinaryOps)RdxOpcode, LHS, RHS,
                                 Name);
    case RecurKind::Add:
    case RecurKind::Mul:
    case RecurKind::Xor:
    case RecurKind::FAdd:
    case RecurKind::FMul:
      return Builder.CreateBinOp((Instruction::BinaryOps)RdxOpcode, LHS, RHS,
                                 Name);
    case RecurKind::FMax:
      return Builder.CreateBinaryIntrinsic(Intrinsic::maxnum, LHS, RHS);
    case RecurKind::FMin:
      return Builder.CreateBinaryIntrinsic(Intrinsic::minnum, LHS, RHS);
    case RecurKind::SMax:
      if (UseSelect) {
        Value *Cmp = Builder.CreateICmpSGT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::smax, LHS, RHS);
    case RecurKind::SMin:
      if (UseSelect) {
        Value *Cmp = Builder.CreateICmpSLT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::smin, LHS, RHS);
    case RecurKind::UMax:
      if (UseSelect) {
        Value *Cmp = Builder.CreateICmpUGT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::umax, LHS, RHS);
    case RecurKind::UMin:
      if (UseSelect) {
        Value *Cmp = Builder.CreateICmpULT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::umin, LHS, RHS);
    default:
      llvm_unreachable("Unknown reduction operation.");
    }
  }

  /// Creates reduction operation with the current opcode with the IR flags
  /// from \p ReductionOps.
  static Value *createOp(IRBuilder<> &Builder, RecurKind RdxKind, Value *LHS,
                         Value *RHS, const Twine &Name,
                         const ReductionOpsListType &ReductionOps) {
    bool UseSelect = ReductionOps.size() == 2 ||
                     // Logical or/and.
                     (ReductionOps.size() == 1 &&
                      isa<SelectInst>(ReductionOps.front().front()));
    assert((!UseSelect || ReductionOps.size() != 2 ||
            isa<SelectInst>(ReductionOps[1][0])) &&
           "Expected cmp + select pairs for reduction");
    Value *Op = createOp(Builder, RdxKind, LHS, RHS, Name, UseSelect);
    if (RecurrenceDescriptor::isIntMinMaxRecurrenceKind(RdxKind)) {
      if (auto *Sel = dyn_cast<SelectInst>(Op)) {
        propagateIRFlags(Sel->getCondition(), ReductionOps[0]);
        propagateIRFlags(Op, ReductionOps[1]);
        return Op;
      }
    }
    propagateIRFlags(Op, ReductionOps[0]);
    return Op;
  }

  /// Creates reduction operation with the current opcode with the IR flags
  /// from \p I.
  static Value *createOp(IRBuilder<> &Builder, RecurKind RdxKind, Value *LHS,
                         Value *RHS, const Twine &Name, Instruction *I) {
    auto *SelI = dyn_cast<SelectInst>(I);
    Value *Op = createOp(Builder, RdxKind, LHS, RHS, Name, SelI != nullptr);
    if (SelI && RecurrenceDescriptor::isIntMinMaxRecurrenceKind(RdxKind)) {
      if (auto *Sel = dyn_cast<SelectInst>(Op))
        propagateIRFlags(Sel->getCondition(), SelI->getCondition());
    }
    propagateIRFlags(Op, I);
    return Op;
  }

  static RecurKind getRdxKind(Instruction *I) {
    assert(I && "Expected instruction for reduction matching");
    if (match(I, m_Add(m_Value(), m_Value())))
      return RecurKind::Add;
    if (match(I, m_Mul(m_Value(), m_Value())))
      return RecurKind::Mul;
    if (match(I, m_And(m_Value(), m_Value())) ||
        match(I, m_LogicalAnd(m_Value(), m_Value())))
      return RecurKind::And;
    if (match(I, m_Or(m_Value(), m_Value())) ||
        match(I, m_LogicalOr(m_Value(), m_Value())))
      return RecurKind::Or;
    if (match(I, m_Xor(m_Value(), m_Value())))
      return RecurKind::Xor;
    if (match(I, m_FAdd(m_Value(), m_Value())))
      return RecurKind::FAdd;
    if (match(I, m_FMul(m_Value(), m_Value())))
      return RecurKind::FMul;

    if (match(I, m_Intrinsic<Intrinsic::maxnum>(m_Value(), m_Value())))
      return RecurKind::FMax;
    if (match(I, m_Intrinsic<Intrinsic::minnum>(m_Value(), m_Value())))
      return RecurKind::FMin;

    // This matches either cmp+select or intrinsics. SLP is expected to handle
    // either form.
    // TODO: If we are canonicalizing to intrinsics, we can remove several
    //       special-case paths that deal with selects.
    if (match(I, m_SMax(m_Value(), m_Value())))
      return RecurKind::SMax;
    if (match(I, m_SMin(m_Value(), m_Value())))
      return RecurKind::SMin;
    if (match(I, m_UMax(m_Value(), m_Value())))
      return RecurKind::UMax;
    if (match(I, m_UMin(m_Value(), m_Value())))
      return RecurKind::UMin;

    if (auto *Select = dyn_cast<SelectInst>(I)) {
      // Try harder: look for min/max pattern based on instructions producing
      // same values such as: select ((cmp Inst1, Inst2), Inst1, Inst2).
      // During the intermediate stages of SLP, it's very common to have
      // pattern like this (since optimizeGatherSequence is run only once
      // at the end):
      // %1 = extractelement <2 x i32> %a, i32 0
      // %2 = extractelement <2 x i32> %a, i32 1
      // %cond = icmp sgt i32 %1, %2
      // %3 = extractelement <2 x i32> %a, i32 0
      // %4 = extractelement <2 x i32> %a, i32 1
      // %select = select i1 %cond, i32 %3, i32 %4
      CmpInst::Predicate Pred;
      Instruction *L1;
      Instruction *L2;

      Value *LHS = Select->getTrueValue();
      Value *RHS = Select->getFalseValue();
      Value *Cond = Select->getCondition();

      // TODO: Support inverse predicates.
      if (match(Cond, m_Cmp(Pred, m_Specific(LHS), m_Instruction(L2)))) {
        if (!isa<ExtractElementInst>(RHS) ||
            !L2->isIdenticalTo(cast<Instruction>(RHS)))
          return RecurKind::None;
      } else if (match(Cond, m_Cmp(Pred, m_Instruction(L1), m_Specific(RHS)))) {
        if (!isa<ExtractElementInst>(LHS) ||
            !L1->isIdenticalTo(cast<Instruction>(LHS)))
          return RecurKind::None;
      } else {
        if (!isa<ExtractElementInst>(LHS) || !isa<ExtractElementInst>(RHS))
          return RecurKind::None;
        if (!match(Cond, m_Cmp(Pred, m_Instruction(L1), m_Instruction(L2))) ||
            !L1->isIdenticalTo(cast<Instruction>(LHS)) ||
            !L2->isIdenticalTo(cast<Instruction>(RHS)))
          return RecurKind::None;
      }

      switch (Pred) {
      default:
        return RecurKind::None;
      case CmpInst::ICMP_SGT:
      case CmpInst::ICMP_SGE:
        return RecurKind::SMax;
      case CmpInst::ICMP_SLT:
      case CmpInst::ICMP_SLE:
        return RecurKind::SMin;
      case CmpInst::ICMP_UGT:
      case CmpInst::ICMP_UGE:
        return RecurKind::UMax;
      case CmpInst::ICMP_ULT:
      case CmpInst::ICMP_ULE:
        return RecurKind::UMin;
      }
    }
    return RecurKind::None;
  }

  /// Get the index of the first operand.
  static unsigned getFirstOperandIndex(Instruction *I) {
    return isCmpSelMinMax(I) ? 1 : 0;
  }

  /// Total number of operands in the reduction operation.
  static unsigned getNumberOfOperands(Instruction *I) {
    return isCmpSelMinMax(I) ? 3 : 2;
  }

  /// Checks if the instruction is in basic block \p BB.
  /// For a cmp+sel min/max reduction check that both ops are in \p BB.
  static bool hasSameParent(Instruction *I, BasicBlock *BB) {
    if (isCmpSelMinMax(I) || (isBoolLogicOp(I) && isa<SelectInst>(I))) {
      auto *Sel = cast<SelectInst>(I);
      auto *Cmp = dyn_cast<Instruction>(Sel->getCondition());
      return Sel->getParent() == BB && Cmp && Cmp->getParent() == BB;
    }
    return I->getParent() == BB;
  }

  /// Expected number of uses for reduction operations/reduced values.
  static bool hasRequiredNumberOfUses(bool IsCmpSelMinMax, Instruction *I) {
    if (IsCmpSelMinMax) {
      // SelectInst must be used twice while the condition op must have single
      // use only.
      if (auto *Sel = dyn_cast<SelectInst>(I))
        return Sel->hasNUses(2) && Sel->getCondition()->hasOneUse();
      return I->hasNUses(2);
    }

    // Arithmetic reduction operation must be used once only.
    return I->hasOneUse();
  }

  /// Initializes the list of reduction operations.
  void initReductionOps(Instruction *I) {
    if (isCmpSelMinMax(I))
      ReductionOps.assign(2, ReductionOpsType());
    else
      ReductionOps.assign(1, ReductionOpsType());
  }

  /// Add all reduction operations for the reduction instruction \p I.
  void addReductionOps(Instruction *I) {
    if (isCmpSelMinMax(I)) {
      ReductionOps[0].emplace_back(cast<SelectInst>(I)->getCondition());
      ReductionOps[1].emplace_back(I);
    } else {
      ReductionOps[0].emplace_back(I);
    }
  }

  static Value *getLHS(RecurKind Kind, Instruction *I) {
    if (Kind == RecurKind::None)
      return nullptr;
    return I->getOperand(getFirstOperandIndex(I));
  }
  static Value *getRHS(RecurKind Kind, Instruction *I) {
    if (Kind == RecurKind::None)
      return nullptr;
    return I->getOperand(getFirstOperandIndex(I) + 1);
  }

public:
  HorizontalReduction() = default;

  /// Try to find a reduction tree.
  bool matchAssociativeReduction(PHINode *Phi, Instruction *Inst) {
    assert((!Phi || is_contained(Phi->operands(), Inst)) &&
           "Phi needs to use the binary operator");
    assert((isa<BinaryOperator>(Inst) || isa<SelectInst>(Inst) ||
            isa<IntrinsicInst>(Inst)) &&
           "Expected binop, select, or intrinsic for reduction matching");
    RdxKind = getRdxKind(Inst);

    // We could have a initial reductions that is not an add.
    //  r *= v1 + v2 + v3 + v4
    // In such a case start looking for a tree rooted in the first '+'.
    if (Phi) {
      if (getLHS(RdxKind, Inst) == Phi) {
        Phi = nullptr;
        Inst = dyn_cast<Instruction>(getRHS(RdxKind, Inst));
        if (!Inst)
          return false;
        RdxKind = getRdxKind(Inst);
      } else if (getRHS(RdxKind, Inst) == Phi) {
        Phi = nullptr;
        Inst = dyn_cast<Instruction>(getLHS(RdxKind, Inst));
        if (!Inst)
          return false;
        RdxKind = getRdxKind(Inst);
      }
    }

    if (!isVectorizable(RdxKind, Inst))
      return false;

    // Analyze "regular" integer/FP types for reductions - no target-specific
    // types or pointers.
    Type *Ty = Inst->getType();
    if (!isValidElementType(Ty) || Ty->isPointerTy())
      return false;

    // Though the ultimate reduction may have multiple uses, its condition must
    // have only single use.
    if (auto *Sel = dyn_cast<SelectInst>(Inst))
      if (!Sel->getCondition()->hasOneUse())
        return false;

    ReductionRoot = Inst;

    // The opcode for leaf values that we perform a reduction on.
    // For example: load(x) + load(y) + load(z) + fptoui(w)
    // The leaf opcode for 'w' does not match, so we don't include it as a
    // potential candidate for the reduction.
    unsigned LeafOpcode = 0;

    // Post-order traverse the reduction tree starting at Inst. We only handle
    // true trees containing binary operators or selects.
    SmallVector<std::pair<Instruction *, unsigned>, 32> Stack;
    Stack.push_back(std::make_pair(Inst, getFirstOperandIndex(Inst)));
    initReductionOps(Inst);
    while (!Stack.empty()) {
      Instruction *TreeN = Stack.back().first;
      unsigned EdgeToVisit = Stack.back().second++;
      const RecurKind TreeRdxKind = getRdxKind(TreeN);
      bool IsReducedValue = TreeRdxKind != RdxKind;

      // Postorder visit.
      if (IsReducedValue || EdgeToVisit >= getNumberOfOperands(TreeN)) {
        if (IsReducedValue)
          ReducedVals.push_back(TreeN);
        else {
          auto ExtraArgsIter = ExtraArgs.find(TreeN);
          if (ExtraArgsIter != ExtraArgs.end() && !ExtraArgsIter->second) {
            // Check if TreeN is an extra argument of its parent operation.
            if (Stack.size() <= 1) {
              // TreeN can't be an extra argument as it is a root reduction
              // operation.
              return false;
            }
            // Yes, TreeN is an extra argument, do not add it to a list of
            // reduction operations.
            // Stack[Stack.size() - 2] always points to the parent operation.
            markExtraArg(Stack[Stack.size() - 2], TreeN);
            ExtraArgs.erase(TreeN);
          } else
            addReductionOps(TreeN);
        }
        // Retract.
        Stack.pop_back();
        continue;
      }

      // Visit operands.
      Value *EdgeVal = getRdxOperand(TreeN, EdgeToVisit);
      auto *EdgeInst = dyn_cast<Instruction>(EdgeVal);
      if (!EdgeInst) {
        // Edge value is not a reduction instruction or a leaf instruction.
        // (It may be a constant, function argument, or something else.)
        markExtraArg(Stack.back(), EdgeVal);
        continue;
      }
      RecurKind EdgeRdxKind = getRdxKind(EdgeInst);
      // Continue analysis if the next operand is a reduction operation or
      // (possibly) a leaf value. If the leaf value opcode is not set,
      // the first met operation != reduction operation is considered as the
      // leaf opcode.
      // Only handle trees in the current basic block.
      // Each tree node needs to have minimal number of users except for the
      // ultimate reduction.
      const bool IsRdxInst = EdgeRdxKind == RdxKind;
      if (EdgeInst != Phi && EdgeInst != Inst &&
          hasSameParent(EdgeInst, Inst->getParent()) &&
          hasRequiredNumberOfUses(isCmpSelMinMax(Inst), EdgeInst) &&
          (!LeafOpcode || LeafOpcode == EdgeInst->getOpcode() || IsRdxInst)) {
        if (IsRdxInst) {
          // We need to be able to reassociate the reduction operations.
          if (!isVectorizable(EdgeRdxKind, EdgeInst)) {
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), EdgeInst);
            continue;
          }
        } else if (!LeafOpcode) {
          LeafOpcode = EdgeInst->getOpcode();
        }
        Stack.push_back(
            std::make_pair(EdgeInst, getFirstOperandIndex(EdgeInst)));
        continue;
      }
      // I is an extra argument for TreeN (its parent operation).
      markExtraArg(Stack.back(), EdgeInst);
    }
    return true;
  }

  /// Attempt to vectorize the tree found by matchAssociativeReduction.
  Value *tryToReduce(BoUpSLP &V, TargetTransformInfo *TTI) {
    // If there are a sufficient number of reduction values, reduce
    // to a nearby power-of-2. We can safely generate oversized
    // vectors and rely on the backend to split them to legal sizes.
    unsigned NumReducedVals = ReducedVals.size();
    if (NumReducedVals < 4)
      return nullptr;

    // Intersect the fast-math-flags from all reduction operations.
    FastMathFlags RdxFMF;
    RdxFMF.set();
    for (ReductionOpsType &RdxOp : ReductionOps) {
      for (Value *RdxVal : RdxOp) {
        if (auto *FPMO = dyn_cast<FPMathOperator>(RdxVal))
          RdxFMF &= FPMO->getFastMathFlags();
      }
    }

    IRBuilder<> Builder(cast<Instruction>(ReductionRoot));
    Builder.setFastMathFlags(RdxFMF);

    BoUpSLP::ExtraValueToDebugLocsMap ExternallyUsedValues;
    // The same extra argument may be used several times, so log each attempt
    // to use it.
    for (const std::pair<Instruction *, Value *> &Pair : ExtraArgs) {
      assert(Pair.first && "DebugLoc must be set.");
      ExternallyUsedValues[Pair.second].push_back(Pair.first);
    }

    // The compare instruction of a min/max is the insertion point for new
    // instructions and may be replaced with a new compare instruction.
    auto getCmpForMinMaxReduction = [](Instruction *RdxRootInst) {
      assert(isa<SelectInst>(RdxRootInst) &&
             "Expected min/max reduction to have select root instruction");
      Value *ScalarCond = cast<SelectInst>(RdxRootInst)->getCondition();
      assert(isa<Instruction>(ScalarCond) &&
             "Expected min/max reduction to have compare condition");
      return cast<Instruction>(ScalarCond);
    };

    // The reduction root is used as the insertion point for new instructions,
    // so set it as externally used to prevent it from being deleted.
    ExternallyUsedValues[ReductionRoot];
    SmallVector<Value *, 16> IgnoreList;
    for (ReductionOpsType &RdxOp : ReductionOps)
      IgnoreList.append(RdxOp.begin(), RdxOp.end());

    unsigned ReduxWidth = PowerOf2Floor(NumReducedVals);
    if (NumReducedVals > ReduxWidth) {
      // In the loop below, we are building a tree based on a window of
      // 'ReduxWidth' values.
      // If the operands of those values have common traits (compare predicate,
      // constant operand, etc), then we want to group those together to
      // minimize the cost of the reduction.

      // TODO: This should be extended to count common operands for
      //       compares and binops.

      // Step 1: Count the number of times each compare predicate occurs.
      SmallDenseMap<unsigned, unsigned> PredCountMap;
      for (Value *RdxVal : ReducedVals) {
        CmpInst::Predicate Pred;
        if (match(RdxVal, m_Cmp(Pred, m_Value(), m_Value())))
          ++PredCountMap[Pred];
      }
      // Step 2: Sort the values so the most common predicates come first.
      stable_sort(ReducedVals, [&PredCountMap](Value *A, Value *B) {
        CmpInst::Predicate PredA, PredB;
        if (match(A, m_Cmp(PredA, m_Value(), m_Value())) &&
            match(B, m_Cmp(PredB, m_Value(), m_Value()))) {
          return PredCountMap[PredA] > PredCountMap[PredB];
        }
        return false;
      });
    }

    Value *VectorizedTree = nullptr;
    unsigned i = 0;
    while (i < NumReducedVals - ReduxWidth + 1 && ReduxWidth > 2) {
      ArrayRef<Value *> VL(&ReducedVals[i], ReduxWidth);
      V.buildTree(VL, IgnoreList);
      if (V.isTreeTinyAndNotFullyVectorizable(/*ForReduction=*/true))
        break;
      if (V.isLoadCombineReductionCandidate(RdxKind))
        break;
      V.reorderTopToBottom();
      V.reorderBottomToTop(/*IgnoreReorder=*/true);
      V.buildExternalUses(ExternallyUsedValues);

      // For a poison-safe boolean logic reduction, do not replace select
      // instructions with logic ops. All reduced values will be frozen (see
      // below) to prevent leaking poison.
      if (isa<SelectInst>(ReductionRoot) &&
          isBoolLogicOp(cast<Instruction>(ReductionRoot)) &&
          NumReducedVals != ReduxWidth)
        break;

      V.computeMinimumValueSizes();

      // Estimate cost.
      InstructionCost TreeCost =
          V.getTreeCost(makeArrayRef(&ReducedVals[i], ReduxWidth));
      InstructionCost ReductionCost =
          getReductionCost(TTI, ReducedVals[i], ReduxWidth, RdxFMF);
      InstructionCost Cost = TreeCost + ReductionCost;
      if (!Cost.isValid()) {
        LLVM_DEBUG(dbgs() << "Encountered invalid baseline cost.\n");
        return nullptr;
      }
      if (Cost >= -SLPCostThreshold) {
        V.getORE()->emit([&]() {
          return OptimizationRemarkMissed(SV_NAME, "HorSLPNotBeneficial",
                                          cast<Instruction>(VL[0]))
                 << "Vectorizing horizontal reduction is possible"
                 << "but not beneficial with cost " << ore::NV("Cost", Cost)
                 << " and threshold "
                 << ore::NV("Threshold", -SLPCostThreshold);
        });
        break;
      }

      LLVM_DEBUG(dbgs() << "SLP: Vectorizing horizontal reduction at cost:"
                        << Cost << ". (HorRdx)\n");
      V.getORE()->emit([&]() {
        return OptimizationRemark(SV_NAME, "VectorizedHorizontalReduction",
                                  cast<Instruction>(VL[0]))
               << "Vectorized horizontal reduction with cost "
               << ore::NV("Cost", Cost) << " and with tree size "
               << ore::NV("TreeSize", V.getTreeSize());
      });

      // Vectorize a tree.
      DebugLoc Loc = cast<Instruction>(ReducedVals[i])->getDebugLoc();
      Value *VectorizedRoot = V.vectorizeTree(ExternallyUsedValues);

      // Emit a reduction. If the root is a select (min/max idiom), the insert
      // point is the compare condition of that select.
      Instruction *RdxRootInst = cast<Instruction>(ReductionRoot);
      if (isCmpSelMinMax(RdxRootInst))
        Builder.SetInsertPoint(getCmpForMinMaxReduction(RdxRootInst));
      else
        Builder.SetInsertPoint(RdxRootInst);

      // To prevent poison from leaking across what used to be sequential, safe,
      // scalar boolean logic operations, the reduction operand must be frozen.
      if (isa<SelectInst>(RdxRootInst) && isBoolLogicOp(RdxRootInst))
        VectorizedRoot = Builder.CreateFreeze(VectorizedRoot);

      Value *ReducedSubTree =
          emitReduction(VectorizedRoot, Builder, ReduxWidth, TTI);

      if (!VectorizedTree) {
        // Initialize the final value in the reduction.
        VectorizedTree = ReducedSubTree;
      } else {
        // Update the final value in the reduction.
        Builder.SetCurrentDebugLocation(Loc);
        VectorizedTree = createOp(Builder, RdxKind, VectorizedTree,
                                  ReducedSubTree, "op.rdx", ReductionOps);
      }
      i += ReduxWidth;
      ReduxWidth = PowerOf2Floor(NumReducedVals - i);
    }

    if (VectorizedTree) {
      // Finish the reduction.
      for (; i < NumReducedVals; ++i) {
        auto *I = cast<Instruction>(ReducedVals[i]);
        Builder.SetCurrentDebugLocation(I->getDebugLoc());
        VectorizedTree =
            createOp(Builder, RdxKind, VectorizedTree, I, "", ReductionOps);
      }
      for (auto &Pair : ExternallyUsedValues) {
        // Add each externally used value to the final reduction.
        for (auto *I : Pair.second) {
          Builder.SetCurrentDebugLocation(I->getDebugLoc());
          VectorizedTree = createOp(Builder, RdxKind, VectorizedTree,
                                    Pair.first, "op.extra", I);
        }
      }

      ReductionRoot->replaceAllUsesWith(VectorizedTree);

      // Mark all scalar reduction ops for deletion, they are replaced by the
      // vector reductions.
      V.eraseInstructions(IgnoreList);
    }
    return VectorizedTree;
  }

  unsigned numReductionValues() const { return ReducedVals.size(); }

private:
  /// Calculate the cost of a reduction.
  InstructionCost getReductionCost(TargetTransformInfo *TTI,
                                   Value *FirstReducedVal, unsigned ReduxWidth,
                                   FastMathFlags FMF) {
    TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
    Type *ScalarTy = FirstReducedVal->getType();
    FixedVectorType *VectorTy = FixedVectorType::get(ScalarTy, ReduxWidth);
    InstructionCost VectorCost, ScalarCost;
    switch (RdxKind) {
    case RecurKind::Add:
    case RecurKind::Mul:
    case RecurKind::Or:
    case RecurKind::And:
    case RecurKind::Xor:
    case RecurKind::FAdd:
    case RecurKind::FMul: {
      unsigned RdxOpcode = RecurrenceDescriptor::getOpcode(RdxKind);
      VectorCost =
          TTI->getArithmeticReductionCost(RdxOpcode, VectorTy, FMF, CostKind);
      ScalarCost = TTI->getArithmeticInstrCost(RdxOpcode, ScalarTy, CostKind);
      break;
    }
    case RecurKind::FMax:
    case RecurKind::FMin: {
      auto *SclCondTy = CmpInst::makeCmpResultType(ScalarTy);
      auto *VecCondTy = cast<VectorType>(CmpInst::makeCmpResultType(VectorTy));
      VectorCost = TTI->getMinMaxReductionCost(VectorTy, VecCondTy,
                                               /*IsUnsigned=*/false, CostKind);
      CmpInst::Predicate RdxPred = getMinMaxReductionPredicate(RdxKind);
      ScalarCost = TTI->getCmpSelInstrCost(Instruction::FCmp, ScalarTy,
                                           SclCondTy, RdxPred, CostKind) +
                   TTI->getCmpSelInstrCost(Instruction::Select, ScalarTy,
                                           SclCondTy, RdxPred, CostKind);
      break;
    }
    case RecurKind::SMax:
    case RecurKind::SMin:
    case RecurKind::UMax:
    case RecurKind::UMin: {
      auto *SclCondTy = CmpInst::makeCmpResultType(ScalarTy);
      auto *VecCondTy = cast<VectorType>(CmpInst::makeCmpResultType(VectorTy));
      bool IsUnsigned =
          RdxKind == RecurKind::UMax || RdxKind == RecurKind::UMin;
      VectorCost = TTI->getMinMaxReductionCost(VectorTy, VecCondTy, IsUnsigned,
                                               CostKind);
      CmpInst::Predicate RdxPred = getMinMaxReductionPredicate(RdxKind);
      ScalarCost = TTI->getCmpSelInstrCost(Instruction::ICmp, ScalarTy,
                                           SclCondTy, RdxPred, CostKind) +
                   TTI->getCmpSelInstrCost(Instruction::Select, ScalarTy,
                                           SclCondTy, RdxPred, CostKind);
      break;
    }
    default:
      llvm_unreachable("Expected arithmetic or min/max reduction operation");
    }

    // Scalar cost is repeated for N-1 elements.
    ScalarCost *= (ReduxWidth - 1);
    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << VectorCost - ScalarCost
                      << " for reduction that starts with " << *FirstReducedVal
                      << " (It is a splitting reduction)\n");
    return VectorCost - ScalarCost;
  }

  /// Emit a horizontal reduction of the vectorized value.
  Value *emitReduction(Value *VectorizedValue, IRBuilder<> &Builder,
                       unsigned ReduxWidth, const TargetTransformInfo *TTI) {
    assert(VectorizedValue && "Need to have a vectorized tree node");
    assert(isPowerOf2_32(ReduxWidth) &&
           "We only handle power-of-two reductions for now");
    assert(RdxKind != RecurKind::FMulAdd &&
           "A call to the llvm.fmuladd intrinsic is not handled yet");

    ++NumVectorInstructions;
    return createSimpleTargetReduction(Builder, TTI, VectorizedValue, RdxKind);
  }
};

} // end anonymous namespace

static Optional<unsigned> getAggregateSize(Instruction *InsertInst) {
  if (auto *IE = dyn_cast<InsertElementInst>(InsertInst))
    return cast<FixedVectorType>(IE->getType())->getNumElements();

  unsigned AggregateSize = 1;
  auto *IV = cast<InsertValueInst>(InsertInst);
  Type *CurrentType = IV->getType();
  do {
    if (auto *ST = dyn_cast<StructType>(CurrentType)) {
      for (auto *Elt : ST->elements())
        if (Elt != ST->getElementType(0)) // check homogeneity
          return None;
      AggregateSize *= ST->getNumElements();
      CurrentType = ST->getElementType(0);
    } else if (auto *AT = dyn_cast<ArrayType>(CurrentType)) {
      AggregateSize *= AT->getNumElements();
      CurrentType = AT->getElementType();
    } else if (auto *VT = dyn_cast<FixedVectorType>(CurrentType)) {
      AggregateSize *= VT->getNumElements();
      return AggregateSize;
    } else if (CurrentType->isSingleValueType()) {
      return AggregateSize;
    } else {
      return None;
    }
  } while (true);
}

static void findBuildAggregate_rec(Instruction *LastInsertInst,
                                   TargetTransformInfo *TTI,
                                   SmallVectorImpl<Value *> &BuildVectorOpds,
                                   SmallVectorImpl<Value *> &InsertElts,
                                   unsigned OperandOffset) {
  do {
    Value *InsertedOperand = LastInsertInst->getOperand(1);
    Optional<unsigned> OperandIndex =
        getInsertIndex(LastInsertInst, OperandOffset);
    if (!OperandIndex)
      return;
    if (isa<InsertElementInst>(InsertedOperand) ||
        isa<InsertValueInst>(InsertedOperand)) {
      findBuildAggregate_rec(cast<Instruction>(InsertedOperand), TTI,
                             BuildVectorOpds, InsertElts, *OperandIndex);

    } else {
      BuildVectorOpds[*OperandIndex] = InsertedOperand;
      InsertElts[*OperandIndex] = LastInsertInst;
    }
    LastInsertInst = dyn_cast<Instruction>(LastInsertInst->getOperand(0));
  } while (LastInsertInst != nullptr &&
           (isa<InsertValueInst>(LastInsertInst) ||
            isa<InsertElementInst>(LastInsertInst)) &&
           LastInsertInst->hasOneUse());
}

/// Recognize construction of vectors like
///  %ra = insertelement <4 x float> poison, float %s0, i32 0
///  %rb = insertelement <4 x float> %ra, float %s1, i32 1
///  %rc = insertelement <4 x float> %rb, float %s2, i32 2
///  %rd = insertelement <4 x float> %rc, float %s3, i32 3
///  starting from the last insertelement or insertvalue instruction.
///
/// Also recognize homogeneous aggregates like {<2 x float>, <2 x float>},
/// {{float, float}, {float, float}}, [2 x {float, float}] and so on.
/// See llvm/test/Transforms/SLPVectorizer/X86/pr42022.ll for examples.
///
/// Assume LastInsertInst is of InsertElementInst or InsertValueInst type.
///
/// \return true if it matches.
static bool findBuildAggregate(Instruction *LastInsertInst,
                               TargetTransformInfo *TTI,
                               SmallVectorImpl<Value *> &BuildVectorOpds,
                               SmallVectorImpl<Value *> &InsertElts) {

  assert((isa<InsertElementInst>(LastInsertInst) ||
          isa<InsertValueInst>(LastInsertInst)) &&
         "Expected insertelement or insertvalue instruction!");

  assert((BuildVectorOpds.empty() && InsertElts.empty()) &&
         "Expected empty result vectors!");

  Optional<unsigned> AggregateSize = getAggregateSize(LastInsertInst);
  if (!AggregateSize)
    return false;
  BuildVectorOpds.resize(*AggregateSize);
  InsertElts.resize(*AggregateSize);

  findBuildAggregate_rec(LastInsertInst, TTI, BuildVectorOpds, InsertElts, 0);
  llvm::erase_value(BuildVectorOpds, nullptr);
  llvm::erase_value(InsertElts, nullptr);
  if (BuildVectorOpds.size() >= 2)
    return true;

  return false;
}

/// Try and get a reduction value from a phi node.
///
/// Given a phi node \p P in a block \p ParentBB, consider possible reductions
/// if they come from either \p ParentBB or a containing loop latch.
///
/// \returns A candidate reduction value if possible, or \code nullptr \endcode
/// if not possible.
static Value *getReductionValue(const DominatorTree *DT, PHINode *P,
                                BasicBlock *ParentBB, LoopInfo *LI) {
  // There are situations where the reduction value is not dominated by the
  // reduction phi. Vectorizing such cases has been reported to cause
  // miscompiles. See PR25787.
  auto DominatedReduxValue = [&](Value *R) {
    return isa<Instruction>(R) &&
           DT->dominates(P->getParent(), cast<Instruction>(R)->getParent());
  };

  Value *Rdx = nullptr;

  // Return the incoming value if it comes from the same BB as the phi node.
  if (P->getIncomingBlock(0) == ParentBB) {
    Rdx = P->getIncomingValue(0);
  } else if (P->getIncomingBlock(1) == ParentBB) {
    Rdx = P->getIncomingValue(1);
  }

  if (Rdx && DominatedReduxValue(Rdx))
    return Rdx;

  // Otherwise, check whether we have a loop latch to look at.
  Loop *BBL = LI->getLoopFor(ParentBB);
  if (!BBL)
    return nullptr;
  BasicBlock *BBLatch = BBL->getLoopLatch();
  if (!BBLatch)
    return nullptr;

  // There is a loop latch, return the incoming value if it comes from
  // that. This reduction pattern occasionally turns up.
  if (P->getIncomingBlock(0) == BBLatch) {
    Rdx = P->getIncomingValue(0);
  } else if (P->getIncomingBlock(1) == BBLatch) {
    Rdx = P->getIncomingValue(1);
  }

  if (Rdx && DominatedReduxValue(Rdx))
    return Rdx;

  return nullptr;
}

static bool matchRdxBop(Instruction *I, Value *&V0, Value *&V1) {
  if (match(I, m_BinOp(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::maxnum>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::minnum>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::smax>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::smin>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::umax>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::umin>(m_Value(V0), m_Value(V1))))
    return true;
  return false;
}

/// Attempt to reduce a horizontal reduction.
/// If it is legal to match a horizontal reduction feeding the phi node \a P
/// with reduction operators \a Root (or one of its operands) in a basic block
/// \a BB, then check if it can be done. If horizontal reduction is not found
/// and root instruction is a binary operation, vectorization of the operands is
/// attempted.
/// \returns true if a horizontal reduction was matched and reduced or operands
/// of one of the binary instruction were vectorized.
/// \returns false if a horizontal reduction was not matched (or not possible)
/// or no vectorization of any binary operation feeding \a Root instruction was
/// performed.
static bool tryToVectorizeHorReductionOrInstOperands(
    PHINode *P, Instruction *Root, BasicBlock *BB, BoUpSLP &R,
    TargetTransformInfo *TTI,
    const function_ref<bool(Instruction *, BoUpSLP &)> Vectorize) {
  if (!ShouldVectorizeHor)
    return false;

  if (!Root)
    return false;

  if (Root->getParent() != BB || isa<PHINode>(Root))
    return false;
  // Start analysis starting from Root instruction. If horizontal reduction is
  // found, try to vectorize it. If it is not a horizontal reduction or
  // vectorization is not possible or not effective, and currently analyzed
  // instruction is a binary operation, try to vectorize the operands, using
  // pre-order DFS traversal order. If the operands were not vectorized, repeat
  // the same procedure considering each operand as a possible root of the
  // horizontal reduction.
  // Interrupt the process if the Root instruction itself was vectorized or all
  // sub-trees not higher that RecursionMaxDepth were analyzed/vectorized.
  // Skip the analysis of CmpInsts.Compiler implements postanalysis of the
  // CmpInsts so we can skip extra attempts in
  // tryToVectorizeHorReductionOrInstOperands and save compile time.
  std::queue<std::pair<Instruction *, unsigned>> Stack;
  Stack.emplace(Root, 0);
  SmallPtrSet<Value *, 8> VisitedInstrs;
  SmallVector<WeakTrackingVH> PostponedInsts;
  bool Res = false;
  auto &&TryToReduce = [TTI, &P, &R](Instruction *Inst, Value *&B0,
                                     Value *&B1) -> Value * {
    bool IsBinop = matchRdxBop(Inst, B0, B1);
    bool IsSelect = match(Inst, m_Select(m_Value(), m_Value(), m_Value()));
    if (IsBinop || IsSelect) {
      HorizontalReduction HorRdx;
      if (HorRdx.matchAssociativeReduction(P, Inst))
        return HorRdx.tryToReduce(R, TTI);
    }
    return nullptr;
  };
  while (!Stack.empty()) {
    Instruction *Inst;
    unsigned Level;
    std::tie(Inst, Level) = Stack.front();
    Stack.pop();
    // Do not try to analyze instruction that has already been vectorized.
    // This may happen when we vectorize instruction operands on a previous
    // iteration while stack was populated before that happened.
    if (R.isDeleted(Inst))
      continue;
    Value *B0 = nullptr, *B1 = nullptr;
    if (Value *V = TryToReduce(Inst, B0, B1)) {
      Res = true;
      // Set P to nullptr to avoid re-analysis of phi node in
      // matchAssociativeReduction function unless this is the root node.
      P = nullptr;
      if (auto *I = dyn_cast<Instruction>(V)) {
        // Try to find another reduction.
        Stack.emplace(I, Level);
        continue;
      }
    } else {
      bool IsBinop = B0 && B1;
      if (P && IsBinop) {
        Inst = dyn_cast<Instruction>(B0);
        if (Inst == P)
          Inst = dyn_cast<Instruction>(B1);
        if (!Inst) {
          // Set P to nullptr to avoid re-analysis of phi node in
          // matchAssociativeReduction function unless this is the root node.
          P = nullptr;
          continue;
        }
      }
      // Set P to nullptr to avoid re-analysis of phi node in
      // matchAssociativeReduction function unless this is the root node.
      P = nullptr;
      // Do not try to vectorize CmpInst operands, this is done separately.
      // Final attempt for binop args vectorization should happen after the loop
      // to try to find reductions.
      if (!isa<CmpInst>(Inst))
        PostponedInsts.push_back(Inst);
    }

    // Try to vectorize operands.
    // Continue analysis for the instruction from the same basic block only to
    // save compile time.
    if (++Level < RecursionMaxDepth)
      for (auto *Op : Inst->operand_values())
        if (VisitedInstrs.insert(Op).second)
          if (auto *I = dyn_cast<Instruction>(Op))
            // Do not try to vectorize CmpInst operands,  this is done
            // separately.
            if (!isa<PHINode>(I) && !isa<CmpInst>(I) && !R.isDeleted(I) &&
                I->getParent() == BB)
              Stack.emplace(I, Level);
  }
  // Try to vectorized binops where reductions were not found.
  for (Value *V : PostponedInsts)
    if (auto *Inst = dyn_cast<Instruction>(V))
      if (!R.isDeleted(Inst))
        Res |= Vectorize(Inst, R);
  return Res;
}

bool SLPVectorizerPass::vectorizeRootInstruction(PHINode *P, Value *V,
                                                 BasicBlock *BB, BoUpSLP &R,
                                                 TargetTransformInfo *TTI) {
  auto *I = dyn_cast_or_null<Instruction>(V);
  if (!I)
    return false;

  if (!isa<BinaryOperator>(I))
    P = nullptr;
  // Try to match and vectorize a horizontal reduction.
  auto &&ExtraVectorization = [this](Instruction *I, BoUpSLP &R) -> bool {
    return tryToVectorize(I, R);
  };
  return tryToVectorizeHorReductionOrInstOperands(P, I, BB, R, TTI,
                                                  ExtraVectorization);
}

bool SLPVectorizerPass::vectorizeInsertValueInst(InsertValueInst *IVI,
                                                 BasicBlock *BB, BoUpSLP &R) {
  const DataLayout &DL = BB->getModule()->getDataLayout();
  if (!R.canMapToVector(IVI->getType(), DL))
    return false;

  SmallVector<Value *, 16> BuildVectorOpds;
  SmallVector<Value *, 16> BuildVectorInsts;
  if (!findBuildAggregate(IVI, TTI, BuildVectorOpds, BuildVectorInsts))
    return false;

  LLVM_DEBUG(dbgs() << "SLP: array mappable to vector: " << *IVI << "\n");
  // Aggregate value is unlikely to be processed in vector register.
  return tryToVectorizeList(BuildVectorOpds, R);
}

bool SLPVectorizerPass::vectorizeInsertElementInst(InsertElementInst *IEI,
                                                   BasicBlock *BB, BoUpSLP &R) {
  SmallVector<Value *, 16> BuildVectorInsts;
  SmallVector<Value *, 16> BuildVectorOpds;
  SmallVector<int> Mask;
  if (!findBuildAggregate(IEI, TTI, BuildVectorOpds, BuildVectorInsts) ||
      (llvm::all_of(
           BuildVectorOpds,
           [](Value *V) { return isa<ExtractElementInst, UndefValue>(V); }) &&
       isFixedVectorShuffle(BuildVectorOpds, Mask)))
    return false;

  LLVM_DEBUG(dbgs() << "SLP: array mappable to vector: " << *IEI << "\n");
  return tryToVectorizeList(BuildVectorInsts, R);
}

template <typename T>
static bool
tryToVectorizeSequence(SmallVectorImpl<T *> &Incoming,
                       function_ref<unsigned(T *)> Limit,
                       function_ref<bool(T *, T *)> Comparator,
                       function_ref<bool(T *, T *)> AreCompatible,
                       function_ref<bool(ArrayRef<T *>, bool)> TryToVectorizeHelper,
                       bool LimitForRegisterSize) {
  bool Changed = false;
  // Sort by type, parent, operands.
  stable_sort(Incoming, Comparator);

  // Try to vectorize elements base on their type.
  SmallVector<T *> Candidates;
  for (auto *IncIt = Incoming.begin(), *E = Incoming.end(); IncIt != E;) {
    // Look for the next elements with the same type, parent and operand
    // kinds.
    auto *SameTypeIt = IncIt;
    while (SameTypeIt != E && AreCompatible(*SameTypeIt, *IncIt))
      ++SameTypeIt;

    // Try to vectorize them.
    unsigned NumElts = (SameTypeIt - IncIt);
    LLVM_DEBUG(dbgs() << "SLP: Trying to vectorize starting at nodes ("
                      << NumElts << ")\n");
    // The vectorization is a 3-state attempt:
    // 1. Try to vectorize instructions with the same/alternate opcodes with the
    // size of maximal register at first.
    // 2. Try to vectorize remaining instructions with the same type, if
    // possible. This may result in the better vectorization results rather than
    // if we try just to vectorize instructions with the same/alternate opcodes.
    // 3. Final attempt to try to vectorize all instructions with the
    // same/alternate ops only, this may result in some extra final
    // vectorization.
    if (NumElts > 1 &&
        TryToVectorizeHelper(makeArrayRef(IncIt, NumElts), LimitForRegisterSize)) {
      // Success start over because instructions might have been changed.
      Changed = true;
    } else if (NumElts < Limit(*IncIt) &&
               (Candidates.empty() ||
                Candidates.front()->getType() == (*IncIt)->getType())) {
      Candidates.append(IncIt, std::next(IncIt, NumElts));
    }
    // Final attempt to vectorize instructions with the same types.
    if (Candidates.size() > 1 &&
        (SameTypeIt == E || (*SameTypeIt)->getType() != (*IncIt)->getType())) {
      if (TryToVectorizeHelper(Candidates, /*LimitForRegisterSize=*/false)) {
        // Success start over because instructions might have been changed.
        Changed = true;
      } else if (LimitForRegisterSize) {
        // Try to vectorize using small vectors.
        for (auto *It = Candidates.begin(), *End = Candidates.end();
             It != End;) {
          auto *SameTypeIt = It;
          while (SameTypeIt != End && AreCompatible(*SameTypeIt, *It))
            ++SameTypeIt;
          unsigned NumElts = (SameTypeIt - It);
          if (NumElts > 1 && TryToVectorizeHelper(makeArrayRef(It, NumElts),
                                            /*LimitForRegisterSize=*/false))
            Changed = true;
          It = SameTypeIt;
        }
      }
      Candidates.clear();
    }

    // Start over at the next instruction of a different type (or the end).
    IncIt = SameTypeIt;
  }
  return Changed;
}

/// Compare two cmp instructions. If IsCompatibility is true, function returns
/// true if 2 cmps have same/swapped predicates and mos compatible corresponding
/// operands. If IsCompatibility is false, function implements strict weak
/// ordering relation between two cmp instructions, returning true if the first
/// instruction is "less" than the second, i.e. its predicate is less than the
/// predicate of the second or the operands IDs are less than the operands IDs
/// of the second cmp instruction.
template <bool IsCompatibility>
static bool compareCmp(Value *V, Value *V2,
                       function_ref<bool(Instruction *)> IsDeleted) {
  auto *CI1 = cast<CmpInst>(V);
  auto *CI2 = cast<CmpInst>(V2);
  if (IsDeleted(CI2) || !isValidElementType(CI2->getType()))
    return false;
  if (CI1->getOperand(0)->getType()->getTypeID() <
      CI2->getOperand(0)->getType()->getTypeID())
    return !IsCompatibility;
  if (CI1->getOperand(0)->getType()->getTypeID() >
      CI2->getOperand(0)->getType()->getTypeID())
    return false;
  CmpInst::Predicate Pred1 = CI1->getPredicate();
  CmpInst::Predicate Pred2 = CI2->getPredicate();
  CmpInst::Predicate SwapPred1 = CmpInst::getSwappedPredicate(Pred1);
  CmpInst::Predicate SwapPred2 = CmpInst::getSwappedPredicate(Pred2);
  CmpInst::Predicate BasePred1 = std::min(Pred1, SwapPred1);
  CmpInst::Predicate BasePred2 = std::min(Pred2, SwapPred2);
  if (BasePred1 < BasePred2)
    return !IsCompatibility;
  if (BasePred1 > BasePred2)
    return false;
  // Compare operands.
  bool LEPreds = Pred1 <= Pred2;
  bool GEPreds = Pred1 >= Pred2;
  for (int I = 0, E = CI1->getNumOperands(); I < E; ++I) {
    auto *Op1 = CI1->getOperand(LEPreds ? I : E - I - 1);
    auto *Op2 = CI2->getOperand(GEPreds ? I : E - I - 1);
    if (Op1->getValueID() < Op2->getValueID())
      return !IsCompatibility;
    if (Op1->getValueID() > Op2->getValueID())
      return false;
    if (auto *I1 = dyn_cast<Instruction>(Op1))
      if (auto *I2 = dyn_cast<Instruction>(Op2)) {
        if (I1->getParent() != I2->getParent())
          return false;
        InstructionsState S = getSameOpcode({I1, I2});
        if (S.getOpcode())
          continue;
        return false;
      }
  }
  return IsCompatibility;
}

bool SLPVectorizerPass::vectorizeSimpleInstructions(
    SmallVectorImpl<Instruction *> &Instructions, BasicBlock *BB, BoUpSLP &R,
    bool AtTerminator) {
  bool OpsChanged = false;
  SmallVector<Instruction *, 4> PostponedCmps;
  for (auto *I : reverse(Instructions)) {
    if (R.isDeleted(I))
      continue;
    if (auto *LastInsertValue = dyn_cast<InsertValueInst>(I))
      OpsChanged |= vectorizeInsertValueInst(LastInsertValue, BB, R);
    else if (auto *LastInsertElem = dyn_cast<InsertElementInst>(I))
      OpsChanged |= vectorizeInsertElementInst(LastInsertElem, BB, R);
    else if (isa<CmpInst>(I))
      PostponedCmps.push_back(I);
  }
  if (AtTerminator) {
    // Try to find reductions first.
    for (Instruction *I : PostponedCmps) {
      if (R.isDeleted(I))
        continue;
      for (Value *Op : I->operands())
        OpsChanged |= vectorizeRootInstruction(nullptr, Op, BB, R, TTI);
    }
    // Try to vectorize operands as vector bundles.
    for (Instruction *I : PostponedCmps) {
      if (R.isDeleted(I))
        continue;
      OpsChanged |= tryToVectorize(I, R);
    }
    // Try to vectorize list of compares.
    // Sort by type, compare predicate, etc.
    auto &&CompareSorter = [&R](Value *V, Value *V2) {
      return compareCmp<false>(V, V2,
                               [&R](Instruction *I) { return R.isDeleted(I); });
    };

    auto &&AreCompatibleCompares = [&R](Value *V1, Value *V2) {
      if (V1 == V2)
        return true;
      return compareCmp<true>(V1, V2,
                              [&R](Instruction *I) { return R.isDeleted(I); });
    };
    auto Limit = [&R](Value *V) {
      unsigned EltSize = R.getVectorElementSize(V);
      return std::max(2U, R.getMaxVecRegSize() / EltSize);
    };

    SmallVector<Value *> Vals(PostponedCmps.begin(), PostponedCmps.end());
    OpsChanged |= tryToVectorizeSequence<Value>(
        Vals, Limit, CompareSorter, AreCompatibleCompares,
        [this, &R](ArrayRef<Value *> Candidates, bool LimitForRegisterSize) {
          // Exclude possible reductions from other blocks.
          bool ArePossiblyReducedInOtherBlock =
              any_of(Candidates, [](Value *V) {
                return any_of(V->users(), [V](User *U) {
                  return isa<SelectInst>(U) &&
                         cast<SelectInst>(U)->getParent() !=
                             cast<Instruction>(V)->getParent();
                });
              });
          if (ArePossiblyReducedInOtherBlock)
            return false;
          return tryToVectorizeList(Candidates, R, LimitForRegisterSize);
        },
        /*LimitForRegisterSize=*/true);
    Instructions.clear();
  } else {
    // Insert in reverse order since the PostponedCmps vector was filled in
    // reverse order.
    Instructions.assign(PostponedCmps.rbegin(), PostponedCmps.rend());
  }
  return OpsChanged;
}

bool SLPVectorizerPass::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  SmallVector<Value *, 4> Incoming;
  SmallPtrSet<Value *, 16> VisitedInstrs;
  // Maps phi nodes to the non-phi nodes found in the use tree for each phi
  // node. Allows better to identify the chains that can be vectorized in the
  // better way.
  DenseMap<Value *, SmallVector<Value *, 4>> PHIToOpcodes;
  auto PHICompare = [this, &PHIToOpcodes](Value *V1, Value *V2) {
    assert(isValidElementType(V1->getType()) &&
           isValidElementType(V2->getType()) &&
           "Expected vectorizable types only.");
    // It is fine to compare type IDs here, since we expect only vectorizable
    // types, like ints, floats and pointers, we don't care about other type.
    if (V1->getType()->getTypeID() < V2->getType()->getTypeID())
      return true;
    if (V1->getType()->getTypeID() > V2->getType()->getTypeID())
      return false;
    ArrayRef<Value *> Opcodes1 = PHIToOpcodes[V1];
    ArrayRef<Value *> Opcodes2 = PHIToOpcodes[V2];
    if (Opcodes1.size() < Opcodes2.size())
      return true;
    if (Opcodes1.size() > Opcodes2.size())
      return false;
    Optional<bool> ConstOrder;
    for (int I = 0, E = Opcodes1.size(); I < E; ++I) {
      // Undefs are compatible with any other value.
      if (isa<UndefValue>(Opcodes1[I]) || isa<UndefValue>(Opcodes2[I])) {
        if (!ConstOrder)
          ConstOrder =
              !isa<UndefValue>(Opcodes1[I]) && isa<UndefValue>(Opcodes2[I]);
        continue;
      }
      if (auto *I1 = dyn_cast<Instruction>(Opcodes1[I]))
        if (auto *I2 = dyn_cast<Instruction>(Opcodes2[I])) {
          DomTreeNodeBase<BasicBlock> *NodeI1 = DT->getNode(I1->getParent());
          DomTreeNodeBase<BasicBlock> *NodeI2 = DT->getNode(I2->getParent());
          if (!NodeI1)
            return NodeI2 != nullptr;
          if (!NodeI2)
            return false;
          assert((NodeI1 == NodeI2) ==
                     (NodeI1->getDFSNumIn() == NodeI2->getDFSNumIn()) &&
                 "Different nodes should have different DFS numbers");
          if (NodeI1 != NodeI2)
            return NodeI1->getDFSNumIn() < NodeI2->getDFSNumIn();
          InstructionsState S = getSameOpcode({I1, I2});
          if (S.getOpcode())
            continue;
          return I1->getOpcode() < I2->getOpcode();
        }
      if (isa<Constant>(Opcodes1[I]) && isa<Constant>(Opcodes2[I])) {
        if (!ConstOrder)
          ConstOrder = Opcodes1[I]->getValueID() < Opcodes2[I]->getValueID();
        continue;
      }
      if (Opcodes1[I]->getValueID() < Opcodes2[I]->getValueID())
        return true;
      if (Opcodes1[I]->getValueID() > Opcodes2[I]->getValueID())
        return false;
    }
    return ConstOrder && *ConstOrder;
  };
  auto AreCompatiblePHIs = [&PHIToOpcodes](Value *V1, Value *V2) {
    if (V1 == V2)
      return true;
    if (V1->getType() != V2->getType())
      return false;
    ArrayRef<Value *> Opcodes1 = PHIToOpcodes[V1];
    ArrayRef<Value *> Opcodes2 = PHIToOpcodes[V2];
    if (Opcodes1.size() != Opcodes2.size())
      return false;
    for (int I = 0, E = Opcodes1.size(); I < E; ++I) {
      // Undefs are compatible with any other value.
      if (isa<UndefValue>(Opcodes1[I]) || isa<UndefValue>(Opcodes2[I]))
        continue;
      if (auto *I1 = dyn_cast<Instruction>(Opcodes1[I]))
        if (auto *I2 = dyn_cast<Instruction>(Opcodes2[I])) {
          if (I1->getParent() != I2->getParent())
            return false;
          InstructionsState S = getSameOpcode({I1, I2});
          if (S.getOpcode())
            continue;
          return false;
        }
      if (isa<Constant>(Opcodes1[I]) && isa<Constant>(Opcodes2[I]))
        continue;
      if (Opcodes1[I]->getValueID() != Opcodes2[I]->getValueID())
        return false;
    }
    return true;
  };
  auto Limit = [&R](Value *V) {
    unsigned EltSize = R.getVectorElementSize(V);
    return std::max(2U, R.getMaxVecRegSize() / EltSize);
  };

  bool HaveVectorizedPhiNodes = false;
  do {
    // Collect the incoming values from the PHIs.
    Incoming.clear();
    for (Instruction &I : *BB) {
      PHINode *P = dyn_cast<PHINode>(&I);
      if (!P)
        break;

      // No need to analyze deleted, vectorized and non-vectorizable
      // instructions.
      if (!VisitedInstrs.count(P) && !R.isDeleted(P) &&
          isValidElementType(P->getType()))
        Incoming.push_back(P);
    }

    // Find the corresponding non-phi nodes for better matching when trying to
    // build the tree.
    for (Value *V : Incoming) {
      SmallVectorImpl<Value *> &Opcodes =
          PHIToOpcodes.try_emplace(V).first->getSecond();
      if (!Opcodes.empty())
        continue;
      SmallVector<Value *, 4> Nodes(1, V);
      SmallPtrSet<Value *, 4> Visited;
      while (!Nodes.empty()) {
        auto *PHI = cast<PHINode>(Nodes.pop_back_val());
        if (!Visited.insert(PHI).second)
          continue;
        for (Value *V : PHI->incoming_values()) {
          if (auto *PHI1 = dyn_cast<PHINode>((V))) {
            Nodes.push_back(PHI1);
            continue;
          }
          Opcodes.emplace_back(V);
        }
      }
    }

    HaveVectorizedPhiNodes = tryToVectorizeSequence<Value>(
        Incoming, Limit, PHICompare, AreCompatiblePHIs,
        [this, &R](ArrayRef<Value *> Candidates, bool LimitForRegisterSize) {
          return tryToVectorizeList(Candidates, R, LimitForRegisterSize);
        },
        /*LimitForRegisterSize=*/true);
    Changed |= HaveVectorizedPhiNodes;
    VisitedInstrs.insert(Incoming.begin(), Incoming.end());
  } while (HaveVectorizedPhiNodes);

  VisitedInstrs.clear();

  SmallVector<Instruction *, 8> PostProcessInstructions;
  SmallDenseSet<Instruction *, 4> KeyNodes;
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    // Skip instructions with scalable type. The num of elements is unknown at
    // compile-time for scalable type.
    if (isa<ScalableVectorType>(it->getType()))
      continue;

    // Skip instructions marked for the deletion.
    if (R.isDeleted(&*it))
      continue;
    // We may go through BB multiple times so skip the one we have checked.
    if (!VisitedInstrs.insert(&*it).second) {
      if (it->use_empty() && KeyNodes.contains(&*it) &&
          vectorizeSimpleInstructions(PostProcessInstructions, BB, R,
                                      it->isTerminator())) {
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        Changed = true;
        it = BB->begin();
        e = BB->end();
      }
      continue;
    }

    if (isa<DbgInfoIntrinsic>(it))
      continue;

    // Try to vectorize reductions that use PHINodes.
    if (PHINode *P = dyn_cast<PHINode>(it)) {
      // Check that the PHI is a reduction PHI.
      if (P->getNumIncomingValues() == 2) {
        // Try to match and vectorize a horizontal reduction.
        if (vectorizeRootInstruction(P, getReductionValue(DT, P, BB, LI), BB, R,
                                     TTI)) {
          Changed = true;
          it = BB->begin();
          e = BB->end();
          continue;
        }
      }
      // Try to vectorize the incoming values of the PHI, to catch reductions
      // that feed into PHIs.
      for (unsigned I = 0, E = P->getNumIncomingValues(); I != E; I++) {
        // Skip if the incoming block is the current BB for now. Also, bypass
        // unreachable IR for efficiency and to avoid crashing.
        // TODO: Collect the skipped incoming values and try to vectorize them
        // after processing BB.
        if (BB == P->getIncomingBlock(I) ||
            !DT->isReachableFromEntry(P->getIncomingBlock(I)))
          continue;

        Changed |= vectorizeRootInstruction(nullptr, P->getIncomingValue(I),
                                            P->getIncomingBlock(I), R, TTI);
      }
      continue;
    }

    // Ran into an instruction without users, like terminator, or function call
    // with ignored return value, store. Ignore unused instructions (basing on
    // instruction type, except for CallInst and InvokeInst).
    if (it->use_empty() && (it->getType()->isVoidTy() || isa<CallInst>(it) ||
                            isa<InvokeInst>(it))) {
      KeyNodes.insert(&*it);
      bool OpsChanged = false;
      if (ShouldStartVectorizeHorAtStore || !isa<StoreInst>(it)) {
        for (auto *V : it->operand_values()) {
          // Try to match and vectorize a horizontal reduction.
          OpsChanged |= vectorizeRootInstruction(nullptr, V, BB, R, TTI);
        }
      }
      // Start vectorization of post-process list of instructions from the
      // top-tree instructions to try to vectorize as many instructions as
      // possible.
      OpsChanged |= vectorizeSimpleInstructions(PostProcessInstructions, BB, R,
                                                it->isTerminator());
      if (OpsChanged) {
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }
    }

    if (isa<InsertElementInst>(it) || isa<CmpInst>(it) ||
        isa<InsertValueInst>(it))
      PostProcessInstructions.push_back(&*it);
  }

  return Changed;
}

bool SLPVectorizerPass::vectorizeGEPIndices(BasicBlock *BB, BoUpSLP &R) {
  auto Changed = false;
  for (auto &Entry : GEPs) {
    // If the getelementptr list has fewer than two elements, there's nothing
    // to do.
    if (Entry.second.size() < 2)
      continue;

    LLVM_DEBUG(dbgs() << "SLP: Analyzing a getelementptr list of length "
                      << Entry.second.size() << ".\n");

    // Process the GEP list in chunks suitable for the target's supported
    // vector size. If a vector register can't hold 1 element, we are done. We
    // are trying to vectorize the index computations, so the maximum number of
    // elements is based on the size of the index expression, rather than the
    // size of the GEP itself (the target's pointer size).
    unsigned MaxVecRegSize = R.getMaxVecRegSize();
    unsigned EltSize = R.getVectorElementSize(*Entry.second[0]->idx_begin());
    if (MaxVecRegSize < EltSize)
      continue;

    unsigned MaxElts = MaxVecRegSize / EltSize;
    for (unsigned BI = 0, BE = Entry.second.size(); BI < BE; BI += MaxElts) {
      auto Len = std::min<unsigned>(BE - BI, MaxElts);
      ArrayRef<GetElementPtrInst *> GEPList(&Entry.second[BI], Len);

      // Initialize a set a candidate getelementptrs. Note that we use a
      // SetVector here to preserve program order. If the index computations
      // are vectorizable and begin with loads, we want to minimize the chance
      // of having to reorder them later.
      SetVector<Value *> Candidates(GEPList.begin(), GEPList.end());

      // Some of the candidates may have already been vectorized after we
      // initially collected them. If so, they are marked as deleted, so remove
      // them from the set of candidates.
      Candidates.remove_if(
          [&R](Value *I) { return R.isDeleted(cast<Instruction>(I)); });

      // Remove from the set of candidates all pairs of getelementptrs with
      // constant differences. Such getelementptrs are likely not good
      // candidates for vectorization in a bottom-up phase since one can be
      // computed from the other. We also ensure all candidate getelementptr
      // indices are unique.
      for (int I = 0, E = GEPList.size(); I < E && Candidates.size() > 1; ++I) {
        auto *GEPI = GEPList[I];
        if (!Candidates.count(GEPI))
          continue;
        auto *SCEVI = SE->getSCEV(GEPList[I]);
        for (int J = I + 1; J < E && Candidates.size() > 1; ++J) {
          auto *GEPJ = GEPList[J];
          auto *SCEVJ = SE->getSCEV(GEPList[J]);
          if (isa<SCEVConstant>(SE->getMinusSCEV(SCEVI, SCEVJ))) {
            Candidates.remove(GEPI);
            Candidates.remove(GEPJ);
          } else if (GEPI->idx_begin()->get() == GEPJ->idx_begin()->get()) {
            Candidates.remove(GEPJ);
          }
        }
      }

      // We break out of the above computation as soon as we know there are
      // fewer than two candidates remaining.
      if (Candidates.size() < 2)
        continue;

      // Add the single, non-constant index of each candidate to the bundle. We
      // ensured the indices met these constraints when we originally collected
      // the getelementptrs.
      SmallVector<Value *, 16> Bundle(Candidates.size());
      auto BundleIndex = 0u;
      for (auto *V : Candidates) {
        auto *GEP = cast<GetElementPtrInst>(V);
        auto *GEPIdx = GEP->idx_begin()->get();
        assert(GEP->getNumIndices() == 1 || !isa<Constant>(GEPIdx));
        Bundle[BundleIndex++] = GEPIdx;
      }

      // Try and vectorize the indices. We are currently only interested in
      // gather-like cases of the form:
      //
      // ... = g[a[0] - b[0]] + g[a[1] - b[1]] + ...
      //
      // where the loads of "a", the loads of "b", and the subtractions can be
      // performed in parallel. It's likely that detecting this pattern in a
      // bottom-up phase will be simpler and less costly than building a
      // full-blown top-down phase beginning at the consecutive loads.
      Changed |= tryToVectorizeList(Bundle, R);
    }
  }
  return Changed;
}

bool SLPVectorizerPass::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Sort by type, base pointers and values operand. Value operands must be
  // compatible (have the same opcode, same parent), otherwise it is
  // definitely not profitable to try to vectorize them.
  auto &&StoreSorter = [this](StoreInst *V, StoreInst *V2) {
    if (V->getPointerOperandType()->getTypeID() <
        V2->getPointerOperandType()->getTypeID())
      return true;
    if (V->getPointerOperandType()->getTypeID() >
        V2->getPointerOperandType()->getTypeID())
      return false;
    // UndefValues are compatible with all other values.
    if (isa<UndefValue>(V->getValueOperand()) ||
        isa<UndefValue>(V2->getValueOperand()))
      return false;
    if (auto *I1 = dyn_cast<Instruction>(V->getValueOperand()))
      if (auto *I2 = dyn_cast<Instruction>(V2->getValueOperand())) {
        DomTreeNodeBase<llvm::BasicBlock> *NodeI1 =
            DT->getNode(I1->getParent());
        DomTreeNodeBase<llvm::BasicBlock> *NodeI2 =
            DT->getNode(I2->getParent());
        assert(NodeI1 && "Should only process reachable instructions");
        assert(NodeI1 && "Should only process reachable instructions");
        assert((NodeI1 == NodeI2) ==
                   (NodeI1->getDFSNumIn() == NodeI2->getDFSNumIn()) &&
               "Different nodes should have different DFS numbers");
        if (NodeI1 != NodeI2)
          return NodeI1->getDFSNumIn() < NodeI2->getDFSNumIn();
        InstructionsState S = getSameOpcode({I1, I2});
        if (S.getOpcode())
          return false;
        return I1->getOpcode() < I2->getOpcode();
      }
    if (isa<Constant>(V->getValueOperand()) &&
        isa<Constant>(V2->getValueOperand()))
      return false;
    return V->getValueOperand()->getValueID() <
           V2->getValueOperand()->getValueID();
  };

  auto &&AreCompatibleStores = [](StoreInst *V1, StoreInst *V2) {
    if (V1 == V2)
      return true;
    if (V1->getPointerOperandType() != V2->getPointerOperandType())
      return false;
    // Undefs are compatible with any other value.
    if (isa<UndefValue>(V1->getValueOperand()) ||
        isa<UndefValue>(V2->getValueOperand()))
      return true;
    if (auto *I1 = dyn_cast<Instruction>(V1->getValueOperand()))
      if (auto *I2 = dyn_cast<Instruction>(V2->getValueOperand())) {
        if (I1->getParent() != I2->getParent())
          return false;
        InstructionsState S = getSameOpcode({I1, I2});
        return S.getOpcode() > 0;
      }
    if (isa<Constant>(V1->getValueOperand()) &&
        isa<Constant>(V2->getValueOperand()))
      return true;
    return V1->getValueOperand()->getValueID() ==
           V2->getValueOperand()->getValueID();
  };
  auto Limit = [&R, this](StoreInst *SI) {
    unsigned EltSize = DL->getTypeSizeInBits(SI->getValueOperand()->getType());
    return R.getMinVF(EltSize);
  };

  // Attempt to sort and vectorize each of the store-groups.
  for (auto &Pair : Stores) {
    if (Pair.second.size() < 2)
      continue;

    LLVM_DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
                      << Pair.second.size() << ".\n");

    if (!isValidElementType(Pair.second.front()->getValueOperand()->getType()))
      continue;

    Changed |= tryToVectorizeSequence<StoreInst>(
        Pair.second, Limit, StoreSorter, AreCompatibleStores,
        [this, &R](ArrayRef<StoreInst *> Candidates, bool) {
          return vectorizeStores(Candidates, R);
        },
        /*LimitForRegisterSize=*/false);
  }
  return Changed;
}

char SLPVectorizer::ID = 0;

static const char lv_name[] = "SLP Vectorizer";

INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DemandedBitsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_DEPENDENCY(InjectTLIMappingsLegacy)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

Pass *llvm::createSLPVectorizerPass() { return new SLPVectorizer(); }
