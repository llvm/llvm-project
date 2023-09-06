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
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#ifdef EXPENSIVE_CHECKS
#include "llvm/IR/Verifier.h"
#endif
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
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
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

// NOTE: If AllowHorRdxIdenityOptimization is true, the optimization will run
// even if we match a reduction but do not vectorize in the end.
static cl::opt<bool> AllowHorRdxIdenityOptimization(
    "slp-optimize-identity-hor-reduction-ops", cl::init(true), cl::Hidden,
    cl::desc("Allow optimization of original scalar identity operations on "
             "matched horizontal reductions."));

static cl::opt<int>
MaxVectorRegSizeOption("slp-max-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

static cl::opt<unsigned>
MaxVFOption("slp-max-vf", cl::init(0), cl::Hidden,
    cl::desc("Maximum SLP vectorization factor (0=unlimited)"));

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

// The maximum depth that the look-ahead score heuristic will explore
// when it probing among candidates for vectorization tree roots.
// The higher this value, the higher the compilation time overhead but unlike
// similar limit for operands ordering this is less frequently used, hence
// impact of higher value is less noticeable.
static cl::opt<int> RootLookAheadMaxDepth(
    "slp-max-root-look-ahead-depth", cl::init(2), cl::Hidden,
    cl::desc("The maximum look-ahead depth for searching best rooting option"));

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
  return isa<Constant>(V) && !isa<ConstantExpr, GlobalValue>(V);
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

#if !defined(NDEBUG)
/// Print a short descriptor of the instruction bundle suitable for debug output.
static std::string shortBundleName(ArrayRef<Value *> VL) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "n=" << VL.size() << " [" << *VL.front() << ", ..]";
  OS.flush();
  return Result;
}
#endif

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

/// \returns inserting index of InsertElement or InsertValue instruction,
/// using Offset as base offset for index.
static std::optional<unsigned> getInsertIndex(const Value *InsertInst,
                                              unsigned Offset = 0) {
  int Index = Offset;
  if (const auto *IE = dyn_cast<InsertElementInst>(InsertInst)) {
    const auto *VT = dyn_cast<FixedVectorType>(IE->getType());
    if (!VT)
      return std::nullopt;
    const auto *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
    if (!CI)
      return std::nullopt;
    if (CI->getValue().uge(VT->getNumElements()))
      return std::nullopt;
    Index *= VT->getNumElements();
    Index += CI->getZExtValue();
    return Index;
  }

  const auto *IV = cast<InsertValueInst>(InsertInst);
  Type *CurrentType = IV->getType();
  for (unsigned I : IV->indices()) {
    if (const auto *ST = dyn_cast<StructType>(CurrentType)) {
      Index *= ST->getNumElements();
      CurrentType = ST->getElementType(I);
    } else if (const auto *AT = dyn_cast<ArrayType>(CurrentType)) {
      Index *= AT->getNumElements();
      CurrentType = AT->getElementType();
    } else {
      return std::nullopt;
    }
    Index += I;
  }
  return Index;
}

namespace {
/// Specifies the way the mask should be analyzed for undefs/poisonous elements
/// in the shuffle mask.
enum class UseMask {
  FirstArg, ///< The mask is expected to be for permutation of 1-2 vectors,
            ///< check for the mask elements for the first argument (mask
            ///< indices are in range [0:VF)).
  SecondArg, ///< The mask is expected to be for permutation of 2 vectors, check
             ///< for the mask elements for the second argument (mask indices
             ///< are in range [VF:2*VF))
  UndefsAsMask ///< Consider undef mask elements (-1) as placeholders for
               ///< future shuffle elements and mark them as ones as being used
               ///< in future. Non-undef elements are considered as unused since
               ///< they're already marked as used in the mask.
};
} // namespace

/// Prepares a use bitset for the given mask either for the first argument or
/// for the second.
static SmallBitVector buildUseMask(int VF, ArrayRef<int> Mask,
                                   UseMask MaskArg) {
  SmallBitVector UseMask(VF, true);
  for (auto [Idx, Value] : enumerate(Mask)) {
    if (Value == PoisonMaskElem) {
      if (MaskArg == UseMask::UndefsAsMask)
        UseMask.reset(Idx);
      continue;
    }
    if (MaskArg == UseMask::FirstArg && Value < VF)
      UseMask.reset(Value);
    else if (MaskArg == UseMask::SecondArg && Value >= VF)
      UseMask.reset(Value - VF);
  }
  return UseMask;
}

/// Checks if the given value is actually an undefined constant vector.
/// Also, if the \p UseMask is not empty, tries to check if the non-masked
/// elements actually mask the insertelement buildvector, if any.
template <bool IsPoisonOnly = false>
static SmallBitVector isUndefVector(const Value *V,
                                    const SmallBitVector &UseMask = {}) {
  SmallBitVector Res(UseMask.empty() ? 1 : UseMask.size(), true);
  using T = std::conditional_t<IsPoisonOnly, PoisonValue, UndefValue>;
  if (isa<T>(V))
    return Res;
  auto *VecTy = dyn_cast<FixedVectorType>(V->getType());
  if (!VecTy)
    return Res.reset();
  auto *C = dyn_cast<Constant>(V);
  if (!C) {
    if (!UseMask.empty()) {
      const Value *Base = V;
      while (auto *II = dyn_cast<InsertElementInst>(Base)) {
        Base = II->getOperand(0);
        if (isa<T>(II->getOperand(1)))
          continue;
        std::optional<unsigned> Idx = getInsertIndex(II);
        if (!Idx)
          continue;
        if (*Idx < UseMask.size() && !UseMask.test(*Idx))
          Res.reset(*Idx);
      }
      // TODO: Add analysis for shuffles here too.
      if (V == Base) {
        Res.reset();
      } else {
        SmallBitVector SubMask(UseMask.size(), false);
        Res &= isUndefVector<IsPoisonOnly>(Base, SubMask);
      }
    } else {
      Res.reset();
    }
    return Res;
  }
  for (unsigned I = 0, E = VecTy->getNumElements(); I != E; ++I) {
    if (Constant *Elem = C->getAggregateElement(I))
      if (!isa<T>(Elem) &&
          (UseMask.empty() || (I < UseMask.size() && !UseMask.test(I))))
        Res.reset(I);
  }
  return Res;
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
/// ShuffleVectorInst/getShuffleCost?
static std::optional<TargetTransformInfo::ShuffleKind>
isFixedVectorShuffle(ArrayRef<Value *> VL, SmallVectorImpl<int> &Mask) {
  const auto *It =
      find_if(VL, [](Value *V) { return isa<ExtractElementInst>(V); });
  if (It == VL.end())
    return std::nullopt;
  auto *EI0 = cast<ExtractElementInst>(*It);
  if (isa<ScalableVectorType>(EI0->getVectorOperandType()))
    return std::nullopt;
  unsigned Size =
      cast<FixedVectorType>(EI0->getVectorOperandType())->getNumElements();
  Value *Vec1 = nullptr;
  Value *Vec2 = nullptr;
  enum ShuffleMode { Unknown, Select, Permute };
  ShuffleMode CommonShuffleMode = Unknown;
  Mask.assign(VL.size(), PoisonMaskElem);
  for (unsigned I = 0, E = VL.size(); I < E; ++I) {
    // Undef can be represented as an undef element in a vector.
    if (isa<UndefValue>(VL[I]))
      continue;
    auto *EI = cast<ExtractElementInst>(VL[I]);
    if (isa<ScalableVectorType>(EI->getVectorOperandType()))
      return std::nullopt;
    auto *Vec = EI->getVectorOperand();
    // We can extractelement from undef or poison vector.
    if (isUndefVector(Vec).all())
      continue;
    // All vector operands must have the same number of vector elements.
    if (cast<FixedVectorType>(Vec->getType())->getNumElements() != Size)
      return std::nullopt;
    if (isa<UndefValue>(EI->getIndexOperand()))
      continue;
    auto *Idx = dyn_cast<ConstantInt>(EI->getIndexOperand());
    if (!Idx)
      return std::nullopt;
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
      return std::nullopt;
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

/// \returns True if Extract{Value,Element} instruction extracts element Idx.
static std::optional<unsigned> getExtractIndex(Instruction *E) {
  unsigned Opcode = E->getOpcode();
  assert((Opcode == Instruction::ExtractElement ||
          Opcode == Instruction::ExtractValue) &&
         "Expected extractelement or extractvalue instruction.");
  if (Opcode == Instruction::ExtractElement) {
    auto *CI = dyn_cast<ConstantInt>(E->getOperand(1));
    if (!CI)
      return std::nullopt;
    return CI->getZExtValue();
  }
  auto *EI = cast<ExtractValueInst>(E);
  if (EI->getNumIndices() != 1)
    return std::nullopt;
  return *EI->idx_begin();
}

/// Tries to find extractelement instructions with constant indices from fixed
/// vector type and gather such instructions into a bunch, which highly likely
/// might be detected as a shuffle of 1 or 2 input vectors. If this attempt was
/// successful, the matched scalars are replaced by poison values in \p VL for
/// future analysis.
static std::optional<TTI::ShuffleKind>
tryToGatherExtractElements(SmallVectorImpl<Value *> &VL,
                           SmallVectorImpl<int> &Mask) {
  // Scan list of gathered scalars for extractelements that can be represented
  // as shuffles.
  MapVector<Value *, SmallVector<int>> VectorOpToIdx;
  SmallVector<int> UndefVectorExtracts;
  for (int I = 0, E = VL.size(); I < E; ++I) {
    auto *EI = dyn_cast<ExtractElementInst>(VL[I]);
    if (!EI) {
      if (isa<UndefValue>(VL[I]))
        UndefVectorExtracts.push_back(I);
      continue;
    }
    auto *VecTy = dyn_cast<FixedVectorType>(EI->getVectorOperandType());
    if (!VecTy || !isa<ConstantInt, UndefValue>(EI->getIndexOperand()))
      continue;
    std::optional<unsigned> Idx = getExtractIndex(EI);
    // Undefined index.
    if (!Idx) {
      UndefVectorExtracts.push_back(I);
      continue;
    }
    SmallBitVector ExtractMask(VecTy->getNumElements(), true);
    ExtractMask.reset(*Idx);
    if (isUndefVector(EI->getVectorOperand(), ExtractMask).all()) {
      UndefVectorExtracts.push_back(I);
      continue;
    }
    VectorOpToIdx[EI->getVectorOperand()].push_back(I);
  }
  // Sort the vector operands by the maximum number of uses in extractelements.
  MapVector<unsigned, SmallVector<Value *>> VFToVector;
  for (const auto &Data : VectorOpToIdx)
    VFToVector[cast<FixedVectorType>(Data.first->getType())->getNumElements()]
        .push_back(Data.first);
  for (auto &Data : VFToVector) {
    stable_sort(Data.second, [&VectorOpToIdx](Value *V1, Value *V2) {
      return VectorOpToIdx.find(V1)->second.size() >
             VectorOpToIdx.find(V2)->second.size();
    });
  }
  // Find the best pair of the vectors with the same number of elements or a
  // single vector.
  const int UndefSz = UndefVectorExtracts.size();
  unsigned SingleMax = 0;
  Value *SingleVec = nullptr;
  unsigned PairMax = 0;
  std::pair<Value *, Value *> PairVec(nullptr, nullptr);
  for (auto &Data : VFToVector) {
    Value *V1 = Data.second.front();
    if (SingleMax < VectorOpToIdx[V1].size() + UndefSz) {
      SingleMax = VectorOpToIdx[V1].size() + UndefSz;
      SingleVec = V1;
    }
    Value *V2 = nullptr;
    if (Data.second.size() > 1)
      V2 = *std::next(Data.second.begin());
    if (V2 && PairMax < VectorOpToIdx[V1].size() + VectorOpToIdx[V2].size() +
                            UndefSz) {
      PairMax = VectorOpToIdx[V1].size() + VectorOpToIdx[V2].size() + UndefSz;
      PairVec = std::make_pair(V1, V2);
    }
  }
  if (SingleMax == 0 && PairMax == 0 && UndefSz == 0)
    return std::nullopt;
  // Check if better to perform a shuffle of 2 vectors or just of a single
  // vector.
  SmallVector<Value *> SavedVL(VL.begin(), VL.end());
  SmallVector<Value *> GatheredExtracts(
      VL.size(), PoisonValue::get(VL.front()->getType()));
  if (SingleMax >= PairMax && SingleMax) {
    for (int Idx : VectorOpToIdx[SingleVec])
      std::swap(GatheredExtracts[Idx], VL[Idx]);
  } else {
    for (Value *V : {PairVec.first, PairVec.second})
      for (int Idx : VectorOpToIdx[V])
        std::swap(GatheredExtracts[Idx], VL[Idx]);
  }
  // Add extracts from undefs too.
  for (int Idx : UndefVectorExtracts)
    std::swap(GatheredExtracts[Idx], VL[Idx]);
  // Check that gather of extractelements can be represented as just a
  // shuffle of a single/two vectors the scalars are extracted from.
  std::optional<TTI::ShuffleKind> Res =
      isFixedVectorShuffle(GatheredExtracts, Mask);
  if (!Res) {
    // TODO: try to check other subsets if possible.
    // Restore the original VL if attempt was not successful.
    VL.swap(SavedVL);
    return std::nullopt;
  }
  // Restore unused scalars from mask, if some of the extractelements were not
  // selected for shuffle.
  for (int I = 0, E = GatheredExtracts.size(); I < E; ++I) {
    auto *EI = dyn_cast<ExtractElementInst>(VL[I]);
    if (!EI || !isa<FixedVectorType>(EI->getVectorOperandType()) ||
        !isa<ConstantInt, UndefValue>(EI->getIndexOperand()) ||
        is_contained(UndefVectorExtracts, I))
      continue;
    if (Mask[I] == PoisonMaskElem && !isa<PoisonValue>(GatheredExtracts[I]))
      std::swap(VL[I], GatheredExtracts[I]);
  }
  return Res;
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

/// \returns true if \p Opcode is allowed as part of the main/alternate
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
                                       const TargetLibraryInfo &TLI,
                                       unsigned BaseIndex = 0);

/// Checks if the provided operands of 2 cmp instructions are compatible, i.e.
/// compatible instructions or constants, or just some other regular values.
static bool areCompatibleCmpOps(Value *BaseOp0, Value *BaseOp1, Value *Op0,
                                Value *Op1, const TargetLibraryInfo &TLI) {
  return (isConstant(BaseOp0) && isConstant(Op0)) ||
         (isConstant(BaseOp1) && isConstant(Op1)) ||
         (!isa<Instruction>(BaseOp0) && !isa<Instruction>(Op0) &&
          !isa<Instruction>(BaseOp1) && !isa<Instruction>(Op1)) ||
         BaseOp0 == Op0 || BaseOp1 == Op1 ||
         getSameOpcode({BaseOp0, Op0}, TLI).getOpcode() ||
         getSameOpcode({BaseOp1, Op1}, TLI).getOpcode();
}

/// \returns true if a compare instruction \p CI has similar "look" and
/// same predicate as \p BaseCI, "as is" or with its operands and predicate
/// swapped, false otherwise.
static bool isCmpSameOrSwapped(const CmpInst *BaseCI, const CmpInst *CI,
                               const TargetLibraryInfo &TLI) {
  assert(BaseCI->getOperand(0)->getType() == CI->getOperand(0)->getType() &&
         "Assessing comparisons of different types?");
  CmpInst::Predicate BasePred = BaseCI->getPredicate();
  CmpInst::Predicate Pred = CI->getPredicate();
  CmpInst::Predicate SwappedPred = CmpInst::getSwappedPredicate(Pred);

  Value *BaseOp0 = BaseCI->getOperand(0);
  Value *BaseOp1 = BaseCI->getOperand(1);
  Value *Op0 = CI->getOperand(0);
  Value *Op1 = CI->getOperand(1);

  return (BasePred == Pred &&
          areCompatibleCmpOps(BaseOp0, BaseOp1, Op0, Op1, TLI)) ||
         (BasePred == SwappedPred &&
          areCompatibleCmpOps(BaseOp0, BaseOp1, Op1, Op0, TLI));
}

/// \returns analysis of the Instructions in \p VL described in
/// InstructionsState, the Opcode that we suppose the whole list
/// could be vectorized even if its structure is diverse.
static InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                       const TargetLibraryInfo &TLI,
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
  auto *IBase = cast<Instruction>(VL[BaseIndex]);
  Intrinsic::ID BaseID = 0;
  SmallVector<VFInfo> BaseMappings;
  if (auto *CallBase = dyn_cast<CallInst>(IBase)) {
    BaseID = getVectorIntrinsicIDForCall(CallBase, &TLI);
    BaseMappings = VFDatabase(*CallBase).getMappings(*CallBase);
    if (!isTriviallyVectorizable(BaseID) && BaseMappings.empty())
      return InstructionsState(VL[BaseIndex], nullptr, nullptr);
  }
  for (int Cnt = 0, E = VL.size(); Cnt < E; Cnt++) {
    auto *I = cast<Instruction>(VL[Cnt]);
    unsigned InstOpcode = I->getOpcode();
    if (IsBinOp && isa<BinaryOperator>(I)) {
      if (InstOpcode == Opcode || InstOpcode == AltOpcode)
        continue;
      if (Opcode == AltOpcode && isValidForAlternation(InstOpcode) &&
          isValidForAlternation(Opcode)) {
        AltOpcode = InstOpcode;
        AltIndex = Cnt;
        continue;
      }
    } else if (IsCastOp && isa<CastInst>(I)) {
      Value *Op0 = IBase->getOperand(0);
      Type *Ty0 = Op0->getType();
      Value *Op1 = I->getOperand(0);
      Type *Ty1 = Op1->getType();
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
    } else if (auto *Inst = dyn_cast<CmpInst>(VL[Cnt]); Inst && IsCmpOp) {
      auto *BaseInst = cast<CmpInst>(VL[BaseIndex]);
      Type *Ty0 = BaseInst->getOperand(0)->getType();
      Type *Ty1 = Inst->getOperand(0)->getType();
      if (Ty0 == Ty1) {
        assert(InstOpcode == Opcode && "Expected same CmpInst opcode.");
        // Check for compatible operands. If the corresponding operands are not
        // compatible - need to perform alternate vectorization.
        CmpInst::Predicate CurrentPred = Inst->getPredicate();
        CmpInst::Predicate SwappedCurrentPred =
            CmpInst::getSwappedPredicate(CurrentPred);

        if (E == 2 &&
            (BasePred == CurrentPred || BasePred == SwappedCurrentPred))
          continue;

        if (isCmpSameOrSwapped(BaseInst, Inst, TLI))
          continue;
        auto *AltInst = cast<CmpInst>(VL[AltIndex]);
        if (AltIndex != BaseIndex) {
          if (isCmpSameOrSwapped(AltInst, Inst, TLI))
            continue;
        } else if (BasePred != CurrentPred) {
          assert(
              isValidForAlternation(InstOpcode) &&
              "CmpInst isn't safe for alternation, logic needs to be updated!");
          AltIndex = Cnt;
          continue;
        }
        CmpInst::Predicate AltPred = AltInst->getPredicate();
        if (BasePred == CurrentPred || BasePred == SwappedCurrentPred ||
            AltPred == CurrentPred || AltPred == SwappedCurrentPred)
          continue;
      }
    } else if (InstOpcode == Opcode || InstOpcode == AltOpcode) {
      if (auto *Gep = dyn_cast<GetElementPtrInst>(I)) {
        if (Gep->getNumOperands() != 2 ||
            Gep->getOperand(0)->getType() != IBase->getOperand(0)->getType())
          return InstructionsState(VL[BaseIndex], nullptr, nullptr);
      } else if (auto *EI = dyn_cast<ExtractElementInst>(I)) {
        if (!isVectorLikeInstWithConstOps(EI))
          return InstructionsState(VL[BaseIndex], nullptr, nullptr);
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        auto *BaseLI = cast<LoadInst>(IBase);
        if (!LI->isSimple() || !BaseLI->isSimple())
          return InstructionsState(VL[BaseIndex], nullptr, nullptr);
      } else if (auto *Call = dyn_cast<CallInst>(I)) {
        auto *CallBase = cast<CallInst>(IBase);
        if (Call->getCalledFunction() != CallBase->getCalledFunction())
          return InstructionsState(VL[BaseIndex], nullptr, nullptr);
        if (Call->hasOperandBundles() &&
            !std::equal(Call->op_begin() + Call->getBundleOperandsStartIndex(),
                        Call->op_begin() + Call->getBundleOperandsEndIndex(),
                        CallBase->op_begin() +
                            CallBase->getBundleOperandsStartIndex()))
          return InstructionsState(VL[BaseIndex], nullptr, nullptr);
        Intrinsic::ID ID = getVectorIntrinsicIDForCall(Call, &TLI);
        if (ID != BaseID)
          return InstructionsState(VL[BaseIndex], nullptr, nullptr);
        if (!ID) {
          SmallVector<VFInfo> Mappings = VFDatabase(*Call).getMappings(*Call);
          if (Mappings.size() != BaseMappings.size() ||
              Mappings.front().ISA != BaseMappings.front().ISA ||
              Mappings.front().ScalarName != BaseMappings.front().ScalarName ||
              Mappings.front().VectorName != BaseMappings.front().VectorName ||
              Mappings.front().Shape.VF != BaseMappings.front().Shape.VF ||
              Mappings.front().Shape.Parameters !=
                  BaseMappings.front().Shape.Parameters)
            return InstructionsState(VL[BaseIndex], nullptr, nullptr);
        }
      }
      continue;
    }
    return InstructionsState(VL[BaseIndex], nullptr, nullptr);
  }

  return InstructionsState(VL[BaseIndex], cast<Instruction>(VL[BaseIndex]),
                           cast<Instruction>(VL[AltIndex]));
}

/// \returns true if all of the values in \p VL have the same type or false
/// otherwise.
static bool allSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL.front()->getType();
  return all_of(VL.drop_front(), [&](Value *V) { return V->getType() == Ty; });
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
    return any_of(enumerate(CI->args()), [&](auto &&Arg) {
      return isVectorIntrinsicWithScalarOpAtArg(ID, Arg.index()) &&
             Arg.value().get() == Scalar;
    });
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
/// \param ExtendingManyInputs Supports reshuffling of the mask with not only
/// one but two input vectors.
static void addMask(SmallVectorImpl<int> &Mask, ArrayRef<int> SubMask,
                    bool ExtendingManyInputs = false) {
  if (SubMask.empty())
    return;
  assert(
      (!ExtendingManyInputs || SubMask.size() > Mask.size() ||
       // Check if input scalars were extended to match the size of other node.
       (SubMask.size() == Mask.size() &&
        std::all_of(std::next(Mask.begin(), Mask.size() / 2), Mask.end(),
                    [](int Idx) { return Idx == PoisonMaskElem; }))) &&
      "SubMask with many inputs support must be larger than the mask.");
  if (Mask.empty()) {
    Mask.append(SubMask.begin(), SubMask.end());
    return;
  }
  SmallVector<int> NewMask(SubMask.size(), PoisonMaskElem);
  int TermValue = std::min(Mask.size(), SubMask.size());
  for (int I = 0, E = SubMask.size(); I < E; ++I) {
    if (SubMask[I] == PoisonMaskElem ||
        (!ExtendingManyInputs &&
         (SubMask[I] >= TermValue || Mask[SubMask[I]] >= TermValue)))
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
  Mask.resize(E, PoisonMaskElem);
  for (unsigned I = 0; I < E; ++I)
    Mask[Indices[I]] = I;
}

/// Reorders the list of scalars in accordance with the given \p Mask.
static void reorderScalars(SmallVectorImpl<Value *> &Scalars,
                           ArrayRef<int> Mask) {
  assert(!Mask.empty() && "Expected non-empty mask.");
  SmallVector<Value *> Prev(Scalars.size(),
                            UndefValue::get(Scalars.front()->getType()));
  Prev.swap(Scalars);
  for (unsigned I = 0, E = Prev.size(); I < E; ++I)
    if (Mask[I] != PoisonMaskElem)
      Scalars[Mask[I]] = Prev[I];
}

/// Checks if the provided value does not require scheduling. It does not
/// require scheduling if this is not an instruction or it is an instruction
/// that does not read/write memory and all operands are either not instructions
/// or phi nodes or instructions from different blocks.
static bool areAllOperandsNonInsts(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return true;
  return !mayHaveNonDefUseDependency(*I) &&
    all_of(I->operands(), [I](Value *V) {
      auto *IO = dyn_cast<Instruction>(V);
      if (!IO)
        return true;
      return isa<PHINode>(IO) || IO->getParent() != I->getParent();
    });
}

/// Checks if the provided value does not require scheduling. It does not
/// require scheduling if this is not an instruction or it is an instruction
/// that does not read/write memory and all users are phi nodes or instructions
/// from the different blocks.
static bool isUsedOutsideBlock(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return true;
  // Limits the number of uses to save compile time.
  constexpr int UsesLimit = 8;
  return !I->mayReadOrWriteMemory() && !I->hasNUsesOrMore(UsesLimit) &&
         all_of(I->users(), [I](User *U) {
           auto *IU = dyn_cast<Instruction>(U);
           if (!IU)
             return true;
           return IU->getParent() != I->getParent() || isa<PHINode>(IU);
         });
}

/// Checks if the specified value does not require scheduling. It does not
/// require scheduling if all operands and all users do not need to be scheduled
/// in the current basic block.
static bool doesNotNeedToBeScheduled(Value *V) {
  return areAllOperandsNonInsts(V) && isUsedOutsideBlock(V);
}

/// Checks if the specified array of instructions does not require scheduling.
/// It is so if all either instructions have operands that do not require
/// scheduling or their users do not require scheduling since they are phis or
/// in other basic blocks.
static bool doesNotNeedToSchedule(ArrayRef<Value *> VL) {
  return !VL.empty() &&
         (all_of(VL, isUsedOutsideBlock) || all_of(VL, areAllOperandsNonInsts));
}

namespace slpvectorizer {

/// Bottom Up SLP Vectorizer.
class BoUpSLP {
  struct TreeEntry;
  struct ScheduleData;
  class ShuffleCostEstimator;
  class ShuffleInstructionBuilder;

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
              .getFixedValue();

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
  /// \param ReplacedExternals containd list of replaced external values
  /// {scalar, replace} after emitting extractelement for external uses.
  Value *
  vectorizeTree(const ExtraValueToDebugLocsMap &ExternallyUsedValues,
                SmallVectorImpl<std::pair<Value *, Value *>> &ReplacedExternals,
                Instruction *ReductionRoot = nullptr);

  /// \returns the cost incurred by unwanted spills and fills, caused by
  /// holding live values over call sites.
  InstructionCost getSpillCost() const;

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  InstructionCost getTreeCost(ArrayRef<Value *> VectorizedVals = std::nullopt);

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst.
  void buildTree(ArrayRef<Value *> Roots,
                 const SmallDenseSet<Value *> &UserIgnoreLst);

  /// Construct a vectorizable tree that starts at \p Roots.
  void buildTree(ArrayRef<Value *> Roots);

  /// Returns whether the root node has in-tree uses.
  bool doesRootHaveInTreeUses() const {
    return !VectorizableTree.empty() &&
           !VectorizableTree.front()->UserTreeIndices.empty();
  }

  /// Return the scalars of the root node.
  ArrayRef<Value *> getRootNodeScalars() const {
    assert(!VectorizableTree.empty() && "No graph to get the first node from");
    return VectorizableTree.front()->Scalars;
  }

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
    EntryToLastInstruction.clear();
    ExternalUses.clear();
    for (auto &Iter : BlocksSchedules) {
      BlockScheduling *BS = Iter.second.get();
      BS->clear();
    }
    MinBWs.clear();
    InstrElementSize.clear();
    UserIgnoreList = nullptr;
    PostponedGathers.clear();
    ValueToGatherNodes.clear();
  }

  unsigned getTreeSize() const { return VectorizableTree.size(); }

  /// Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  /// Checks if the specified gather tree entry \p TE can be represented as a
  /// shuffled vector entry + (possibly) permutation with other gathers. It
  /// implements the checks only for possibly ordered scalars (Loads,
  /// ExtractElement, ExtractValue), which can be part of the graph.
  std::optional<OrdersType> findReusedOrderedScalars(const TreeEntry &TE);

  /// Sort loads into increasing pointers offsets to allow greater clustering.
  std::optional<OrdersType> findPartiallyOrderedLoads(const TreeEntry &TE);

  /// Gets reordering data for the given tree entry. If the entry is vectorized
  /// - just return ReorderIndices, otherwise check if the scalars can be
  /// reordered and return the most optimal order.
  /// \return std::nullopt if ordering is not important, empty order, if
  /// identity order is important, or the actual order.
  /// \param TopToBottom If true, include the order of vectorized stores and
  /// insertelement nodes, otherwise skip them.
  std::optional<OrdersType> getReorderingData(const TreeEntry &TE,
                                              bool TopToBottom);

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

  /// A helper class used for scoring candidates for two consecutive lanes.
  class LookAheadHeuristics {
    const TargetLibraryInfo &TLI;
    const DataLayout &DL;
    ScalarEvolution &SE;
    const BoUpSLP &R;
    int NumLanes; // Total number of lanes (aka vectorization factor).
    int MaxLevel; // The maximum recursion depth for accumulating score.

  public:
    LookAheadHeuristics(const TargetLibraryInfo &TLI, const DataLayout &DL,
                        ScalarEvolution &SE, const BoUpSLP &R, int NumLanes,
                        int MaxLevel)
        : TLI(TLI), DL(DL), SE(SE), R(R), NumLanes(NumLanes),
          MaxLevel(MaxLevel) {}

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
    /// The same load multiple times. This should have a better score than
    /// `ScoreSplat` because it in x86 for a 2-lane vector we can represent it
    /// with `movddup (%reg), xmm0` which has a throughput of 0.5 versus 0.5 for
    /// a vector load and 1.0 for a broadcast.
    static const int ScoreSplatLoads = 3;
    /// Loads from reversed memory addresses, e.g. load(A[i+1]), load(A[i]).
    static const int ScoreReversedLoads = 3;
    /// A load candidate for masked gather.
    static const int ScoreMaskedGatherCandidate = 1;
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
    /// \p U1 and \p U2 are the users of \p V1 and \p V2.
    /// Also, checks if \p V1 and \p V2 are compatible with instructions in \p
    /// MainAltOps.
    int getShallowScore(Value *V1, Value *V2, Instruction *U1, Instruction *U2,
                        ArrayRef<Value *> MainAltOps) const {
      if (!isValidElementType(V1->getType()) ||
          !isValidElementType(V2->getType()))
        return LookAheadHeuristics::ScoreFail;

      if (V1 == V2) {
        if (isa<LoadInst>(V1)) {
          // Retruns true if the users of V1 and V2 won't need to be extracted.
          auto AllUsersAreInternal = [U1, U2, this](Value *V1, Value *V2) {
            // Bail out if we have too many uses to save compilation time.
            static constexpr unsigned Limit = 8;
            if (V1->hasNUsesOrMore(Limit) || V2->hasNUsesOrMore(Limit))
              return false;

            auto AllUsersVectorized = [U1, U2, this](Value *V) {
              return llvm::all_of(V->users(), [U1, U2, this](Value *U) {
                return U == U1 || U == U2 || R.getTreeEntry(U) != nullptr;
              });
            };
            return AllUsersVectorized(V1) && AllUsersVectorized(V2);
          };
          // A broadcast of a load can be cheaper on some targets.
          if (R.TTI->isLegalBroadcastLoad(V1->getType(),
                                          ElementCount::getFixed(NumLanes)) &&
              ((int)V1->getNumUses() == NumLanes ||
               AllUsersAreInternal(V1, V2)))
            return LookAheadHeuristics::ScoreSplatLoads;
        }
        return LookAheadHeuristics::ScoreSplat;
      }

      auto *LI1 = dyn_cast<LoadInst>(V1);
      auto *LI2 = dyn_cast<LoadInst>(V2);
      if (LI1 && LI2) {
        if (LI1->getParent() != LI2->getParent() || !LI1->isSimple() ||
            !LI2->isSimple())
          return LookAheadHeuristics::ScoreFail;

        std::optional<int> Dist = getPointersDiff(
            LI1->getType(), LI1->getPointerOperand(), LI2->getType(),
            LI2->getPointerOperand(), DL, SE, /*StrictCheck=*/true);
        if (!Dist || *Dist == 0) {
          if (getUnderlyingObject(LI1->getPointerOperand()) ==
                  getUnderlyingObject(LI2->getPointerOperand()) &&
              R.TTI->isLegalMaskedGather(
                  FixedVectorType::get(LI1->getType(), NumLanes),
                  LI1->getAlign()))
            return LookAheadHeuristics::ScoreMaskedGatherCandidate;
          return LookAheadHeuristics::ScoreFail;
        }
        // The distance is too large - still may be profitable to use masked
        // loads/gathers.
        if (std::abs(*Dist) > NumLanes / 2)
          return LookAheadHeuristics::ScoreMaskedGatherCandidate;
        // This still will detect consecutive loads, but we might have "holes"
        // in some cases. It is ok for non-power-2 vectorization and may produce
        // better results. It should not affect current vectorization.
        return (*Dist > 0) ? LookAheadHeuristics::ScoreConsecutiveLoads
                           : LookAheadHeuristics::ScoreReversedLoads;
      }

      auto *C1 = dyn_cast<Constant>(V1);
      auto *C2 = dyn_cast<Constant>(V2);
      if (C1 && C2)
        return LookAheadHeuristics::ScoreConstants;

      // Extracts from consecutive indexes of the same vector better score as
      // the extracts could be optimized away.
      Value *EV1;
      ConstantInt *Ex1Idx;
      if (match(V1, m_ExtractElt(m_Value(EV1), m_ConstantInt(Ex1Idx)))) {
        // Undefs are always profitable for extractelements.
        // Compiler can easily combine poison and extractelement <non-poison> or
        // undef and extractelement <poison>. But combining undef +
        // extractelement <non-poison-but-may-produce-poison> requires some
        // extra operations.
        if (isa<UndefValue>(V2))
          return (isa<PoisonValue>(V2) || isUndefVector(EV1).all())
                     ? LookAheadHeuristics::ScoreConsecutiveExtracts
                     : LookAheadHeuristics::ScoreSameOpcode;
        Value *EV2 = nullptr;
        ConstantInt *Ex2Idx = nullptr;
        if (match(V2,
                  m_ExtractElt(m_Value(EV2), m_CombineOr(m_ConstantInt(Ex2Idx),
                                                         m_Undef())))) {
          // Undefs are always profitable for extractelements.
          if (!Ex2Idx)
            return LookAheadHeuristics::ScoreConsecutiveExtracts;
          if (isUndefVector(EV2).all() && EV2->getType() == EV1->getType())
            return LookAheadHeuristics::ScoreConsecutiveExtracts;
          if (EV2 == EV1) {
            int Idx1 = Ex1Idx->getZExtValue();
            int Idx2 = Ex2Idx->getZExtValue();
            int Dist = Idx2 - Idx1;
            // The distance is too large - still may be profitable to use
            // shuffles.
            if (std::abs(Dist) == 0)
              return LookAheadHeuristics::ScoreSplat;
            if (std::abs(Dist) > NumLanes / 2)
              return LookAheadHeuristics::ScoreSameOpcode;
            return (Dist > 0) ? LookAheadHeuristics::ScoreConsecutiveExtracts
                              : LookAheadHeuristics::ScoreReversedExtracts;
          }
          return LookAheadHeuristics::ScoreAltOpcodes;
        }
        return LookAheadHeuristics::ScoreFail;
      }

      auto *I1 = dyn_cast<Instruction>(V1);
      auto *I2 = dyn_cast<Instruction>(V2);
      if (I1 && I2) {
        if (I1->getParent() != I2->getParent())
          return LookAheadHeuristics::ScoreFail;
        SmallVector<Value *, 4> Ops(MainAltOps.begin(), MainAltOps.end());
        Ops.push_back(I1);
        Ops.push_back(I2);
        InstructionsState S = getSameOpcode(Ops, TLI);
        // Note: Only consider instructions with <= 2 operands to avoid
        // complexity explosion.
        if (S.getOpcode() &&
            (S.MainOp->getNumOperands() <= 2 || !MainAltOps.empty() ||
             !S.isAltShuffle()) &&
            all_of(Ops, [&S](Value *V) {
              return cast<Instruction>(V)->getNumOperands() ==
                     S.MainOp->getNumOperands();
            }))
          return S.isAltShuffle() ? LookAheadHeuristics::ScoreAltOpcodes
                                  : LookAheadHeuristics::ScoreSameOpcode;
      }

      if (isa<UndefValue>(V2))
        return LookAheadHeuristics::ScoreUndef;

      return LookAheadHeuristics::ScoreFail;
    }

    /// Go through the operands of \p LHS and \p RHS recursively until
    /// MaxLevel, and return the cummulative score. \p U1 and \p U2 are
    /// the users of \p LHS and \p RHS (that is \p LHS and \p RHS are operands
    /// of \p U1 and \p U2), except at the beginning of the recursion where
    /// these are set to nullptr.
    ///
    /// For example:
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
    /// {B[0],B[1]} match with LookAheadHeuristics::ScoreConsecutiveLoads, while
    /// {A[0],C[0]} has a score of LookAheadHeuristics::ScoreFail.
    /// Please note that the order of the operands does not matter, as we
    /// evaluate the score of all profitable combinations of operands. In
    /// other words the score of G1 and G4 is the same as G1 and G2. This
    /// heuristic is based on ideas described in:
    ///   Look-ahead SLP: Auto-vectorization in the presence of commutative
    ///   operations, CGO 2018 by Vasileios Porpodas, Rodrigo C. O. Rocha,
    ///   Luís F. W. Góes
    int getScoreAtLevelRec(Value *LHS, Value *RHS, Instruction *U1,
                           Instruction *U2, int CurrLevel,
                           ArrayRef<Value *> MainAltOps) const {

      // Get the shallow score of V1 and V2.
      int ShallowScoreAtThisLevel =
          getShallowScore(LHS, RHS, U1, U2, MainAltOps);

      // If reached MaxLevel,
      //  or if V1 and V2 are not instructions,
      //  or if they are SPLAT,
      //  or if they are not consecutive,
      //  or if profitable to vectorize loads or extractelements, early return
      //  the current cost.
      auto *I1 = dyn_cast<Instruction>(LHS);
      auto *I2 = dyn_cast<Instruction>(RHS);
      if (CurrLevel == MaxLevel || !(I1 && I2) || I1 == I2 ||
          ShallowScoreAtThisLevel == LookAheadHeuristics::ScoreFail ||
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
                                 I1, I2, CurrLevel + 1, std::nullopt);
          // Look for the best score.
          if (TmpScore > LookAheadHeuristics::ScoreFail &&
              TmpScore > MaxTmpScore) {
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

    const TargetLibraryInfo &TLI;
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
        return LookAheadHeuristics::ScoreAllUserVectorized;
      auto *IdxLaneI = dyn_cast<Instruction>(IdxLaneV);
      if (!IdxLaneI || !isa<Instruction>(OpIdxLaneV))
        return 0;
      return R.areAllUsersVectorized(IdxLaneI)
                 ? LookAheadHeuristics::ScoreAllUserVectorized
                 : 0;
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
      LookAheadHeuristics LookAhead(TLI, DL, SE, R, getNumLanes(),
                                    LookAheadMaxDepth);
      // Keep track of the instruction stack as we recurse into the operands
      // during the look-ahead score exploration.
      int Score =
          LookAhead.getScoreAtLevelRec(LHS, RHS, /*U1=*/nullptr, /*U2=*/nullptr,
                                       /*CurrLevel=*/1, MainAltOps);
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
    // If no good match can be found, return std::nullopt.
    std::optional<unsigned>
    getBestOperand(unsigned OpIdx, int Lane, int LastLane,
                   ArrayRef<ReorderingMode> ReorderingModes,
                   ArrayRef<Value *> MainAltOps) {
      unsigned NumOperands = getNumOperands();

      // The operand of the previous lane at OpIdx.
      Value *OpLastLane = getData(OpIdx, LastLane).V;

      // Our strategy mode for OpIdx.
      ReorderingMode RMode = ReorderingModes[OpIdx];
      if (RMode == ReorderingMode::Failed)
        return std::nullopt;

      // The linearized opcode of the operand at OpIdx, Lane.
      bool OpIdxAPO = getData(OpIdx, Lane).APO;

      // The best operand index and its score.
      // Sometimes we have more than one option (e.g., Opcode and Undefs), so we
      // are using the score to differentiate between the two.
      struct BestOpData {
        std::optional<unsigned> Idx;
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
        getData(*BestOp.Idx, Lane).IsUsed = IsUsed;
        return BestOp.Idx;
      }
      // If we could not find a good match return std::nullopt.
      return std::nullopt;
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
          if (!OpcodeI || !getSameOpcode({OpcodeI, I}, TLI).getOpcode() ||
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
    VLOperands(ArrayRef<Value *> RootVL, const TargetLibraryInfo &TLI,
               const DataLayout &DL, ScalarEvolution &SE, const BoUpSLP &R)
        : TLI(TLI), DL(DL), SE(SE), R(R) {
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
        // Need to do it to avoid extra external use cost counting for
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
              std::optional<unsigned> BestIdx = getBestOperand(
                  OpIdx, Lane, LastLane, ReorderingModes, MainAltOps[OpIdx]);
              // By not selecting a value, we allow the operands that follow to
              // select a better matching value. We will get a non-null value in
              // the next run of getBestOperand().
              if (BestIdx) {
                // Swap the current operand with the one returned by
                // getBestOperand().
                swap(OpIdx, *BestIdx, Lane);
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
                    getSameOpcode({MainAltOps[OpIdx].front(), AltOp.V}, TLI);
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

  /// Evaluate each pair in \p Candidates and return index into \p Candidates
  /// for a pair which have highest score deemed to have best chance to form
  /// root of profitable tree to vectorize. Return std::nullopt if no candidate
  /// scored above the LookAheadHeuristics::ScoreFail. \param Limit Lower limit
  /// of the cost, considered to be good enough score.
  std::optional<int>
  findBestRootPair(ArrayRef<std::pair<Value *, Value *>> Candidates,
                   int Limit = LookAheadHeuristics::ScoreFail) {
    LookAheadHeuristics LookAhead(*TLI, *DL, *SE, *this, /*NumLanes=*/2,
                                  RootLookAheadMaxDepth);
    int BestScore = Limit;
    std::optional<int> Index;
    for (int I : seq<int>(0, Candidates.size())) {
      int Score = LookAhead.getScoreAtLevelRec(Candidates[I].first,
                                               Candidates[I].second,
                                               /*U1=*/nullptr, /*U2=*/nullptr,
                                               /*Level=*/1, std::nullopt);
      if (Score > BestScore) {
        BestScore = Score;
        Index = I;
      }
    }
    return Index;
  }

  /// Checks if the instruction is marked for deletion.
  bool isDeleted(Instruction *I) const { return DeletedInstructions.count(I); }

  /// Removes an instruction from its block and eventually deletes it.
  /// It's like Instruction::eraseFromParent() except that the actual deletion
  /// is delayed until BoUpSLP is destructed.
  void eraseInstruction(Instruction *I) {
    DeletedInstructions.insert(I);
  }

  /// Checks if the instruction was already analyzed for being possible
  /// reduction root.
  bool isAnalyzedReductionRoot(Instruction *I) const {
    return AnalyzedReductionsRoots.count(I);
  }
  /// Register given instruction as already analyzed for being possible
  /// reduction root.
  void analyzedReductionRoot(Instruction *I) {
    AnalyzedReductionsRoots.insert(I);
  }
  /// Checks if the provided list of reduced values was checked already for
  /// vectorization.
  bool areAnalyzedReductionVals(ArrayRef<Value *> VL) const {
    return AnalyzedReductionVals.contains(hash_value(VL));
  }
  /// Adds the list of reduced values to list of already checked values for the
  /// vectorization.
  void analyzedReductionVals(ArrayRef<Value *> VL) {
    AnalyzedReductionVals.insert(hash_value(VL));
  }
  /// Clear the list of the analyzed reduction root instructions.
  void clearReductionData() {
    AnalyzedReductionsRoots.clear();
    AnalyzedReductionVals.clear();
  }
  /// Checks if the given value is gathered in one of the nodes.
  bool isAnyGathered(const SmallDenseSet<Value *> &Vals) const {
    return any_of(MustGather, [&](Value *V) { return Vals.contains(V); });
  }

  /// Check if the value is vectorized in the tree.
  bool isVectorized(Value *V) const { return getTreeEntry(V); }

  ~BoUpSLP();

private:
  /// Check if the operands on the edges \p Edges of the \p UserTE allows
  /// reordering (i.e. the operands can be reordered because they have only one
  /// user and reordarable).
  /// \param ReorderableGathers List of all gather nodes that require reordering
  /// (e.g., gather of extractlements or partially vectorizable loads).
  /// \param GatherOps List of gather operand nodes for \p UserTE that require
  /// reordering, subset of \p NonVectorized.
  bool
  canReorderOperands(TreeEntry *UserTE,
                     SmallVectorImpl<std::pair<unsigned, TreeEntry *>> &Edges,
                     ArrayRef<TreeEntry *> ReorderableGathers,
                     SmallVectorImpl<TreeEntry *> &GatherOps);

  /// Checks if the given \p TE is a gather node with clustered reused scalars
  /// and reorders it per given \p Mask.
  void reorderNodeWithReuses(TreeEntry &TE, ArrayRef<int> Mask) const;

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
  bool areAllUsersVectorized(
      Instruction *I,
      const SmallDenseSet<Value *> *VectorizedVals = nullptr) const;

  /// Return information about the vector formed for the specified index
  /// of a vector of (the same) instruction.
  TargetTransformInfo::OperandValueInfo getOperandInfo(const TreeEntry &E,
                                                       unsigned OpIdx);

  /// \returns the cost of the vectorizable entry.
  InstructionCost getEntryCost(const TreeEntry *E,
                               ArrayRef<Value *> VectorizedVals,
                               SmallPtrSetImpl<Value *> &CheckedExtracts);

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

  /// Vectorize a single entry in the tree, the \p Idx-th operand of the entry
  /// \p E.
  Value *vectorizeOperand(TreeEntry *E, unsigned NodeIdx);

  /// Create a new vector from a list of scalar values.  Produces a sequence
  /// which exploits values reused across lanes, and arranges the inserts
  /// for ease of later optimization.
  template <typename BVTy, typename ResTy, typename... Args>
  ResTy processBuildVector(const TreeEntry *E, Args &...Params);

  /// Create a new vector from a list of scalar values.  Produces a sequence
  /// which exploits values reused across lanes, and arranges the inserts
  /// for ease of later optimization.
  Value *createBuildVector(const TreeEntry *E);

  /// Returns the instruction in the bundle, which can be used as a base point
  /// for scheduling. Usually it is the last instruction in the bundle, except
  /// for the case when all operands are external (in this case, it is the first
  /// instruction in the list).
  Instruction &getLastInstructionInBundle(const TreeEntry *E);

  /// Checks if the gathered \p VL can be represented as shuffle(s) of previous
  /// tree entries.
  /// \param TE Tree entry checked for permutation.
  /// \param VL List of scalars (a subset of the TE scalar), checked for
  /// permutations.
  /// \returns ShuffleKind, if gathered values can be represented as shuffles of
  /// previous tree entries. \p Mask is filled with the shuffle mask.
  std::optional<TargetTransformInfo::ShuffleKind>
  isGatherShuffledEntry(const TreeEntry *TE, ArrayRef<Value *> VL,
                        SmallVectorImpl<int> &Mask,
                        SmallVectorImpl<const TreeEntry *> &Entries);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  /// \param ForPoisonSrc true if initial vector is poison, false otherwise.
  InstructionCost getGatherCost(ArrayRef<Value *> VL, bool ForPoisonSrc) const;

  /// Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(const TreeEntry *E);

  /// \returns a vector from a collection of scalars in \p VL. if \p Root is not
  /// specified, the starting vector value is poison.
  Value *gather(ArrayRef<Value *> VL, Value *Root);

  /// \returns whether the VectorizableTree is fully vectorizable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree(bool ForReduction) const;

  /// Reorder commutative or alt operands to get better probability of
  /// generating vectorized code.
  static void reorderInputsAccordingToOpcode(
      ArrayRef<Value *> VL, SmallVectorImpl<Value *> &Left,
      SmallVectorImpl<Value *> &Right, const TargetLibraryInfo &TLI,
      const DataLayout &DL, ScalarEvolution &SE, const BoUpSLP &R);

  /// Helper for `findExternalStoreUsersReorderIndices()`. It iterates over the
  /// users of \p TE and collects the stores. It returns the map from the store
  /// pointers to the collected stores.
  DenseMap<Value *, SmallVector<StoreInst *>>
  collectUserStores(const BoUpSLP::TreeEntry *TE) const;

  /// Helper for `findExternalStoreUsersReorderIndices()`. It checks if the
  /// stores in \p StoresVec can form a vector instruction. If so it returns
  /// true and populates \p ReorderIndices with the shuffle indices of the
  /// stores when compared to the sorted vector.
  bool canFormVector(ArrayRef<StoreInst *> StoresVec,
                     OrdersType &ReorderIndices) const;

  /// Iterates through the users of \p TE, looking for scalar stores that can be
  /// potentially vectorized in a future SLP-tree. If found, it keeps track of
  /// their order and builds an order index vector for each store bundle. It
  /// returns all these order vectors found.
  /// We run this after the tree has formed, otherwise we may come across user
  /// instructions that are not yet in the tree.
  SmallVector<OrdersType, 1>
  findExternalStoreUsersReorderIndices(TreeEntry *TE) const;

  struct TreeEntry {
    using VecTreeTy = SmallVector<std::unique_ptr<TreeEntry>, 8>;
    TreeEntry(VecTreeTy &Container) : Container(Container) {}

    /// \returns Common mask for reorder indices and reused scalars.
    SmallVector<int> getCommonMask() const {
      SmallVector<int> Mask;
      inversePermutation(ReorderIndices, Mask);
      ::addMask(Mask, ReuseShuffleIndices);
      return Mask;
    }

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) const {
      auto &&IsSame = [VL](ArrayRef<Value *> Scalars, ArrayRef<int> Mask) {
        if (Mask.size() != VL.size() && VL.size() == Scalars.size())
          return std::equal(VL.begin(), VL.end(), Scalars.begin());
        return VL.size() == Mask.size() &&
               std::equal(VL.begin(), VL.end(), Mask.begin(),
                          [Scalars](Value *V, int Idx) {
                            return (isa<UndefValue>(V) &&
                                    Idx == PoisonMaskElem) ||
                                   (Idx != PoisonMaskElem && V == Scalars[Idx]);
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

    bool isOperandGatherNode(const EdgeInfo &UserEI) const {
      return State == TreeEntry::NeedToGather &&
             UserTreeIndices.front().EdgeIdx == UserEI.EdgeIdx &&
             UserTreeIndices.front().UserTE == UserEI.UserTE;
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
    WeakTrackingVH VectorizedValue = nullptr;

    /// Do we need to gather this sequence or vectorize it
    /// (either with vector instruction or with scatter/gather
    /// intrinsics for store/load)?
    enum EntryState {
      Vectorize,
      ScatterVectorize,
      PossibleStridedVectorize,
      NeedToGather
    };
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

    /// Build a shuffle mask for graph entry which represents a merge of main
    /// and alternate operations.
    void
    buildAltOpShuffleMask(const function_ref<bool(Instruction *)> IsAltOp,
                          SmallVectorImpl<int> &Mask,
                          SmallVectorImpl<Value *> *OpScalars = nullptr,
                          SmallVectorImpl<Value *> *AltScalars = nullptr) const;

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
      case PossibleStridedVectorize:
        dbgs() << "PossibleStridedVectorize\n";
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
                     InstructionCost VecCost, InstructionCost ScalarCost,
                     StringRef Banner) const {
    dbgs() << "SLP: " << Banner << ":\n";
    E->dump();
    dbgs() << "SLP: Costs:\n";
    dbgs() << "SLP:     ReuseShuffleCost = " << ReuseShuffleCost << "\n";
    dbgs() << "SLP:     VectorCost = " << VecCost << "\n";
    dbgs() << "SLP:     ScalarCost = " << ScalarCost << "\n";
    dbgs() << "SLP:     ReuseShuffleCost + VecCost - ScalarCost = "
           << ReuseShuffleCost + VecCost - ScalarCost << "\n";
  }
#endif

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL,
                          std::optional<ScheduleData *> Bundle,
                          const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<int> ReuseShuffleIndices = std::nullopt,
                          ArrayRef<unsigned> ReorderIndices = std::nullopt) {
    TreeEntry::EntryState EntryState =
        Bundle ? TreeEntry::Vectorize : TreeEntry::NeedToGather;
    return newTreeEntry(VL, EntryState, Bundle, S, UserTreeIdx,
                        ReuseShuffleIndices, ReorderIndices);
  }

  TreeEntry *newTreeEntry(ArrayRef<Value *> VL,
                          TreeEntry::EntryState EntryState,
                          std::optional<ScheduleData *> Bundle,
                          const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<int> ReuseShuffleIndices = std::nullopt,
                          ArrayRef<unsigned> ReorderIndices = std::nullopt) {
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
      InstructionsState S = getSameOpcode(Last->Scalars, *TLI);
      Last->setOperations(S);
      Last->ReorderIndices.append(ReorderIndices.begin(), ReorderIndices.end());
    }
    if (Last->State != TreeEntry::NeedToGather) {
      for (Value *V : VL) {
        assert(!getTreeEntry(V) && "Scalar already in tree!");
        ScalarToTreeEntry[V] = Last;
      }
      // Update the scheduler bundle to point to this TreeEntry.
      ScheduleData *BundleMember = *Bundle;
      assert((BundleMember || isa<PHINode>(S.MainOp) ||
              isVectorLikeInstWithConstOps(S.MainOp) ||
              doesNotNeedToSchedule(VL)) &&
             "Bundle and VL out of sync");
      if (BundleMember) {
        for (Value *V : VL) {
          if (doesNotNeedToBeScheduled(V))
            continue;
          assert(BundleMember && "Unexpected end of bundle.");
          BundleMember->TE = Last;
          BundleMember = BundleMember->NextInBundle;
        }
      }
      assert(!BundleMember && "Bundle and VL out of sync");
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

  /// Checks if the specified list of the instructions/values can be vectorized
  /// and fills required data before actual scheduling of the instructions.
  TreeEntry::EntryState getScalarsVectorizationState(
      InstructionsState &S, ArrayRef<Value *> VL, bool IsScatterVectorizeUserTE,
      OrdersType &CurrentOrder, SmallVectorImpl<Value *> &PointerOps) const;

  /// Maps a specific scalar to its tree entry.
  SmallDenseMap<Value *, TreeEntry *> ScalarToTreeEntry;

  /// Maps a value to the proposed vectorizable size.
  SmallDenseMap<Value *, unsigned> InstrElementSize;

  /// A list of scalars that we found that we need to keep as scalars.
  ValueSet MustGather;

  /// A map between the vectorized entries and the last instructions in the
  /// bundles. The bundles are built in use order, not in the def order of the
  /// instructions. So, we cannot rely directly on the last instruction in the
  /// bundle being the last instruction in the program order during
  /// vectorization process since the basic blocks are affected, need to
  /// pre-gather them before.
  DenseMap<const TreeEntry *, Instruction *> EntryToLastInstruction;

  /// List of gather nodes, depending on other gather/vector nodes, which should
  /// be emitted after the vector instruction emission process to correctly
  /// handle order of the vector instructions and shuffles.
  SetVector<const TreeEntry *> PostponedGathers;

  using ValueToGatherNodesMap =
      DenseMap<Value *, SmallPtrSet<const TreeEntry *, 4>>;
  ValueToGatherNodesMap ValueToGatherNodes;

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
    std::optional<bool> &result = AliasCache[key];
    if (result) {
      return *result;
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
  DenseMap<AliasCacheKey, std::optional<bool>> AliasCache;

  // Cache for pointerMayBeCaptured calls inside AA.  This is preserved
  // globally through SLP because we don't perform any action which
  // invalidates capture results.
  BatchAAResults BatchAA;

  /// Temporary store for deleted instructions. Instructions will be deleted
  /// eventually when the BoUpSLP is destructed.  The deferral is required to
  /// ensure that there are no incorrect collisions in the AliasCache, which
  /// can happen if a new instruction is allocated at the same address as a
  /// previously deleted instruction.
  DenseSet<Instruction *> DeletedInstructions;

  /// Set of the instruction, being analyzed already for reductions.
  SmallPtrSet<Instruction *, 16> AnalyzedReductionsRoots;

  /// Set of hashes for the list of reduction values already being analyzed.
  DenseSet<size_t> AnalyzedReductionVals;

  /// A list of values that need to extracted out of the tree.
  /// This list holds pairs of (Internal Scalar : External User). External User
  /// can be nullptr, it means that this Internal Scalar will be used later,
  /// after vectorization.
  UserList ExternalUses;

  /// Values used only by @llvm.assume calls.
  SmallPtrSet<const Value *, 32> EphValues;

  /// Holds all of the instructions that we gathered, shuffle instructions and
  /// extractelements.
  SetVector<Instruction *> GatherShuffleExtractSeq;

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
      return NextInBundle != nullptr || FirstInBundle != this || TE;
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
      ControlDependencies.clear();
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

    /// List of instructions which this instruction could be control dependent
    /// on.  Allowing such nodes to be scheduled below this one could introduce
    /// a runtime fault which didn't exist in the original program.
    /// ex: this is a load or udiv following a readonly call which inf loops
    SmallVector<ScheduleData *, 4> ControlDependencies;

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
  /// It does not schedules instructions, which are not memory read/write
  /// instructions and their operands are either constants, or arguments, or
  /// phis, or instructions from others blocks, or their users are phis or from
  /// the other blocks. The resulting vector instructions can be placed at the
  /// beginning of the basic block without scheduling (if operands does not need
  /// to be scheduled) or at the end of the block (if users are outside of the
  /// block). It allows to save some compile time and memory used by the
  /// compiler.
  /// ScheduleData is assigned for each instruction in between the boundaries of
  /// the tree entry, even for those, which are not part of the graph. It is
  /// required to correctly follow the dependencies between the instructions and
  /// their correct scheduling. The ScheduleData is not allocated for the
  /// instructions, which do not require scheduling, like phis, nodes with
  /// extractelements/insertelements only or nodes with instructions, with
  /// uses/operands outside of the block.
  struct BlockScheduling {
    BlockScheduling(BasicBlock *BB)
        : BB(BB), ChunkSize(BB->size()), ChunkPos(ChunkSize) {}

    void clear() {
      ReadyInsts.clear();
      ScheduleStart = nullptr;
      ScheduleEnd = nullptr;
      FirstLoadStoreInRegion = nullptr;
      LastLoadStoreInRegion = nullptr;
      RegionHasStackSave = false;

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
      ScheduleData *SD = ScheduleDataMap.lookup(I);
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
        ScheduleData *SD = I->second.lookup(Key);
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
          // Need to search for the lane since the tree entry can be reordered.
          int Lane = std::distance(TE->Scalars.begin(),
                                   find(TE->Scalars, BundleMember->Inst));
          assert(Lane >= 0 && "Lane not set");

          // Since vectorization tree is being built recursively this assertion
          // ensures that the tree entry has all operands set before reaching
          // this code. Couple of exceptions known at the moment are extracts
          // where their second (immediate) operand is not added. Since
          // immediates do not affect scheduler behavior this is considered
          // okay.
          auto *In = BundleMember->Inst;
          assert(In &&
                 (isa<ExtractValueInst, ExtractElementInst>(In) ||
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
          if (MemoryDepSD->hasValidDependencies() &&
              MemoryDepSD->incrementUnscheduledDeps(-1) == 0) {
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
        // Handle the control dependencies.
        for (ScheduleData *DepSD : BundleMember->ControlDependencies) {
          if (DepSD->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after decrementing,
            // so we can put the dependent instruction into the ready list.
            ScheduleData *DepBundle = DepSD->FirstInBundle;
            assert(!DepBundle->IsScheduled &&
                   "already scheduled bundle gets ready");
            ReadyList.insert(DepBundle);
            LLVM_DEBUG(dbgs()
                       << "SLP:    gets ready (ctl): " << *DepBundle << "\n");
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
        if (!SD)
          continue;
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
          if (SD->isSchedulingEntity() && SD->hasValidDependencies() &&
              SD->isReady()) {
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
    /// \returns the scheduling bundle. The returned Optional value is not
    /// std::nullopt if \p VL is allowed to be scheduled.
    std::optional<ScheduleData *>
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

    /// Is there an llvm.stacksave or llvm.stackrestore in the scheduling
    /// region?  Used to optimize the dependence calculation for the
    /// common case where there isn't.
    bool RegionHasStackSave = false;

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
  const SmallDenseSet<Value *> *UserIgnoreList = nullptr;

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
  DenseMap<Value *, std::pair<uint64_t, bool>> MinBWs;
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
    OS << Entry->Idx << ".\n";
    if (isSplat(Entry->Scalars))
      OS << "<splat> ";
    for (auto *V : Entry->Scalars) {
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
    if (Entry->State == TreeEntry::ScatterVectorize ||
        Entry->State == TreeEntry::PossibleStridedVectorize)
      return "color=blue";
    return "";
  }
};

} // end namespace llvm

BoUpSLP::~BoUpSLP() {
  SmallVector<WeakTrackingVH> DeadInsts;
  for (auto *I : DeletedInstructions) {
    for (Use &U : I->operands()) {
      auto *Op = dyn_cast<Instruction>(U.get());
      if (Op && !DeletedInstructions.count(Op) && Op->hasOneUser() &&
          wouldInstructionBeTriviallyDead(Op, TLI))
        DeadInsts.emplace_back(Op);
    }
    I->dropAllReferences();
  }
  for (auto *I : DeletedInstructions) {
    assert(I->use_empty() &&
           "trying to erase instruction with users.");
    I->eraseFromParent();
  }

  // Cleanup any dead scalar code feeding the vectorized instructions
  RecursivelyDeleteTriviallyDeadInstructions(DeadInsts, TLI);

#ifdef EXPENSIVE_CHECKS
  // If we could guarantee that this call is not extremely slow, we could
  // remove the ifdef limitation (see PR47712).
  assert(!verifyFunction(*F, &dbgs()));
#endif
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
    if (Mask[I] != PoisonMaskElem)
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
    if (MaskOrder[I] != PoisonMaskElem)
      Order[MaskOrder[I]] = I;
  fixupOrderingIndices(Order);
}

std::optional<BoUpSLP::OrdersType>
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
        return std::nullopt;
      unsigned Lane =
          std::distance(STE->Scalars.begin(), find(STE->Scalars, V));
      if (Lane >= NumScalars)
        return std::nullopt;
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
    if (IsIdentityOrder(CurrentOrder))
      return OrdersType();
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
    return std::move(CurrentOrder);
  }
  return std::nullopt;
}

namespace {
/// Tracks the state we can represent the loads in the given sequence.
enum class LoadsState {
  Gather,
  Vectorize,
  ScatterVectorize,
  PossibleStridedVectorize
};
} // anonymous namespace

static bool arePointersCompatible(Value *Ptr1, Value *Ptr2,
                                  const TargetLibraryInfo &TLI,
                                  bool CompareOpcodes = true) {
  if (getUnderlyingObject(Ptr1) != getUnderlyingObject(Ptr2))
    return false;
  auto *GEP1 = dyn_cast<GetElementPtrInst>(Ptr1);
  if (!GEP1)
    return false;
  auto *GEP2 = dyn_cast<GetElementPtrInst>(Ptr2);
  if (!GEP2)
    return false;
  return GEP1->getNumOperands() == 2 && GEP2->getNumOperands() == 2 &&
         ((isConstant(GEP1->getOperand(1)) &&
           isConstant(GEP2->getOperand(1))) ||
          !CompareOpcodes ||
          getSameOpcode({GEP1->getOperand(1), GEP2->getOperand(1)}, TLI)
              .getOpcode());
}

/// Checks if the given array of loads can be represented as a vectorized,
/// scatter or just simple gather.
static LoadsState canVectorizeLoads(ArrayRef<Value *> VL, const Value *VL0,
                                    const TargetTransformInfo &TTI,
                                    const DataLayout &DL, ScalarEvolution &SE,
                                    LoopInfo &LI, const TargetLibraryInfo &TLI,
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
  // Check the order of pointer operands or that all pointers are the same.
  bool IsSorted = sortPtrAccesses(PointerOps, ScalarTy, DL, SE, Order);
  if (IsSorted || all_of(PointerOps, [&](Value *P) {
        return arePointersCompatible(P, PointerOps.front(), TLI);
      })) {
    bool IsPossibleStrided = false;
    if (IsSorted) {
      Value *Ptr0;
      Value *PtrN;
      if (Order.empty()) {
        Ptr0 = PointerOps.front();
        PtrN = PointerOps.back();
      } else {
        Ptr0 = PointerOps[Order.front()];
        PtrN = PointerOps[Order.back()];
      }
      std::optional<int> Diff =
          getPointersDiff(ScalarTy, Ptr0, ScalarTy, PtrN, DL, SE);
      // Check that the sorted loads are consecutive.
      if (static_cast<unsigned>(*Diff) == VL.size() - 1)
        return LoadsState::Vectorize;
      // Simple check if not a strided access - clear order.
      IsPossibleStrided = *Diff % (VL.size() - 1) == 0;
    }
    // TODO: need to improve analysis of the pointers, if not all of them are
    // GEPs or have > 2 operands, we end up with a gather node, which just
    // increases the cost.
    Loop *L = LI.getLoopFor(cast<LoadInst>(VL0)->getParent());
    bool ProfitableGatherPointers =
        static_cast<unsigned>(count_if(PointerOps, [L](Value *V) {
          return L && L->isLoopInvariant(V);
        })) <= VL.size() / 2 && VL.size() > 2;
    if (ProfitableGatherPointers || all_of(PointerOps, [IsSorted](Value *P) {
          auto *GEP = dyn_cast<GetElementPtrInst>(P);
          return (IsSorted && !GEP && doesNotNeedToBeScheduled(P)) ||
                 (GEP && GEP->getNumOperands() == 2);
        })) {
      Align CommonAlignment = cast<LoadInst>(VL0)->getAlign();
      for (Value *V : VL)
        CommonAlignment =
            std::min(CommonAlignment, cast<LoadInst>(V)->getAlign());
      auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());
      if (TTI.isLegalMaskedGather(VecTy, CommonAlignment) &&
          !TTI.forceScalarizeMaskedGather(VecTy, CommonAlignment))
        return IsPossibleStrided ? LoadsState::PossibleStridedVectorize
                                 : LoadsState::ScatterVectorize;
    }
  }

  return LoadsState::Gather;
}

static bool clusterSortPtrAccesses(ArrayRef<Value *> VL, Type *ElemTy,
                                   const DataLayout &DL, ScalarEvolution &SE,
                                   SmallVectorImpl<unsigned> &SortedIndices) {
  assert(llvm::all_of(
             VL, [](const Value *V) { return V->getType()->isPointerTy(); }) &&
         "Expected list of pointer operands.");
  // Map from bases to a vector of (Ptr, Offset, OrigIdx), which we insert each
  // Ptr into, sort and return the sorted indices with values next to one
  // another.
  MapVector<Value *, SmallVector<std::tuple<Value *, int, unsigned>>> Bases;
  Bases[VL[0]].push_back(std::make_tuple(VL[0], 0U, 0U));

  unsigned Cnt = 1;
  for (Value *Ptr : VL.drop_front()) {
    bool Found = any_of(Bases, [&](auto &Base) {
      std::optional<int> Diff =
          getPointersDiff(ElemTy, Base.first, ElemTy, Ptr, DL, SE,
                          /*StrictCheck=*/true);
      if (!Diff)
        return false;

      Base.second.emplace_back(Ptr, *Diff, Cnt++);
      return true;
    });

    if (!Found) {
      // If we haven't found enough to usefully cluster, return early.
      if (Bases.size() > VL.size() / 2 - 1)
        return false;

      // Not found already - add a new Base
      Bases[Ptr].emplace_back(Ptr, 0, Cnt++);
    }
  }

  // For each of the bases sort the pointers by Offset and check if any of the
  // base become consecutively allocated.
  bool AnyConsecutive = false;
  for (auto &Base : Bases) {
    auto &Vec = Base.second;
    if (Vec.size() > 1) {
      llvm::stable_sort(Vec, [](const std::tuple<Value *, int, unsigned> &X,
                                const std::tuple<Value *, int, unsigned> &Y) {
        return std::get<1>(X) < std::get<1>(Y);
      });
      int InitialOffset = std::get<1>(Vec[0]);
      AnyConsecutive |= all_of(enumerate(Vec), [InitialOffset](const auto &P) {
        return std::get<1>(P.value()) == int(P.index()) + InitialOffset;
      });
    }
  }

  // Fill SortedIndices array only if it looks worth-while to sort the ptrs.
  SortedIndices.clear();
  if (!AnyConsecutive)
    return false;

  for (auto &Base : Bases) {
    for (auto &T : Base.second)
      SortedIndices.push_back(std::get<2>(T));
  }

  assert(SortedIndices.size() == VL.size() &&
         "Expected SortedIndices to be the size of VL");
  return true;
}

std::optional<BoUpSLP::OrdersType>
BoUpSLP::findPartiallyOrderedLoads(const BoUpSLP::TreeEntry &TE) {
  assert(TE.State == TreeEntry::NeedToGather && "Expected gather node only.");
  Type *ScalarTy = TE.Scalars[0]->getType();

  SmallVector<Value *> Ptrs;
  Ptrs.reserve(TE.Scalars.size());
  for (Value *V : TE.Scalars) {
    auto *L = dyn_cast<LoadInst>(V);
    if (!L || !L->isSimple())
      return std::nullopt;
    Ptrs.push_back(L->getPointerOperand());
  }

  BoUpSLP::OrdersType Order;
  if (clusterSortPtrAccesses(Ptrs, ScalarTy, *DL, *SE, Order))
    return std::move(Order);
  return std::nullopt;
}

/// Check if two insertelement instructions are from the same buildvector.
static bool areTwoInsertFromSameBuildVector(
    InsertElementInst *VU, InsertElementInst *V,
    function_ref<Value *(InsertElementInst *)> GetBaseOperand) {
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
  std::optional<unsigned> Idx1 = getInsertIndex(IE1);
  std::optional<unsigned> Idx2 = getInsertIndex(IE2);
  if (Idx1 == std::nullopt || Idx2 == std::nullopt)
    return false;
  // Go through the vector operand of insertelement instructions trying to find
  // either VU as the original vector for IE2 or V as the original vector for
  // IE1.
  SmallSet<int, 8> ReusedIdx;
  bool IsReusedIdx = false;
  do {
    if (IE2 == VU && !IE1)
      return VU->hasOneUse();
    if (IE1 == V && !IE2)
      return V->hasOneUse();
    if (IE1 && IE1 != V) {
      IsReusedIdx |=
          !ReusedIdx.insert(getInsertIndex(IE1).value_or(*Idx2)).second;
      if ((IE1 != VU && !IE1->hasOneUse()) || IsReusedIdx)
        IE1 = nullptr;
      else
        IE1 = dyn_cast_or_null<InsertElementInst>(GetBaseOperand(IE1));
    }
    if (IE2 && IE2 != VU) {
      IsReusedIdx |=
          !ReusedIdx.insert(getInsertIndex(IE2).value_or(*Idx1)).second;
      if ((IE2 != V && !IE2->hasOneUse()) || IsReusedIdx)
        IE2 = nullptr;
      else
        IE2 = dyn_cast_or_null<InsertElementInst>(GetBaseOperand(IE2));
    }
  } while (!IsReusedIdx && (IE1 || IE2));
  return false;
}

std::optional<BoUpSLP::OrdersType>
BoUpSLP::getReorderingData(const TreeEntry &TE, bool TopToBottom) {
  // No need to reorder if need to shuffle reuses, still need to shuffle the
  // node.
  if (!TE.ReuseShuffleIndices.empty()) {
    // Check if reuse shuffle indices can be improved by reordering.
    // For this, check that reuse mask is "clustered", i.e. each scalar values
    // is used once in each submask of size <number_of_scalars>.
    // Example: 4 scalar values.
    // ReuseShuffleIndices mask: 0, 1, 2, 3, 3, 2, 0, 1 - clustered.
    //                           0, 1, 2, 3, 3, 3, 1, 0 - not clustered, because
    //                           element 3 is used twice in the second submask.
    unsigned Sz = TE.Scalars.size();
    if (!ShuffleVectorInst::isOneUseSingleSourceMask(TE.ReuseShuffleIndices,
                                                     Sz))
      return std::nullopt;
    unsigned VF = TE.getVectorFactor();
    // Try build correct order for extractelement instructions.
    SmallVector<int> ReusedMask(TE.ReuseShuffleIndices.begin(),
                                TE.ReuseShuffleIndices.end());
    if (TE.getOpcode() == Instruction::ExtractElement && !TE.isAltShuffle() &&
        all_of(TE.Scalars, [Sz](Value *V) {
          std::optional<unsigned> Idx = getExtractIndex(cast<Instruction>(V));
          return Idx && *Idx < Sz;
        })) {
      SmallVector<int> ReorderMask(Sz, PoisonMaskElem);
      if (TE.ReorderIndices.empty())
        std::iota(ReorderMask.begin(), ReorderMask.end(), 0);
      else
        inversePermutation(TE.ReorderIndices, ReorderMask);
      for (unsigned I = 0; I < VF; ++I) {
        int &Idx = ReusedMask[I];
        if (Idx == PoisonMaskElem)
          continue;
        Value *V = TE.Scalars[ReorderMask[Idx]];
        std::optional<unsigned> EI = getExtractIndex(cast<Instruction>(V));
        Idx = std::distance(ReorderMask.begin(), find(ReorderMask, *EI));
      }
    }
    // Build the order of the VF size, need to reorder reuses shuffles, they are
    // always of VF size.
    OrdersType ResOrder(VF);
    std::iota(ResOrder.begin(), ResOrder.end(), 0);
    auto *It = ResOrder.begin();
    for (unsigned K = 0; K < VF; K += Sz) {
      OrdersType CurrentOrder(TE.ReorderIndices);
      SmallVector<int> SubMask{ArrayRef(ReusedMask).slice(K, Sz)};
      if (SubMask.front() == PoisonMaskElem)
        std::iota(SubMask.begin(), SubMask.end(), 0);
      reorderOrder(CurrentOrder, SubMask);
      transform(CurrentOrder, It, [K](unsigned Pos) { return Pos + K; });
      std::advance(It, Sz);
    }
    if (all_of(enumerate(ResOrder),
               [](const auto &Data) { return Data.index() == Data.value(); }))
      return std::nullopt; // No need to reorder.
    return std::move(ResOrder);
  }
  if ((TE.State == TreeEntry::Vectorize ||
       TE.State == TreeEntry::PossibleStridedVectorize) &&
      (isa<LoadInst, ExtractElementInst, ExtractValueInst>(TE.getMainOp()) ||
       (TopToBottom && isa<StoreInst, InsertElementInst>(TE.getMainOp()))) &&
      !TE.isAltShuffle())
    return TE.ReorderIndices;
  if (TE.State == TreeEntry::Vectorize && TE.getOpcode() == Instruction::PHI) {
    auto PHICompare = [&](unsigned I1, unsigned I2) {
      Value *V1 = TE.Scalars[I1];
      Value *V2 = TE.Scalars[I2];
      if (V1 == V2)
        return false;
      if (!V1->hasOneUse() || !V2->hasOneUse())
        return false;
      auto *FirstUserOfPhi1 = cast<Instruction>(*V1->user_begin());
      auto *FirstUserOfPhi2 = cast<Instruction>(*V2->user_begin());
      if (auto *IE1 = dyn_cast<InsertElementInst>(FirstUserOfPhi1))
        if (auto *IE2 = dyn_cast<InsertElementInst>(FirstUserOfPhi2)) {
          if (!areTwoInsertFromSameBuildVector(
                  IE1, IE2,
                  [](InsertElementInst *II) { return II->getOperand(0); }))
            return false;
          std::optional<unsigned> Idx1 = getInsertIndex(IE1);
          std::optional<unsigned> Idx2 = getInsertIndex(IE2);
          if (Idx1 == std::nullopt || Idx2 == std::nullopt)
            return false;
          return *Idx1 < *Idx2;
        }
      if (auto *EE1 = dyn_cast<ExtractElementInst>(FirstUserOfPhi1))
        if (auto *EE2 = dyn_cast<ExtractElementInst>(FirstUserOfPhi2)) {
          if (EE1->getOperand(0) != EE2->getOperand(0))
            return false;
          std::optional<unsigned> Idx1 = getExtractIndex(EE1);
          std::optional<unsigned> Idx2 = getExtractIndex(EE2);
          if (Idx1 == std::nullopt || Idx2 == std::nullopt)
            return false;
          return *Idx1 < *Idx2;
        }
      return false;
    };
    auto IsIdentityOrder = [](const OrdersType &Order) {
      for (unsigned Idx : seq<unsigned>(0, Order.size()))
        if (Idx != Order[Idx])
          return false;
      return true;
    };
    if (!TE.ReorderIndices.empty())
      return TE.ReorderIndices;
    DenseMap<unsigned, unsigned> PhiToId;
    SmallVector<unsigned> Phis(TE.Scalars.size());
    std::iota(Phis.begin(), Phis.end(), 0);
    OrdersType ResOrder(TE.Scalars.size());
    for (unsigned Id = 0, Sz = TE.Scalars.size(); Id < Sz; ++Id)
      PhiToId[Id] = Id;
    stable_sort(Phis, PHICompare);
    for (unsigned Id = 0, Sz = Phis.size(); Id < Sz; ++Id)
      ResOrder[Id] = PhiToId[Phis[Id]];
    if (IsIdentityOrder(ResOrder))
      return std::nullopt; // No need to reorder.
    return std::move(ResOrder);
  }
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
        return std::move(CurrentOrder);
      }
    }
    // If the gather node is <undef, v, .., poison> and
    // insertelement poison, v, 0 [+ permute]
    // is cheaper than
    // insertelement poison, v, n - try to reorder.
    // If rotating the whole graph, exclude the permute cost, the whole graph
    // might be transformed.
    int Sz = TE.Scalars.size();
    if (isSplat(TE.Scalars) && !allConstant(TE.Scalars) &&
        count_if(TE.Scalars, UndefValue::classof) == Sz - 1) {
      const auto *It =
          find_if(TE.Scalars, [](Value *V) { return !isConstant(V); });
      if (It == TE.Scalars.begin())
        return OrdersType();
      auto *Ty = FixedVectorType::get(TE.Scalars.front()->getType(), Sz);
      if (It != TE.Scalars.end()) {
        OrdersType Order(Sz, Sz);
        unsigned Idx = std::distance(TE.Scalars.begin(), It);
        Order[Idx] = 0;
        fixupOrderingIndices(Order);
        SmallVector<int> Mask;
        inversePermutation(Order, Mask);
        InstructionCost PermuteCost =
            TopToBottom
                ? 0
                : TTI->getShuffleCost(TTI::SK_PermuteSingleSrc, Ty, Mask);
        InstructionCost InsertFirstCost = TTI->getVectorInstrCost(
            Instruction::InsertElement, Ty, TTI::TCK_RecipThroughput, 0,
            PoisonValue::get(Ty), *It);
        InstructionCost InsertIdxCost = TTI->getVectorInstrCost(
            Instruction::InsertElement, Ty, TTI::TCK_RecipThroughput, Idx,
            PoisonValue::get(Ty), *It);
        if (InsertFirstCost + PermuteCost < InsertIdxCost)
          return std::move(Order);
      }
    }
    if (std::optional<OrdersType> CurrentOrder = findReusedOrderedScalars(TE))
      return CurrentOrder;
    if (TE.Scalars.size() >= 4)
      if (std::optional<OrdersType> Order = findPartiallyOrderedLoads(TE))
        return Order;
  }
  return std::nullopt;
}

/// Checks if the given mask is a "clustered" mask with the same clusters of
/// size \p Sz, which are not identity submasks.
static bool isRepeatedNonIdentityClusteredMask(ArrayRef<int> Mask,
                                               unsigned Sz) {
  ArrayRef<int> FirstCluster = Mask.slice(0, Sz);
  if (ShuffleVectorInst::isIdentityMask(FirstCluster))
    return false;
  for (unsigned I = Sz, E = Mask.size(); I < E; I += Sz) {
    ArrayRef<int> Cluster = Mask.slice(I, Sz);
    if (Cluster != FirstCluster)
      return false;
  }
  return true;
}

void BoUpSLP::reorderNodeWithReuses(TreeEntry &TE, ArrayRef<int> Mask) const {
  // Reorder reuses mask.
  reorderReuses(TE.ReuseShuffleIndices, Mask);
  const unsigned Sz = TE.Scalars.size();
  // For vectorized and non-clustered reused no need to do anything else.
  if (TE.State != TreeEntry::NeedToGather ||
      !ShuffleVectorInst::isOneUseSingleSourceMask(TE.ReuseShuffleIndices,
                                                   Sz) ||
      !isRepeatedNonIdentityClusteredMask(TE.ReuseShuffleIndices, Sz))
    return;
  SmallVector<int> NewMask;
  inversePermutation(TE.ReorderIndices, NewMask);
  addMask(NewMask, TE.ReuseShuffleIndices);
  // Clear reorder since it is going to be applied to the new mask.
  TE.ReorderIndices.clear();
  // Try to improve gathered nodes with clustered reuses, if possible.
  ArrayRef<int> Slice = ArrayRef(NewMask).slice(0, Sz);
  SmallVector<unsigned> NewOrder(Slice.begin(), Slice.end());
  inversePermutation(NewOrder, NewMask);
  reorderScalars(TE.Scalars, NewMask);
  // Fill the reuses mask with the identity submasks.
  for (auto *It = TE.ReuseShuffleIndices.begin(),
            *End = TE.ReuseShuffleIndices.end();
       It != End; std::advance(It, Sz))
    std::iota(It, std::next(It, Sz), 0);
}

void BoUpSLP::reorderTopToBottom() {
  // Maps VF to the graph nodes.
  DenseMap<unsigned, SetVector<TreeEntry *>> VFToOrderedEntries;
  // ExtractElement gather nodes which can be vectorized and need to handle
  // their ordering.
  DenseMap<const TreeEntry *, OrdersType> GathersToOrders;

  // Phi nodes can have preferred ordering based on their result users
  DenseMap<const TreeEntry *, OrdersType> PhisToOrders;

  // AltShuffles can also have a preferred ordering that leads to fewer
  // instructions, e.g., the addsub instruction in x86.
  DenseMap<const TreeEntry *, OrdersType> AltShufflesToOrders;

  // Maps a TreeEntry to the reorder indices of external users.
  DenseMap<const TreeEntry *, SmallVector<OrdersType, 1>>
      ExternalUserReorderMap;
  // FIXME: Workaround for syntax error reported by MSVC buildbots.
  TargetTransformInfo &TTIRef = *TTI;
  // Find all reorderable nodes with the given VF.
  // Currently the are vectorized stores,loads,extracts + some gathering of
  // extracts.
  for_each(VectorizableTree, [this, &TTIRef, &VFToOrderedEntries,
                              &GathersToOrders, &ExternalUserReorderMap,
                              &AltShufflesToOrders, &PhisToOrders](
                                 const std::unique_ptr<TreeEntry> &TE) {
    // Look for external users that will probably be vectorized.
    SmallVector<OrdersType, 1> ExternalUserReorderIndices =
        findExternalStoreUsersReorderIndices(TE.get());
    if (!ExternalUserReorderIndices.empty()) {
      VFToOrderedEntries[TE->getVectorFactor()].insert(TE.get());
      ExternalUserReorderMap.try_emplace(TE.get(),
                                         std::move(ExternalUserReorderIndices));
    }

    // Patterns like [fadd,fsub] can be combined into a single instruction in
    // x86. Reordering them into [fsub,fadd] blocks this pattern. So we need
    // to take into account their order when looking for the most used order.
    if (TE->isAltShuffle()) {
      VectorType *VecTy =
          FixedVectorType::get(TE->Scalars[0]->getType(), TE->Scalars.size());
      unsigned Opcode0 = TE->getOpcode();
      unsigned Opcode1 = TE->getAltOpcode();
      // The opcode mask selects between the two opcodes.
      SmallBitVector OpcodeMask(TE->Scalars.size(), false);
      for (unsigned Lane : seq<unsigned>(0, TE->Scalars.size()))
        if (cast<Instruction>(TE->Scalars[Lane])->getOpcode() == Opcode1)
          OpcodeMask.set(Lane);
      // If this pattern is supported by the target then we consider the order.
      if (TTIRef.isLegalAltInstr(VecTy, Opcode0, Opcode1, OpcodeMask)) {
        VFToOrderedEntries[TE->getVectorFactor()].insert(TE.get());
        AltShufflesToOrders.try_emplace(TE.get(), OrdersType());
      }
      // TODO: Check the reverse order too.
    }

    if (std::optional<OrdersType> CurrentOrder =
            getReorderingData(*TE, /*TopToBottom=*/true)) {
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
        UserTE = UserTE->UserTreeIndices.back().UserTE;
        ++Cnt;
      }
      VFToOrderedEntries[TE->getVectorFactor()].insert(TE.get());
      if (!(TE->State == TreeEntry::Vectorize ||
            TE->State == TreeEntry::PossibleStridedVectorize) ||
          !TE->ReuseShuffleIndices.empty())
        GathersToOrders.try_emplace(TE.get(), *CurrentOrder);
      if (TE->State == TreeEntry::Vectorize &&
          TE->getOpcode() == Instruction::PHI)
        PhisToOrders.try_emplace(TE.get(), *CurrentOrder);
    }
  });

  // Reorder the graph nodes according to their vectorization factor.
  for (unsigned VF = VectorizableTree.front()->getVectorFactor(); VF > 1;
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
    // Last chance orders - scatter vectorize. Try to use their orders if no
    // other orders or the order is counted already.
    SmallVector<OrdersType> StridedVectorizeOrders;
    SmallPtrSet<const TreeEntry *, 4> VisitedOps;
    for (const TreeEntry *OpTE : OrderedEntries) {
      // No need to reorder this nodes, still need to extend and to use shuffle,
      // just need to merge reordering shuffle and the reuse shuffle.
      if (!OpTE->ReuseShuffleIndices.empty() && !GathersToOrders.count(OpTE))
        continue;
      // Count number of orders uses.
      const auto &Order = [OpTE, &GathersToOrders, &AltShufflesToOrders,
                           &PhisToOrders]() -> const OrdersType & {
        if (OpTE->State == TreeEntry::NeedToGather ||
            !OpTE->ReuseShuffleIndices.empty()) {
          auto It = GathersToOrders.find(OpTE);
          if (It != GathersToOrders.end())
            return It->second;
        }
        if (OpTE->isAltShuffle()) {
          auto It = AltShufflesToOrders.find(OpTE);
          if (It != AltShufflesToOrders.end())
            return It->second;
        }
        if (OpTE->State == TreeEntry::Vectorize &&
            OpTE->getOpcode() == Instruction::PHI) {
          auto It = PhisToOrders.find(OpTE);
          if (It != PhisToOrders.end())
            return It->second;
        }
        return OpTE->ReorderIndices;
      }();
      // First consider the order of the external scalar users.
      auto It = ExternalUserReorderMap.find(OpTE);
      if (It != ExternalUserReorderMap.end()) {
        const auto &ExternalUserReorderIndices = It->second;
        // If the OpTE vector factor != number of scalars - use natural order,
        // it is an attempt to reorder node with reused scalars but with
        // external uses.
        if (OpTE->getVectorFactor() != OpTE->Scalars.size()) {
          OrdersUses.insert(std::make_pair(OrdersType(), 0)).first->second +=
              ExternalUserReorderIndices.size();
        } else {
          for (const OrdersType &ExtOrder : ExternalUserReorderIndices)
            ++OrdersUses.insert(std::make_pair(ExtOrder, 0)).first->second;
        }
        // No other useful reorder data in this entry.
        if (Order.empty())
          continue;
      }
      // Postpone scatter orders.
      if (OpTE->State == TreeEntry::PossibleStridedVectorize) {
        StridedVectorizeOrders.push_back(Order);
        continue;
      }
      // Stores actually store the mask, not the order, need to invert.
      if (OpTE->State == TreeEntry::Vectorize && !OpTE->isAltShuffle() &&
          OpTE->getOpcode() == Instruction::Store && !Order.empty()) {
        SmallVector<int> Mask;
        inversePermutation(Order, Mask);
        unsigned E = Order.size();
        OrdersType CurrentOrder(E, E);
        transform(Mask, CurrentOrder.begin(), [E](int Idx) {
          return Idx == PoisonMaskElem ? E : static_cast<unsigned>(Idx);
        });
        fixupOrderingIndices(CurrentOrder);
        ++OrdersUses.insert(std::make_pair(CurrentOrder, 0)).first->second;
      } else {
        ++OrdersUses.insert(std::make_pair(Order, 0)).first->second;
      }
    }
    // Set order of the user node.
    if (OrdersUses.empty()) {
      if (StridedVectorizeOrders.empty())
        continue;
      // Add (potentially!) strided vectorize orders.
      for (OrdersType &Order : StridedVectorizeOrders)
        ++OrdersUses.insert(std::make_pair(Order, 0)).first->second;
    } else {
      // Account (potentially!) strided vectorize orders only if it was used
      // already.
      for (OrdersType &Order : StridedVectorizeOrders) {
        auto *It = OrdersUses.find(Order);
        if (It != OrdersUses.end())
          ++It->second;
      }
    }
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
    SmallVector<int> MaskOrder(BestOrder.size(), PoisonMaskElem);
    unsigned E = BestOrder.size();
    transform(BestOrder, MaskOrder.begin(), [E](unsigned I) {
      return I < E ? static_cast<int>(I) : PoisonMaskElem;
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
          reorderNodeWithReuses(*TE, Mask);
        }
        continue;
      }
      if ((TE->State == TreeEntry::Vectorize ||
           TE->State == TreeEntry::PossibleStridedVectorize) &&
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
      // Add ScatterVectorize nodes to the list of operands, where just
      // reordering of the scalars is required. Similar to the gathers, so
      // simply add to the list of gathered ops.
      // If there are reused scalars, process this node as a regular vectorize
      // node, just reorder reuses mask.
      if (TE->State != TreeEntry::Vectorize &&
          TE->State != TreeEntry::PossibleStridedVectorize &&
          TE->ReuseShuffleIndices.empty() && TE->ReorderIndices.empty())
        GatherOps.push_back(TE);
      continue;
    }
    TreeEntry *Gather = nullptr;
    if (count_if(ReorderableGathers,
                 [&Gather, UserTE, I](TreeEntry *TE) {
                   assert(TE->State != TreeEntry::Vectorize &&
                          "Only non-vectorized nodes are expected.");
                   if (any_of(TE->UserTreeIndices,
                              [UserTE, I](const EdgeInfo &EI) {
                                return EI.UserTE == UserTE && EI.EdgeIdx == I;
                              })) {
                     assert(TE->isSame(UserTE->getOperand(I)) &&
                            "Operand entry does not match operands.");
                     Gather = TE;
                     return true;
                   }
                   return false;
                 }) > 1 &&
        !allConstant(UserTE->getOperand(I)))
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
    if (TE->State != TreeEntry::Vectorize &&
        TE->State != TreeEntry::PossibleStridedVectorize)
      NonVectorized.push_back(TE.get());
    if (std::optional<OrdersType> CurrentOrder =
            getReorderingData(*TE, /*TopToBottom=*/false)) {
      OrderedEntries.insert(TE.get());
      if (!(TE->State == TreeEntry::Vectorize ||
            TE->State == TreeEntry::PossibleStridedVectorize) ||
          !TE->ReuseShuffleIndices.empty())
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
    DenseMap<TreeEntry *, SmallVector<std::pair<unsigned, TreeEntry *>>> Users;
    SmallVector<TreeEntry *> Filtered;
    for (TreeEntry *TE : OrderedEntries) {
      if (!(TE->State == TreeEntry::Vectorize ||
            TE->State == TreeEntry::PossibleStridedVectorize ||
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
    SmallVector<
        std::pair<TreeEntry *, SmallVector<std::pair<unsigned, TreeEntry *>>>>
        UsersVec(Users.begin(), Users.end());
    sort(UsersVec, [](const auto &Data1, const auto &Data2) {
      return Data1.first->Idx > Data2.first->Idx;
    });
    for (auto &Data : UsersVec) {
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
      // Last chance orders - scatter vectorize. Try to use their orders if no
      // other orders or the order is counted already.
      SmallVector<std::pair<OrdersType, unsigned>> StridedVectorizeOrders;
      // Do the analysis for each tree entry only once, otherwise the order of
      // the same node my be considered several times, though might be not
      // profitable.
      SmallPtrSet<const TreeEntry *, 4> VisitedOps;
      SmallPtrSet<const TreeEntry *, 4> VisitedUsers;
      for (const auto &Op : Data.second) {
        TreeEntry *OpTE = Op.second;
        if (!VisitedOps.insert(OpTE).second)
          continue;
        if (!OpTE->ReuseShuffleIndices.empty() && !GathersToOrders.count(OpTE))
          continue;
        const auto &Order = [OpTE, &GathersToOrders]() -> const OrdersType & {
          if (OpTE->State == TreeEntry::NeedToGather ||
              !OpTE->ReuseShuffleIndices.empty())
            return GathersToOrders.find(OpTE)->second;
          return OpTE->ReorderIndices;
        }();
        unsigned NumOps = count_if(
            Data.second, [OpTE](const std::pair<unsigned, TreeEntry *> &P) {
              return P.second == OpTE;
            });
        // Postpone scatter orders.
        if (OpTE->State == TreeEntry::PossibleStridedVectorize) {
          StridedVectorizeOrders.emplace_back(Order, NumOps);
          continue;
        }
        // Stores actually store the mask, not the order, need to invert.
        if (OpTE->State == TreeEntry::Vectorize && !OpTE->isAltShuffle() &&
            OpTE->getOpcode() == Instruction::Store && !Order.empty()) {
          SmallVector<int> Mask;
          inversePermutation(Order, Mask);
          unsigned E = Order.size();
          OrdersType CurrentOrder(E, E);
          transform(Mask, CurrentOrder.begin(), [E](int Idx) {
            return Idx == PoisonMaskElem ? E : static_cast<unsigned>(Idx);
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
        if (StridedVectorizeOrders.empty() ||
            (Data.first->ReorderIndices.empty() &&
             Data.first->ReuseShuffleIndices.empty() &&
             !(IgnoreReorder &&
               Data.first == VectorizableTree.front().get()))) {
          for_each(
              Data.second,
              [&OrderedEntries](const std::pair<unsigned, TreeEntry *> &Op) {
                OrderedEntries.remove(Op.second);
              });
          continue;
        }
        // Add (potentially!) strided vectorize orders.
        for (std::pair<OrdersType, unsigned> &Pair : StridedVectorizeOrders)
          OrdersUses.insert(std::make_pair(Pair.first, 0)).first->second +=
              Pair.second;
      } else {
        // Account (potentially!) strided vectorize orders only if it was used
        // already.
        for (std::pair<OrdersType, unsigned> &Pair : StridedVectorizeOrders) {
          auto *It = OrdersUses.find(Pair.first);
          if (It != OrdersUses.end())
            It->second += Pair.second;
        }
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
      SmallVector<int> MaskOrder(BestOrder.size(), PoisonMaskElem);
      unsigned E = BestOrder.size();
      transform(BestOrder, MaskOrder.begin(), [E](unsigned I) {
        return I < E ? static_cast<int>(I) : PoisonMaskElem;
      });
      for (const std::pair<unsigned, TreeEntry *> &Op : Data.second) {
        TreeEntry *TE = Op.second;
        OrderedEntries.remove(TE);
        if (!VisitedOps.insert(TE).second)
          continue;
        if (TE->ReuseShuffleIndices.size() == BestOrder.size()) {
          reorderNodeWithReuses(*TE, Mask);
          continue;
        }
        // Gathers are processed separately.
        if (TE->State != TreeEntry::Vectorize &&
            TE->State != TreeEntry::PossibleStridedVectorize &&
            (TE->State != TreeEntry::ScatterVectorize ||
             TE->ReorderIndices.empty()))
          continue;
        assert((BestOrder.size() == TE->ReorderIndices.size() ||
                TE->ReorderIndices.empty()) &&
               "Non-matching sizes of user/operand entries.");
        reorderOrder(TE->ReorderIndices, Mask);
        if (IgnoreReorder && TE == VectorizableTree.front().get())
          IgnoreReorder = false;
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
          Data.first->isAltShuffle() ||
          Data.first->State == TreeEntry::PossibleStridedVectorize) {
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
              UseEntry->State == TreeEntry::PossibleStridedVectorize ||
              !InTreeUserNeedToExtract(Scalar, UserInst, TLI)) {
            LLVM_DEBUG(dbgs() << "SLP: \tInternal user will be removed:" << *U
                              << ".\n");
            assert(UseEntry->State != TreeEntry::NeedToGather && "Bad state");
            continue;
          }
        }

        // Ignore users in the user ignore list.
        if (UserIgnoreList && UserIgnoreList->contains(UserInst))
          continue;

        LLVM_DEBUG(dbgs() << "SLP: Need to extract:" << *U << " from lane "
                          << Lane << " from " << *Scalar << ".\n");
        ExternalUses.push_back(ExternalUser(Scalar, U, FoundLane));
      }
    }
  }
}

DenseMap<Value *, SmallVector<StoreInst *>>
BoUpSLP::collectUserStores(const BoUpSLP::TreeEntry *TE) const {
  DenseMap<Value *, SmallVector<StoreInst *>> PtrToStoresMap;
  for (unsigned Lane : seq<unsigned>(0, TE->Scalars.size())) {
    Value *V = TE->Scalars[Lane];
    // To save compilation time we don't visit if we have too many users.
    static constexpr unsigned UsersLimit = 4;
    if (V->hasNUsesOrMore(UsersLimit))
      break;

    // Collect stores per pointer object.
    for (User *U : V->users()) {
      auto *SI = dyn_cast<StoreInst>(U);
      if (SI == nullptr || !SI->isSimple() ||
          !isValidElementType(SI->getValueOperand()->getType()))
        continue;
      // Skip entry if already
      if (getTreeEntry(U))
        continue;

      Value *Ptr = getUnderlyingObject(SI->getPointerOperand());
      auto &StoresVec = PtrToStoresMap[Ptr];
      // For now just keep one store per pointer object per lane.
      // TODO: Extend this to support multiple stores per pointer per lane
      if (StoresVec.size() > Lane)
        continue;
      // Skip if in different BBs.
      if (!StoresVec.empty() &&
          SI->getParent() != StoresVec.back()->getParent())
        continue;
      // Make sure that the stores are of the same type.
      if (!StoresVec.empty() &&
          SI->getValueOperand()->getType() !=
              StoresVec.back()->getValueOperand()->getType())
        continue;
      StoresVec.push_back(SI);
    }
  }
  return PtrToStoresMap;
}

bool BoUpSLP::canFormVector(ArrayRef<StoreInst *> StoresVec,
                            OrdersType &ReorderIndices) const {
  // We check whether the stores in StoreVec can form a vector by sorting them
  // and checking whether they are consecutive.

  // To avoid calling getPointersDiff() while sorting we create a vector of
  // pairs {store, offset from first} and sort this instead.
  SmallVector<std::pair<StoreInst *, int>> StoreOffsetVec(StoresVec.size());
  StoreInst *S0 = StoresVec[0];
  StoreOffsetVec[0] = {S0, 0};
  Type *S0Ty = S0->getValueOperand()->getType();
  Value *S0Ptr = S0->getPointerOperand();
  for (unsigned Idx : seq<unsigned>(1, StoresVec.size())) {
    StoreInst *SI = StoresVec[Idx];
    std::optional<int> Diff =
        getPointersDiff(S0Ty, S0Ptr, SI->getValueOperand()->getType(),
                        SI->getPointerOperand(), *DL, *SE,
                        /*StrictCheck=*/true);
    // We failed to compare the pointers so just abandon this StoresVec.
    if (!Diff)
      return false;
    StoreOffsetVec[Idx] = {StoresVec[Idx], *Diff};
  }

  // Sort the vector based on the pointers. We create a copy because we may
  // need the original later for calculating the reorder (shuffle) indices.
  stable_sort(StoreOffsetVec, [](const std::pair<StoreInst *, int> &Pair1,
                                 const std::pair<StoreInst *, int> &Pair2) {
    int Offset1 = Pair1.second;
    int Offset2 = Pair2.second;
    return Offset1 < Offset2;
  });

  // Check if the stores are consecutive by checking if their difference is 1.
  for (unsigned Idx : seq<unsigned>(1, StoreOffsetVec.size()))
    if (StoreOffsetVec[Idx].second != StoreOffsetVec[Idx - 1].second + 1)
      return false;

  // Calculate the shuffle indices according to their offset against the sorted
  // StoreOffsetVec.
  ReorderIndices.reserve(StoresVec.size());
  for (StoreInst *SI : StoresVec) {
    unsigned Idx = find_if(StoreOffsetVec,
                           [SI](const std::pair<StoreInst *, int> &Pair) {
                             return Pair.first == SI;
                           }) -
                   StoreOffsetVec.begin();
    ReorderIndices.push_back(Idx);
  }
  // Identity order (e.g., {0,1,2,3}) is modeled as an empty OrdersType in
  // reorderTopToBottom() and reorderBottomToTop(), so we are following the
  // same convention here.
  auto IsIdentityOrder = [](const OrdersType &Order) {
    for (unsigned Idx : seq<unsigned>(0, Order.size()))
      if (Idx != Order[Idx])
        return false;
    return true;
  };
  if (IsIdentityOrder(ReorderIndices))
    ReorderIndices.clear();

  return true;
}

#ifndef NDEBUG
LLVM_DUMP_METHOD static void dumpOrder(const BoUpSLP::OrdersType &Order) {
  for (unsigned Idx : Order)
    dbgs() << Idx << ", ";
  dbgs() << "\n";
}
#endif

SmallVector<BoUpSLP::OrdersType, 1>
BoUpSLP::findExternalStoreUsersReorderIndices(TreeEntry *TE) const {
  unsigned NumLanes = TE->Scalars.size();

  DenseMap<Value *, SmallVector<StoreInst *>> PtrToStoresMap =
      collectUserStores(TE);

  // Holds the reorder indices for each candidate store vector that is a user of
  // the current TreeEntry.
  SmallVector<OrdersType, 1> ExternalReorderIndices;

  // Now inspect the stores collected per pointer and look for vectorization
  // candidates. For each candidate calculate the reorder index vector and push
  // it into `ExternalReorderIndices`
  for (const auto &Pair : PtrToStoresMap) {
    auto &StoresVec = Pair.second;
    // If we have fewer than NumLanes stores, then we can't form a vector.
    if (StoresVec.size() != NumLanes)
      continue;

    // If the stores are not consecutive then abandon this StoresVec.
    OrdersType ReorderIndices;
    if (!canFormVector(StoresVec, ReorderIndices))
      continue;

    // We now know that the scalars in StoresVec can form a vector instruction,
    // so set the reorder indices.
    ExternalReorderIndices.push_back(ReorderIndices);
  }
  return ExternalReorderIndices;
}

void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        const SmallDenseSet<Value *> &UserIgnoreLst) {
  deleteTree();
  UserIgnoreList = &UserIgnoreLst;
  if (!allSameType(Roots))
    return;
  buildTree_rec(Roots, 0, EdgeInfo());
}

void BoUpSLP::buildTree(ArrayRef<Value *> Roots) {
  deleteTree();
  if (!allSameType(Roots))
    return;
  buildTree_rec(Roots, 0, EdgeInfo());
}

/// \return true if the specified list of values has only one instruction that
/// requires scheduling, false otherwise.
#ifndef NDEBUG
static bool needToScheduleSingleInstruction(ArrayRef<Value *> VL) {
  Value *NeedsScheduling = nullptr;
  for (Value *V : VL) {
    if (doesNotNeedToBeScheduled(V))
      continue;
    if (!NeedsScheduling) {
      NeedsScheduling = V;
      continue;
    }
    return false;
  }
  return NeedsScheduling;
}
#endif

/// Generates key/subkey pair for the given value to provide effective sorting
/// of the values and better detection of the vectorizable values sequences. The
/// keys/subkeys can be used for better sorting of the values themselves (keys)
/// and in values subgroups (subkeys).
static std::pair<size_t, size_t> generateKeySubkey(
    Value *V, const TargetLibraryInfo *TLI,
    function_ref<hash_code(size_t, LoadInst *)> LoadsSubkeyGenerator,
    bool AllowAlternate) {
  hash_code Key = hash_value(V->getValueID() + 2);
  hash_code SubKey = hash_value(0);
  // Sort the loads by the distance between the pointers.
  if (auto *LI = dyn_cast<LoadInst>(V)) {
    Key = hash_combine(LI->getType(), hash_value(Instruction::Load), Key);
    if (LI->isSimple())
      SubKey = hash_value(LoadsSubkeyGenerator(Key, LI));
    else
      Key = SubKey = hash_value(LI);
  } else if (isVectorLikeInstWithConstOps(V)) {
    // Sort extracts by the vector operands.
    if (isa<ExtractElementInst, UndefValue>(V))
      Key = hash_value(Value::UndefValueVal + 1);
    if (auto *EI = dyn_cast<ExtractElementInst>(V)) {
      if (!isUndefVector(EI->getVectorOperand()).all() &&
          !isa<UndefValue>(EI->getIndexOperand()))
        SubKey = hash_value(EI->getVectorOperand());
    }
  } else if (auto *I = dyn_cast<Instruction>(V)) {
    // Sort other instructions just by the opcodes except for CMPInst.
    // For CMP also sort by the predicate kind.
    if ((isa<BinaryOperator, CastInst>(I)) &&
        isValidForAlternation(I->getOpcode())) {
      if (AllowAlternate)
        Key = hash_value(isa<BinaryOperator>(I) ? 1 : 0);
      else
        Key = hash_combine(hash_value(I->getOpcode()), Key);
      SubKey = hash_combine(
          hash_value(I->getOpcode()), hash_value(I->getType()),
          hash_value(isa<BinaryOperator>(I)
                         ? I->getType()
                         : cast<CastInst>(I)->getOperand(0)->getType()));
      // For casts, look through the only operand to improve compile time.
      if (isa<CastInst>(I)) {
        std::pair<size_t, size_t> OpVals =
            generateKeySubkey(I->getOperand(0), TLI, LoadsSubkeyGenerator,
                              /*AllowAlternate=*/true);
        Key = hash_combine(OpVals.first, Key);
        SubKey = hash_combine(OpVals.first, SubKey);
      }
    } else if (auto *CI = dyn_cast<CmpInst>(I)) {
      CmpInst::Predicate Pred = CI->getPredicate();
      if (CI->isCommutative())
        Pred = std::min(Pred, CmpInst::getInversePredicate(Pred));
      CmpInst::Predicate SwapPred = CmpInst::getSwappedPredicate(Pred);
      SubKey = hash_combine(hash_value(I->getOpcode()), hash_value(Pred),
                            hash_value(SwapPred),
                            hash_value(CI->getOperand(0)->getType()));
    } else if (auto *Call = dyn_cast<CallInst>(I)) {
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(Call, TLI);
      if (isTriviallyVectorizable(ID)) {
        SubKey = hash_combine(hash_value(I->getOpcode()), hash_value(ID));
      } else if (!VFDatabase(*Call).getMappings(*Call).empty()) {
        SubKey = hash_combine(hash_value(I->getOpcode()),
                              hash_value(Call->getCalledFunction()));
      } else {
        Key = hash_combine(hash_value(Call), Key);
        SubKey = hash_combine(hash_value(I->getOpcode()), hash_value(Call));
      }
      for (const CallBase::BundleOpInfo &Op : Call->bundle_op_infos())
        SubKey = hash_combine(hash_value(Op.Begin), hash_value(Op.End),
                              hash_value(Op.Tag), SubKey);
    } else if (auto *Gep = dyn_cast<GetElementPtrInst>(I)) {
      if (Gep->getNumOperands() == 2 && isa<ConstantInt>(Gep->getOperand(1)))
        SubKey = hash_value(Gep->getPointerOperand());
      else
        SubKey = hash_value(Gep);
    } else if (BinaryOperator::isIntDivRem(I->getOpcode()) &&
               !isa<ConstantInt>(I->getOperand(1))) {
      // Do not try to vectorize instructions with potentially high cost.
      SubKey = hash_value(I);
    } else {
      SubKey = hash_value(I->getOpcode());
    }
    Key = hash_combine(hash_value(I->getParent()), Key);
  }
  return std::make_pair(Key, SubKey);
}

/// Checks if the specified instruction \p I is an alternate operation for
/// the given \p MainOp and \p AltOp instructions.
static bool isAlternateInstruction(const Instruction *I,
                                   const Instruction *MainOp,
                                   const Instruction *AltOp,
                                   const TargetLibraryInfo &TLI);

BoUpSLP::TreeEntry::EntryState BoUpSLP::getScalarsVectorizationState(
    InstructionsState &S, ArrayRef<Value *> VL, bool IsScatterVectorizeUserTE,
    OrdersType &CurrentOrder, SmallVectorImpl<Value *> &PointerOps) const {
  assert(S.MainOp && "Expected instructions with same/alternate opcodes only.");

  unsigned ShuffleOrOp =
      S.isAltShuffle() ? (unsigned)Instruction::ShuffleVector : S.getOpcode();
  auto *VL0 = cast<Instruction>(S.OpValue);
  switch (ShuffleOrOp) {
  case Instruction::PHI: {
    // Check for terminator values (e.g. invoke).
    for (Value *V : VL)
      for (Value *Incoming : cast<PHINode>(V)->incoming_values()) {
        Instruction *Term = dyn_cast<Instruction>(Incoming);
        if (Term && Term->isTerminator()) {
          LLVM_DEBUG(dbgs()
                     << "SLP: Need to swizzle PHINodes (terminator use).\n");
          return TreeEntry::NeedToGather;
        }
      }

    return TreeEntry::Vectorize;
  }
  case Instruction::ExtractValue:
  case Instruction::ExtractElement: {
    bool Reuse = canReuseExtract(VL, VL0, CurrentOrder);
    if (Reuse || !CurrentOrder.empty())
      return TreeEntry::Vectorize;
    LLVM_DEBUG(dbgs() << "SLP: Gather extract sequence.\n");
    return TreeEntry::NeedToGather;
  }
  case Instruction::InsertElement: {
    // Check that we have a buildvector and not a shuffle of 2 or more
    // different vectors.
    ValueSet SourceVectors;
    for (Value *V : VL) {
      SourceVectors.insert(cast<Instruction>(V)->getOperand(0));
      assert(getInsertIndex(V) != std::nullopt &&
             "Non-constant or undef index?");
    }

    if (count_if(VL, [&SourceVectors](Value *V) {
          return !SourceVectors.contains(V);
        }) >= 2) {
      // Found 2nd source vector - cancel.
      LLVM_DEBUG(dbgs() << "SLP: Gather of insertelement vectors with "
                           "different source vectors.\n");
      return TreeEntry::NeedToGather;
    }

    return TreeEntry::Vectorize;
  }
  case Instruction::Load: {
    // Check that a vectorized load would load the same memory as a scalar
    // load. For example, we don't want to vectorize loads that are smaller
    // than 8-bit. Even though we have a packed struct {<i2, i2, i2, i2>} LLVM
    // treats loading/storing it as an i8 struct. If we vectorize loads/stores
    // from such a struct, we read/write packed bits disagreeing with the
    // unvectorized version.
    switch (canVectorizeLoads(VL, VL0, *TTI, *DL, *SE, *LI, *TLI, CurrentOrder,
                              PointerOps)) {
    case LoadsState::Vectorize:
      return TreeEntry::Vectorize;
    case LoadsState::ScatterVectorize:
      return TreeEntry::ScatterVectorize;
    case LoadsState::PossibleStridedVectorize:
      return TreeEntry::PossibleStridedVectorize;
    case LoadsState::Gather:
#ifndef NDEBUG
      Type *ScalarTy = VL0->getType();
      if (DL->getTypeSizeInBits(ScalarTy) !=
          DL->getTypeAllocSizeInBits(ScalarTy))
        LLVM_DEBUG(dbgs() << "SLP: Gathering loads of non-packed type.\n");
      else if (any_of(VL,
                      [](Value *V) { return !cast<LoadInst>(V)->isSimple(); }))
        LLVM_DEBUG(dbgs() << "SLP: Gathering non-simple loads.\n");
      else
        LLVM_DEBUG(dbgs() << "SLP: Gathering non-consecutive loads.\n");
#endif // NDEBUG
      return TreeEntry::NeedToGather;
    }
    llvm_unreachable("Unexpected state of loads");
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
        LLVM_DEBUG(
            dbgs() << "SLP: Gathering casts with different src types.\n");
        return TreeEntry::NeedToGather;
      }
    }
    return TreeEntry::Vectorize;
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
        LLVM_DEBUG(dbgs() << "SLP: Gathering cmp with different predicate.\n");
        return TreeEntry::NeedToGather;
      }
    }
    return TreeEntry::Vectorize;
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
  case Instruction::Xor:
    return TreeEntry::Vectorize;
  case Instruction::GetElementPtr: {
    // We don't combine GEPs with complicated (nested) indexing.
    for (Value *V : VL) {
      auto *I = dyn_cast<GetElementPtrInst>(V);
      if (!I)
        continue;
      if (I->getNumOperands() != 2) {
        LLVM_DEBUG(dbgs() << "SLP: not-vectorizable GEP (nested indexes).\n");
        return TreeEntry::NeedToGather;
      }
    }

    // We can't combine several GEPs into one vector if they operate on
    // different types.
    Type *Ty0 = cast<GEPOperator>(VL0)->getSourceElementType();
    for (Value *V : VL) {
      auto *GEP = dyn_cast<GEPOperator>(V);
      if (!GEP)
        continue;
      Type *CurTy = GEP->getSourceElementType();
      if (Ty0 != CurTy) {
        LLVM_DEBUG(dbgs() << "SLP: not-vectorizable GEP (different types).\n");
        return TreeEntry::NeedToGather;
      }
    }

    // We don't combine GEPs with non-constant indexes.
    Type *Ty1 = VL0->getOperand(1)->getType();
    for (Value *V : VL) {
      auto *I = dyn_cast<GetElementPtrInst>(V);
      if (!I)
        continue;
      auto *Op = I->getOperand(1);
      if ((!IsScatterVectorizeUserTE && !isa<ConstantInt>(Op)) ||
          (Op->getType() != Ty1 &&
           ((IsScatterVectorizeUserTE && !isa<ConstantInt>(Op)) ||
            Op->getType()->getScalarSizeInBits() >
                DL->getIndexSizeInBits(
                    V->getType()->getPointerAddressSpace())))) {
        LLVM_DEBUG(
            dbgs() << "SLP: not-vectorizable GEP (non-constant indexes).\n");
        return TreeEntry::NeedToGather;
      }
    }

    return TreeEntry::Vectorize;
  }
  case Instruction::Store: {
    // Check if the stores are consecutive or if we need to swizzle them.
    llvm::Type *ScalarTy = cast<StoreInst>(VL0)->getValueOperand()->getType();
    // Avoid types that are padded when being allocated as scalars, while
    // being packed together in a vector (such as i1).
    if (DL->getTypeSizeInBits(ScalarTy) !=
        DL->getTypeAllocSizeInBits(ScalarTy)) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering stores of non-packed type.\n");
      return TreeEntry::NeedToGather;
    }
    // Make sure all stores in the bundle are simple - we can't vectorize
    // atomic or volatile stores.
    for (Value *V : VL) {
      auto *SI = cast<StoreInst>(V);
      if (!SI->isSimple()) {
        LLVM_DEBUG(dbgs() << "SLP: Gathering non-simple stores.\n");
        return TreeEntry::NeedToGather;
      }
      PointerOps.push_back(SI->getPointerOperand());
    }

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
      std::optional<int> Dist =
          getPointersDiff(ScalarTy, Ptr0, ScalarTy, PtrN, *DL, *SE);
      // Check that the sorted pointer operands are consecutive.
      if (static_cast<unsigned>(*Dist) == VL.size() - 1)
        return TreeEntry::Vectorize;
    }

    LLVM_DEBUG(dbgs() << "SLP: Non-consecutive store.\n");
    return TreeEntry::NeedToGather;
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
      LLVM_DEBUG(dbgs() << "SLP: Non-vectorizable call.\n");
      return TreeEntry::NeedToGather;
    }
    Function *F = CI->getCalledFunction();
    unsigned NumArgs = CI->arg_size();
    SmallVector<Value *, 4> ScalarArgs(NumArgs, nullptr);
    for (unsigned J = 0; J != NumArgs; ++J)
      if (isVectorIntrinsicWithScalarOpAtArg(ID, J))
        ScalarArgs[J] = CI->getArgOperand(J);
    for (Value *V : VL) {
      CallInst *CI2 = dyn_cast<CallInst>(V);
      if (!CI2 || CI2->getCalledFunction() != F ||
          getVectorIntrinsicIDForCall(CI2, TLI) != ID ||
          (VecFunc &&
           VecFunc != VFDatabase(*CI2).getVectorizedFunction(Shape)) ||
          !CI->hasIdenticalOperandBundleSchema(*CI2)) {
        LLVM_DEBUG(dbgs() << "SLP: mismatched calls:" << *CI << "!=" << *V
                          << "\n");
        return TreeEntry::NeedToGather;
      }
      // Some intrinsics have scalar arguments and should be same in order for
      // them to be vectorized.
      for (unsigned J = 0; J != NumArgs; ++J) {
        if (isVectorIntrinsicWithScalarOpAtArg(ID, J)) {
          Value *A1J = CI2->getArgOperand(J);
          if (ScalarArgs[J] != A1J) {
            LLVM_DEBUG(dbgs()
                       << "SLP: mismatched arguments in call:" << *CI
                       << " argument " << ScalarArgs[J] << "!=" << A1J << "\n");
            return TreeEntry::NeedToGather;
          }
        }
      }
      // Verify that the bundle operands are identical between the two calls.
      if (CI->hasOperandBundles() &&
          !std::equal(CI->op_begin() + CI->getBundleOperandsStartIndex(),
                      CI->op_begin() + CI->getBundleOperandsEndIndex(),
                      CI2->op_begin() + CI2->getBundleOperandsStartIndex())) {
        LLVM_DEBUG(dbgs() << "SLP: mismatched bundle operands in calls:" << *CI
                          << "!=" << *V << '\n');
        return TreeEntry::NeedToGather;
      }
    }

    return TreeEntry::Vectorize;
  }
  case Instruction::ShuffleVector: {
    // If this is not an alternate sequence of opcode like add-sub
    // then do not vectorize this instruction.
    if (!S.isAltShuffle()) {
      LLVM_DEBUG(dbgs() << "SLP: ShuffleVector are not vectorized.\n");
      return TreeEntry::NeedToGather;
    }
    return TreeEntry::Vectorize;
  }
  default:
    LLVM_DEBUG(dbgs() << "SLP: Gathering unknown instruction.\n");
    return TreeEntry::NeedToGather;
  }
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
    DenseMap<Value *, unsigned> UniquePositions(VL.size());
    for (Value *V : VL) {
      if (isConstant(V)) {
        ReuseShuffleIndicies.emplace_back(
            isa<UndefValue>(V) ? PoisonMaskElem : UniqueValues.size());
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
          !llvm::has_single_bit<uint32_t>(NumUniqueScalarValues)) {
        LLVM_DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
        newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx);
        return false;
      }
      VL = UniqueValues;
    }
    return true;
  };

  InstructionsState S = getSameOpcode(VL, *TLI);

  // Gather if we hit the RecursionMaxDepth, unless this is a load (or z/sext of
  // a load), in which case peek through to include it in the tree, without
  // ballooning over-budget.
  if (Depth >= RecursionMaxDepth &&
      !(S.MainOp && isa<Instruction>(S.MainOp) && S.MainOp == S.AltOp &&
        VL.size() >= 4 &&
        (match(S.MainOp, m_Load(m_Value())) || all_of(VL, [&S](const Value *I) {
           return match(I,
                        m_OneUse(m_ZExtOrSExt(m_OneUse(m_Load(m_Value()))))) &&
                  cast<Instruction>(I)->getOpcode() ==
                      cast<Instruction>(S.MainOp)->getOpcode();
         })))) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    if (TryToFindDuplicates(S))
      newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
    return;
  }

  // Don't handle scalable vectors
  if (S.getOpcode() == Instruction::ExtractElement &&
      isa<ScalableVectorType>(
          cast<ExtractElementInst>(S.OpValue)->getVectorOperandType())) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to scalable vector type.\n");
    if (TryToFindDuplicates(S))
      newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
    return;
  }

  // Don't handle vectors.
  if (S.OpValue->getType()->isVectorTy() &&
      !isa<InsertElementInst>(S.OpValue)) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(S.OpValue))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx);
      return;
    }

  // If all of the operands are identical or constant we have a simple solution.
  // If we deal with insert/extract instructions, they all must have constant
  // indices, otherwise we should gather them, not try to vectorize.
  // If alternate op node with 2 elements with gathered operands - do not
  // vectorize.
  auto &&NotProfitableForVectorization = [&S, this,
                                          Depth](ArrayRef<Value *> VL) {
    if (!S.getOpcode() || !S.isAltShuffle() || VL.size() > 2)
      return false;
    if (VectorizableTree.size() < MinTreeSize)
      return false;
    if (Depth >= RecursionMaxDepth - 1)
      return true;
    // Check if all operands are extracts, part of vector node or can build a
    // regular vectorize node.
    SmallVector<unsigned, 2> InstsCount(VL.size(), 0);
    for (Value *V : VL) {
      auto *I = cast<Instruction>(V);
      InstsCount.push_back(count_if(I->operand_values(), [](Value *Op) {
        return isa<Instruction>(Op) || isVectorLikeInstWithConstOps(Op);
      }));
    }
    bool IsCommutative = isCommutative(S.MainOp) || isCommutative(S.AltOp);
    if ((IsCommutative &&
         std::accumulate(InstsCount.begin(), InstsCount.end(), 0) < 2) ||
        (!IsCommutative &&
         all_of(InstsCount, [](unsigned ICnt) { return ICnt < 2; })))
      return true;
    assert(VL.size() == 2 && "Expected only 2 alternate op instructions.");
    SmallVector<SmallVector<std::pair<Value *, Value *>>> Candidates;
    auto *I1 = cast<Instruction>(VL.front());
    auto *I2 = cast<Instruction>(VL.back());
    for (int Op = 0, E = S.MainOp->getNumOperands(); Op < E; ++Op)
      Candidates.emplace_back().emplace_back(I1->getOperand(Op),
                                             I2->getOperand(Op));
    if (static_cast<unsigned>(count_if(
            Candidates, [this](ArrayRef<std::pair<Value *, Value *>> Cand) {
              return findBestRootPair(Cand, LookAheadHeuristics::ScoreSplat);
            })) >= S.MainOp->getNumOperands() / 2)
      return false;
    if (S.MainOp->getNumOperands() > 2)
      return true;
    if (IsCommutative) {
      // Check permuted operands.
      Candidates.clear();
      for (int Op = 0, E = S.MainOp->getNumOperands(); Op < E; ++Op)
        Candidates.emplace_back().emplace_back(I1->getOperand(Op),
                                               I2->getOperand((Op + 1) % E));
      if (any_of(
              Candidates, [this](ArrayRef<std::pair<Value *, Value *>> Cand) {
                return findBestRootPair(Cand, LookAheadHeuristics::ScoreSplat);
              }))
        return false;
    }
    return true;
  };
  SmallVector<unsigned> SortedIndices;
  BasicBlock *BB = nullptr;
  bool IsScatterVectorizeUserTE =
      UserTreeIdx.UserTE &&
      (UserTreeIdx.UserTE->State == TreeEntry::ScatterVectorize ||
       UserTreeIdx.UserTE->State == TreeEntry::PossibleStridedVectorize);
  bool AreAllSameInsts =
      (S.getOpcode() && allSameBlock(VL)) ||
      (S.OpValue->getType()->isPointerTy() && IsScatterVectorizeUserTE &&
       VL.size() > 2 &&
       all_of(VL,
              [&BB](Value *V) {
                auto *I = dyn_cast<GetElementPtrInst>(V);
                if (!I)
                  return doesNotNeedToBeScheduled(V);
                if (!BB)
                  BB = I->getParent();
                return BB == I->getParent() && I->getNumOperands() == 2;
              }) &&
       BB &&
       sortPtrAccesses(VL, UserTreeIdx.UserTE->getMainOp()->getType(), *DL, *SE,
                       SortedIndices));
  if (!AreAllSameInsts || allConstant(VL) || isSplat(VL) ||
      (isa<InsertElementInst, ExtractValueInst, ExtractElementInst>(
           S.OpValue) &&
       !all_of(VL, isVectorLikeInstWithConstOps)) ||
      NotProfitableForVectorization(VL)) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O, small shuffle. \n");
    if (TryToFindDuplicates(S))
      newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
    return;
  }

  // We now know that this is a vector of instructions of the same type from
  // the same block.

  // Don't vectorize ephemeral values.
  if (!EphValues.empty()) {
    for (Value *V : VL) {
      if (EphValues.count(V)) {
        LLVM_DEBUG(dbgs() << "SLP: The instruction (" << *V
                          << ") is ephemeral.\n");
        newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx);
        return;
      }
    }
  }

  // Check if this is a duplicate of another entry.
  if (TreeEntry *E = getTreeEntry(S.OpValue)) {
    LLVM_DEBUG(dbgs() << "SLP: \tChecking bundle: " << *S.OpValue << ".\n");
    if (!E->isSame(VL)) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to partial overlap.\n");
      if (TryToFindDuplicates(S))
        newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
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
    if (!IsScatterVectorizeUserTE && !isa<Instruction>(V))
      continue;
    if (getTreeEntry(V)) {
      LLVM_DEBUG(dbgs() << "SLP: The instruction (" << *V
                        << ") is already in tree.\n");
      if (TryToFindDuplicates(S))
        newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
      return;
    }
  }

  // The reduction nodes (stored in UserIgnoreList) also should stay scalar.
  if (UserIgnoreList && !UserIgnoreList->empty()) {
    for (Value *V : VL) {
      if (UserIgnoreList && UserIgnoreList->contains(V)) {
        LLVM_DEBUG(dbgs() << "SLP: Gathering due to gathered scalar.\n");
        if (TryToFindDuplicates(S))
          newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
        return;
      }
    }
  }

  // Special processing for sorted pointers for ScatterVectorize node with
  // constant indeces only.
  if (AreAllSameInsts && UserTreeIdx.UserTE &&
      (UserTreeIdx.UserTE->State == TreeEntry::ScatterVectorize ||
       UserTreeIdx.UserTE->State == TreeEntry::PossibleStridedVectorize) &&
      !(S.getOpcode() && allSameBlock(VL))) {
    assert(S.OpValue->getType()->isPointerTy() &&
           count_if(VL, [](Value *V) { return isa<GetElementPtrInst>(V); }) >=
               2 &&
           "Expected pointers only.");
    // Reset S to make it GetElementPtr kind of node.
    const auto *It = find_if(VL, [](Value *V) { return isa<GetElementPtrInst>(V); });
    assert(It != VL.end() && "Expected at least one GEP.");
    S = getSameOpcode(*It, *TLI);
  }

  // Check that all of the users of the scalars that we want to vectorize are
  // schedulable.
  auto *VL0 = cast<Instruction>(S.OpValue);
  BB = VL0->getParent();

  if (!DT->isReachableFromEntry(BB)) {
    // Don't go into unreachable blocks. They may contain instructions with
    // dependency cycles which confuse the final scheduling.
    LLVM_DEBUG(dbgs() << "SLP: bundle in unreachable block.\n");
    newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  // Don't go into catchswitch blocks, which can happen with PHIs.
  // Such blocks can only have PHIs and the catchswitch.  There is no
  // place to insert a shuffle if we need to, so just avoid that issue.
  if (isa<CatchSwitchInst>(BB->getTerminator())) {
    LLVM_DEBUG(dbgs() << "SLP: bundle in catchswitch block.\n");
    newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  // Check that every instruction appears once in this bundle.
  if (!TryToFindDuplicates(S))
    return;

  // Perform specific checks for each particular instruction kind.
  OrdersType CurrentOrder;
  SmallVector<Value *> PointerOps;
  TreeEntry::EntryState State = getScalarsVectorizationState(
      S, VL, IsScatterVectorizeUserTE, CurrentOrder, PointerOps);
  if (State == TreeEntry::NeedToGather) {
    newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                 ReuseShuffleIndicies);
    return;
  }

  auto &BSRef = BlocksSchedules[BB];
  if (!BSRef)
    BSRef = std::make_unique<BlockScheduling>(BB);

  BlockScheduling &BS = *BSRef;

  std::optional<ScheduleData *> Bundle = BS.tryScheduleBundle(VL, this, S);
#ifdef EXPENSIVE_CHECKS
  // Make sure we didn't break any internal invariants
  BS.verify();
#endif
  if (!Bundle) {
    LLVM_DEBUG(dbgs() << "SLP: We are not able to schedule this bundle!\n");
    assert((!BS.getScheduleData(VL0) ||
            !BS.getScheduleData(VL0)->isPartOfBundle()) &&
           "tryScheduleBundle should cancelScheduling on failure");
    newTreeEntry(VL, std::nullopt /*not vectorized*/, S, UserTreeIdx,
                 ReuseShuffleIndicies);
    return;
  }
  LLVM_DEBUG(dbgs() << "SLP: We are able to schedule this bundle.\n");

  unsigned ShuffleOrOp = S.isAltShuffle() ?
                (unsigned) Instruction::ShuffleVector : S.getOpcode();
  switch (ShuffleOrOp) {
    case Instruction::PHI: {
      auto *PH = cast<PHINode>(VL0);

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
      if (CurrentOrder.empty()) {
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
    case Instruction::InsertElement: {
      assert(ReuseShuffleIndicies.empty() && "All inserts should be unique");

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
                                   std::nullopt, CurrentOrder);
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
      TreeEntry *TE = nullptr;
      fixupOrderingIndices(CurrentOrder);
      switch (State) {
      case TreeEntry::Vectorize:
        if (CurrentOrder.empty()) {
          // Original loads are consecutive and does not require reordering.
          TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                            ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs() << "SLP: added a vector of loads.\n");
        } else {
          // Need to reorder.
          TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                            ReuseShuffleIndicies, CurrentOrder);
          LLVM_DEBUG(dbgs() << "SLP: added a vector of jumbled loads.\n");
        }
        TE->setOperandsInOrder();
        break;
      case TreeEntry::PossibleStridedVectorize:
        // Vectorizing non-consecutive loads with `llvm.masked.gather`.
        if (CurrentOrder.empty()) {
          TE = newTreeEntry(VL, TreeEntry::PossibleStridedVectorize, Bundle, S,
                            UserTreeIdx, ReuseShuffleIndicies);
        } else {
          TE = newTreeEntry(VL, TreeEntry::PossibleStridedVectorize, Bundle, S,
                            UserTreeIdx, ReuseShuffleIndicies, CurrentOrder);
        }
        TE->setOperandsInOrder();
        buildTree_rec(PointerOps, Depth + 1, {TE, 0});
        LLVM_DEBUG(dbgs() << "SLP: added a vector of non-consecutive loads.\n");
        break;
      case TreeEntry::ScatterVectorize:
        // Vectorizing non-consecutive loads with `llvm.masked.gather`.
        TE = newTreeEntry(VL, TreeEntry::ScatterVectorize, Bundle, S,
                          UserTreeIdx, ReuseShuffleIndicies);
        TE->setOperandsInOrder();
        buildTree_rec(PointerOps, Depth + 1, {TE, 0});
        LLVM_DEBUG(dbgs() << "SLP: added a vector of non-consecutive loads.\n");
        break;
      case TreeEntry::NeedToGather:
        llvm_unreachable("Unexpected loads state.");
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
      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of compares.\n");

      ValueList Left, Right;
      if (cast<CmpInst>(VL0)->isCommutative()) {
        // Commutative predicate - collect + sort operands of the instructions
        // so that each side is more likely to have the same opcode.
        assert(P0 == CmpInst::getSwappedPredicate(P0) &&
               "Commutative Predicate mismatch");
        reorderInputsAccordingToOpcode(VL, Left, Right, *TLI, *DL, *SE, *this);
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
        reorderInputsAccordingToOpcode(VL, Left, Right, *TLI, *DL, *SE, *this);
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
      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      LLVM_DEBUG(dbgs() << "SLP: added a vector of GEPs.\n");
      SmallVector<ValueList, 2> Operands(2);
      // Prepare the operand vector for pointer operands.
      for (Value *V : VL) {
        auto *GEP = dyn_cast<GetElementPtrInst>(V);
        if (!GEP) {
          Operands.front().push_back(V);
          continue;
        }
        Operands.front().push_back(GEP->getPointerOperand());
      }
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
                          auto *GEP = dyn_cast<GetElementPtrInst>(V);
                          if (!GEP)
                            return true;
                          return VL0Ty == GEP->getOperand(IndexIdx)->getType();
                        })
                     ? VL0Ty
                     : DL->getIndexType(cast<GetElementPtrInst>(VL0)
                                            ->getPointerOperandType()
                                            ->getScalarType());
      // Prepare the operand vector.
      for (Value *V : VL) {
        auto *I = dyn_cast<GetElementPtrInst>(V);
        if (!I) {
          Operands.back().push_back(
              ConstantInt::get(Ty, 0, /*isSigned=*/false));
          continue;
        }
        auto *Op = I->getOperand(IndexIdx);
        auto *CI = dyn_cast<ConstantInt>(Op);
        if (!CI)
          Operands.back().push_back(Op);
        else
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
      ValueList Operands(VL.size());
      auto *OIter = Operands.begin();
      for (Value *V : VL) {
        auto *SI = cast<StoreInst>(V);
        *OIter = SI->getValueOperand();
        ++OIter;
      }
      // Check that the sorted pointer operands are consecutive.
      if (CurrentOrder.empty()) {
        // Original stores are consecutive and does not require reordering.
        TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                     ReuseShuffleIndicies);
        TE->setOperandsInOrder();
        buildTree_rec(Operands, Depth + 1, {TE, 0});
        LLVM_DEBUG(dbgs() << "SLP: added a vector of stores.\n");
      } else {
        fixupOrderingIndices(CurrentOrder);
        TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                     ReuseShuffleIndicies, CurrentOrder);
        TE->setOperandsInOrder();
        buildTree_rec(Operands, Depth + 1, {TE, 0});
        LLVM_DEBUG(dbgs() << "SLP: added a vector of jumbled stores.\n");
      }
      return;
    }
    case Instruction::Call: {
      // Check if the calls are all to the same vectorizable intrinsic or
      // library function.
      CallInst *CI = cast<CallInst>(VL0);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                                   ReuseShuffleIndicies);
      TE->setOperandsInOrder();
      for (unsigned i = 0, e = CI->arg_size(); i != e; ++i) {
        // For scalar operands no need to create an entry since no need to
        // vectorize it.
        if (isVectorIntrinsicWithScalarOpAtArg(ID, i))
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
          reorderInputsAccordingToOpcode(VL, Left, Right, *TLI, *DL, *SE,
                                         *this);
        } else {
          auto *MainCI = cast<CmpInst>(S.MainOp);
          auto *AltCI = cast<CmpInst>(S.AltOp);
          CmpInst::Predicate MainP = MainCI->getPredicate();
          CmpInst::Predicate AltP = AltCI->getPredicate();
          assert(MainP != AltP &&
                 "Expected different main/alternate predicates.");
          // Collect operands - commute if it uses the swapped predicate or
          // alternate operation.
          for (Value *V : VL) {
            auto *Cmp = cast<CmpInst>(V);
            Value *LHS = Cmp->getOperand(0);
            Value *RHS = Cmp->getOperand(1);

            if (isAlternateInstruction(Cmp, MainCI, AltCI, *TLI)) {
              if (AltP == CmpInst::getSwappedPredicate(Cmp->getPredicate()))
                std::swap(LHS, RHS);
            } else {
              if (MainP == CmpInst::getSwappedPredicate(Cmp->getPredicate()))
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
      break;
  }
  llvm_unreachable("Unexpected vectorization of the instructions.");
}

unsigned BoUpSLP::canMapToVector(Type *T, const DataLayout &DL) const {
  unsigned N = 1;
  Type *EltTy = T;

  while (isa<StructType, ArrayType, FixedVectorType>(EltTy)) {
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
  if (VTSize < MinVecRegSize || VTSize > MaxVecRegSize ||
      VTSize != DL.getTypeStoreSizeInBits(T))
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
    std::optional<unsigned> Idx = getExtractIndex(Inst);
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

bool BoUpSLP::areAllUsersVectorized(
    Instruction *I, const SmallDenseSet<Value *> *VectorizedVals) const {
  return (I->hasOneUse() && (!VectorizedVals || VectorizedVals->contains(I))) ||
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

void BoUpSLP::TreeEntry::buildAltOpShuffleMask(
    const function_ref<bool(Instruction *)> IsAltOp, SmallVectorImpl<int> &Mask,
    SmallVectorImpl<Value *> *OpScalars,
    SmallVectorImpl<Value *> *AltScalars) const {
  unsigned Sz = Scalars.size();
  Mask.assign(Sz, PoisonMaskElem);
  SmallVector<int> OrderMask;
  if (!ReorderIndices.empty())
    inversePermutation(ReorderIndices, OrderMask);
  for (unsigned I = 0; I < Sz; ++I) {
    unsigned Idx = I;
    if (!ReorderIndices.empty())
      Idx = OrderMask[I];
    auto *OpInst = cast<Instruction>(Scalars[Idx]);
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
  if (!ReuseShuffleIndices.empty()) {
    SmallVector<int> NewMask(ReuseShuffleIndices.size(), PoisonMaskElem);
    transform(ReuseShuffleIndices, NewMask.begin(), [&Mask](int Idx) {
      return Idx != PoisonMaskElem ? Mask[Idx] : PoisonMaskElem;
    });
    Mask.swap(NewMask);
  }
}

static bool isAlternateInstruction(const Instruction *I,
                                   const Instruction *MainOp,
                                   const Instruction *AltOp,
                                   const TargetLibraryInfo &TLI) {
  if (auto *MainCI = dyn_cast<CmpInst>(MainOp)) {
    auto *AltCI = cast<CmpInst>(AltOp);
    CmpInst::Predicate MainP = MainCI->getPredicate();
    CmpInst::Predicate AltP = AltCI->getPredicate();
    assert(MainP != AltP && "Expected different main/alternate predicates.");
    auto *CI = cast<CmpInst>(I);
    if (isCmpSameOrSwapped(MainCI, CI, TLI))
      return false;
    if (isCmpSameOrSwapped(AltCI, CI, TLI))
      return true;
    CmpInst::Predicate P = CI->getPredicate();
    CmpInst::Predicate SwappedP = CmpInst::getSwappedPredicate(P);

    assert((MainP == P || AltP == P || MainP == SwappedP || AltP == SwappedP) &&
           "CmpInst expected to match either main or alternate predicate or "
           "their swap.");
    (void)AltP;
    return MainP != P && MainP != SwappedP;
  }
  return I->getOpcode() == AltOp->getOpcode();
}

TTI::OperandValueInfo BoUpSLP::getOperandInfo(const TreeEntry &E,
                                              unsigned OpIdx) {
  ArrayRef<Value*> VL = E.getOperand(OpIdx);
  assert(!VL.empty());
  const auto *Op0 = VL.front();

  const bool IsConstant = all_of(VL, [](Value *V) {
    // TODO: We should allow undef elements here
    return isConstant(V) && !isa<UndefValue>(V);
  });
  const bool IsUniform = all_of(VL, [=](Value *V) {
    // TODO: We should allow undef elements here
    return V == Op0;
  });
  const bool IsPowerOfTwo = all_of(VL, [](Value *V) {
    // TODO: We should allow undef elements here
    if (auto *CI = dyn_cast<ConstantInt>(V))
      return CI->getValue().isPowerOf2();
    return false;
  });
  const bool IsNegatedPowerOfTwo = all_of(VL, [](Value *V) {
    // TODO: We should allow undef elements here
    if (auto *CI = dyn_cast<ConstantInt>(V))
      return CI->getValue().isNegatedPowerOf2();
    return false;
  });

  TTI::OperandValueKind VK = TTI::OK_AnyValue;
  if (IsConstant && IsUniform)
    VK = TTI::OK_UniformConstantValue;
  else if (IsConstant)
    VK = TTI::OK_NonUniformConstantValue;
  else if (IsUniform)
    VK = TTI::OK_UniformValue;

  TTI::OperandValueProperties VP = TTI::OP_None;
  VP = IsPowerOfTwo ? TTI::OP_PowerOf2 : VP;
  VP = IsNegatedPowerOfTwo ? TTI::OP_NegatedPowerOf2 : VP;

  return {VK, VP};
}

namespace {
/// The base class for shuffle instruction emission and shuffle cost estimation.
class BaseShuffleAnalysis {
protected:
  /// Checks if the mask is an identity mask.
  /// \param IsStrict if is true the function returns false if mask size does
  /// not match vector size.
  static bool isIdentityMask(ArrayRef<int> Mask, const FixedVectorType *VecTy,
                             bool IsStrict) {
    int Limit = Mask.size();
    int VF = VecTy->getNumElements();
    return (VF == Limit || !IsStrict) &&
           all_of(Mask, [Limit](int Idx) { return Idx < Limit; }) &&
           ShuffleVectorInst::isIdentityMask(Mask);
  }

  /// Tries to combine 2 different masks into single one.
  /// \param LocalVF Vector length of the permuted input vector. \p Mask may
  /// change the size of the vector, \p LocalVF is the original size of the
  /// shuffled vector.
  static void combineMasks(unsigned LocalVF, SmallVectorImpl<int> &Mask,
                           ArrayRef<int> ExtMask) {
    unsigned VF = Mask.size();
    SmallVector<int> NewMask(ExtMask.size(), PoisonMaskElem);
    for (int I = 0, Sz = ExtMask.size(); I < Sz; ++I) {
      if (ExtMask[I] == PoisonMaskElem)
        continue;
      int MaskedIdx = Mask[ExtMask[I] % VF];
      NewMask[I] =
          MaskedIdx == PoisonMaskElem ? PoisonMaskElem : MaskedIdx % LocalVF;
    }
    Mask.swap(NewMask);
  }

  /// Looks through shuffles trying to reduce final number of shuffles in the
  /// code. The function looks through the previously emitted shuffle
  /// instructions and properly mark indices in mask as undef.
  /// For example, given the code
  /// \code
  /// %s1 = shufflevector <2 x ty> %0, poison, <1, 0>
  /// %s2 = shufflevector <2 x ty> %1, poison, <1, 0>
  /// \endcode
  /// and if need to emit shuffle of %s1 and %s2 with mask <1, 0, 3, 2>, it will
  /// look through %s1 and %s2 and select vectors %0 and %1 with mask
  /// <0, 1, 2, 3> for the shuffle.
  /// If 2 operands are of different size, the smallest one will be resized and
  /// the mask recalculated properly.
  /// For example, given the code
  /// \code
  /// %s1 = shufflevector <2 x ty> %0, poison, <1, 0, 1, 0>
  /// %s2 = shufflevector <2 x ty> %1, poison, <1, 0, 1, 0>
  /// \endcode
  /// and if need to emit shuffle of %s1 and %s2 with mask <1, 0, 5, 4>, it will
  /// look through %s1 and %s2 and select vectors %0 and %1 with mask
  /// <0, 1, 2, 3> for the shuffle.
  /// So, it tries to transform permutations to simple vector merge, if
  /// possible.
  /// \param V The input vector which must be shuffled using the given \p Mask.
  /// If the better candidate is found, \p V is set to this best candidate
  /// vector.
  /// \param Mask The input mask for the shuffle. If the best candidate is found
  /// during looking-through-shuffles attempt, it is updated accordingly.
  /// \param SinglePermute true if the shuffle operation is originally a
  /// single-value-permutation. In this case the look-through-shuffles procedure
  /// may look for resizing shuffles as the best candidates.
  /// \return true if the shuffle results in the non-resizing identity shuffle
  /// (and thus can be ignored), false - otherwise.
  static bool peekThroughShuffles(Value *&V, SmallVectorImpl<int> &Mask,
                                  bool SinglePermute) {
    Value *Op = V;
    ShuffleVectorInst *IdentityOp = nullptr;
    SmallVector<int> IdentityMask;
    while (auto *SV = dyn_cast<ShuffleVectorInst>(Op)) {
      // Exit if not a fixed vector type or changing size shuffle.
      auto *SVTy = dyn_cast<FixedVectorType>(SV->getType());
      if (!SVTy)
        break;
      // Remember the identity or broadcast mask, if it is not a resizing
      // shuffle. If no better candidates are found, this Op and Mask will be
      // used in the final shuffle.
      if (isIdentityMask(Mask, SVTy, /*IsStrict=*/false)) {
        if (!IdentityOp || !SinglePermute ||
            (isIdentityMask(Mask, SVTy, /*IsStrict=*/true) &&
             !ShuffleVectorInst::isZeroEltSplatMask(IdentityMask))) {
          IdentityOp = SV;
          // Store current mask in the IdentityMask so later we did not lost
          // this info if IdentityOp is selected as the best candidate for the
          // permutation.
          IdentityMask.assign(Mask);
        }
      }
      // Remember the broadcast mask. If no better candidates are found, this Op
      // and Mask will be used in the final shuffle.
      // Zero splat can be used as identity too, since it might be used with
      // mask <0, 1, 2, ...>, i.e. identity mask without extra reshuffling.
      // E.g. if need to shuffle the vector with the mask <3, 1, 2, 0>, which is
      // expensive, the analysis founds out, that the source vector is just a
      // broadcast, this original mask can be transformed to identity mask <0,
      // 1, 2, 3>.
      // \code
      // %0 = shuffle %v, poison, zeroinitalizer
      // %res = shuffle %0, poison, <3, 1, 2, 0>
      // \endcode
      // may be transformed to
      // \code
      // %0 = shuffle %v, poison, zeroinitalizer
      // %res = shuffle %0, poison, <0, 1, 2, 3>
      // \endcode
      if (SV->isZeroEltSplat()) {
        IdentityOp = SV;
        IdentityMask.assign(Mask);
      }
      int LocalVF = Mask.size();
      if (auto *SVOpTy =
              dyn_cast<FixedVectorType>(SV->getOperand(0)->getType()))
        LocalVF = SVOpTy->getNumElements();
      SmallVector<int> ExtMask(Mask.size(), PoisonMaskElem);
      for (auto [Idx, I] : enumerate(Mask)) {
        if (I == PoisonMaskElem ||
            static_cast<unsigned>(I) >= SV->getShuffleMask().size())
          continue;
        ExtMask[Idx] = SV->getMaskValue(I);
      }
      bool IsOp1Undef =
          isUndefVector(SV->getOperand(0),
                        buildUseMask(LocalVF, ExtMask, UseMask::FirstArg))
              .all();
      bool IsOp2Undef =
          isUndefVector(SV->getOperand(1),
                        buildUseMask(LocalVF, ExtMask, UseMask::SecondArg))
              .all();
      if (!IsOp1Undef && !IsOp2Undef) {
        // Update mask and mark undef elems.
        for (int &I : Mask) {
          if (I == PoisonMaskElem)
            continue;
          if (SV->getMaskValue(I % SV->getShuffleMask().size()) ==
              PoisonMaskElem)
            I = PoisonMaskElem;
        }
        break;
      }
      SmallVector<int> ShuffleMask(SV->getShuffleMask().begin(),
                                   SV->getShuffleMask().end());
      combineMasks(LocalVF, ShuffleMask, Mask);
      Mask.swap(ShuffleMask);
      if (IsOp2Undef)
        Op = SV->getOperand(0);
      else
        Op = SV->getOperand(1);
    }
    if (auto *OpTy = dyn_cast<FixedVectorType>(Op->getType());
        !OpTy || !isIdentityMask(Mask, OpTy, SinglePermute) ||
        ShuffleVectorInst::isZeroEltSplatMask(Mask)) {
      if (IdentityOp) {
        V = IdentityOp;
        assert(Mask.size() == IdentityMask.size() &&
               "Expected masks of same sizes.");
        // Clear known poison elements.
        for (auto [I, Idx] : enumerate(Mask))
          if (Idx == PoisonMaskElem)
            IdentityMask[I] = PoisonMaskElem;
        Mask.swap(IdentityMask);
        auto *Shuffle = dyn_cast<ShuffleVectorInst>(V);
        return SinglePermute &&
               (isIdentityMask(Mask, cast<FixedVectorType>(V->getType()),
                               /*IsStrict=*/true) ||
                (Shuffle && Mask.size() == Shuffle->getShuffleMask().size() &&
                 Shuffle->isZeroEltSplat() &&
                 ShuffleVectorInst::isZeroEltSplatMask(Mask)));
      }
      V = Op;
      return false;
    }
    V = Op;
    return true;
  }

  /// Smart shuffle instruction emission, walks through shuffles trees and
  /// tries to find the best matching vector for the actual shuffle
  /// instruction.
  template <typename T, typename ShuffleBuilderTy>
  static T createShuffle(Value *V1, Value *V2, ArrayRef<int> Mask,
                         ShuffleBuilderTy &Builder) {
    assert(V1 && "Expected at least one vector value.");
    if (V2)
      Builder.resizeToMatch(V1, V2);
    int VF = Mask.size();
    if (auto *FTy = dyn_cast<FixedVectorType>(V1->getType()))
      VF = FTy->getNumElements();
    if (V2 &&
        !isUndefVector(V2, buildUseMask(VF, Mask, UseMask::SecondArg)).all()) {
      // Peek through shuffles.
      Value *Op1 = V1;
      Value *Op2 = V2;
      int VF =
          cast<VectorType>(V1->getType())->getElementCount().getKnownMinValue();
      SmallVector<int> CombinedMask1(Mask.size(), PoisonMaskElem);
      SmallVector<int> CombinedMask2(Mask.size(), PoisonMaskElem);
      for (int I = 0, E = Mask.size(); I < E; ++I) {
        if (Mask[I] < VF)
          CombinedMask1[I] = Mask[I];
        else
          CombinedMask2[I] = Mask[I] - VF;
      }
      Value *PrevOp1;
      Value *PrevOp2;
      do {
        PrevOp1 = Op1;
        PrevOp2 = Op2;
        (void)peekThroughShuffles(Op1, CombinedMask1, /*SinglePermute=*/false);
        (void)peekThroughShuffles(Op2, CombinedMask2, /*SinglePermute=*/false);
        // Check if we have 2 resizing shuffles - need to peek through operands
        // again.
        if (auto *SV1 = dyn_cast<ShuffleVectorInst>(Op1))
          if (auto *SV2 = dyn_cast<ShuffleVectorInst>(Op2)) {
            SmallVector<int> ExtMask1(Mask.size(), PoisonMaskElem);
            for (auto [Idx, I] : enumerate(CombinedMask1)) {
                if (I == PoisonMaskElem)
                continue;
                ExtMask1[Idx] = SV1->getMaskValue(I);
            }
            SmallBitVector UseMask1 = buildUseMask(
                cast<FixedVectorType>(SV1->getOperand(1)->getType())
                    ->getNumElements(),
                ExtMask1, UseMask::SecondArg);
            SmallVector<int> ExtMask2(CombinedMask2.size(), PoisonMaskElem);
            for (auto [Idx, I] : enumerate(CombinedMask2)) {
                if (I == PoisonMaskElem)
                continue;
                ExtMask2[Idx] = SV2->getMaskValue(I);
            }
            SmallBitVector UseMask2 = buildUseMask(
                cast<FixedVectorType>(SV2->getOperand(1)->getType())
                    ->getNumElements(),
                ExtMask2, UseMask::SecondArg);
            if (SV1->getOperand(0)->getType() ==
                    SV2->getOperand(0)->getType() &&
                SV1->getOperand(0)->getType() != SV1->getType() &&
                isUndefVector(SV1->getOperand(1), UseMask1).all() &&
                isUndefVector(SV2->getOperand(1), UseMask2).all()) {
              Op1 = SV1->getOperand(0);
              Op2 = SV2->getOperand(0);
              SmallVector<int> ShuffleMask1(SV1->getShuffleMask().begin(),
                                            SV1->getShuffleMask().end());
              int LocalVF = ShuffleMask1.size();
              if (auto *FTy = dyn_cast<FixedVectorType>(Op1->getType()))
                LocalVF = FTy->getNumElements();
              combineMasks(LocalVF, ShuffleMask1, CombinedMask1);
              CombinedMask1.swap(ShuffleMask1);
              SmallVector<int> ShuffleMask2(SV2->getShuffleMask().begin(),
                                            SV2->getShuffleMask().end());
              LocalVF = ShuffleMask2.size();
              if (auto *FTy = dyn_cast<FixedVectorType>(Op2->getType()))
                LocalVF = FTy->getNumElements();
              combineMasks(LocalVF, ShuffleMask2, CombinedMask2);
              CombinedMask2.swap(ShuffleMask2);
            }
          }
      } while (PrevOp1 != Op1 || PrevOp2 != Op2);
      Builder.resizeToMatch(Op1, Op2);
      VF = std::max(cast<VectorType>(Op1->getType())
                        ->getElementCount()
                        .getKnownMinValue(),
                    cast<VectorType>(Op2->getType())
                        ->getElementCount()
                        .getKnownMinValue());
      for (int I = 0, E = Mask.size(); I < E; ++I) {
        if (CombinedMask2[I] != PoisonMaskElem) {
          assert(CombinedMask1[I] == PoisonMaskElem &&
                 "Expected undefined mask element");
          CombinedMask1[I] = CombinedMask2[I] + (Op1 == Op2 ? 0 : VF);
        }
      }
      const int Limit = CombinedMask1.size() * 2;
      if (Op1 == Op2 && Limit == 2 * VF &&
          all_of(CombinedMask1, [=](int Idx) { return Idx < Limit; }) &&
          (ShuffleVectorInst::isIdentityMask(CombinedMask1) ||
           (ShuffleVectorInst::isZeroEltSplatMask(CombinedMask1) &&
            isa<ShuffleVectorInst>(Op1) &&
            cast<ShuffleVectorInst>(Op1)->getShuffleMask() ==
                ArrayRef(CombinedMask1))))
        return Builder.createIdentity(Op1);
      return Builder.createShuffleVector(
          Op1, Op1 == Op2 ? PoisonValue::get(Op1->getType()) : Op2,
          CombinedMask1);
    }
    if (isa<PoisonValue>(V1))
      return Builder.createPoison(
          cast<VectorType>(V1->getType())->getElementType(), Mask.size());
    SmallVector<int> NewMask(Mask.begin(), Mask.end());
    bool IsIdentity = peekThroughShuffles(V1, NewMask, /*SinglePermute=*/true);
    assert(V1 && "Expected non-null value after looking through shuffles.");

    if (!IsIdentity)
      return Builder.createShuffleVector(V1, NewMask);
    return Builder.createIdentity(V1);
  }
};
} // namespace

/// Merges shuffle masks and emits final shuffle instruction, if required. It
/// supports shuffling of 2 input vectors. It implements lazy shuffles emission,
/// when the actual shuffle instruction is generated only if this is actually
/// required. Otherwise, the shuffle instruction emission is delayed till the
/// end of the process, to reduce the number of emitted instructions and further
/// analysis/transformations.
class BoUpSLP::ShuffleCostEstimator : public BaseShuffleAnalysis {
  bool IsFinalized = false;
  SmallVector<int> CommonMask;
  SmallVector<PointerUnion<Value *, const TreeEntry *>, 2> InVectors;
  const TargetTransformInfo &TTI;
  InstructionCost Cost = 0;
  SmallDenseSet<Value *> VectorizedVals;
  BoUpSLP &R;
  SmallPtrSetImpl<Value *> &CheckedExtracts;
  constexpr static TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  InstructionCost getBuildVectorCost(ArrayRef<Value *> VL, Value *Root) {
    if ((!Root && allConstant(VL)) || all_of(VL, UndefValue::classof))
      return TTI::TCC_Free;
    auto *VecTy = FixedVectorType::get(VL.front()->getType(), VL.size());
    InstructionCost GatherCost = 0;
    SmallVector<Value *> Gathers(VL.begin(), VL.end());
    // Improve gather cost for gather of loads, if we can group some of the
    // loads into vector loads.
    InstructionsState S = getSameOpcode(VL, *R.TLI);
    if (VL.size() > 2 && S.getOpcode() == Instruction::Load &&
        !S.isAltShuffle() &&
        !all_of(Gathers, [&](Value *V) { return R.getTreeEntry(V); }) &&
        !isSplat(Gathers)) {
      BoUpSLP::ValueSet VectorizedLoads;
      unsigned StartIdx = 0;
      unsigned VF = VL.size() / 2;
      unsigned VectorizedCnt = 0;
      unsigned ScatterVectorizeCnt = 0;
      const unsigned Sz = R.DL->getTypeSizeInBits(S.MainOp->getType());
      for (unsigned MinVF = R.getMinVF(2 * Sz); VF >= MinVF; VF /= 2) {
        for (unsigned Cnt = StartIdx, End = VL.size(); Cnt + VF <= End;
             Cnt += VF) {
          ArrayRef<Value *> Slice = VL.slice(Cnt, VF);
          if (!VectorizedLoads.count(Slice.front()) &&
              !VectorizedLoads.count(Slice.back()) && allSameBlock(Slice)) {
            SmallVector<Value *> PointerOps;
            OrdersType CurrentOrder;
            LoadsState LS =
                canVectorizeLoads(Slice, Slice.front(), TTI, *R.DL, *R.SE,
                                  *R.LI, *R.TLI, CurrentOrder, PointerOps);
            switch (LS) {
            case LoadsState::Vectorize:
            case LoadsState::ScatterVectorize:
            case LoadsState::PossibleStridedVectorize:
              // Mark the vectorized loads so that we don't vectorize them
              // again.
              if (LS == LoadsState::Vectorize)
                ++VectorizedCnt;
              else
                ++ScatterVectorizeCnt;
              VectorizedLoads.insert(Slice.begin(), Slice.end());
              // If we vectorized initial block, no need to try to vectorize
              // it again.
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
        unsigned NumParts = TTI.getNumberOfParts(VecTy);
        bool NeedInsertSubvectorAnalysis =
            !NumParts || (VL.size() / VF) > NumParts;
        // Get the cost for gathered loads.
        for (unsigned I = 0, End = VL.size(); I < End; I += VF) {
          if (VectorizedLoads.contains(VL[I]))
            continue;
          GatherCost += getBuildVectorCost(VL.slice(I, VF), Root);
        }
        // Exclude potentially vectorized loads from list of gathered
        // scalars.
        auto *LI = cast<LoadInst>(S.MainOp);
        Gathers.assign(Gathers.size(), PoisonValue::get(LI->getType()));
        // The cost for vectorized loads.
        InstructionCost ScalarsCost = 0;
        for (Value *V : VectorizedLoads) {
          auto *LI = cast<LoadInst>(V);
          ScalarsCost +=
              TTI.getMemoryOpCost(Instruction::Load, LI->getType(),
                                  LI->getAlign(), LI->getPointerAddressSpace(),
                                  CostKind, TTI::OperandValueInfo(), LI);
        }
        auto *LoadTy = FixedVectorType::get(LI->getType(), VF);
        Align Alignment = LI->getAlign();
        GatherCost +=
            VectorizedCnt *
            TTI.getMemoryOpCost(Instruction::Load, LoadTy, Alignment,
                                LI->getPointerAddressSpace(), CostKind,
                                TTI::OperandValueInfo(), LI);
        GatherCost += ScatterVectorizeCnt *
                      TTI.getGatherScatterOpCost(
                          Instruction::Load, LoadTy, LI->getPointerOperand(),
                          /*VariableMask=*/false, Alignment, CostKind, LI);
        if (NeedInsertSubvectorAnalysis) {
          // Add the cost for the subvectors insert.
          for (int I = VF, E = VL.size(); I < E; I += VF)
            GatherCost += TTI.getShuffleCost(TTI::SK_InsertSubvector, VecTy,
                                             std::nullopt, CostKind, I, LoadTy);
        }
        GatherCost -= ScalarsCost;
      }
    } else if (!Root && isSplat(VL)) {
      // Found the broadcasting of the single scalar, calculate the cost as
      // the broadcast.
      const auto *It =
          find_if(VL, [](Value *V) { return !isa<UndefValue>(V); });
      assert(It != VL.end() && "Expected at least one non-undef value.");
      // Add broadcast for non-identity shuffle only.
      bool NeedShuffle =
          count(VL, *It) > 1 &&
          (VL.front() != *It || !all_of(VL.drop_front(), UndefValue::classof));
      InstructionCost InsertCost = TTI.getVectorInstrCost(
          Instruction::InsertElement, VecTy, CostKind,
          NeedShuffle ? 0 : std::distance(VL.begin(), It),
          PoisonValue::get(VecTy), *It);
      return InsertCost +
             (NeedShuffle ? TTI.getShuffleCost(
                                TargetTransformInfo::SK_Broadcast, VecTy,
                                /*Mask=*/std::nullopt, CostKind, /*Index=*/0,
                                /*SubTp=*/nullptr, /*Args=*/*It)
                          : TTI::TCC_Free);
    }
    return GatherCost +
           (all_of(Gathers, UndefValue::classof)
                ? TTI::TCC_Free
                : R.getGatherCost(Gathers, !Root && VL.equals(Gathers)));
  };

  /// Compute the cost of creating a vector of type \p VecTy containing the
  /// extracted values from \p VL.
  InstructionCost computeExtractCost(ArrayRef<Value *> VL, ArrayRef<int> Mask,
                                     TTI::ShuffleKind ShuffleKind) {
    auto *VecTy = FixedVectorType::get(VL.front()->getType(), VL.size());
    unsigned NumOfParts = TTI.getNumberOfParts(VecTy);

    if (ShuffleKind != TargetTransformInfo::SK_PermuteSingleSrc ||
        !NumOfParts || VecTy->getNumElements() < NumOfParts)
      return TTI.getShuffleCost(ShuffleKind, VecTy, Mask);

    bool AllConsecutive = true;
    unsigned EltsPerVector = VecTy->getNumElements() / NumOfParts;
    unsigned Idx = -1;
    InstructionCost Cost = 0;

    // Process extracts in blocks of EltsPerVector to check if the source vector
    // operand can be re-used directly. If not, add the cost of creating a
    // shuffle to extract the values into a vector register.
    SmallVector<int> RegMask(EltsPerVector, PoisonMaskElem);
    for (auto *V : VL) {
      ++Idx;

      // Reached the start of a new vector registers.
      if (Idx % EltsPerVector == 0) {
        RegMask.assign(EltsPerVector, PoisonMaskElem);
        AllConsecutive = true;
        continue;
      }

      // Need to exclude undefs from analysis.
      if (isa<UndefValue>(V) || Mask[Idx] == PoisonMaskElem)
        continue;

      // Check all extracts for a vector register on the target directly
      // extract values in order.
      unsigned CurrentIdx = *getExtractIndex(cast<Instruction>(V));
      if (!isa<UndefValue>(VL[Idx - 1]) && Mask[Idx - 1] != PoisonMaskElem) {
        unsigned PrevIdx = *getExtractIndex(cast<Instruction>(VL[Idx - 1]));
        AllConsecutive &= PrevIdx + 1 == CurrentIdx &&
                          CurrentIdx % EltsPerVector == Idx % EltsPerVector;
        RegMask[Idx % EltsPerVector] = CurrentIdx % EltsPerVector;
      }

      if (AllConsecutive)
        continue;

      // Skip all indices, except for the last index per vector block.
      if ((Idx + 1) % EltsPerVector != 0 && Idx + 1 != VL.size())
        continue;

      // If we have a series of extracts which are not consecutive and hence
      // cannot re-use the source vector register directly, compute the shuffle
      // cost to extract the vector with EltsPerVector elements.
      Cost += TTI.getShuffleCost(
          TargetTransformInfo::SK_PermuteSingleSrc,
          FixedVectorType::get(VecTy->getElementType(), EltsPerVector),
          RegMask);
    }
    return Cost;
  }

  class ShuffleCostBuilder {
    const TargetTransformInfo &TTI;

    static bool isEmptyOrIdentity(ArrayRef<int> Mask, unsigned VF) {
      int Limit = 2 * VF;
      return Mask.empty() ||
             (VF == Mask.size() &&
              all_of(Mask, [Limit](int Idx) { return Idx < Limit; }) &&
              ShuffleVectorInst::isIdentityMask(Mask));
    }

  public:
    ShuffleCostBuilder(const TargetTransformInfo &TTI) : TTI(TTI) {}
    ~ShuffleCostBuilder() = default;
    InstructionCost createShuffleVector(Value *V1, Value *,
                                        ArrayRef<int> Mask) const {
      // Empty mask or identity mask are free.
      unsigned VF =
          cast<VectorType>(V1->getType())->getElementCount().getKnownMinValue();
      if (isEmptyOrIdentity(Mask, VF))
        return TTI::TCC_Free;
      return TTI.getShuffleCost(
          TTI::SK_PermuteTwoSrc,
          FixedVectorType::get(
              cast<VectorType>(V1->getType())->getElementType(), Mask.size()),
          Mask);
    }
    InstructionCost createShuffleVector(Value *V1, ArrayRef<int> Mask) const {
      // Empty mask or identity mask are free.
      if (isEmptyOrIdentity(Mask, Mask.size()))
        return TTI::TCC_Free;
      return TTI.getShuffleCost(
          TTI::SK_PermuteSingleSrc,
          FixedVectorType::get(
              cast<VectorType>(V1->getType())->getElementType(), Mask.size()),
          Mask);
    }
    InstructionCost createIdentity(Value *) const { return TTI::TCC_Free; }
    InstructionCost createPoison(Type *Ty, unsigned VF) const {
      return TTI::TCC_Free;
    }
    void resizeToMatch(Value *&, Value *&) const {}
  };

  /// Smart shuffle instruction emission, walks through shuffles trees and
  /// tries to find the best matching vector for the actual shuffle
  /// instruction.
  InstructionCost
  createShuffle(const PointerUnion<Value *, const TreeEntry *> &P1,
                const PointerUnion<Value *, const TreeEntry *> &P2,
                ArrayRef<int> Mask) {
    ShuffleCostBuilder Builder(TTI);
    Value *V1 = P1.dyn_cast<Value *>(), *V2 = P2.dyn_cast<Value *>();
    unsigned CommonVF = 0;
    if (!V1) {
      const TreeEntry *E = P1.get<const TreeEntry *>();
      unsigned VF = E->getVectorFactor();
      if (V2) {
        unsigned V2VF = cast<FixedVectorType>(V2->getType())->getNumElements();
        if (V2VF != VF && V2VF == E->Scalars.size())
          VF = E->Scalars.size();
      } else if (!P2.isNull()) {
        const TreeEntry *E2 = P2.get<const TreeEntry *>();
        if (E->Scalars.size() == E2->Scalars.size())
          CommonVF = VF = E->Scalars.size();
      } else {
        // P2 is empty, check that we have same node + reshuffle (if any).
        if (E->Scalars.size() == Mask.size() && VF != Mask.size()) {
          VF = E->Scalars.size();
          SmallVector<int> CommonMask(Mask.begin(), Mask.end());
          ::addMask(CommonMask, E->getCommonMask());
          V1 = Constant::getNullValue(
              FixedVectorType::get(E->Scalars.front()->getType(), VF));
          return BaseShuffleAnalysis::createShuffle<InstructionCost>(
              V1, nullptr, CommonMask, Builder);
        }
      }
      V1 = Constant::getNullValue(
          FixedVectorType::get(E->Scalars.front()->getType(), VF));
    }
    if (!V2 && !P2.isNull()) {
      const TreeEntry *E = P2.get<const TreeEntry *>();
      unsigned VF = E->getVectorFactor();
      unsigned V1VF = cast<FixedVectorType>(V1->getType())->getNumElements();
      if (!CommonVF && V1VF == E->Scalars.size())
        CommonVF = E->Scalars.size();
      if (CommonVF)
        VF = CommonVF;
      V2 = Constant::getNullValue(
          FixedVectorType::get(E->Scalars.front()->getType(), VF));
    }
    return BaseShuffleAnalysis::createShuffle<InstructionCost>(V1, V2, Mask,
                                                               Builder);
  }

public:
  ShuffleCostEstimator(TargetTransformInfo &TTI,
                       ArrayRef<Value *> VectorizedVals, BoUpSLP &R,
                       SmallPtrSetImpl<Value *> &CheckedExtracts)
      : TTI(TTI), VectorizedVals(VectorizedVals.begin(), VectorizedVals.end()),
        R(R), CheckedExtracts(CheckedExtracts) {}
  Value *adjustExtracts(const TreeEntry *E, ArrayRef<int> Mask,
                        TTI::ShuffleKind ShuffleKind) {
    if (Mask.empty())
      return nullptr;
    Value *VecBase = nullptr;
    ArrayRef<Value *> VL = E->Scalars;
    auto *VecTy = FixedVectorType::get(VL.front()->getType(), VL.size());
    // If the resulting type is scalarized, do not adjust the cost.
    unsigned VecNumParts = TTI.getNumberOfParts(VecTy);
    if (VecNumParts == VecTy->getNumElements())
      return nullptr;
    DenseMap<Value *, int> ExtractVectorsTys;
    for (auto [I, V] : enumerate(VL)) {
      // Ignore non-extractelement scalars.
      if (isa<UndefValue>(V) || (!Mask.empty() && Mask[I] == PoisonMaskElem))
        continue;
      // If all users of instruction are going to be vectorized and this
      // instruction itself is not going to be vectorized, consider this
      // instruction as dead and remove its cost from the final cost of the
      // vectorized tree.
      // Also, avoid adjusting the cost for extractelements with multiple uses
      // in different graph entries.
      const TreeEntry *VE = R.getTreeEntry(V);
      if (!CheckedExtracts.insert(V).second ||
          !R.areAllUsersVectorized(cast<Instruction>(V), &VectorizedVals) ||
          (VE && VE != E))
        continue;
      auto *EE = cast<ExtractElementInst>(V);
      VecBase = EE->getVectorOperand();
      std::optional<unsigned> EEIdx = getExtractIndex(EE);
      if (!EEIdx)
        continue;
      unsigned Idx = *EEIdx;
      if (VecNumParts != TTI.getNumberOfParts(EE->getVectorOperandType())) {
        auto It =
            ExtractVectorsTys.try_emplace(EE->getVectorOperand(), Idx).first;
        It->getSecond() = std::min<int>(It->second, Idx);
      }
      // Take credit for instruction that will become dead.
      if (EE->hasOneUse()) {
        Instruction *Ext = EE->user_back();
        if (isa<SExtInst, ZExtInst>(Ext) && all_of(Ext->users(), [](User *U) {
              return isa<GetElementPtrInst>(U);
            })) {
          // Use getExtractWithExtendCost() to calculate the cost of
          // extractelement/ext pair.
          Cost -= TTI.getExtractWithExtendCost(Ext->getOpcode(), Ext->getType(),
                                               EE->getVectorOperandType(), Idx);
          // Add back the cost of s|zext which is subtracted separately.
          Cost += TTI.getCastInstrCost(
              Ext->getOpcode(), Ext->getType(), EE->getType(),
              TTI::getCastContextHint(Ext), CostKind, Ext);
          continue;
        }
      }
      Cost -= TTI.getVectorInstrCost(*EE, EE->getVectorOperandType(), CostKind,
                                     Idx);
    }
    // Add a cost for subvector extracts/inserts if required.
    for (const auto &Data : ExtractVectorsTys) {
      auto *EEVTy = cast<FixedVectorType>(Data.first->getType());
      unsigned NumElts = VecTy->getNumElements();
      if (Data.second % NumElts == 0)
        continue;
      if (TTI.getNumberOfParts(EEVTy) > VecNumParts) {
        unsigned Idx = (Data.second / NumElts) * NumElts;
        unsigned EENumElts = EEVTy->getNumElements();
        if (Idx % NumElts == 0)
          continue;
        if (Idx + NumElts <= EENumElts) {
          Cost += TTI.getShuffleCost(TargetTransformInfo::SK_ExtractSubvector,
                                     EEVTy, std::nullopt, CostKind, Idx, VecTy);
        } else {
          // Need to round up the subvector type vectorization factor to avoid a
          // crash in cost model functions. Make SubVT so that Idx + VF of SubVT
          // <= EENumElts.
          auto *SubVT =
              FixedVectorType::get(VecTy->getElementType(), EENumElts - Idx);
          Cost += TTI.getShuffleCost(TargetTransformInfo::SK_ExtractSubvector,
                                     EEVTy, std::nullopt, CostKind, Idx, SubVT);
        }
      } else {
        Cost += TTI.getShuffleCost(TargetTransformInfo::SK_InsertSubvector,
                                   VecTy, std::nullopt, CostKind, 0, EEVTy);
      }
    }
    // Check that gather of extractelements can be represented as just a
    // shuffle of a single/two vectors the scalars are extracted from.
    // Found the bunch of extractelement instructions that must be gathered
    // into a vector and can be represented as a permutation elements in a
    // single input vector or of 2 input vectors.
    Cost += computeExtractCost(VL, Mask, ShuffleKind);
    return VecBase;
  }
  void add(const TreeEntry *E1, const TreeEntry *E2, ArrayRef<int> Mask) {
    CommonMask.assign(Mask.begin(), Mask.end());
    InVectors.assign({E1, E2});
  }
  void add(const TreeEntry *E1, ArrayRef<int> Mask) {
    CommonMask.assign(Mask.begin(), Mask.end());
    InVectors.assign(1, E1);
  }
  /// Adds another one input vector and the mask for the shuffling.
  void add(Value *V1, ArrayRef<int> Mask) {
    assert(CommonMask.empty() && InVectors.empty() &&
           "Expected empty input mask/vectors.");
    CommonMask.assign(Mask.begin(), Mask.end());
    InVectors.assign(1, V1);
  }
  Value *gather(ArrayRef<Value *> VL, Value *Root = nullptr) {
    Cost += getBuildVectorCost(VL, Root);
    if (!Root) {
      assert(InVectors.empty() && "Unexpected input vectors for buildvector.");
      // FIXME: Need to find a way to avoid use of getNullValue here.
      SmallVector<Constant *> Vals;
      for (Value *V : VL) {
        if (isa<UndefValue>(V)) {
          Vals.push_back(cast<Constant>(V));
          continue;
        }
        Vals.push_back(Constant::getNullValue(V->getType()));
      }
      return ConstantVector::get(Vals);
    }
    return ConstantVector::getSplat(
        ElementCount::getFixed(VL.size()),
        Constant::getNullValue(VL.front()->getType()));
  }
  /// Finalize emission of the shuffles.
  InstructionCost
  finalize(ArrayRef<int> ExtMask, unsigned VF = 0,
           function_ref<void(Value *&, SmallVectorImpl<int> &)> Action = {}) {
    IsFinalized = true;
    if (Action) {
      const PointerUnion<Value *, const TreeEntry *> &Vec = InVectors.front();
      if (InVectors.size() == 2) {
        Cost += createShuffle(Vec, InVectors.back(), CommonMask);
        InVectors.pop_back();
      } else {
        Cost += createShuffle(Vec, nullptr, CommonMask);
      }
      for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
        if (CommonMask[Idx] != PoisonMaskElem)
          CommonMask[Idx] = Idx;
      assert(VF > 0 &&
             "Expected vector length for the final value before action.");
      Value *V = Vec.dyn_cast<Value *>();
      if (!Vec.isNull() && !V)
        V = Constant::getNullValue(FixedVectorType::get(
            Vec.get<const TreeEntry *>()->Scalars.front()->getType(),
            CommonMask.size()));
      Action(V, CommonMask);
    }
    ::addMask(CommonMask, ExtMask, /*ExtendingManyInputs=*/true);
    if (CommonMask.empty())
      return Cost;
    int Limit = CommonMask.size() * 2;
    if (all_of(CommonMask, [=](int Idx) { return Idx < Limit; }) &&
        ShuffleVectorInst::isIdentityMask(CommonMask))
      return Cost;
    return Cost +
           createShuffle(InVectors.front(),
                         InVectors.size() == 2 ? InVectors.back() : nullptr,
                         CommonMask);
  }

  ~ShuffleCostEstimator() {
    assert((IsFinalized || CommonMask.empty()) &&
           "Shuffle construction must be finalized.");
  }
};

InstructionCost
BoUpSLP::getEntryCost(const TreeEntry *E, ArrayRef<Value *> VectorizedVals,
                      SmallPtrSetImpl<Value *> &CheckedExtracts) {
  ArrayRef<Value *> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (auto *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  else if (auto *CI = dyn_cast<CmpInst>(VL[0]))
    ScalarTy = CI->getOperand(0)->getType();
  else if (auto *IE = dyn_cast<InsertElementInst>(VL[0]))
    ScalarTy = IE->getOperand(1)->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  // If we have computed a smaller type for the expression, update VecTy so
  // that the costs will be accurate.
  auto It = MinBWs.find(VL.front());
  if (It != MinBWs.end()) {
    ScalarTy = IntegerType::get(F->getContext(), It->second.first);
    VecTy = FixedVectorType::get(ScalarTy, VL.size());
  }
  unsigned EntryVF = E->getVectorFactor();
  auto *FinalVecTy = FixedVectorType::get(VecTy->getElementType(), EntryVF);

  bool NeedToShuffleReuses = !E->ReuseShuffleIndices.empty();
  if (E->State == TreeEntry::NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isa<InsertElementInst>(VL[0]))
      return InstructionCost::getInvalid();
    // The gather nodes use small bitwidth only if all operands use the same
    // bitwidth. Otherwise - use the original one.
    if (It != MinBWs.end() && any_of(VL.drop_front(), [&](Value *V) {
          auto VIt = MinBWs.find(V);
          return VIt == MinBWs.end() || VIt->second.first != It->second.first ||
                 VIt->second.second != It->second.second;
        })) {
      ScalarTy = VL.front()->getType();
      VecTy = FixedVectorType::get(ScalarTy, VL.size());
    }
    ShuffleCostEstimator Estimator(*TTI, VectorizedVals, *this,
                                   CheckedExtracts);
    unsigned VF = E->getVectorFactor();
    SmallVector<int> ReuseShuffleIndicies(E->ReuseShuffleIndices.begin(),
                                          E->ReuseShuffleIndices.end());
    SmallVector<Value *> GatheredScalars(E->Scalars.begin(), E->Scalars.end());
    // Build a mask out of the reorder indices and reorder scalars per this
    // mask.
    SmallVector<int> ReorderMask;
    inversePermutation(E->ReorderIndices, ReorderMask);
    if (!ReorderMask.empty())
      reorderScalars(GatheredScalars, ReorderMask);
    SmallVector<int> Mask;
    SmallVector<int> ExtractMask;
    std::optional<TargetTransformInfo::ShuffleKind> ExtractShuffle;
    std::optional<TargetTransformInfo::ShuffleKind> GatherShuffle;
    SmallVector<const TreeEntry *> Entries;
    // Check for gathered extracts.
    ExtractShuffle = tryToGatherExtractElements(GatheredScalars, ExtractMask);
    SmallVector<Value *> IgnoredVals;
    if (UserIgnoreList)
      IgnoredVals.assign(UserIgnoreList->begin(), UserIgnoreList->end());

    bool Resized = false;
    if (Value *VecBase = Estimator.adjustExtracts(
            E, ExtractMask, ExtractShuffle.value_or(TTI::SK_PermuteTwoSrc)))
      if (auto *VecBaseTy = dyn_cast<FixedVectorType>(VecBase->getType()))
        if (VF == VecBaseTy->getNumElements() && GatheredScalars.size() != VF) {
          Resized = true;
          GatheredScalars.append(VF - GatheredScalars.size(),
                                 PoisonValue::get(ScalarTy));
        }

    // Do not try to look for reshuffled loads for gathered loads (they will be
    // handled later), for vectorized scalars, and cases, which are definitely
    // not profitable (splats and small gather nodes.)
    if (ExtractShuffle || E->getOpcode() != Instruction::Load ||
        E->isAltShuffle() ||
        all_of(E->Scalars, [this](Value *V) { return getTreeEntry(V); }) ||
        isSplat(E->Scalars) ||
        (E->Scalars != GatheredScalars && GatheredScalars.size() <= 2))
      GatherShuffle = isGatherShuffledEntry(E, GatheredScalars, Mask, Entries);
    if (GatherShuffle) {
      assert((Entries.size() == 1 || Entries.size() == 2) &&
             "Expected shuffle of 1 or 2 entries.");
      if (*GatherShuffle == TTI::SK_PermuteSingleSrc &&
          Entries.front()->isSame(E->Scalars)) {
        // Perfect match in the graph, will reuse the previously vectorized
        // node. Cost is 0.
        LLVM_DEBUG(
            dbgs()
            << "SLP: perfect diamond match for gather bundle "
            << shortBundleName(VL) << ".\n");
        // Restore the mask for previous partially matched values.
        for (auto [I, V] : enumerate(E->Scalars)) {
          if (isa<PoisonValue>(V)) {
            Mask[I] = PoisonMaskElem;
            continue;
          }
          if (Mask[I] == PoisonMaskElem)
            Mask[I] = Entries.front()->findLaneForValue(V);
        }
        Estimator.add(Entries.front(), Mask);
        return Estimator.finalize(E->ReuseShuffleIndices);
      }
      if (!Resized) {
        unsigned VF1 = Entries.front()->getVectorFactor();
        unsigned VF2 = Entries.back()->getVectorFactor();
        if ((VF == VF1 || VF == VF2) && GatheredScalars.size() != VF)
          GatheredScalars.append(VF - GatheredScalars.size(),
                                 PoisonValue::get(ScalarTy));
      }
      // Remove shuffled elements from list of gathers.
      for (int I = 0, Sz = Mask.size(); I < Sz; ++I) {
        if (Mask[I] != PoisonMaskElem)
          GatheredScalars[I] = PoisonValue::get(ScalarTy);
      }
      LLVM_DEBUG(dbgs() << "SLP: shuffled " << Entries.size()
                        << " entries for bundle "
                        << shortBundleName(VL) << ".\n");
      if (Entries.size() == 1)
        Estimator.add(Entries.front(), Mask);
      else
        Estimator.add(Entries.front(), Entries.back(), Mask);
      if (all_of(GatheredScalars, PoisonValue ::classof))
        return Estimator.finalize(E->ReuseShuffleIndices);
      return Estimator.finalize(
          E->ReuseShuffleIndices, E->Scalars.size(),
          [&](Value *&Vec, SmallVectorImpl<int> &Mask) {
            Vec = Estimator.gather(GatheredScalars,
                                   Constant::getNullValue(FixedVectorType::get(
                                       ScalarTy, GatheredScalars.size())));
          });
    }
    if (!all_of(GatheredScalars, PoisonValue::classof)) {
      auto Gathers = ArrayRef(GatheredScalars).take_front(VL.size());
      bool SameGathers = VL.equals(Gathers);
      Value *BV = Estimator.gather(
          Gathers, SameGathers ? nullptr
                               : Constant::getNullValue(FixedVectorType::get(
                                     ScalarTy, GatheredScalars.size())));
      SmallVector<int> ReuseMask(Gathers.size(), PoisonMaskElem);
      std::iota(ReuseMask.begin(), ReuseMask.end(), 0);
      Estimator.add(BV, ReuseMask);
    }
    if (ExtractShuffle)
      Estimator.add(E, std::nullopt);
    return Estimator.finalize(E->ReuseShuffleIndices);
  }
  InstructionCost CommonCost = 0;
  SmallVector<int> Mask;
  if (!E->ReorderIndices.empty() &&
      E->State != TreeEntry::PossibleStridedVectorize) {
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
          E->State == TreeEntry::ScatterVectorize ||
          E->State == TreeEntry::PossibleStridedVectorize) &&
         "Unhandled state");
  assert(E->getOpcode() &&
         ((allSameType(VL) && allSameBlock(VL)) ||
          (E->getOpcode() == Instruction::GetElementPtr &&
           E->getMainOp()->getType()->isPointerTy())) &&
         "Invalid VL");
  Instruction *VL0 = E->getMainOp();
  unsigned ShuffleOrOp =
      E->isAltShuffle() ? (unsigned)Instruction::ShuffleVector : E->getOpcode();
  const unsigned Sz = VL.size();
  auto GetCostDiff =
      [=](function_ref<InstructionCost(unsigned)> ScalarEltCost,
          function_ref<InstructionCost(InstructionCost)> VectorCost) {
        // Calculate the cost of this instruction.
        InstructionCost ScalarCost = 0;
        if (isa<CastInst, CmpInst, SelectInst, CallInst>(VL0)) {
          // For some of the instructions no need to calculate cost for each
          // particular instruction, we can use the cost of the single
          // instruction x total number of scalar instructions.
          ScalarCost = Sz * ScalarEltCost(0);
        } else {
          for (unsigned I = 0; I < Sz; ++I)
            ScalarCost += ScalarEltCost(I);
        }

        InstructionCost VecCost = VectorCost(CommonCost);
        LLVM_DEBUG(dumpTreeCosts(E, CommonCost, VecCost - CommonCost,
                                 ScalarCost, "Calculated costs for Tree"));
        return VecCost - ScalarCost;
      };
  // Calculate cost difference from vectorizing set of GEPs.
  // Negative value means vectorizing is profitable.
  auto GetGEPCostDiff = [=](ArrayRef<Value *> Ptrs, Value *BasePtr) {
    InstructionCost ScalarCost = 0;
    InstructionCost VecCost = 0;
    // Here we differentiate two cases: (1) when Ptrs represent a regular
    // vectorization tree node (as they are pointer arguments of scattered
    // loads) or (2) when Ptrs are the arguments of loads or stores being
    // vectorized as plane wide unit-stride load/store since all the
    // loads/stores are known to be from/to adjacent locations.
    assert(E->State == TreeEntry::Vectorize &&
           "Entry state expected to be Vectorize here.");
    if (isa<LoadInst, StoreInst>(VL0)) {
      // Case 2: estimate costs for pointer related costs when vectorizing to
      // a wide load/store.
      // Scalar cost is estimated as a set of pointers with known relationship
      // between them.
      // For vector code we will use BasePtr as argument for the wide load/store
      // but we also need to account all the instructions which are going to
      // stay in vectorized code due to uses outside of these scalar
      // loads/stores.
      ScalarCost = TTI->getPointersChainCost(
          Ptrs, BasePtr, TTI::PointersChainInfo::getUnitStride(), ScalarTy,
          CostKind);

      SmallVector<const Value *> PtrsRetainedInVecCode;
      for (Value *V : Ptrs) {
        if (V == BasePtr) {
          PtrsRetainedInVecCode.push_back(V);
          continue;
        }
        auto *Ptr = dyn_cast<GetElementPtrInst>(V);
        // For simplicity assume Ptr to stay in vectorized code if it's not a
        // GEP instruction. We don't care since it's cost considered free.
        // TODO: We should check for any uses outside of vectorizable tree
        // rather than just single use.
        if (!Ptr || !Ptr->hasOneUse())
          PtrsRetainedInVecCode.push_back(V);
      }

      if (PtrsRetainedInVecCode.size() == Ptrs.size()) {
        // If all pointers stay in vectorized code then we don't have
        // any savings on that.
        LLVM_DEBUG(dumpTreeCosts(E, 0, ScalarCost, ScalarCost,
                                 "Calculated GEPs cost for Tree"));
        return InstructionCost{TTI::TCC_Free};
      }
      VecCost = TTI->getPointersChainCost(
          PtrsRetainedInVecCode, BasePtr,
          TTI::PointersChainInfo::getKnownStride(), VecTy, CostKind);
    } else {
      // Case 1: Ptrs are the arguments of loads that we are going to transform
      // into masked gather load intrinsic.
      // All the scalar GEPs will be removed as a result of vectorization.
      // For any external uses of some lanes extract element instructions will
      // be generated (which cost is estimated separately).
      TTI::PointersChainInfo PtrsInfo =
          all_of(Ptrs,
                 [](const Value *V) {
                   auto *Ptr = dyn_cast<GetElementPtrInst>(V);
                   return Ptr && !Ptr->hasAllConstantIndices();
                 })
              ? TTI::PointersChainInfo::getUnknownStride()
              : TTI::PointersChainInfo::getKnownStride();

      ScalarCost = TTI->getPointersChainCost(Ptrs, BasePtr, PtrsInfo, ScalarTy,
                                             CostKind);
      if (auto *BaseGEP = dyn_cast<GEPOperator>(BasePtr)) {
        SmallVector<const Value *> Indices(BaseGEP->indices());
        VecCost = TTI->getGEPCost(BaseGEP->getSourceElementType(),
                                  BaseGEP->getPointerOperand(), Indices, VecTy,
                                  CostKind);
      }
    }

    LLVM_DEBUG(dumpTreeCosts(E, 0, VecCost, ScalarCost,
                             "Calculated GEPs cost for Tree"));

    return VecCost - ScalarCost;
  };

  switch (ShuffleOrOp) {
  case Instruction::PHI: {
    // Count reused scalars.
    InstructionCost ScalarCost = 0;
    SmallPtrSet<const TreeEntry *, 4> CountedOps;
    for (Value *V : VL) {
      auto *PHI = dyn_cast<PHINode>(V);
      if (!PHI)
        continue;

      ValueList Operands(PHI->getNumIncomingValues(), nullptr);
      for (unsigned I = 0, N = PHI->getNumIncomingValues(); I < N; ++I) {
        Value *Op = PHI->getIncomingValue(I);
        Operands[I] = Op;
      }
      if (const TreeEntry *OpTE = getTreeEntry(Operands.front()))
        if (OpTE->isSame(Operands) && CountedOps.insert(OpTE).second)
          if (!OpTE->ReuseShuffleIndices.empty())
            ScalarCost += TTI::TCC_Basic * (OpTE->ReuseShuffleIndices.size() -
                                            OpTE->Scalars.size());
    }

    return CommonCost - ScalarCost;
  }
  case Instruction::ExtractValue:
  case Instruction::ExtractElement: {
    auto GetScalarCost = [=](unsigned Idx) {
      auto *I = cast<Instruction>(VL[Idx]);
      VectorType *SrcVecTy;
      if (ShuffleOrOp == Instruction::ExtractElement) {
        auto *EE = cast<ExtractElementInst>(I);
        SrcVecTy = EE->getVectorOperandType();
      } else {
        auto *EV = cast<ExtractValueInst>(I);
        Type *AggregateTy = EV->getAggregateOperand()->getType();
        unsigned NumElts;
        if (auto *ATy = dyn_cast<ArrayType>(AggregateTy))
          NumElts = ATy->getNumElements();
        else
          NumElts = AggregateTy->getStructNumElements();
        SrcVecTy = FixedVectorType::get(ScalarTy, NumElts);
      }
      if (I->hasOneUse()) {
        Instruction *Ext = I->user_back();
        if ((isa<SExtInst>(Ext) || isa<ZExtInst>(Ext)) &&
            all_of(Ext->users(),
                   [](User *U) { return isa<GetElementPtrInst>(U); })) {
          // Use getExtractWithExtendCost() to calculate the cost of
          // extractelement/ext pair.
          InstructionCost Cost = TTI->getExtractWithExtendCost(
              Ext->getOpcode(), Ext->getType(), SrcVecTy, *getExtractIndex(I));
          // Subtract the cost of s|zext which is subtracted separately.
          Cost -= TTI->getCastInstrCost(
              Ext->getOpcode(), Ext->getType(), I->getType(),
              TTI::getCastContextHint(Ext), CostKind, Ext);
          return Cost;
        }
      }
      return TTI->getVectorInstrCost(Instruction::ExtractElement, SrcVecTy,
                                     CostKind, *getExtractIndex(I));
    };
    auto GetVectorCost = [](InstructionCost CommonCost) { return CommonCost; };
    return GetCostDiff(GetScalarCost, GetVectorCost);
  }
  case Instruction::InsertElement: {
    assert(E->ReuseShuffleIndices.empty() &&
           "Unique insertelements only are expected.");
    auto *SrcVecTy = cast<FixedVectorType>(VL0->getType());
    unsigned const NumElts = SrcVecTy->getNumElements();
    unsigned const NumScalars = VL.size();

    unsigned NumOfParts = TTI->getNumberOfParts(SrcVecTy);

    SmallVector<int> InsertMask(NumElts, PoisonMaskElem);
    unsigned OffsetBeg = *getInsertIndex(VL.front());
    unsigned OffsetEnd = OffsetBeg;
    InsertMask[OffsetBeg] = 0;
    for (auto [I, V] : enumerate(VL.drop_front())) {
      unsigned Idx = *getInsertIndex(V);
      if (OffsetBeg > Idx)
        OffsetBeg = Idx;
      else if (OffsetEnd < Idx)
        OffsetEnd = Idx;
      InsertMask[Idx] = I + 1;
    }
    unsigned VecScalarsSz = PowerOf2Ceil(NumElts);
    if (NumOfParts > 0)
      VecScalarsSz = PowerOf2Ceil((NumElts + NumOfParts - 1) / NumOfParts);
    unsigned VecSz = (1 + OffsetEnd / VecScalarsSz - OffsetBeg / VecScalarsSz) *
                     VecScalarsSz;
    unsigned Offset = VecScalarsSz * (OffsetBeg / VecScalarsSz);
    unsigned InsertVecSz = std::min<unsigned>(
        PowerOf2Ceil(OffsetEnd - OffsetBeg + 1),
        ((OffsetEnd - OffsetBeg + VecScalarsSz) / VecScalarsSz) * VecScalarsSz);
    bool IsWholeSubvector =
        OffsetBeg == Offset && ((OffsetEnd + 1) % VecScalarsSz == 0);
    // Check if we can safely insert a subvector. If it is not possible, just
    // generate a whole-sized vector and shuffle the source vector and the new
    // subvector.
    if (OffsetBeg + InsertVecSz > VecSz) {
      // Align OffsetBeg to generate correct mask.
      OffsetBeg = alignDown(OffsetBeg, VecSz, Offset);
      InsertVecSz = VecSz;
    }

    APInt DemandedElts = APInt::getZero(NumElts);
    // TODO: Add support for Instruction::InsertValue.
    SmallVector<int> Mask;
    if (!E->ReorderIndices.empty()) {
      inversePermutation(E->ReorderIndices, Mask);
      Mask.append(InsertVecSz - Mask.size(), PoisonMaskElem);
    } else {
      Mask.assign(VecSz, PoisonMaskElem);
      std::iota(Mask.begin(), std::next(Mask.begin(), InsertVecSz), 0);
    }
    bool IsIdentity = true;
    SmallVector<int> PrevMask(InsertVecSz, PoisonMaskElem);
    Mask.swap(PrevMask);
    for (unsigned I = 0; I < NumScalars; ++I) {
      unsigned InsertIdx = *getInsertIndex(VL[PrevMask[I]]);
      DemandedElts.setBit(InsertIdx);
      IsIdentity &= InsertIdx - OffsetBeg == I;
      Mask[InsertIdx - OffsetBeg] = I;
    }
    assert(Offset < NumElts && "Failed to find vector index offset");

    InstructionCost Cost = 0;
    Cost -= TTI->getScalarizationOverhead(SrcVecTy, DemandedElts,
                                          /*Insert*/ true, /*Extract*/ false,
                                          CostKind);

    // First cost - resize to actual vector size if not identity shuffle or
    // need to shift the vector.
    // Do not calculate the cost if the actual size is the register size and
    // we can merge this shuffle with the following SK_Select.
    auto *InsertVecTy = FixedVectorType::get(ScalarTy, InsertVecSz);
    if (!IsIdentity)
      Cost += TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                  InsertVecTy, Mask);
    auto *FirstInsert = cast<Instruction>(*find_if(E->Scalars, [E](Value *V) {
      return !is_contained(E->Scalars, cast<Instruction>(V)->getOperand(0));
    }));
    // Second cost - permutation with subvector, if some elements are from the
    // initial vector or inserting a subvector.
    // TODO: Implement the analysis of the FirstInsert->getOperand(0)
    // subvector of ActualVecTy.
    SmallBitVector InMask =
        isUndefVector(FirstInsert->getOperand(0),
                      buildUseMask(NumElts, InsertMask, UseMask::UndefsAsMask));
    if (!InMask.all() && NumScalars != NumElts && !IsWholeSubvector) {
      if (InsertVecSz != VecSz) {
        auto *ActualVecTy = FixedVectorType::get(ScalarTy, VecSz);
        Cost += TTI->getShuffleCost(TTI::SK_InsertSubvector, ActualVecTy,
                                    std::nullopt, CostKind, OffsetBeg - Offset,
                                    InsertVecTy);
      } else {
        for (unsigned I = 0, End = OffsetBeg - Offset; I < End; ++I)
          Mask[I] = InMask.test(I) ? PoisonMaskElem : I;
        for (unsigned I = OffsetBeg - Offset, End = OffsetEnd - Offset;
             I <= End; ++I)
          if (Mask[I] != PoisonMaskElem)
            Mask[I] = I + VecSz;
        for (unsigned I = OffsetEnd + 1 - Offset; I < VecSz; ++I)
          Mask[I] =
              ((I >= InMask.size()) || InMask.test(I)) ? PoisonMaskElem : I;
        Cost += TTI->getShuffleCost(TTI::SK_PermuteTwoSrc, InsertVecTy, Mask);
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
    auto SrcIt = MinBWs.find(VL0->getOperand(0));
    Type *SrcScalarTy = VL0->getOperand(0)->getType();
    auto *SrcVecTy = FixedVectorType::get(SrcScalarTy, VL.size());
    unsigned Opcode = ShuffleOrOp;
    if (!ScalarTy->isFloatingPointTy() && !SrcScalarTy->isFloatingPointTy() &&
        (SrcIt != MinBWs.end() || It != MinBWs.end())) {
      // Check if the values are candidates to demote.
      unsigned SrcBWSz = DL->getTypeSizeInBits(SrcScalarTy);
      if (SrcIt != MinBWs.end()) {
        SrcBWSz = SrcIt->second.first;
        SrcScalarTy = IntegerType::get(F->getContext(), SrcBWSz);
        SrcVecTy = FixedVectorType::get(SrcScalarTy, VL.size());
      }
      unsigned BWSz = DL->getTypeSizeInBits(ScalarTy);
      if (BWSz == SrcBWSz) {
        Opcode = Instruction::BitCast;
      } else if (BWSz < SrcBWSz) {
        Opcode = Instruction::Trunc;
      } else if (It != MinBWs.end()) {
        assert(BWSz > SrcBWSz && "Invalid cast!");
        Opcode = It->second.second ? Instruction::SExt : Instruction::ZExt;
      }
    }
    auto GetScalarCost = [=](unsigned Idx) {
      auto *VI =
          VL0->getOpcode() == Opcode ? cast<Instruction>(VL[Idx]) : nullptr;
      return TTI->getCastInstrCost(Opcode, ScalarTy, SrcScalarTy,
                                   TTI::getCastContextHint(VI), CostKind, VI);
    };
    TTI::CastContextHint CCH = TTI::CastContextHint::None;
    if (const TreeEntry *OpTE = getTreeEntry(VL0->getOperand(0))) {
      if (OpTE->State == TreeEntry::ScatterVectorize) {
        CCH = TTI::CastContextHint::GatherScatter;
      } else if (OpTE->State == TreeEntry::Vectorize &&
                 OpTE->getOpcode() == Instruction::Load &&
                 !OpTE->isAltShuffle()) {
        if (OpTE->ReorderIndices.empty()) {
          CCH = TTI::CastContextHint::Normal;
        } else {
          SmallVector<int> Mask;
          inversePermutation(OpTE->ReorderIndices, Mask);
          if (ShuffleVectorInst::isReverseMask(Mask))
            CCH = TTI::CastContextHint::Reversed;
        }
      }
    } else {
      InstructionsState SrcState = getSameOpcode(E->getOperand(0), *TLI);
      if (SrcState.getOpcode() == Instruction::Load && !SrcState.isAltShuffle())
        CCH = TTI::CastContextHint::GatherScatter;
    }
    auto GetVectorCost = [=](InstructionCost CommonCost) {
      auto *VI = VL0->getOpcode() == Opcode ? VL0 : nullptr;
      return CommonCost +
             TTI->getCastInstrCost(Opcode, VecTy, SrcVecTy, CCH, CostKind, VI);
    };
    return GetCostDiff(GetScalarCost, GetVectorCost);
  }
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select: {
    CmpInst::Predicate VecPred, SwappedVecPred;
    auto MatchCmp = m_Cmp(VecPred, m_Value(), m_Value());
    if (match(VL0, m_Select(MatchCmp, m_Value(), m_Value())) ||
        match(VL0, MatchCmp))
      SwappedVecPred = CmpInst::getSwappedPredicate(VecPred);
    else
      SwappedVecPred = VecPred = ScalarTy->isFloatingPointTy()
                                     ? CmpInst::BAD_FCMP_PREDICATE
                                     : CmpInst::BAD_ICMP_PREDICATE;
    auto GetScalarCost = [&](unsigned Idx) {
      auto *VI = cast<Instruction>(VL[Idx]);
      CmpInst::Predicate CurrentPred = ScalarTy->isFloatingPointTy()
                                           ? CmpInst::BAD_FCMP_PREDICATE
                                           : CmpInst::BAD_ICMP_PREDICATE;
      auto MatchCmp = m_Cmp(CurrentPred, m_Value(), m_Value());
      if ((!match(VI, m_Select(MatchCmp, m_Value(), m_Value())) &&
           !match(VI, MatchCmp)) ||
          (CurrentPred != VecPred && CurrentPred != SwappedVecPred))
        VecPred = SwappedVecPred = ScalarTy->isFloatingPointTy()
                                       ? CmpInst::BAD_FCMP_PREDICATE
                                       : CmpInst::BAD_ICMP_PREDICATE;

      return TTI->getCmpSelInstrCost(E->getOpcode(), ScalarTy,
                                     Builder.getInt1Ty(), CurrentPred, CostKind,
                                     VI);
    };
    auto GetVectorCost = [&](InstructionCost CommonCost) {
      auto *MaskTy = FixedVectorType::get(Builder.getInt1Ty(), VL.size());

      InstructionCost VecCost = TTI->getCmpSelInstrCost(
          E->getOpcode(), VecTy, MaskTy, VecPred, CostKind, VL0);
      // Check if it is possible and profitable to use min/max for selects
      // in VL.
      //
      auto IntrinsicAndUse = canConvertToMinOrMaxIntrinsic(VL);
      if (IntrinsicAndUse.first != Intrinsic::not_intrinsic) {
        IntrinsicCostAttributes CostAttrs(IntrinsicAndUse.first, VecTy,
                                          {VecTy, VecTy});
        InstructionCost IntrinsicCost =
            TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
        // If the selects are the only uses of the compares, they will be
        // dead and we can adjust the cost by removing their cost.
        if (IntrinsicAndUse.second)
          IntrinsicCost -= TTI->getCmpSelInstrCost(Instruction::ICmp, VecTy,
                                                   MaskTy, VecPred, CostKind);
        VecCost = std::min(VecCost, IntrinsicCost);
      }
      return VecCost + CommonCost;
    };
    return GetCostDiff(GetScalarCost, GetVectorCost);
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
    auto GetScalarCost = [=](unsigned Idx) {
      auto *VI = cast<Instruction>(VL[Idx]);
      unsigned OpIdx = isa<UnaryOperator>(VI) ? 0 : 1;
      TTI::OperandValueInfo Op1Info = TTI::getOperandInfo(VI->getOperand(0));
      TTI::OperandValueInfo Op2Info =
          TTI::getOperandInfo(VI->getOperand(OpIdx));
      SmallVector<const Value *> Operands(VI->operand_values());
      return TTI->getArithmeticInstrCost(ShuffleOrOp, ScalarTy, CostKind,
                                         Op1Info, Op2Info, Operands, VI);
    };
    auto GetVectorCost = [=](InstructionCost CommonCost) {
      unsigned OpIdx = isa<UnaryOperator>(VL0) ? 0 : 1;
      TTI::OperandValueInfo Op1Info = getOperandInfo(*E, 0);
      TTI::OperandValueInfo Op2Info = getOperandInfo(*E, OpIdx);
      return TTI->getArithmeticInstrCost(ShuffleOrOp, VecTy, CostKind, Op1Info,
                                         Op2Info) +
             CommonCost;
    };
    return GetCostDiff(GetScalarCost, GetVectorCost);
  }
  case Instruction::GetElementPtr: {
    return CommonCost + GetGEPCostDiff(VL, VL0);
  }
  case Instruction::Load: {
    auto GetScalarCost = [=](unsigned Idx) {
      auto *VI = cast<LoadInst>(VL[Idx]);
      return TTI->getMemoryOpCost(Instruction::Load, ScalarTy, VI->getAlign(),
                                  VI->getPointerAddressSpace(), CostKind,
                                  TTI::OperandValueInfo(), VI);
    };
    auto *LI0 = cast<LoadInst>(VL0);
    auto GetVectorCost = [=](InstructionCost CommonCost) {
      InstructionCost VecLdCost;
      if (E->State == TreeEntry::Vectorize) {
        VecLdCost = TTI->getMemoryOpCost(
            Instruction::Load, VecTy, LI0->getAlign(),
            LI0->getPointerAddressSpace(), CostKind, TTI::OperandValueInfo());
      } else {
        assert((E->State == TreeEntry::ScatterVectorize ||
                E->State == TreeEntry::PossibleStridedVectorize) &&
               "Unknown EntryState");
        Align CommonAlignment = LI0->getAlign();
        for (Value *V : VL)
          CommonAlignment =
              std::min(CommonAlignment, cast<LoadInst>(V)->getAlign());
        VecLdCost = TTI->getGatherScatterOpCost(
            Instruction::Load, VecTy, LI0->getPointerOperand(),
            /*VariableMask=*/false, CommonAlignment, CostKind);
      }
      return VecLdCost + CommonCost;
    };

    InstructionCost Cost = GetCostDiff(GetScalarCost, GetVectorCost);
    // If this node generates masked gather load then it is not a terminal node.
    // Hence address operand cost is estimated separately.
    if (E->State == TreeEntry::ScatterVectorize ||
        E->State == TreeEntry::PossibleStridedVectorize)
      return Cost;

    // Estimate cost of GEPs since this tree node is a terminator.
    SmallVector<Value *> PointerOps(VL.size());
    for (auto [I, V] : enumerate(VL))
      PointerOps[I] = cast<LoadInst>(V)->getPointerOperand();
    return Cost + GetGEPCostDiff(PointerOps, LI0->getPointerOperand());
  }
  case Instruction::Store: {
    bool IsReorder = !E->ReorderIndices.empty();
    auto GetScalarCost = [=](unsigned Idx) {
      auto *VI = cast<StoreInst>(VL[Idx]);
      TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(VI->getValueOperand());
      return TTI->getMemoryOpCost(Instruction::Store, ScalarTy, VI->getAlign(),
                                  VI->getPointerAddressSpace(), CostKind,
                                  OpInfo, VI);
    };
    auto *BaseSI =
        cast<StoreInst>(IsReorder ? VL[E->ReorderIndices.front()] : VL0);
    auto GetVectorCost = [=](InstructionCost CommonCost) {
      // We know that we can merge the stores. Calculate the cost.
      TTI::OperandValueInfo OpInfo = getOperandInfo(*E, 0);
      return TTI->getMemoryOpCost(Instruction::Store, VecTy, BaseSI->getAlign(),
                                  BaseSI->getPointerAddressSpace(), CostKind,
                                  OpInfo) +
             CommonCost;
    };
    SmallVector<Value *> PointerOps(VL.size());
    for (auto [I, V] : enumerate(VL)) {
      unsigned Idx = IsReorder ? E->ReorderIndices[I] : I;
      PointerOps[Idx] = cast<StoreInst>(V)->getPointerOperand();
    }

    return GetCostDiff(GetScalarCost, GetVectorCost) +
           GetGEPCostDiff(PointerOps, BaseSI->getPointerOperand());
  }
  case Instruction::Call: {
    auto GetScalarCost = [=](unsigned Idx) {
      auto *CI = cast<CallInst>(VL[Idx]);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
      if (ID != Intrinsic::not_intrinsic) {
        IntrinsicCostAttributes CostAttrs(ID, *CI, 1);
        return TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
      }
      return TTI->getCallInstrCost(CI->getCalledFunction(),
                                   CI->getFunctionType()->getReturnType(),
                                   CI->getFunctionType()->params(), CostKind);
    };
    auto GetVectorCost = [=](InstructionCost CommonCost) {
      auto *CI = cast<CallInst>(VL0);
      auto VecCallCosts = getVectorCallCosts(CI, VecTy, TTI, TLI);
      return std::min(VecCallCosts.first, VecCallCosts.second) + CommonCost;
    };
    return GetCostDiff(GetScalarCost, GetVectorCost);
  }
  case Instruction::ShuffleVector: {
    assert(E->isAltShuffle() &&
           ((Instruction::isBinaryOp(E->getOpcode()) &&
             Instruction::isBinaryOp(E->getAltOpcode())) ||
            (Instruction::isCast(E->getOpcode()) &&
             Instruction::isCast(E->getAltOpcode())) ||
            (isa<CmpInst>(VL0) && isa<CmpInst>(E->getAltOp()))) &&
           "Invalid Shuffle Vector Operand");
    // Try to find the previous shuffle node with the same operands and same
    // main/alternate ops.
    auto TryFindNodeWithEqualOperands = [=]() {
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
    auto GetScalarCost = [=](unsigned Idx) {
      auto *VI = cast<Instruction>(VL[Idx]);
      assert(E->isOpcodeOrAlt(VI) && "Unexpected main/alternate opcode");
      (void)E;
      return TTI->getInstructionCost(VI, CostKind);
    };
    // Need to clear CommonCost since the final shuffle cost is included into
    // vector cost.
    auto GetVectorCost = [&](InstructionCost) {
      // VecCost is equal to sum of the cost of creating 2 vectors
      // and the cost of creating shuffle.
      InstructionCost VecCost = 0;
      if (TryFindNodeWithEqualOperands()) {
        LLVM_DEBUG({
          dbgs() << "SLP: diamond match for alternate node found.\n";
          E->dump();
        });
        // No need to add new vector costs here since we're going to reuse
        // same main/alternate vector ops, just do different shuffling.
      } else if (Instruction::isBinaryOp(E->getOpcode())) {
        VecCost = TTI->getArithmeticInstrCost(E->getOpcode(), VecTy, CostKind);
        VecCost +=
            TTI->getArithmeticInstrCost(E->getAltOpcode(), VecTy, CostKind);
      } else if (auto *CI0 = dyn_cast<CmpInst>(VL0)) {
        auto *MaskTy = FixedVectorType::get(Builder.getInt1Ty(), VL.size());
        VecCost = TTI->getCmpSelInstrCost(E->getOpcode(), VecTy, MaskTy,
                                          CI0->getPredicate(), CostKind, VL0);
        VecCost += TTI->getCmpSelInstrCost(
            E->getOpcode(), VecTy, MaskTy,
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
      E->buildAltOpShuffleMask(
          [E](Instruction *I) {
            assert(E->isOpcodeOrAlt(I) && "Unexpected main/alternate opcode");
            return I->getOpcode() == E->getAltOpcode();
          },
          Mask);
      VecCost += TTI->getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc,
                                     FinalVecTy, Mask);
      return VecCost;
    };
    return GetCostDiff(GetScalarCost, GetVectorCost);
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
       VectorizableTree[0]->State != TreeEntry::ScatterVectorize &&
       VectorizableTree[0]->State != TreeEntry::PossibleStridedVectorize))
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
      VectorizableTree[1]->State == TreeEntry::NeedToGather &&
      (VectorizableTree[1]->getVectorFactor() <= 2 ||
       !(isSplat(VectorizableTree[1]->Scalars) ||
         allConstant(VectorizableTree[1]->Scalars))))
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

  SmallPtrSet<Instruction *, 4> LiveValues;
  Instruction *PrevInst = nullptr;

  // The entries in VectorizableTree are not necessarily ordered by their
  // position in basic blocks. Collect them and order them by dominance so later
  // instructions are guaranteed to be visited first. For instructions in
  // different basic blocks, we only scan to the beginning of the block, so
  // their order does not matter, as long as all instructions in a basic block
  // are grouped together. Using dominance ensures a deterministic order.
  SmallVector<Instruction *, 16> OrderedScalars;
  for (const auto &TEPtr : VectorizableTree) {
    if (TEPtr->State != TreeEntry::Vectorize)
      continue;
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
      return NodeA->getDFSNumIn() > NodeB->getDFSNumIn();
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

      auto NoCallIntrinsic = [this](Instruction *I) {
        if (auto *II = dyn_cast<IntrinsicInst>(I)) {
          if (II->isAssumeLikeIntrinsic())
            return true;
          FastMathFlags FMF;
          SmallVector<Type *, 4> Tys;
          for (auto &ArgOp : II->args())
            Tys.push_back(ArgOp->getType());
          if (auto *FPMO = dyn_cast<FPMathOperator>(II))
            FMF = FPMO->getFastMathFlags();
          IntrinsicCostAttributes ICA(II->getIntrinsicID(), II->getType(), Tys,
                                      FMF);
          InstructionCost IntrCost =
              TTI->getIntrinsicInstrCost(ICA, TTI::TCK_RecipThroughput);
          InstructionCost CallCost = TTI->getCallInstrCost(
              nullptr, II->getType(), Tys, TTI::TCK_RecipThroughput);
          if (IntrCost < CallCost)
            return true;
        }
        return false;
      };

      // Debug information does not impact spill cost.
      if (isa<CallBase>(&*PrevInstIt) && !NoCallIntrinsic(&*PrevInstIt) &&
          &*PrevInstIt != PrevInst)
        NumCalls++;

      ++PrevInstIt;
    }

    if (NumCalls) {
      SmallVector<Type *, 4> V;
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

/// Checks if the \p IE1 instructions is followed by \p IE2 instruction in the
/// buildvector sequence.
static bool isFirstInsertElement(const InsertElementInst *IE1,
                                 const InsertElementInst *IE2) {
  if (IE1 == IE2)
    return false;
  const auto *I1 = IE1;
  const auto *I2 = IE2;
  const InsertElementInst *PrevI1;
  const InsertElementInst *PrevI2;
  unsigned Idx1 = *getInsertIndex(IE1);
  unsigned Idx2 = *getInsertIndex(IE2);
  do {
    if (I2 == IE1)
      return true;
    if (I1 == IE2)
      return false;
    PrevI1 = I1;
    PrevI2 = I2;
    if (I1 && (I1 == IE1 || I1->hasOneUse()) &&
        getInsertIndex(I1).value_or(Idx2) != Idx2)
      I1 = dyn_cast<InsertElementInst>(I1->getOperand(0));
    if (I2 && ((I2 == IE2 || I2->hasOneUse())) &&
        getInsertIndex(I2).value_or(Idx1) != Idx1)
      I2 = dyn_cast<InsertElementInst>(I2->getOperand(0));
  } while ((I1 && PrevI1 != I1) || (I2 && PrevI2 != I2));
  llvm_unreachable("Two different buildvectors not expected.");
}

namespace {
/// Returns incoming Value *, if the requested type is Value * too, or a default
/// value, otherwise.
struct ValueSelect {
  template <typename U>
  static std::enable_if_t<std::is_same_v<Value *, U>, Value *> get(Value *V) {
    return V;
  }
  template <typename U>
  static std::enable_if_t<!std::is_same_v<Value *, U>, U> get(Value *) {
    return U();
  }
};
} // namespace

/// Does the analysis of the provided shuffle masks and performs the requested
/// actions on the vectors with the given shuffle masks. It tries to do it in
/// several steps.
/// 1. If the Base vector is not undef vector, resizing the very first mask to
/// have common VF and perform action for 2 input vectors (including non-undef
/// Base). Other shuffle masks are combined with the resulting after the 1 stage
/// and processed as a shuffle of 2 elements.
/// 2. If the Base is undef vector and have only 1 shuffle mask, perform the
/// action only for 1 vector with the given mask, if it is not the identity
/// mask.
/// 3. If > 2 masks are used, perform the remaining shuffle actions for 2
/// vectors, combing the masks properly between the steps.
template <typename T>
static T *performExtractsShuffleAction(
    MutableArrayRef<std::pair<T *, SmallVector<int>>> ShuffleMask, Value *Base,
    function_ref<unsigned(T *)> GetVF,
    function_ref<std::pair<T *, bool>(T *, ArrayRef<int>, bool)> ResizeAction,
    function_ref<T *(ArrayRef<int>, ArrayRef<T *>)> Action) {
  assert(!ShuffleMask.empty() && "Empty list of shuffles for inserts.");
  SmallVector<int> Mask(ShuffleMask.begin()->second);
  auto VMIt = std::next(ShuffleMask.begin());
  T *Prev = nullptr;
  SmallBitVector UseMask =
      buildUseMask(Mask.size(), Mask, UseMask::UndefsAsMask);
  SmallBitVector IsBaseUndef = isUndefVector(Base, UseMask);
  if (!IsBaseUndef.all()) {
    // Base is not undef, need to combine it with the next subvectors.
    std::pair<T *, bool> Res =
        ResizeAction(ShuffleMask.begin()->first, Mask, /*ForSingleMask=*/false);
    SmallBitVector IsBasePoison = isUndefVector<true>(Base, UseMask);
    for (unsigned Idx = 0, VF = Mask.size(); Idx < VF; ++Idx) {
      if (Mask[Idx] == PoisonMaskElem)
        Mask[Idx] = IsBasePoison.test(Idx) ? PoisonMaskElem : Idx;
      else
        Mask[Idx] = (Res.second ? Idx : Mask[Idx]) + VF;
    }
    auto *V = ValueSelect::get<T *>(Base);
    (void)V;
    assert((!V || GetVF(V) == Mask.size()) &&
           "Expected base vector of VF number of elements.");
    Prev = Action(Mask, {nullptr, Res.first});
  } else if (ShuffleMask.size() == 1) {
    // Base is undef and only 1 vector is shuffled - perform the action only for
    // single vector, if the mask is not the identity mask.
    std::pair<T *, bool> Res = ResizeAction(ShuffleMask.begin()->first, Mask,
                                            /*ForSingleMask=*/true);
    if (Res.second)
      // Identity mask is found.
      Prev = Res.first;
    else
      Prev = Action(Mask, {ShuffleMask.begin()->first});
  } else {
    // Base is undef and at least 2 input vectors shuffled - perform 2 vectors
    // shuffles step by step, combining shuffle between the steps.
    unsigned Vec1VF = GetVF(ShuffleMask.begin()->first);
    unsigned Vec2VF = GetVF(VMIt->first);
    if (Vec1VF == Vec2VF) {
      // No need to resize the input vectors since they are of the same size, we
      // can shuffle them directly.
      ArrayRef<int> SecMask = VMIt->second;
      for (unsigned I = 0, VF = Mask.size(); I < VF; ++I) {
        if (SecMask[I] != PoisonMaskElem) {
          assert(Mask[I] == PoisonMaskElem && "Multiple uses of scalars.");
          Mask[I] = SecMask[I] + Vec1VF;
        }
      }
      Prev = Action(Mask, {ShuffleMask.begin()->first, VMIt->first});
    } else {
      // Vectors of different sizes - resize and reshuffle.
      std::pair<T *, bool> Res1 = ResizeAction(ShuffleMask.begin()->first, Mask,
                                               /*ForSingleMask=*/false);
      std::pair<T *, bool> Res2 =
          ResizeAction(VMIt->first, VMIt->second, /*ForSingleMask=*/false);
      ArrayRef<int> SecMask = VMIt->second;
      for (unsigned I = 0, VF = Mask.size(); I < VF; ++I) {
        if (Mask[I] != PoisonMaskElem) {
          assert(SecMask[I] == PoisonMaskElem && "Multiple uses of scalars.");
          if (Res1.second)
            Mask[I] = I;
        } else if (SecMask[I] != PoisonMaskElem) {
          assert(Mask[I] == PoisonMaskElem && "Multiple uses of scalars.");
          Mask[I] = (Res2.second ? I : SecMask[I]) + VF;
        }
      }
      Prev = Action(Mask, {Res1.first, Res2.first});
    }
    VMIt = std::next(VMIt);
  }
  bool IsBaseNotUndef = !IsBaseUndef.all();
  (void)IsBaseNotUndef;
  // Perform requested actions for the remaining masks/vectors.
  for (auto E = ShuffleMask.end(); VMIt != E; ++VMIt) {
    // Shuffle other input vectors, if any.
    std::pair<T *, bool> Res =
        ResizeAction(VMIt->first, VMIt->second, /*ForSingleMask=*/false);
    ArrayRef<int> SecMask = VMIt->second;
    for (unsigned I = 0, VF = Mask.size(); I < VF; ++I) {
      if (SecMask[I] != PoisonMaskElem) {
        assert((Mask[I] == PoisonMaskElem || IsBaseNotUndef) &&
               "Multiple uses of scalars.");
        Mask[I] = (Res.second ? I : SecMask[I]) + VF;
      } else if (Mask[I] != PoisonMaskElem) {
        Mask[I] = I;
      }
    }
    Prev = Action(Mask, {Prev, Res.first});
  }
  return Prev;
}

InstructionCost BoUpSLP::getTreeCost(ArrayRef<Value *> VectorizedVals) {
  // Build a map for gathered scalars to the nodes where they are used.
  ValueToGatherNodes.clear();
  for (const std::unique_ptr<TreeEntry> &EntryPtr : VectorizableTree) {
    if (EntryPtr->State != TreeEntry::NeedToGather)
      continue;
    for (Value *V : EntryPtr->Scalars)
      if (!isConstant(V))
        ValueToGatherNodes.try_emplace(V).first->getSecond().insert(
            EntryPtr.get());
  }
  InstructionCost Cost = 0;
  LLVM_DEBUG(dbgs() << "SLP: Calculating cost for tree of size "
                    << VectorizableTree.size() << ".\n");

  unsigned BundleWidth = VectorizableTree[0]->Scalars.size();

  SmallPtrSet<Value *, 4> CheckedExtracts;
  for (unsigned I = 0, E = VectorizableTree.size(); I < E; ++I) {
    TreeEntry &TE = *VectorizableTree[I];
    if (TE.State == TreeEntry::NeedToGather) {
      if (const TreeEntry *E = getTreeEntry(TE.getMainOp());
          E && E->getVectorFactor() == TE.getVectorFactor() &&
          E->isSame(TE.Scalars)) {
        // Some gather nodes might be absolutely the same as some vectorizable
        // nodes after reordering, need to handle it.
        LLVM_DEBUG(dbgs() << "SLP: Adding cost 0 for bundle "
                          << shortBundleName(TE.Scalars) << ".\n"
                          << "SLP: Current total cost = " << Cost << "\n");
        continue;
      }
    }

    InstructionCost C = getEntryCost(&TE, VectorizedVals, CheckedExtracts);
    Cost += C;
    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C << " for bundle "
                      << shortBundleName(TE.Scalars) << ".\n"
                      << "SLP: Current total cost = " << Cost << "\n");
  }

  SmallPtrSet<Value *, 16> ExtractCostCalculated;
  InstructionCost ExtractCost = 0;
  SmallVector<MapVector<const TreeEntry *, SmallVector<int>>> ShuffleMasks;
  SmallVector<std::pair<Value *, const TreeEntry *>> FirstUsers;
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

    // If found user is an insertelement, do not calculate extract cost but try
    // to detect it as a final shuffled/identity match.
    if (auto *VU = dyn_cast_or_null<InsertElementInst>(EU.User)) {
      if (auto *FTy = dyn_cast<FixedVectorType>(VU->getType())) {
        std::optional<unsigned> InsertIdx = getInsertIndex(VU);
        if (InsertIdx) {
          const TreeEntry *ScalarTE = getTreeEntry(EU.Scalar);
          auto *It = find_if(
              FirstUsers,
              [this, VU](const std::pair<Value *, const TreeEntry *> &Pair) {
                return areTwoInsertFromSameBuildVector(
                    VU, cast<InsertElementInst>(Pair.first),
                    [this](InsertElementInst *II) -> Value * {
                      Value *Op0 = II->getOperand(0);
                      if (getTreeEntry(II) && !getTreeEntry(Op0))
                        return nullptr;
                      return Op0;
                    });
              });
          int VecId = -1;
          if (It == FirstUsers.end()) {
            (void)ShuffleMasks.emplace_back();
            SmallVectorImpl<int> &Mask = ShuffleMasks.back()[ScalarTE];
            if (Mask.empty())
              Mask.assign(FTy->getNumElements(), PoisonMaskElem);
            // Find the insertvector, vectorized in tree, if any.
            Value *Base = VU;
            while (auto *IEBase = dyn_cast<InsertElementInst>(Base)) {
              if (IEBase != EU.User &&
                  (!IEBase->hasOneUse() ||
                   getInsertIndex(IEBase).value_or(*InsertIdx) == *InsertIdx))
                break;
              // Build the mask for the vectorized insertelement instructions.
              if (const TreeEntry *E = getTreeEntry(IEBase)) {
                VU = IEBase;
                do {
                  IEBase = cast<InsertElementInst>(Base);
                  int Idx = *getInsertIndex(IEBase);
                  assert(Mask[Idx] == PoisonMaskElem &&
                         "InsertElementInstruction used already.");
                  Mask[Idx] = Idx;
                  Base = IEBase->getOperand(0);
                } while (E == getTreeEntry(Base));
                break;
              }
              Base = cast<InsertElementInst>(Base)->getOperand(0);
            }
            FirstUsers.emplace_back(VU, ScalarTE);
            DemandedElts.push_back(APInt::getZero(FTy->getNumElements()));
            VecId = FirstUsers.size() - 1;
          } else {
            if (isFirstInsertElement(VU, cast<InsertElementInst>(It->first)))
              It->first = VU;
            VecId = std::distance(FirstUsers.begin(), It);
          }
          int InIdx = *InsertIdx;
          SmallVectorImpl<int> &Mask = ShuffleMasks[VecId][ScalarTE];
          if (Mask.empty())
            Mask.assign(FTy->getNumElements(), PoisonMaskElem);
          Mask[InIdx] = EU.Lane;
          DemandedElts[VecId].setBit(InIdx);
          continue;
        }
      }
    }

    // If we plan to rewrite the tree in a smaller type, we will need to sign
    // extend the extracted value back to the original type. Here, we account
    // for the extract and the added cost of the sign extend if needed.
    auto *VecTy = FixedVectorType::get(EU.Scalar->getType(), BundleWidth);
    TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
    auto *ScalarRoot = VectorizableTree[0]->Scalars[0];
    auto It = MinBWs.find(ScalarRoot);
    if (It != MinBWs.end()) {
      auto *MinTy = IntegerType::get(F->getContext(), It->second.first);
      unsigned Extend =
          It->second.second ? Instruction::SExt : Instruction::ZExt;
      VecTy = FixedVectorType::get(MinTy, BundleWidth);
      ExtractCost += TTI->getExtractWithExtendCost(Extend, EU.Scalar->getType(),
                                                   VecTy, EU.Lane);
    } else {
      ExtractCost += TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy,
                                             CostKind, EU.Lane);
    }
  }

  InstructionCost SpillCost = getSpillCost();
  Cost += SpillCost + ExtractCost;
  auto &&ResizeToVF = [this, &Cost](const TreeEntry *TE, ArrayRef<int> Mask,
                                    bool) {
    InstructionCost C = 0;
    unsigned VF = Mask.size();
    unsigned VecVF = TE->getVectorFactor();
    if (VF != VecVF &&
        (any_of(Mask, [VF](int Idx) { return Idx >= static_cast<int>(VF); }) ||
         (all_of(Mask,
                 [VF](int Idx) { return Idx < 2 * static_cast<int>(VF); }) &&
          !ShuffleVectorInst::isIdentityMask(Mask)))) {
      SmallVector<int> OrigMask(VecVF, PoisonMaskElem);
      std::copy(Mask.begin(), std::next(Mask.begin(), std::min(VF, VecVF)),
                OrigMask.begin());
      C = TTI->getShuffleCost(
          TTI::SK_PermuteSingleSrc,
          FixedVectorType::get(TE->getMainOp()->getType(), VecVF), OrigMask);
      LLVM_DEBUG(
          dbgs() << "SLP: Adding cost " << C
                 << " for final shuffle of insertelement external users.\n";
          TE->dump(); dbgs() << "SLP: Current total cost = " << Cost << "\n");
      Cost += C;
      return std::make_pair(TE, true);
    }
    return std::make_pair(TE, false);
  };
  // Calculate the cost of the reshuffled vectors, if any.
  for (int I = 0, E = FirstUsers.size(); I < E; ++I) {
    Value *Base = cast<Instruction>(FirstUsers[I].first)->getOperand(0);
    unsigned VF = ShuffleMasks[I].begin()->second.size();
    auto *FTy = FixedVectorType::get(
        cast<VectorType>(FirstUsers[I].first->getType())->getElementType(), VF);
    auto Vector = ShuffleMasks[I].takeVector();
    auto &&EstimateShufflesCost = [this, FTy,
                                   &Cost](ArrayRef<int> Mask,
                                          ArrayRef<const TreeEntry *> TEs) {
      assert((TEs.size() == 1 || TEs.size() == 2) &&
             "Expected exactly 1 or 2 tree entries.");
      if (TEs.size() == 1) {
        int Limit = 2 * Mask.size();
        if (!all_of(Mask, [Limit](int Idx) { return Idx < Limit; }) ||
            !ShuffleVectorInst::isIdentityMask(Mask)) {
          InstructionCost C =
              TTI->getShuffleCost(TTI::SK_PermuteSingleSrc, FTy, Mask);
          LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                            << " for final shuffle of insertelement "
                               "external users.\n";
                     TEs.front()->dump();
                     dbgs() << "SLP: Current total cost = " << Cost << "\n");
          Cost += C;
        }
      } else {
        InstructionCost C =
            TTI->getShuffleCost(TTI::SK_PermuteTwoSrc, FTy, Mask);
        LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                          << " for final shuffle of vector node and external "
                             "insertelement users.\n";
                   if (TEs.front()) { TEs.front()->dump(); } TEs.back()->dump();
                   dbgs() << "SLP: Current total cost = " << Cost << "\n");
        Cost += C;
      }
      return TEs.back();
    };
    (void)performExtractsShuffleAction<const TreeEntry>(
        MutableArrayRef(Vector.data(), Vector.size()), Base,
        [](const TreeEntry *E) { return E->getVectorFactor(); }, ResizeToVF,
        EstimateShufflesCost);
    InstructionCost InsertCost = TTI->getScalarizationOverhead(
        cast<FixedVectorType>(FirstUsers[I].first->getType()), DemandedElts[I],
        /*Insert*/ true, /*Extract*/ false, TTI::TCK_RecipThroughput);
    Cost -= InsertCost;
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

std::optional<TargetTransformInfo::ShuffleKind>
BoUpSLP::isGatherShuffledEntry(const TreeEntry *TE, ArrayRef<Value *> VL,
                               SmallVectorImpl<int> &Mask,
                               SmallVectorImpl<const TreeEntry *> &Entries) {
  Entries.clear();
  // No need to check for the topmost gather node.
  if (TE == VectorizableTree.front().get())
    return std::nullopt;
  Mask.assign(VL.size(), PoisonMaskElem);
  assert(TE->UserTreeIndices.size() == 1 &&
         "Expected only single user of the gather node.");
  // TODO: currently checking only for Scalars in the tree entry, need to count
  // reused elements too for better cost estimation.
  Instruction &UserInst =
      getLastInstructionInBundle(TE->UserTreeIndices.front().UserTE);
  BasicBlock *ParentBB = nullptr;
  // Main node of PHI entries keeps the correct order of operands/incoming
  // blocks.
  if (auto *PHI =
          dyn_cast<PHINode>(TE->UserTreeIndices.front().UserTE->getMainOp())) {
    ParentBB = PHI->getIncomingBlock(TE->UserTreeIndices.front().EdgeIdx);
  } else {
    ParentBB = UserInst.getParent();
  }
  auto *NodeUI = DT->getNode(ParentBB);
  assert(NodeUI && "Should only process reachable instructions");
  SmallPtrSet<Value *, 4> GatheredScalars(VL.begin(), VL.end());
  auto CheckOrdering = [&](Instruction *LastEI) {
    // Check if the user node of the TE comes after user node of EntryPtr,
    // otherwise EntryPtr depends on TE.
    // Gather nodes usually are not scheduled and inserted before their first
    // user node. So, instead of checking dependency between the gather nodes
    // themselves, we check the dependency between their user nodes.
    // If one user node comes before the second one, we cannot use the second
    // gather node as the source vector for the first gather node, because in
    // the list of instructions it will be emitted later.
    auto *EntryParent = LastEI->getParent();
    auto *NodeEUI = DT->getNode(EntryParent);
    if (!NodeEUI)
      return false;
    assert((NodeUI == NodeEUI) ==
               (NodeUI->getDFSNumIn() == NodeEUI->getDFSNumIn()) &&
           "Different nodes should have different DFS numbers");
    // Check the order of the gather nodes users.
    if (UserInst.getParent() != EntryParent &&
        (DT->dominates(NodeUI, NodeEUI) || !DT->dominates(NodeEUI, NodeUI)))
      return false;
    if (UserInst.getParent() == EntryParent && UserInst.comesBefore(LastEI))
      return false;
    return true;
  };
  // Find all tree entries used by the gathered values. If no common entries
  // found - not a shuffle.
  // Here we build a set of tree nodes for each gathered value and trying to
  // find the intersection between these sets. If we have at least one common
  // tree node for each gathered value - we have just a permutation of the
  // single vector. If we have 2 different sets, we're in situation where we
  // have a permutation of 2 input vectors.
  SmallVector<SmallPtrSet<const TreeEntry *, 4>> UsedTEs;
  DenseMap<Value *, int> UsedValuesEntry;
  for (Value *V : VL) {
    if (isConstant(V))
      continue;
    // Build a list of tree entries where V is used.
    SmallPtrSet<const TreeEntry *, 4> VToTEs;
    for (const TreeEntry *TEPtr : ValueToGatherNodes.find(V)->second) {
      if (TEPtr == TE)
        continue;
      assert(any_of(TEPtr->Scalars,
                    [&](Value *V) { return GatheredScalars.contains(V); }) &&
             "Must contain at least single gathered value.");
      assert(TEPtr->UserTreeIndices.size() == 1 &&
             "Expected only single user of the gather node.");
      PHINode *EntryPHI =
          dyn_cast<PHINode>(TEPtr->UserTreeIndices.front().UserTE->getMainOp());
      Instruction *EntryUserInst =
          EntryPHI ? nullptr
                   : &getLastInstructionInBundle(
                         TEPtr->UserTreeIndices.front().UserTE);
      if (&UserInst == EntryUserInst) {
        assert(!EntryPHI && "Unexpected phi node entry.");
        // If 2 gathers are operands of the same entry, compare operands
        // indices, use the earlier one as the base.
        if (TE->UserTreeIndices.front().UserTE ==
                TEPtr->UserTreeIndices.front().UserTE &&
            TE->UserTreeIndices.front().EdgeIdx <
                TEPtr->UserTreeIndices.front().EdgeIdx)
          continue;
      }
      // Check if the user node of the TE comes after user node of EntryPtr,
      // otherwise EntryPtr depends on TE.
      auto *EntryI =
          EntryPHI
              ? EntryPHI
                    ->getIncomingBlock(TEPtr->UserTreeIndices.front().EdgeIdx)
                    ->getTerminator()
              : EntryUserInst;
      if ((ParentBB != EntryI->getParent() ||
           TE->UserTreeIndices.front().EdgeIdx <
               TEPtr->UserTreeIndices.front().EdgeIdx ||
           TE->UserTreeIndices.front().UserTE !=
               TEPtr->UserTreeIndices.front().UserTE) &&
          !CheckOrdering(EntryI))
        continue;
      VToTEs.insert(TEPtr);
    }
    if (const TreeEntry *VTE = getTreeEntry(V)) {
      Instruction &EntryUserInst = getLastInstructionInBundle(VTE);
      if (&EntryUserInst == &UserInst || !CheckOrdering(&EntryUserInst))
        continue;
      VToTEs.insert(VTE);
    }
    if (VToTEs.empty())
      continue;
    if (UsedTEs.empty()) {
      // The first iteration, just insert the list of nodes to vector.
      UsedTEs.push_back(VToTEs);
      UsedValuesEntry.try_emplace(V, 0);
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
        // TODO: support multiple reshuffled nodes.
        if (UsedTEs.size() == 2)
          continue;
        UsedTEs.push_back(SavedVToTEs);
        Idx = UsedTEs.size() - 1;
      }
      UsedValuesEntry.try_emplace(V, Idx);
    }
  }

  if (UsedTEs.empty())
    return std::nullopt;

  unsigned VF = 0;
  if (UsedTEs.size() == 1) {
    // Keep the order to avoid non-determinism.
    SmallVector<const TreeEntry *> FirstEntries(UsedTEs.front().begin(),
                                                UsedTEs.front().end());
    sort(FirstEntries, [](const TreeEntry *TE1, const TreeEntry *TE2) {
      return TE1->Idx < TE2->Idx;
    });
    // Try to find the perfect match in another gather node at first.
    auto *It = find_if(FirstEntries, [=](const TreeEntry *EntryPtr) {
      return EntryPtr->isSame(VL) || EntryPtr->isSame(TE->Scalars);
    });
    if (It != FirstEntries.end() && (*It)->getVectorFactor() == VL.size()) {
      Entries.push_back(*It);
      std::iota(Mask.begin(), Mask.end(), 0);
      // Clear undef scalars.
      for (int I = 0, Sz = VL.size(); I < Sz; ++I)
        if (isa<PoisonValue>(VL[I]))
          Mask[I] = PoisonMaskElem;
      return TargetTransformInfo::SK_PermuteSingleSrc;
    }
    // No perfect match, just shuffle, so choose the first tree node from the
    // tree.
    Entries.push_back(FirstEntries.front());
  } else {
    // Try to find nodes with the same vector factor.
    assert(UsedTEs.size() == 2 && "Expected at max 2 permuted entries.");
    // Keep the order of tree nodes to avoid non-determinism.
    DenseMap<int, const TreeEntry *> VFToTE;
    for (const TreeEntry *TE : UsedTEs.front()) {
      unsigned VF = TE->getVectorFactor();
      auto It = VFToTE.find(VF);
      if (It != VFToTE.end()) {
        if (It->second->Idx > TE->Idx)
          It->getSecond() = TE;
        continue;
      }
      VFToTE.try_emplace(VF, TE);
    }
    // Same, keep the order to avoid non-determinism.
    SmallVector<const TreeEntry *> SecondEntries(UsedTEs.back().begin(),
                                                 UsedTEs.back().end());
    sort(SecondEntries, [](const TreeEntry *TE1, const TreeEntry *TE2) {
      return TE1->Idx < TE2->Idx;
    });
    for (const TreeEntry *TE : SecondEntries) {
      auto It = VFToTE.find(TE->getVectorFactor());
      if (It != VFToTE.end()) {
        VF = It->first;
        Entries.push_back(It->second);
        Entries.push_back(TE);
        break;
      }
    }
    // No 2 source vectors with the same vector factor - just choose 2 with max
    // index.
    if (Entries.empty()) {
      Entries.push_back(
          *std::max_element(UsedTEs.front().begin(), UsedTEs.front().end(),
                            [](const TreeEntry *TE1, const TreeEntry *TE2) {
                              return TE1->Idx < TE2->Idx;
                            }));
      Entries.push_back(SecondEntries.front());
      VF = std::max(Entries.front()->getVectorFactor(),
                    Entries.back()->getVectorFactor());
    }
  }

  bool IsSplatOrUndefs = isSplat(VL) || all_of(VL, UndefValue::classof);
  // Checks if the 2 PHIs are compatible in terms of high possibility to be
  // vectorized.
  auto AreCompatiblePHIs = [&](Value *V, Value *V1) {
    auto *PHI = cast<PHINode>(V);
    auto *PHI1 = cast<PHINode>(V1);
    // Check that all incoming values are compatible/from same parent (if they
    // are instructions).
    // The incoming values are compatible if they all are constants, or
    // instruction with the same/alternate opcodes from the same basic block.
    for (int I = 0, E = PHI->getNumIncomingValues(); I < E; ++I) {
      Value *In = PHI->getIncomingValue(I);
      Value *In1 = PHI1->getIncomingValue(I);
      if (isConstant(In) && isConstant(In1))
        continue;
      if (!getSameOpcode({In, In1}, *TLI).getOpcode())
        return false;
      if (cast<Instruction>(In)->getParent() !=
          cast<Instruction>(In1)->getParent())
        return false;
    }
    return true;
  };
  // Check if the value can be ignored during analysis for shuffled gathers.
  // We suppose it is better to ignore instruction, which do not form splats,
  // are not vectorized/not extractelements (these instructions will be handled
  // by extractelements processing) or may form vector node in future.
  auto MightBeIgnored = [=](Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    return I && !IsSplatOrUndefs && !ScalarToTreeEntry.count(I) &&
           !isVectorLikeInstWithConstOps(I) &&
           !areAllUsersVectorized(I, UserIgnoreList) && isSimple(I);
  };
  // Check that the neighbor instruction may form a full vector node with the
  // current instruction V. It is possible, if they have same/alternate opcode
  // and same parent basic block.
  auto NeighborMightBeIgnored = [&](Value *V, int Idx) {
    Value *V1 = VL[Idx];
    bool UsedInSameVTE = false;
    auto It = UsedValuesEntry.find(V1);
    if (It != UsedValuesEntry.end())
      UsedInSameVTE = It->second == UsedValuesEntry.find(V)->second;
    return V != V1 && MightBeIgnored(V1) && !UsedInSameVTE &&
           getSameOpcode({V, V1}, *TLI).getOpcode() &&
           cast<Instruction>(V)->getParent() ==
               cast<Instruction>(V1)->getParent() &&
           (!isa<PHINode>(V1) || AreCompatiblePHIs(V, V1));
  };
  // Build a shuffle mask for better cost estimation and vector emission.
  SmallBitVector UsedIdxs(Entries.size());
  SmallVector<std::pair<unsigned, int>> EntryLanes;
  for (int I = 0, E = VL.size(); I < E; ++I) {
    Value *V = VL[I];
    auto It = UsedValuesEntry.find(V);
    if (It == UsedValuesEntry.end())
      continue;
    // Do not try to shuffle scalars, if they are constants, or instructions
    // that can be vectorized as a result of the following vector build
    // vectorization.
    if (isConstant(V) || (MightBeIgnored(V) &&
                          ((I > 0 && NeighborMightBeIgnored(V, I - 1)) ||
                           (I != E - 1 && NeighborMightBeIgnored(V, I + 1)))))
      continue;
    unsigned Idx = It->second;
    EntryLanes.emplace_back(Idx, I);
    UsedIdxs.set(Idx);
  }
  // Iterate through all shuffled scalars and select entries, which can be used
  // for final shuffle.
  SmallVector<const TreeEntry *> TempEntries;
  for (unsigned I = 0, Sz = Entries.size(); I < Sz; ++I) {
    if (!UsedIdxs.test(I))
      continue;
    // Fix the entry number for the given scalar. If it is the first entry, set
    // Pair.first to 0, otherwise to 1 (currently select at max 2 nodes).
    // These indices are used when calculating final shuffle mask as the vector
    // offset.
    for (std::pair<unsigned, int> &Pair : EntryLanes)
      if (Pair.first == I)
        Pair.first = TempEntries.size();
    TempEntries.push_back(Entries[I]);
  }
  Entries.swap(TempEntries);
  if (EntryLanes.size() == Entries.size() && !VL.equals(TE->Scalars)) {
    // We may have here 1 or 2 entries only. If the number of scalars is equal
    // to the number of entries, no need to do the analysis, it is not very
    // profitable. Since VL is not the same as TE->Scalars, it means we already
    // have some shuffles before. Cut off not profitable case.
    Entries.clear();
    return std::nullopt;
  }
  // Build the final mask, check for the identity shuffle, if possible.
  bool IsIdentity = Entries.size() == 1;
  // Pair.first is the offset to the vector, while Pair.second is the index of
  // scalar in the list.
  for (const std::pair<unsigned, int> &Pair : EntryLanes) {
    Mask[Pair.second] = Pair.first * VF +
                        Entries[Pair.first]->findLaneForValue(VL[Pair.second]);
    IsIdentity &= Mask[Pair.second] == Pair.second;
  }
  switch (Entries.size()) {
  case 1:
    if (IsIdentity || EntryLanes.size() > 1 || VL.size() <= 2)
      return TargetTransformInfo::SK_PermuteSingleSrc;
    break;
  case 2:
    if (EntryLanes.size() > 2 || VL.size() <= 2)
      return TargetTransformInfo::SK_PermuteTwoSrc;
    break;
  default:
    break;
  }
  Entries.clear();
  return std::nullopt;
}

InstructionCost BoUpSLP::getGatherCost(ArrayRef<Value *> VL,
                                       bool ForPoisonSrc) const {
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
  constexpr TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  InstructionCost Cost;
  auto EstimateInsertCost = [&](unsigned I, Value *V) {
    if (!ForPoisonSrc)
      Cost +=
          TTI->getVectorInstrCost(Instruction::InsertElement, VecTy, CostKind,
                                  I, Constant::getNullValue(VecTy), V);
  };
  for (unsigned I = 0, E = VL.size(); I < E; ++I) {
    Value *V = VL[I];
    // No need to shuffle duplicates for constants.
    if ((ForPoisonSrc && isConstant(V)) || isa<UndefValue>(V)) {
      ShuffledElements.setBit(I);
      continue;
    }
    if (!UniqueElements.insert(V).second) {
      DuplicateNonConst = true;
      ShuffledElements.setBit(I);
      continue;
    }
    EstimateInsertCost(I, V);
  }
  if (ForPoisonSrc)
    Cost =
        TTI->getScalarizationOverhead(VecTy, ~ShuffledElements, /*Insert*/ true,
                                      /*Extract*/ false, CostKind);
  if (DuplicateNonConst)
    Cost +=
        TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, VecTy);
  return Cost;
}

// Perform operand reordering on the instructions in VL and return the reordered
// operands in Left and Right.
void BoUpSLP::reorderInputsAccordingToOpcode(
    ArrayRef<Value *> VL, SmallVectorImpl<Value *> &Left,
    SmallVectorImpl<Value *> &Right, const TargetLibraryInfo &TLI,
    const DataLayout &DL, ScalarEvolution &SE, const BoUpSLP &R) {
  if (VL.empty())
    return;
  VLOperands Ops(VL, TLI, DL, SE, R);
  // Reorder the operands in place.
  Ops.reorder();
  Left = Ops.getVL(0);
  Right = Ops.getVL(1);
}

Instruction &BoUpSLP::getLastInstructionInBundle(const TreeEntry *E) {
  auto &Res = EntryToLastInstruction.FindAndConstruct(E);
  if (Res.second)
    return *Res.second;
  // Get the basic block this bundle is in. All instructions in the bundle
  // should be in this block (except for extractelement-like instructions with
  // constant indeces).
  auto *Front = E->getMainOp();
  auto *BB = Front->getParent();
  assert(llvm::all_of(E->Scalars, [=](Value *V) -> bool {
    if (E->getOpcode() == Instruction::GetElementPtr &&
        !isa<GetElementPtrInst>(V))
      return true;
    auto *I = cast<Instruction>(V);
    return !E->isOpcodeOrAlt(I) || I->getParent() == BB ||
           isVectorLikeInstWithConstOps(I);
  }));

  auto FindLastInst = [&]() {
    Instruction *LastInst = Front;
    for (Value *V : E->Scalars) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        continue;
      if (LastInst->getParent() == I->getParent()) {
        if (LastInst->comesBefore(I))
          LastInst = I;
        continue;
      }
      assert(((E->getOpcode() == Instruction::GetElementPtr &&
               !isa<GetElementPtrInst>(I)) ||
              (isVectorLikeInstWithConstOps(LastInst) &&
               isVectorLikeInstWithConstOps(I))) &&
             "Expected vector-like or non-GEP in GEP node insts only.");
      if (!DT->isReachableFromEntry(LastInst->getParent())) {
        LastInst = I;
        continue;
      }
      if (!DT->isReachableFromEntry(I->getParent()))
        continue;
      auto *NodeA = DT->getNode(LastInst->getParent());
      auto *NodeB = DT->getNode(I->getParent());
      assert(NodeA && "Should only process reachable instructions");
      assert(NodeB && "Should only process reachable instructions");
      assert((NodeA == NodeB) ==
                 (NodeA->getDFSNumIn() == NodeB->getDFSNumIn()) &&
             "Different nodes should have different DFS numbers");
      if (NodeA->getDFSNumIn() < NodeB->getDFSNumIn())
        LastInst = I;
    }
    BB = LastInst->getParent();
    return LastInst;
  };

  auto FindFirstInst = [&]() {
    Instruction *FirstInst = Front;
    for (Value *V : E->Scalars) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        continue;
      if (FirstInst->getParent() == I->getParent()) {
        if (I->comesBefore(FirstInst))
          FirstInst = I;
        continue;
      }
      assert(((E->getOpcode() == Instruction::GetElementPtr &&
              !isa<GetElementPtrInst>(I)) ||
             (isVectorLikeInstWithConstOps(FirstInst) &&
              isVectorLikeInstWithConstOps(I))) &&
                 "Expected vector-like or non-GEP in GEP node insts only.");
      if (!DT->isReachableFromEntry(FirstInst->getParent())) {
        FirstInst = I;
        continue;
      }
      if (!DT->isReachableFromEntry(I->getParent()))
        continue;
      auto *NodeA = DT->getNode(FirstInst->getParent());
      auto *NodeB = DT->getNode(I->getParent());
      assert(NodeA && "Should only process reachable instructions");
      assert(NodeB && "Should only process reachable instructions");
      assert((NodeA == NodeB) ==
                 (NodeA->getDFSNumIn() == NodeB->getDFSNumIn()) &&
             "Different nodes should have different DFS numbers");
      if (NodeA->getDFSNumIn() > NodeB->getDFSNumIn())
        FirstInst = I;
    }
    return FirstInst;
  };

  // Set the insert point to the beginning of the basic block if the entry
  // should not be scheduled.
  if (doesNotNeedToSchedule(E->Scalars) ||
      (E->State != TreeEntry::NeedToGather &&
       all_of(E->Scalars, isVectorLikeInstWithConstOps))) {
    if ((E->getOpcode() == Instruction::GetElementPtr &&
         any_of(E->Scalars,
                [](Value *V) {
                  return !isa<GetElementPtrInst>(V) && isa<Instruction>(V);
                })) ||
        all_of(E->Scalars, [](Value *V) {
          return !isVectorLikeInstWithConstOps(V) && isUsedOutsideBlock(V);
        }))
      Res.second = FindLastInst();
    else
      Res.second = FindFirstInst();
    return *Res.second;
  }

  // Find the last instruction. The common case should be that BB has been
  // scheduled, and the last instruction is VL.back(). So we start with
  // VL.back() and iterate over schedule data until we reach the end of the
  // bundle. The end of the bundle is marked by null ScheduleData.
  if (BlocksSchedules.count(BB)) {
    Value *V = E->isOneOf(E->Scalars.back());
    if (doesNotNeedToBeScheduled(V))
      V = *find_if_not(E->Scalars, doesNotNeedToBeScheduled);
    auto *Bundle = BlocksSchedules[BB]->getScheduleData(V);
    if (Bundle && Bundle->isPartOfBundle())
      for (; Bundle; Bundle = Bundle->NextInBundle)
        if (Bundle->OpValue == Bundle->Inst)
          Res.second = Bundle->Inst;
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
  if (!Res.second)
    Res.second = FindLastInst();
  assert(Res.second && "Failed to find last instruction in bundle");
  return *Res.second;
}

void BoUpSLP::setInsertPointAfterBundle(const TreeEntry *E) {
  auto *Front = E->getMainOp();
  Instruction *LastInst = &getLastInstructionInBundle(E);
  assert(LastInst && "Failed to find last instruction in bundle");
  // If the instruction is PHI, set the insert point after all the PHIs.
  bool IsPHI = isa<PHINode>(LastInst);
  if (IsPHI)
    LastInst = LastInst->getParent()->getFirstNonPHI();
  if (IsPHI || (E->State != TreeEntry::NeedToGather &&
                doesNotNeedToSchedule(E->Scalars))) {
    Builder.SetInsertPoint(LastInst);
  } else {
    // Set the insertion point after the last instruction in the bundle. Set the
    // debug location to Front.
    Builder.SetInsertPoint(LastInst->getParent(),
                           std::next(LastInst->getIterator()));
  }
  Builder.SetCurrentDebugLocation(Front->getDebugLoc());
}

Value *BoUpSLP::gather(ArrayRef<Value *> VL, Value *Root) {
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
           getTreeEntry(Inst) ||
           (L && (!Root || L->isLoopInvariant(Root)) && L->contains(Inst))) &&
          PostponedIndices.insert(I).second)
        PostponedInsts.emplace_back(Inst, I);
  }

  auto &&CreateInsertElement = [this](Value *Vec, Value *V, unsigned Pos) {
    Vec = Builder.CreateInsertElement(Vec, V, Builder.getInt32(Pos));
    auto *InsElt = dyn_cast<InsertElementInst>(Vec);
    if (!InsElt)
      return Vec;
    GatherShuffleExtractSeq.insert(InsElt);
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
  Value *Vec = Root ? Root : PoisonValue::get(VecTy);
  SmallVector<int> NonConsts;
  // Insert constant values at first.
  for (int I = 0, E = VL.size(); I < E; ++I) {
    if (PostponedIndices.contains(I))
      continue;
    if (!isConstant(VL[I])) {
      NonConsts.push_back(I);
      continue;
    }
    if (Root) {
      if (!isa<UndefValue>(VL[I])) {
        NonConsts.push_back(I);
        continue;
      }
      if (isa<PoisonValue>(VL[I]))
        continue;
      if (auto *SV = dyn_cast<ShuffleVectorInst>(Root)) {
        if (SV->getMaskValue(I) == PoisonMaskElem)
          continue;
      }
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

/// Merges shuffle masks and emits final shuffle instruction, if required. It
/// supports shuffling of 2 input vectors. It implements lazy shuffles emission,
/// when the actual shuffle instruction is generated only if this is actually
/// required. Otherwise, the shuffle instruction emission is delayed till the
/// end of the process, to reduce the number of emitted instructions and further
/// analysis/transformations.
/// The class also will look through the previously emitted shuffle instructions
/// and properly mark indices in mask as undef.
/// For example, given the code
/// \code
/// %s1 = shufflevector <2 x ty> %0, poison, <1, 0>
/// %s2 = shufflevector <2 x ty> %1, poison, <1, 0>
/// \endcode
/// and if need to emit shuffle of %s1 and %s2 with mask <1, 0, 3, 2>, it will
/// look through %s1 and %s2 and emit
/// \code
/// %res = shufflevector <2 x ty> %0, %1, <0, 1, 2, 3>
/// \endcode
/// instead.
/// If 2 operands are of different size, the smallest one will be resized and
/// the mask recalculated properly.
/// For example, given the code
/// \code
/// %s1 = shufflevector <2 x ty> %0, poison, <1, 0, 1, 0>
/// %s2 = shufflevector <2 x ty> %1, poison, <1, 0, 1, 0>
/// \endcode
/// and if need to emit shuffle of %s1 and %s2 with mask <1, 0, 5, 4>, it will
/// look through %s1 and %s2 and emit
/// \code
/// %res = shufflevector <2 x ty> %0, %1, <0, 1, 2, 3>
/// \endcode
/// instead.
class BoUpSLP::ShuffleInstructionBuilder final : public BaseShuffleAnalysis {
  bool IsFinalized = false;
  /// Combined mask for all applied operands and masks. It is built during
  /// analysis and actual emission of shuffle vector instructions.
  SmallVector<int> CommonMask;
  /// List of operands for the shuffle vector instruction. It hold at max 2
  /// operands, if the 3rd is going to be added, the first 2 are combined into
  /// shuffle with \p CommonMask mask, the first operand sets to be the
  /// resulting shuffle and the second operand sets to be the newly added
  /// operand. The \p CommonMask is transformed in the proper way after that.
  SmallVector<Value *, 2> InVectors;
  IRBuilderBase &Builder;
  BoUpSLP &R;

  class ShuffleIRBuilder {
    IRBuilderBase &Builder;
    /// Holds all of the instructions that we gathered.
    SetVector<Instruction *> &GatherShuffleExtractSeq;
    /// A list of blocks that we are going to CSE.
    SetVector<BasicBlock *> &CSEBlocks;

  public:
    ShuffleIRBuilder(IRBuilderBase &Builder,
                     SetVector<Instruction *> &GatherShuffleExtractSeq,
                     SetVector<BasicBlock *> &CSEBlocks)
        : Builder(Builder), GatherShuffleExtractSeq(GatherShuffleExtractSeq),
          CSEBlocks(CSEBlocks) {}
    ~ShuffleIRBuilder() = default;
    /// Creates shufflevector for the 2 operands with the given mask.
    Value *createShuffleVector(Value *V1, Value *V2, ArrayRef<int> Mask) {
      Value *Vec = Builder.CreateShuffleVector(V1, V2, Mask);
      if (auto *I = dyn_cast<Instruction>(Vec)) {
        GatherShuffleExtractSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }
      return Vec;
    }
    /// Creates permutation of the single vector operand with the given mask, if
    /// it is not identity mask.
    Value *createShuffleVector(Value *V1, ArrayRef<int> Mask) {
      if (Mask.empty())
        return V1;
      unsigned VF = Mask.size();
      unsigned LocalVF = cast<FixedVectorType>(V1->getType())->getNumElements();
      if (VF == LocalVF && ShuffleVectorInst::isIdentityMask(Mask))
        return V1;
      Value *Vec = Builder.CreateShuffleVector(V1, Mask);
      if (auto *I = dyn_cast<Instruction>(Vec)) {
        GatherShuffleExtractSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }
      return Vec;
    }
    Value *createIdentity(Value *V) { return V; }
    Value *createPoison(Type *Ty, unsigned VF) {
      return PoisonValue::get(FixedVectorType::get(Ty, VF));
    }
    /// Resizes 2 input vector to match the sizes, if the they are not equal
    /// yet. The smallest vector is resized to the size of the larger vector.
    void resizeToMatch(Value *&V1, Value *&V2) {
      if (V1->getType() == V2->getType())
        return;
      int V1VF = cast<FixedVectorType>(V1->getType())->getNumElements();
      int V2VF = cast<FixedVectorType>(V2->getType())->getNumElements();
      int VF = std::max(V1VF, V2VF);
      int MinVF = std::min(V1VF, V2VF);
      SmallVector<int> IdentityMask(VF, PoisonMaskElem);
      std::iota(IdentityMask.begin(), std::next(IdentityMask.begin(), MinVF),
                0);
      Value *&Op = MinVF == V1VF ? V1 : V2;
      Op = Builder.CreateShuffleVector(Op, IdentityMask);
      if (auto *I = dyn_cast<Instruction>(Op)) {
        GatherShuffleExtractSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }
      if (MinVF == V1VF)
        V1 = Op;
      else
        V2 = Op;
    }
  };

  /// Smart shuffle instruction emission, walks through shuffles trees and
  /// tries to find the best matching vector for the actual shuffle
  /// instruction.
  Value *createShuffle(Value *V1, Value *V2, ArrayRef<int> Mask) {
    assert(V1 && "Expected at least one vector value.");
    ShuffleIRBuilder ShuffleBuilder(Builder, R.GatherShuffleExtractSeq,
                                    R.CSEBlocks);
    return BaseShuffleAnalysis::createShuffle<Value *>(V1, V2, Mask,
                                                       ShuffleBuilder);
  }

  /// Transforms mask \p CommonMask per given \p Mask to make proper set after
  /// shuffle emission.
  static void transformMaskAfterShuffle(MutableArrayRef<int> CommonMask,
                                        ArrayRef<int> Mask) {
    for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
      if (Mask[Idx] != PoisonMaskElem)
        CommonMask[Idx] = Idx;
  }

public:
  ShuffleInstructionBuilder(IRBuilderBase &Builder, BoUpSLP &R)
      : Builder(Builder), R(R) {}

  /// Adjusts extractelements after reusing them.
  Value *adjustExtracts(const TreeEntry *E, ArrayRef<int> Mask) {
    Value *VecBase = nullptr;
    for (int I = 0, Sz = Mask.size(); I < Sz; ++I) {
      int Idx = Mask[I];
      if (Idx == PoisonMaskElem)
        continue;
      auto *EI = cast<ExtractElementInst>(E->Scalars[I]);
      VecBase = EI->getVectorOperand();
      // If the only one use is vectorized - can delete the extractelement
      // itself.
      if (!EI->hasOneUse() || any_of(EI->users(), [&](User *U) {
            return !R.ScalarToTreeEntry.count(U);
          }))
        continue;
      R.eraseInstruction(EI);
    }
    return VecBase;
  }
  /// Checks if the specified entry \p E needs to be delayed because of its
  /// dependency nodes.
  Value *needToDelay(const TreeEntry *E, ArrayRef<const TreeEntry *> Deps) {
    // No need to delay emission if all deps are ready.
    if (all_of(Deps, [](const TreeEntry *TE) { return TE->VectorizedValue; }))
      return nullptr;
    // Postpone gather emission, will be emitted after the end of the
    // process to keep correct order.
    auto *VecTy = FixedVectorType::get(E->Scalars.front()->getType(),
                                       E->getVectorFactor());
    return Builder.CreateAlignedLoad(
        VecTy, PoisonValue::get(PointerType::getUnqual(VecTy->getContext())),
        MaybeAlign());
  }
  /// Adds 2 input vectors and the mask for their shuffling.
  void add(Value *V1, Value *V2, ArrayRef<int> Mask) {
    assert(V1 && V2 && !Mask.empty() && "Expected non-empty input vectors.");
    if (InVectors.empty()) {
      InVectors.push_back(V1);
      InVectors.push_back(V2);
      CommonMask.assign(Mask.begin(), Mask.end());
      return;
    }
    Value *Vec = InVectors.front();
    if (InVectors.size() == 2) {
      Vec = createShuffle(Vec, InVectors.back(), CommonMask);
      transformMaskAfterShuffle(CommonMask, CommonMask);
    } else if (cast<FixedVectorType>(Vec->getType())->getNumElements() !=
               Mask.size()) {
      Vec = createShuffle(Vec, nullptr, CommonMask);
      transformMaskAfterShuffle(CommonMask, CommonMask);
    }
    V1 = createShuffle(V1, V2, Mask);
    for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
      if (Mask[Idx] != PoisonMaskElem)
        CommonMask[Idx] = Idx + Sz;
    InVectors.front() = Vec;
    if (InVectors.size() == 2)
      InVectors.back() = V1;
    else
      InVectors.push_back(V1);
  }
  /// Adds another one input vector and the mask for the shuffling.
  void add(Value *V1, ArrayRef<int> Mask) {
    if (InVectors.empty()) {
      if (!isa<FixedVectorType>(V1->getType())) {
        V1 = createShuffle(V1, nullptr, CommonMask);
        CommonMask.assign(Mask.size(), PoisonMaskElem);
        transformMaskAfterShuffle(CommonMask, Mask);
      }
      InVectors.push_back(V1);
      CommonMask.assign(Mask.begin(), Mask.end());
      return;
    }
    const auto *It = find(InVectors, V1);
    if (It == InVectors.end()) {
      if (InVectors.size() == 2 ||
          InVectors.front()->getType() != V1->getType() ||
          !isa<FixedVectorType>(V1->getType())) {
        Value *V = InVectors.front();
        if (InVectors.size() == 2) {
          V = createShuffle(InVectors.front(), InVectors.back(), CommonMask);
          transformMaskAfterShuffle(CommonMask, CommonMask);
        } else if (cast<FixedVectorType>(V->getType())->getNumElements() !=
                   CommonMask.size()) {
          V = createShuffle(InVectors.front(), nullptr, CommonMask);
          transformMaskAfterShuffle(CommonMask, CommonMask);
        }
        for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
          if (CommonMask[Idx] == PoisonMaskElem && Mask[Idx] != PoisonMaskElem)
            CommonMask[Idx] =
                V->getType() != V1->getType()
                    ? Idx + Sz
                    : Mask[Idx] + cast<FixedVectorType>(V1->getType())
                                      ->getNumElements();
        if (V->getType() != V1->getType())
          V1 = createShuffle(V1, nullptr, Mask);
        InVectors.front() = V;
        if (InVectors.size() == 2)
          InVectors.back() = V1;
        else
          InVectors.push_back(V1);
        return;
      }
      // Check if second vector is required if the used elements are already
      // used from the first one.
      for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
        if (Mask[Idx] != PoisonMaskElem && CommonMask[Idx] == PoisonMaskElem) {
          InVectors.push_back(V1);
          break;
        }
    }
    int VF = CommonMask.size();
    if (auto *FTy = dyn_cast<FixedVectorType>(V1->getType()))
      VF = FTy->getNumElements();
    for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
      if (Mask[Idx] != PoisonMaskElem && CommonMask[Idx] == PoisonMaskElem)
        CommonMask[Idx] = Mask[Idx] + (It == InVectors.begin() ? 0 : VF);
  }
  /// Adds another one input vector and the mask for the shuffling.
  void addOrdered(Value *V1, ArrayRef<unsigned> Order) {
    SmallVector<int> NewMask;
    inversePermutation(Order, NewMask);
    add(V1, NewMask);
  }
  Value *gather(ArrayRef<Value *> VL, Value *Root = nullptr) {
    return R.gather(VL, Root);
  }
  Value *createFreeze(Value *V) { return Builder.CreateFreeze(V); }
  /// Finalize emission of the shuffles.
  /// \param Action the action (if any) to be performed before final applying of
  /// the \p ExtMask mask.
  Value *
  finalize(ArrayRef<int> ExtMask, unsigned VF = 0,
           function_ref<void(Value *&, SmallVectorImpl<int> &)> Action = {}) {
    IsFinalized = true;
    if (Action) {
      Value *Vec = InVectors.front();
      if (InVectors.size() == 2) {
        Vec = createShuffle(Vec, InVectors.back(), CommonMask);
        InVectors.pop_back();
      } else {
        Vec = createShuffle(Vec, nullptr, CommonMask);
      }
      for (unsigned Idx = 0, Sz = CommonMask.size(); Idx < Sz; ++Idx)
        if (CommonMask[Idx] != PoisonMaskElem)
          CommonMask[Idx] = Idx;
      assert(VF > 0 &&
             "Expected vector length for the final value before action.");
      unsigned VecVF = cast<FixedVectorType>(Vec->getType())->getNumElements();
      if (VecVF < VF) {
        SmallVector<int> ResizeMask(VF, PoisonMaskElem);
        std::iota(ResizeMask.begin(), std::next(ResizeMask.begin(), VecVF), 0);
        Vec = createShuffle(Vec, nullptr, ResizeMask);
      }
      Action(Vec, CommonMask);
      InVectors.front() = Vec;
    }
    if (!ExtMask.empty()) {
      if (CommonMask.empty()) {
        CommonMask.assign(ExtMask.begin(), ExtMask.end());
      } else {
        SmallVector<int> NewMask(ExtMask.size(), PoisonMaskElem);
        for (int I = 0, Sz = ExtMask.size(); I < Sz; ++I) {
          if (ExtMask[I] == PoisonMaskElem)
            continue;
          NewMask[I] = CommonMask[ExtMask[I]];
        }
        CommonMask.swap(NewMask);
      }
    }
    if (CommonMask.empty()) {
      assert(InVectors.size() == 1 && "Expected only one vector with no mask");
      return InVectors.front();
    }
    if (InVectors.size() == 2)
      return createShuffle(InVectors.front(), InVectors.back(), CommonMask);
    return createShuffle(InVectors.front(), nullptr, CommonMask);
  }

  ~ShuffleInstructionBuilder() {
    assert((IsFinalized || CommonMask.empty()) &&
           "Shuffle construction must be finalized.");
  }
};

Value *BoUpSLP::vectorizeOperand(TreeEntry *E, unsigned NodeIdx) {
  ValueList &VL = E->getOperand(NodeIdx);
  if (E->State == TreeEntry::PossibleStridedVectorize &&
      !E->ReorderIndices.empty()) {
    SmallVector<int> Mask(E->ReorderIndices.begin(), E->ReorderIndices.end());
    reorderScalars(VL, Mask);
  }
  const unsigned VF = VL.size();
  InstructionsState S = getSameOpcode(VL, *TLI);
  // Special processing for GEPs bundle, which may include non-gep values.
  if (!S.getOpcode() && VL.front()->getType()->isPointerTy()) {
    const auto *It =
        find_if(VL, [](Value *V) { return isa<GetElementPtrInst>(V); });
    if (It != VL.end())
      S = getSameOpcode(*It, *TLI);
  }
  if (S.getOpcode()) {
    if (TreeEntry *VE = getTreeEntry(S.OpValue);
        VE && VE->isSame(VL) &&
        (any_of(VE->UserTreeIndices,
                [E, NodeIdx](const EdgeInfo &EI) {
                  return EI.UserTE == E && EI.EdgeIdx == NodeIdx;
                }) ||
         any_of(VectorizableTree,
                [E, NodeIdx, VE](const std::unique_ptr<TreeEntry> &TE) {
                  return TE->isOperandGatherNode({E, NodeIdx}) &&
                         VE->isSame(TE->Scalars);
                }))) {
      auto FinalShuffle = [&](Value *V, ArrayRef<int> Mask) {
        ShuffleInstructionBuilder ShuffleBuilder(Builder, *this);
        ShuffleBuilder.add(V, Mask);
        return ShuffleBuilder.finalize(std::nullopt);
      };
      Value *V = vectorizeTree(VE);
      if (VF != cast<FixedVectorType>(V->getType())->getNumElements()) {
        if (!VE->ReuseShuffleIndices.empty()) {
          // Reshuffle to get only unique values.
          // If some of the scalars are duplicated in the vectorization
          // tree entry, we do not vectorize them but instead generate a
          // mask for the reuses. But if there are several users of the
          // same entry, they may have different vectorization factors.
          // This is especially important for PHI nodes. In this case, we
          // need to adapt the resulting instruction for the user
          // vectorization factor and have to reshuffle it again to take
          // only unique elements of the vector. Without this code the
          // function incorrectly returns reduced vector instruction with
          // the same elements, not with the unique ones.

          // block:
          // %phi = phi <2 x > { .., %entry} {%shuffle, %block}
          // %2 = shuffle <2 x > %phi, poison, <4 x > <1, 1, 0, 0>
          // ... (use %2)
          // %shuffle = shuffle <2 x> %2, poison, <2 x> {2, 0}
          // br %block
          SmallVector<int> UniqueIdxs(VF, PoisonMaskElem);
          SmallSet<int, 4> UsedIdxs;
          int Pos = 0;
          for (int Idx : VE->ReuseShuffleIndices) {
            if (Idx != static_cast<int>(VF) && Idx != PoisonMaskElem &&
                UsedIdxs.insert(Idx).second)
              UniqueIdxs[Idx] = Pos;
            ++Pos;
          }
          assert(VF >= UsedIdxs.size() && "Expected vectorization factor "
                                          "less than original vector size.");
          UniqueIdxs.append(VF - UsedIdxs.size(), PoisonMaskElem);
          V = FinalShuffle(V, UniqueIdxs);
        } else {
          assert(VF < cast<FixedVectorType>(V->getType())->getNumElements() &&
                 "Expected vectorization factor less "
                 "than original vector size.");
          SmallVector<int> UniformMask(VF, 0);
          std::iota(UniformMask.begin(), UniformMask.end(), 0);
          V = FinalShuffle(V, UniformMask);
        }
      }
      // Need to update the operand gather node, if actually the operand is not a
      // vectorized node, but the buildvector/gather node, which matches one of
      // the vectorized nodes.
      if (find_if(VE->UserTreeIndices, [&](const EdgeInfo &EI) {
            return EI.UserTE == E && EI.EdgeIdx == NodeIdx;
          }) == VE->UserTreeIndices.end()) {
        auto *It = find_if(
            VectorizableTree, [&](const std::unique_ptr<TreeEntry> &TE) {
              return TE->State == TreeEntry::NeedToGather &&
                     TE->UserTreeIndices.front().UserTE == E &&
                     TE->UserTreeIndices.front().EdgeIdx == NodeIdx;
            });
        assert(It != VectorizableTree.end() && "Expected gather node operand.");
        (*It)->VectorizedValue = V;
      }
      return V;
    }
  }

  // Find the corresponding gather entry and vectorize it.
  // Allows to be more accurate with tree/graph transformations, checks for the
  // correctness of the transformations in many cases.
  auto *I = find_if(VectorizableTree,
                    [E, NodeIdx](const std::unique_ptr<TreeEntry> &TE) {
                      return TE->isOperandGatherNode({E, NodeIdx});
                    });
  assert(I != VectorizableTree.end() && "Gather node is not in the graph.");
  assert(I->get()->UserTreeIndices.size() == 1 &&
         "Expected only single user for the gather node.");
  assert(I->get()->isSame(VL) && "Expected same list of scalars.");
  IRBuilder<>::InsertPointGuard Guard(Builder);
  if (E->getOpcode() != Instruction::InsertElement &&
      E->getOpcode() != Instruction::PHI) {
    Instruction *LastInst = &getLastInstructionInBundle(E);
    assert(LastInst && "Failed to find last instruction in bundle");
    Builder.SetInsertPoint(LastInst);
  }
  return vectorizeTree(I->get());
}

template <typename BVTy, typename ResTy, typename... Args>
ResTy BoUpSLP::processBuildVector(const TreeEntry *E, Args &...Params) {
  assert(E->State == TreeEntry::NeedToGather && "Expected gather node.");
  unsigned VF = E->getVectorFactor();

  bool NeedFreeze = false;
  SmallVector<int> ReuseShuffleIndicies(E->ReuseShuffleIndices.begin(),
                                        E->ReuseShuffleIndices.end());
  SmallVector<Value *> GatheredScalars(E->Scalars.begin(), E->Scalars.end());
  // Build a mask out of the reorder indices and reorder scalars per this
  // mask.
  SmallVector<int> ReorderMask;
  inversePermutation(E->ReorderIndices, ReorderMask);
  if (!ReorderMask.empty())
    reorderScalars(GatheredScalars, ReorderMask);
  auto FindReusedSplat = [&](SmallVectorImpl<int> &Mask) {
    if (!isSplat(E->Scalars) || none_of(E->Scalars, [](Value *V) {
          return isa<UndefValue>(V) && !isa<PoisonValue>(V);
        }))
      return false;
    TreeEntry *UserTE = E->UserTreeIndices.back().UserTE;
    unsigned EdgeIdx = E->UserTreeIndices.back().EdgeIdx;
    if (UserTE->getNumOperands() != 2)
      return false;
    auto *It =
        find_if(VectorizableTree, [=](const std::unique_ptr<TreeEntry> &TE) {
          return find_if(TE->UserTreeIndices, [=](const EdgeInfo &EI) {
                   return EI.UserTE == UserTE && EI.EdgeIdx != EdgeIdx;
                 }) != TE->UserTreeIndices.end();
        });
    if (It == VectorizableTree.end())
      return false;
    unsigned I =
        *find_if_not(Mask, [](int Idx) { return Idx == PoisonMaskElem; });
    int Sz = Mask.size();
    if (all_of(Mask, [Sz](int Idx) { return Idx < 2 * Sz; }) &&
        ShuffleVectorInst::isIdentityMask(Mask))
      std::iota(Mask.begin(), Mask.end(), 0);
    else
      std::fill(Mask.begin(), Mask.end(), I);
    return true;
  };
  BVTy ShuffleBuilder(Params...);
  ResTy Res = ResTy();
  SmallVector<int> Mask;
  SmallVector<int> ExtractMask;
  std::optional<TargetTransformInfo::ShuffleKind> ExtractShuffle;
  std::optional<TargetTransformInfo::ShuffleKind> GatherShuffle;
  SmallVector<const TreeEntry *> Entries;
  Type *ScalarTy = GatheredScalars.front()->getType();
  if (!all_of(GatheredScalars, UndefValue::classof)) {
    // Check for gathered extracts.
    ExtractShuffle = tryToGatherExtractElements(GatheredScalars, ExtractMask);
    SmallVector<Value *> IgnoredVals;
    if (UserIgnoreList)
      IgnoredVals.assign(UserIgnoreList->begin(), UserIgnoreList->end());
    bool Resized = false;
    if (Value *VecBase = ShuffleBuilder.adjustExtracts(E, ExtractMask))
      if (auto *VecBaseTy = dyn_cast<FixedVectorType>(VecBase->getType()))
        if (VF == VecBaseTy->getNumElements() && GatheredScalars.size() != VF) {
          Resized = true;
          GatheredScalars.append(VF - GatheredScalars.size(),
                                 PoisonValue::get(ScalarTy));
        }
    // Gather extracts after we check for full matched gathers only.
    if (ExtractShuffle || E->getOpcode() != Instruction::Load ||
        E->isAltShuffle() ||
        all_of(E->Scalars, [this](Value *V) { return getTreeEntry(V); }) ||
        isSplat(E->Scalars) ||
        (E->Scalars != GatheredScalars && GatheredScalars.size() <= 2)) {
      GatherShuffle = isGatherShuffledEntry(E, GatheredScalars, Mask, Entries);
    }
    if (GatherShuffle) {
      if (Value *Delayed = ShuffleBuilder.needToDelay(E, Entries)) {
        // Delay emission of gathers which are not ready yet.
        PostponedGathers.insert(E);
        // Postpone gather emission, will be emitted after the end of the
        // process to keep correct order.
        return Delayed;
      }
      assert((Entries.size() == 1 || Entries.size() == 2) &&
             "Expected shuffle of 1 or 2 entries.");
      if (*GatherShuffle == TTI::SK_PermuteSingleSrc &&
          Entries.front()->isSame(E->Scalars)) {
        // Perfect match in the graph, will reuse the previously vectorized
        // node. Cost is 0.
        LLVM_DEBUG(
            dbgs()
            << "SLP: perfect diamond match for gather bundle "
            << shortBundleName(E->Scalars) << ".\n");
        // Restore the mask for previous partially matched values.
        if (Entries.front()->ReorderIndices.empty() &&
            ((Entries.front()->ReuseShuffleIndices.empty() &&
              E->Scalars.size() == Entries.front()->Scalars.size()) ||
             (E->Scalars.size() ==
              Entries.front()->ReuseShuffleIndices.size()))) {
          std::iota(Mask.begin(), Mask.end(), 0);
        } else {
          for (auto [I, V] : enumerate(E->Scalars)) {
            if (isa<PoisonValue>(V)) {
              Mask[I] = PoisonMaskElem;
              continue;
            }
            Mask[I] = Entries.front()->findLaneForValue(V);
          }
        }
        ShuffleBuilder.add(Entries.front()->VectorizedValue, Mask);
        Res = ShuffleBuilder.finalize(E->getCommonMask());
        return Res;
      }
      if (!Resized) {
        unsigned VF1 = Entries.front()->getVectorFactor();
        unsigned VF2 = Entries.back()->getVectorFactor();
        if ((VF == VF1 || VF == VF2) && GatheredScalars.size() != VF)
          GatheredScalars.append(VF - GatheredScalars.size(),
                                 PoisonValue::get(ScalarTy));
      }
      // Remove shuffled elements from list of gathers.
      for (int I = 0, Sz = Mask.size(); I < Sz; ++I) {
        if (Mask[I] != PoisonMaskElem)
          GatheredScalars[I] = PoisonValue::get(ScalarTy);
      }
    }
  }
  auto TryPackScalars = [&](SmallVectorImpl<Value *> &Scalars,
                            SmallVectorImpl<int> &ReuseMask,
                            bool IsRootPoison) {
    // For splats with can emit broadcasts instead of gathers, so try to find
    // such sequences.
    bool IsSplat = IsRootPoison && isSplat(Scalars) &&
                   (Scalars.size() > 2 || Scalars.front() == Scalars.back());
    Scalars.append(VF - Scalars.size(), PoisonValue::get(ScalarTy));
    SmallVector<int> UndefPos;
    DenseMap<Value *, unsigned> UniquePositions;
    // Gather unique non-const values and all constant values.
    // For repeated values, just shuffle them.
    int NumNonConsts = 0;
    int SinglePos = 0;
    for (auto [I, V] : enumerate(Scalars)) {
      if (isa<UndefValue>(V)) {
        if (!isa<PoisonValue>(V)) {
          ReuseMask[I] = I;
          UndefPos.push_back(I);
        }
        continue;
      }
      if (isConstant(V)) {
        ReuseMask[I] = I;
        continue;
      }
      ++NumNonConsts;
      SinglePos = I;
      Value *OrigV = V;
      Scalars[I] = PoisonValue::get(ScalarTy);
      if (IsSplat) {
        Scalars.front() = OrigV;
        ReuseMask[I] = 0;
      } else {
        const auto Res = UniquePositions.try_emplace(OrigV, I);
        Scalars[Res.first->second] = OrigV;
        ReuseMask[I] = Res.first->second;
      }
    }
    if (NumNonConsts == 1) {
      // Restore single insert element.
      if (IsSplat) {
        ReuseMask.assign(VF, PoisonMaskElem);
        std::swap(Scalars.front(), Scalars[SinglePos]);
        if (!UndefPos.empty() && UndefPos.front() == 0)
          Scalars.front() = UndefValue::get(ScalarTy);
      }
      ReuseMask[SinglePos] = SinglePos;
    } else if (!UndefPos.empty() && IsSplat) {
      // For undef values, try to replace them with the simple broadcast.
      // We can do it if the broadcasted value is guaranteed to be
      // non-poisonous, or by freezing the incoming scalar value first.
      auto *It = find_if(Scalars, [this, E](Value *V) {
        return !isa<UndefValue>(V) &&
               (getTreeEntry(V) || isGuaranteedNotToBePoison(V) ||
                (E->UserTreeIndices.size() == 1 &&
                 any_of(V->uses(), [E](const Use &U) {
                   // Check if the value already used in the same operation in
                   // one of the nodes already.
                   return E->UserTreeIndices.front().EdgeIdx !=
                              U.getOperandNo() &&
                          is_contained(
                              E->UserTreeIndices.front().UserTE->Scalars,
                              U.getUser());
                 })));
      });
      if (It != Scalars.end()) {
        // Replace undefs by the non-poisoned scalars and emit broadcast.
        int Pos = std::distance(Scalars.begin(), It);
        for (int I : UndefPos) {
          // Set the undef position to the non-poisoned scalar.
          ReuseMask[I] = Pos;
          // Replace the undef by the poison, in the mask it is replaced by
          // non-poisoned scalar already.
          if (I != Pos)
            Scalars[I] = PoisonValue::get(ScalarTy);
        }
      } else {
        // Replace undefs by the poisons, emit broadcast and then emit
        // freeze.
        for (int I : UndefPos) {
          ReuseMask[I] = PoisonMaskElem;
          if (isa<UndefValue>(Scalars[I]))
            Scalars[I] = PoisonValue::get(ScalarTy);
        }
        NeedFreeze = true;
      }
    }
  };
  if (ExtractShuffle || GatherShuffle) {
    bool IsNonPoisoned = true;
    bool IsUsedInExpr = false;
    Value *Vec1 = nullptr;
    if (ExtractShuffle) {
      // Gather of extractelements can be represented as just a shuffle of
      // a single/two vectors the scalars are extracted from.
      // Find input vectors.
      Value *Vec2 = nullptr;
      for (unsigned I = 0, Sz = ExtractMask.size(); I < Sz; ++I) {
        if (ExtractMask[I] == PoisonMaskElem ||
            (!Mask.empty() && Mask[I] != PoisonMaskElem)) {
          ExtractMask[I] = PoisonMaskElem;
          continue;
        }
        if (isa<UndefValue>(E->Scalars[I]))
          continue;
        auto *EI = cast<ExtractElementInst>(E->Scalars[I]);
        if (!Vec1) {
          Vec1 = EI->getVectorOperand();
        } else if (Vec1 != EI->getVectorOperand()) {
          assert((!Vec2 || Vec2 == EI->getVectorOperand()) &&
                 "Expected only 1 or 2 vectors shuffle.");
          Vec2 = EI->getVectorOperand();
        }
      }
      if (Vec2) {
        IsNonPoisoned &=
            isGuaranteedNotToBePoison(Vec1) && isGuaranteedNotToBePoison(Vec2);
        ShuffleBuilder.add(Vec1, Vec2, ExtractMask);
      } else if (Vec1) {
        IsUsedInExpr = FindReusedSplat(ExtractMask);
        ShuffleBuilder.add(Vec1, ExtractMask);
        IsNonPoisoned &= isGuaranteedNotToBePoison(Vec1);
      } else {
        ShuffleBuilder.add(PoisonValue::get(FixedVectorType::get(
                               ScalarTy, GatheredScalars.size())),
                           ExtractMask);
      }
    }
    if (GatherShuffle) {
      if (Entries.size() == 1) {
        IsUsedInExpr = FindReusedSplat(Mask);
        ShuffleBuilder.add(Entries.front()->VectorizedValue, Mask);
        IsNonPoisoned &=
            isGuaranteedNotToBePoison(Entries.front()->VectorizedValue);
      } else {
        ShuffleBuilder.add(Entries.front()->VectorizedValue,
                           Entries.back()->VectorizedValue, Mask);
        IsNonPoisoned &=
            isGuaranteedNotToBePoison(Entries.front()->VectorizedValue) &&
            isGuaranteedNotToBePoison(Entries.back()->VectorizedValue);
      }
    }
    // Try to figure out best way to combine values: build a shuffle and insert
    // elements or just build several shuffles.
    // Insert non-constant scalars.
    SmallVector<Value *> NonConstants(GatheredScalars);
    int EMSz = ExtractMask.size();
    int MSz = Mask.size();
    // Try to build constant vector and shuffle with it only if currently we
    // have a single permutation and more than 1 scalar constants.
    bool IsSingleShuffle = !ExtractShuffle || !GatherShuffle;
    bool IsIdentityShuffle =
        (ExtractShuffle.value_or(TTI::SK_PermuteTwoSrc) ==
             TTI::SK_PermuteSingleSrc &&
         none_of(ExtractMask, [&](int I) { return I >= EMSz; }) &&
         ShuffleVectorInst::isIdentityMask(ExtractMask)) ||
        (GatherShuffle.value_or(TTI::SK_PermuteTwoSrc) ==
             TTI::SK_PermuteSingleSrc &&
         none_of(Mask, [&](int I) { return I >= MSz; }) &&
         ShuffleVectorInst::isIdentityMask(Mask));
    bool EnoughConstsForShuffle =
        IsSingleShuffle &&
        (none_of(GatheredScalars,
                 [](Value *V) {
                   return isa<UndefValue>(V) && !isa<PoisonValue>(V);
                 }) ||
         any_of(GatheredScalars,
                [](Value *V) {
                  return isa<Constant>(V) && !isa<UndefValue>(V);
                })) &&
        (!IsIdentityShuffle ||
         (GatheredScalars.size() == 2 &&
          any_of(GatheredScalars,
                 [](Value *V) { return !isa<UndefValue>(V); })) ||
         count_if(GatheredScalars, [](Value *V) {
           return isa<Constant>(V) && !isa<PoisonValue>(V);
         }) > 1);
    // NonConstants array contains just non-constant values, GatheredScalars
    // contains only constant to build final vector and then shuffle.
    for (int I = 0, Sz = GatheredScalars.size(); I < Sz; ++I) {
      if (EnoughConstsForShuffle && isa<Constant>(GatheredScalars[I]))
        NonConstants[I] = PoisonValue::get(ScalarTy);
      else
        GatheredScalars[I] = PoisonValue::get(ScalarTy);
    }
    // Generate constants for final shuffle and build a mask for them.
    if (!all_of(GatheredScalars, PoisonValue::classof)) {
      SmallVector<int> BVMask(GatheredScalars.size(), PoisonMaskElem);
      TryPackScalars(GatheredScalars, BVMask, /*IsRootPoison=*/true);
      Value *BV = ShuffleBuilder.gather(GatheredScalars);
      ShuffleBuilder.add(BV, BVMask);
    }
    if (all_of(NonConstants, [=](Value *V) {
          return isa<PoisonValue>(V) ||
                 (IsSingleShuffle && ((IsIdentityShuffle &&
                  IsNonPoisoned) || IsUsedInExpr) && isa<UndefValue>(V));
        }))
      Res = ShuffleBuilder.finalize(E->ReuseShuffleIndices);
    else
      Res = ShuffleBuilder.finalize(
          E->ReuseShuffleIndices, E->Scalars.size(),
          [&](Value *&Vec, SmallVectorImpl<int> &Mask) {
            TryPackScalars(NonConstants, Mask, /*IsRootPoison=*/false);
            Vec = ShuffleBuilder.gather(NonConstants, Vec);
          });
  } else if (!allConstant(GatheredScalars)) {
    // Gather unique scalars and all constants.
    SmallVector<int> ReuseMask(GatheredScalars.size(), PoisonMaskElem);
    TryPackScalars(GatheredScalars, ReuseMask, /*IsRootPoison=*/true);
    Value *BV = ShuffleBuilder.gather(GatheredScalars);
    ShuffleBuilder.add(BV, ReuseMask);
    Res = ShuffleBuilder.finalize(E->ReuseShuffleIndices);
  } else {
    // Gather all constants.
    SmallVector<int> Mask(E->Scalars.size(), PoisonMaskElem);
    for (auto [I, V] : enumerate(E->Scalars)) {
      if (!isa<PoisonValue>(V))
        Mask[I] = I;
    }
    Value *BV = ShuffleBuilder.gather(E->Scalars);
    ShuffleBuilder.add(BV, Mask);
    Res = ShuffleBuilder.finalize(E->ReuseShuffleIndices);
  }

  if (NeedFreeze)
    Res = ShuffleBuilder.createFreeze(Res);
  return Res;
}

Value *BoUpSLP::createBuildVector(const TreeEntry *E) {
  return processBuildVector<ShuffleInstructionBuilder, Value *>(E, Builder,
                                                                *this);
}

Value *BoUpSLP::vectorizeTree(TreeEntry *E) {
  IRBuilder<>::InsertPointGuard Guard(Builder);

  if (E->VectorizedValue) {
    LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *E->Scalars[0] << ".\n");
    return E->VectorizedValue;
  }

  if (E->State == TreeEntry::NeedToGather) {
    if (E->getMainOp() && E->Idx == 0)
      setInsertPointAfterBundle(E);
    Value *Vec = createBuildVector(E);
    E->VectorizedValue = Vec;
    return Vec;
  }

  auto FinalShuffle = [&](Value *V, const TreeEntry *E) {
    ShuffleInstructionBuilder ShuffleBuilder(Builder, *this);
    if (E->getOpcode() == Instruction::Store) {
      ArrayRef<int> Mask =
          ArrayRef(reinterpret_cast<const int *>(E->ReorderIndices.begin()),
                   E->ReorderIndices.size());
      ShuffleBuilder.add(V, Mask);
    } else if (E->State == TreeEntry::PossibleStridedVectorize) {
      ShuffleBuilder.addOrdered(V, std::nullopt);
    } else {
      ShuffleBuilder.addOrdered(V, E->ReorderIndices);
    }
    return ShuffleBuilder.finalize(E->ReuseShuffleIndices);
  };

  assert((E->State == TreeEntry::Vectorize ||
          E->State == TreeEntry::ScatterVectorize ||
          E->State == TreeEntry::PossibleStridedVectorize) &&
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
      assert((E->ReorderIndices.empty() ||
              E != VectorizableTree.front().get() ||
              !E->UserTreeIndices.empty()) &&
             "PHI reordering is free.");
      auto *PH = cast<PHINode>(VL0);
      Builder.SetInsertPoint(PH->getParent()->getFirstNonPHI());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());
      PHINode *NewPhi = Builder.CreatePHI(VecTy, PH->getNumIncomingValues());
      Value *V = NewPhi;

      // Adjust insertion point once all PHI's have been generated.
      Builder.SetInsertPoint(&*PH->getParent()->getFirstInsertionPt());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());

      V = FinalShuffle(V, E);

      E->VectorizedValue = V;

      // PHINodes may have multiple entries from the same block. We want to
      // visit every block once.
      SmallPtrSet<BasicBlock *, 4> VisitedBBs;

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        BasicBlock *IBB = PH->getIncomingBlock(i);

        // Stop emission if all incoming values are generated.
        if (NewPhi->getNumIncomingValues() == PH->getNumIncomingValues()) {
          LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
          return V;
        }

        if (!VisitedBBs.insert(IBB).second) {
          NewPhi->addIncoming(NewPhi->getIncomingValueForBlock(IBB), IBB);
          continue;
        }

        Builder.SetInsertPoint(IBB->getTerminator());
        Builder.SetCurrentDebugLocation(PH->getDebugLoc());
        Value *Vec = vectorizeOperand(E, i);
        NewPhi->addIncoming(Vec, IBB);
      }

      assert(NewPhi->getNumIncomingValues() == PH->getNumIncomingValues() &&
             "Invalid number of incoming values");
      return V;
    }

    case Instruction::ExtractElement: {
      Value *V = E->getSingleOperand(0);
      setInsertPointAfterBundle(E);
      V = FinalShuffle(V, E);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::ExtractValue: {
      auto *LI = cast<LoadInst>(E->getSingleOperand(0));
      Builder.SetInsertPoint(LI);
      Value *Ptr = LI->getPointerOperand();
      LoadInst *V = Builder.CreateAlignedLoad(VecTy, Ptr, LI->getAlign());
      Value *NewV = propagateMetadata(V, E->Scalars);
      NewV = FinalShuffle(NewV, E);
      E->VectorizedValue = NewV;
      return NewV;
    }
    case Instruction::InsertElement: {
      assert(E->ReuseShuffleIndices.empty() && "All inserts should be unique");
      Builder.SetInsertPoint(cast<Instruction>(E->Scalars.back()));
      Value *V = vectorizeOperand(E, 1);

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
        Mask.append(NumElts - NumScalars, PoisonMaskElem);
      } else {
        Mask.assign(NumElts, PoisonMaskElem);
        std::iota(Mask.begin(), std::next(Mask.begin(), NumScalars), 0);
      }
      // Create InsertVector shuffle if necessary
      bool IsIdentity = true;
      SmallVector<int> PrevMask(NumElts, PoisonMaskElem);
      Mask.swap(PrevMask);
      for (unsigned I = 0; I < NumScalars; ++I) {
        Value *Scalar = E->Scalars[PrevMask[I]];
        unsigned InsertIdx = *getInsertIndex(Scalar);
        IsIdentity &= InsertIdx - Offset == I;
        Mask[InsertIdx - Offset] = I;
      }
      if (!IsIdentity || NumElts != NumScalars) {
        Value *V2 = nullptr;
        bool IsVNonPoisonous = isGuaranteedNotToBePoison(V) && !isConstant(V);
        SmallVector<int> InsertMask(Mask);
        if (NumElts != NumScalars && Offset == 0) {
          // Follow all insert element instructions from the current buildvector
          // sequence.
          InsertElementInst *Ins = cast<InsertElementInst>(VL0);
          do {
            std::optional<unsigned> InsertIdx = getInsertIndex(Ins);
            if (!InsertIdx)
              break;
            if (InsertMask[*InsertIdx] == PoisonMaskElem)
              InsertMask[*InsertIdx] = *InsertIdx;
            if (!Ins->hasOneUse())
              break;
            Ins = dyn_cast_or_null<InsertElementInst>(
                Ins->getUniqueUndroppableUser());
          } while (Ins);
          SmallBitVector UseMask =
              buildUseMask(NumElts, InsertMask, UseMask::UndefsAsMask);
          SmallBitVector IsFirstPoison =
              isUndefVector<true>(FirstInsert->getOperand(0), UseMask);
          SmallBitVector IsFirstUndef =
              isUndefVector(FirstInsert->getOperand(0), UseMask);
          if (!IsFirstPoison.all()) {
            unsigned Idx = 0;
            for (unsigned I = 0; I < NumElts; I++) {
              if (InsertMask[I] == PoisonMaskElem && !IsFirstPoison.test(I) &&
                  IsFirstUndef.test(I)) {
                if (IsVNonPoisonous) {
                  InsertMask[I] = I < NumScalars ? I : 0;
                  continue;
                }
                if (!V2)
                  V2 = UndefValue::get(V->getType());
                if (Idx >= NumScalars)
                  Idx = NumScalars - 1;
                InsertMask[I] = NumScalars + Idx;
                ++Idx;
              } else if (InsertMask[I] != PoisonMaskElem &&
                         Mask[I] == PoisonMaskElem) {
                InsertMask[I] = PoisonMaskElem;
              }
            }
          } else {
            InsertMask = Mask;
          }
        }
        if (!V2)
          V2 = PoisonValue::get(V->getType());
        V = Builder.CreateShuffleVector(V, V2, InsertMask);
        if (auto *I = dyn_cast<Instruction>(V)) {
          GatherShuffleExtractSeq.insert(I);
          CSEBlocks.insert(I->getParent());
        }
      }

      SmallVector<int> InsertMask(NumElts, PoisonMaskElem);
      for (unsigned I = 0; I < NumElts; I++) {
        if (Mask[I] != PoisonMaskElem)
          InsertMask[Offset + I] = I;
      }
      SmallBitVector UseMask =
          buildUseMask(NumElts, InsertMask, UseMask::UndefsAsMask);
      SmallBitVector IsFirstUndef =
          isUndefVector(FirstInsert->getOperand(0), UseMask);
      if ((!IsIdentity || Offset != 0 || !IsFirstUndef.all()) &&
          NumElts != NumScalars) {
        if (IsFirstUndef.all()) {
          if (!ShuffleVectorInst::isIdentityMask(InsertMask)) {
            SmallBitVector IsFirstPoison =
                isUndefVector<true>(FirstInsert->getOperand(0), UseMask);
            if (!IsFirstPoison.all()) {
              for (unsigned I = 0; I < NumElts; I++) {
                if (InsertMask[I] == PoisonMaskElem && !IsFirstPoison.test(I))
                  InsertMask[I] = I + NumElts;
              }
            }
            V = Builder.CreateShuffleVector(
                V,
                IsFirstPoison.all() ? PoisonValue::get(V->getType())
                                    : FirstInsert->getOperand(0),
                InsertMask, cast<Instruction>(E->Scalars.back())->getName());
            if (auto *I = dyn_cast<Instruction>(V)) {
              GatherShuffleExtractSeq.insert(I);
              CSEBlocks.insert(I->getParent());
            }
          }
        } else {
          SmallBitVector IsFirstPoison =
              isUndefVector<true>(FirstInsert->getOperand(0), UseMask);
          for (unsigned I = 0; I < NumElts; I++) {
            if (InsertMask[I] == PoisonMaskElem)
              InsertMask[I] = IsFirstPoison.test(I) ? PoisonMaskElem : I;
            else
              InsertMask[I] += NumElts;
          }
          V = Builder.CreateShuffleVector(
              FirstInsert->getOperand(0), V, InsertMask,
              cast<Instruction>(E->Scalars.back())->getName());
          if (auto *I = dyn_cast<Instruction>(V)) {
            GatherShuffleExtractSeq.insert(I);
            CSEBlocks.insert(I->getParent());
          }
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

      Value *InVec = vectorizeOperand(E, 0);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      auto *CI = cast<CastInst>(VL0);
      Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
      V = FinalShuffle(V, E);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::FCmp:
    case Instruction::ICmp: {
      setInsertPointAfterBundle(E);

      Value *L = vectorizeOperand(E, 0);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }
      Value *R = vectorizeOperand(E, 1);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      Value *V = Builder.CreateCmp(P0, L, R);
      propagateIRFlags(V, E->Scalars, VL0);
      V = FinalShuffle(V, E);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Select: {
      setInsertPointAfterBundle(E);

      Value *Cond = vectorizeOperand(E, 0);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }
      Value *True = vectorizeOperand(E, 1);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }
      Value *False = vectorizeOperand(E, 2);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      Value *V = Builder.CreateSelect(Cond, True, False);
      V = FinalShuffle(V, E);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::FNeg: {
      setInsertPointAfterBundle(E);

      Value *Op = vectorizeOperand(E, 0);

      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      Value *V = Builder.CreateUnOp(
          static_cast<Instruction::UnaryOps>(E->getOpcode()), Op);
      propagateIRFlags(V, E->Scalars, VL0);
      if (auto *I = dyn_cast<Instruction>(V))
        V = propagateMetadata(I, E->Scalars);

      V = FinalShuffle(V, E);

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

      Value *LHS = vectorizeOperand(E, 0);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }
      Value *RHS = vectorizeOperand(E, 1);
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

      V = FinalShuffle(V, E);

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
      Value *PO = LI->getPointerOperand();
      if (E->State == TreeEntry::Vectorize) {
        NewLI = Builder.CreateAlignedLoad(VecTy, PO, LI->getAlign());

        // The pointer operand uses an in-tree scalar so we add the new
        // LoadInst to ExternalUses list to make sure that an extract will
        // be generated in the future.
        if (TreeEntry *Entry = getTreeEntry(PO)) {
          // Find which lane we need to extract.
          unsigned FoundLane = Entry->findLaneForValue(PO);
          ExternalUses.emplace_back(PO, NewLI, FoundLane);
        }
      } else {
        assert((E->State == TreeEntry::ScatterVectorize ||
                E->State == TreeEntry::PossibleStridedVectorize) &&
               "Unhandled state");
        Value *VecPtr = vectorizeOperand(E, 0);
        if (E->VectorizedValue) {
          LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
          return E->VectorizedValue;
        }
        // Use the minimum alignment of the gathered loads.
        Align CommonAlignment = LI->getAlign();
        for (Value *V : E->Scalars)
          CommonAlignment =
              std::min(CommonAlignment, cast<LoadInst>(V)->getAlign());
        NewLI = Builder.CreateMaskedGather(VecTy, VecPtr, CommonAlignment);
      }
      Value *V = propagateMetadata(NewLI, E->Scalars);

      V = FinalShuffle(V, E);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Store: {
      auto *SI = cast<StoreInst>(VL0);

      setInsertPointAfterBundle(E);

      Value *VecValue = vectorizeOperand(E, 0);
      VecValue = FinalShuffle(VecValue, E);

      Value *Ptr = SI->getPointerOperand();
      StoreInst *ST =
          Builder.CreateAlignedStore(VecValue, Ptr, SI->getAlign());

      // The pointer operand uses an in-tree scalar, so add the new StoreInst to
      // ExternalUses to make sure that an extract will be generated in the
      // future.
      if (TreeEntry *Entry = getTreeEntry(Ptr)) {
        // Find which lane we need to extract.
        unsigned FoundLane = Entry->findLaneForValue(Ptr);
        ExternalUses.push_back(ExternalUser(Ptr, ST, FoundLane));
      }

      Value *V = propagateMetadata(ST, E->Scalars);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::GetElementPtr: {
      auto *GEP0 = cast<GetElementPtrInst>(VL0);
      setInsertPointAfterBundle(E);

      Value *Op0 = vectorizeOperand(E, 0);
      if (E->VectorizedValue) {
        LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
        return E->VectorizedValue;
      }

      SmallVector<Value *> OpVecs;
      for (int J = 1, N = GEP0->getNumOperands(); J < N; ++J) {
        Value *OpVec = vectorizeOperand(E, J);
        if (E->VectorizedValue) {
          LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
          return E->VectorizedValue;
        }
        OpVecs.push_back(OpVec);
      }

      Value *V = Builder.CreateGEP(GEP0->getSourceElementType(), Op0, OpVecs);
      if (Instruction *I = dyn_cast<GetElementPtrInst>(V)) {
        SmallVector<Value *> GEPs;
        for (Value *V : E->Scalars) {
          if (isa<GetElementPtrInst>(V))
            GEPs.push_back(V);
        }
        V = propagateMetadata(I, GEPs);
      }

      V = FinalShuffle(V, E);

      E->VectorizedValue = V;
      ++NumVectorInstructions;

      return V;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      setInsertPointAfterBundle(E);

      Intrinsic::ID IID = Intrinsic::not_intrinsic;
      if (Function *FI = CI->getCalledFunction())
        IID = FI->getIntrinsicID();

      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      auto VecCallCosts = getVectorCallCosts(CI, VecTy, TTI, TLI);
      bool UseIntrinsic = ID != Intrinsic::not_intrinsic &&
                          VecCallCosts.first <= VecCallCosts.second;

      Value *ScalarArg = nullptr;
      std::vector<Value *> OpVecs;
      SmallVector<Type *, 2> TysForDecl;
      // Add return type if intrinsic is overloaded on it.
      if (isVectorIntrinsicWithOverloadTypeAtArg(IID, -1))
        TysForDecl.push_back(
            FixedVectorType::get(CI->getType(), E->Scalars.size()));
      for (int j = 0, e = CI->arg_size(); j < e; ++j) {
        ValueList OpVL;
        // Some intrinsics have scalar arguments. This argument should not be
        // vectorized.
        if (UseIntrinsic && isVectorIntrinsicWithScalarOpAtArg(IID, j)) {
          CallInst *CEI = cast<CallInst>(VL0);
          ScalarArg = CEI->getArgOperand(j);
          OpVecs.push_back(CEI->getArgOperand(j));
          if (isVectorIntrinsicWithOverloadTypeAtArg(IID, j))
            TysForDecl.push_back(ScalarArg->getType());
          continue;
        }

        Value *OpVec = vectorizeOperand(E, j);
        if (E->VectorizedValue) {
          LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
          return E->VectorizedValue;
        }
        LLVM_DEBUG(dbgs() << "SLP: OpVec[" << j << "]: " << *OpVec << "\n");
        OpVecs.push_back(OpVec);
        if (isVectorIntrinsicWithOverloadTypeAtArg(IID, j))
          TysForDecl.push_back(OpVec->getType());
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
      V = FinalShuffle(V, E);

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
        LHS = vectorizeOperand(E, 0);
        if (E->VectorizedValue) {
          LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *VL0 << ".\n");
          return E->VectorizedValue;
        }
        RHS = vectorizeOperand(E, 1);
      } else {
        setInsertPointAfterBundle(E);
        LHS = vectorizeOperand(E, 0);
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
          GatherShuffleExtractSeq.insert(I);
          CSEBlocks.insert(I->getParent());
        }
      }

      // Create shuffle to take alternate operations from the vector.
      // Also, gather up main and alt scalar ops to propagate IR flags to
      // each vector operation.
      ValueList OpScalars, AltScalars;
      SmallVector<int> Mask;
      E->buildAltOpShuffleMask(
          [E, this](Instruction *I) {
            assert(E->isOpcodeOrAlt(I) && "Unexpected main/alternate opcode");
            return isAlternateInstruction(I, E->getMainOp(), E->getAltOp(),
                                          *TLI);
          },
          Mask, &OpScalars, &AltScalars);

      propagateIRFlags(V0, OpScalars);
      propagateIRFlags(V1, AltScalars);

      Value *V = Builder.CreateShuffleVector(V0, V1, Mask);
      if (auto *I = dyn_cast<Instruction>(V)) {
        V = propagateMetadata(I, E->Scalars);
        GatherShuffleExtractSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }

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
  SmallVector<std::pair<Value *, Value *>> ReplacedExternals;
  return vectorizeTree(ExternallyUsedValues, ReplacedExternals);
}

namespace {
/// Data type for handling buildvector sequences with the reused scalars from
/// other tree entries.
struct ShuffledInsertData {
  /// List of insertelements to be replaced by shuffles.
  SmallVector<InsertElementInst *> InsertElements;
  /// The parent vectors and shuffle mask for the given list of inserts.
  MapVector<Value *, SmallVector<int>> ValueMasks;
};
} // namespace

Value *BoUpSLP::vectorizeTree(
    const ExtraValueToDebugLocsMap &ExternallyUsedValues,
    SmallVectorImpl<std::pair<Value *, Value *>> &ReplacedExternals,
    Instruction *ReductionRoot) {
  // All blocks must be scheduled before any instructions are inserted.
  for (auto &BSIter : BlocksSchedules) {
    scheduleBlock(BSIter.second.get());
  }
  // Clean Entry-to-LastInstruction table. It can be affected after scheduling,
  // need to rebuild it.
  EntryToLastInstruction.clear();

  Builder.SetInsertPoint(ReductionRoot ? ReductionRoot
                                       : &F->getEntryBlock().front());
  auto *VectorRoot = vectorizeTree(VectorizableTree[0].get());
  // Run through the list of postponed gathers and emit them, replacing the temp
  // emitted allocas with actual vector instructions.
  ArrayRef<const TreeEntry *> PostponedNodes = PostponedGathers.getArrayRef();
  DenseMap<Value *, SmallVector<TreeEntry *>> PostponedValues;
  for (const TreeEntry *E : PostponedNodes) {
    auto *TE = const_cast<TreeEntry *>(E);
    if (auto *VecTE = getTreeEntry(TE->Scalars.front()))
      if (VecTE->isSame(TE->UserTreeIndices.front().UserTE->getOperand(
              TE->UserTreeIndices.front().EdgeIdx)))
        // Found gather node which is absolutely the same as one of the
        // vectorized nodes. It may happen after reordering.
        continue;
    auto *PrevVec = cast<Instruction>(TE->VectorizedValue);
    TE->VectorizedValue = nullptr;
    auto *UserI =
        cast<Instruction>(TE->UserTreeIndices.front().UserTE->VectorizedValue);
    Builder.SetInsertPoint(PrevVec);
    Builder.SetCurrentDebugLocation(UserI->getDebugLoc());
    Value *Vec = vectorizeTree(TE);
    PrevVec->replaceAllUsesWith(Vec);
    PostponedValues.try_emplace(Vec).first->second.push_back(TE);
    // Replace the stub vector node, if it was used before for one of the
    // buildvector nodes already.
    auto It = PostponedValues.find(PrevVec);
    if (It != PostponedValues.end()) {
      for (TreeEntry *VTE : It->getSecond())
        VTE->VectorizedValue = Vec;
    }
    eraseInstruction(PrevVec);
  }

  // If the vectorized tree can be rewritten in a smaller type, we truncate the
  // vectorized root. InstCombine will then rewrite the entire expression. We
  // sign extend the extracted values below.
  auto *ScalarRoot = VectorizableTree[0]->Scalars[0];
  auto It = MinBWs.find(ScalarRoot);
  if (It != MinBWs.end()) {
    if (auto *I = dyn_cast<Instruction>(VectorRoot)) {
      // If current instr is a phi and not the last phi, insert it after the
      // last phi node.
      if (isa<PHINode>(I))
        Builder.SetInsertPoint(&*I->getParent()->getFirstInsertionPt());
      else
        Builder.SetInsertPoint(&*++BasicBlock::iterator(I));
    }
    auto BundleWidth = VectorizableTree[0]->Scalars.size();
    auto *MinTy = IntegerType::get(F->getContext(), It->second.first);
    auto *VecTy = FixedVectorType::get(MinTy, BundleWidth);
    auto *Trunc = Builder.CreateTrunc(VectorRoot, VecTy);
    VectorizableTree[0]->VectorizedValue = Trunc;
  }

  LLVM_DEBUG(dbgs() << "SLP: Extracting " << ExternalUses.size()
                    << " values .\n");

  SmallVector<ShuffledInsertData> ShuffledInserts;
  // Maps vector instruction to original insertelement instruction
  DenseMap<Value *, InsertElementInst *> VectorToInsertElement;
  // Maps extract Scalar to the corresponding extractelement instruction in the
  // basic block. Only one extractelement per block should be emitted.
  DenseMap<Value *, DenseMap<BasicBlock *, Instruction *>> ScalarToEEs;
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
    // Non-instruction pointers are not deleted, just skip them.
    if (E->getOpcode() == Instruction::GetElementPtr &&
        !isa<GetElementPtrInst>(Scalar))
      continue;

    Value *Vec = E->VectorizedValue;
    assert(Vec && "Can't find vectorizable value");

    Value *Lane = Builder.getInt32(ExternalUse.Lane);
    auto ExtractAndExtendIfNeeded = [&](Value *Vec) {
      if (Scalar->getType() != Vec->getType()) {
        Value *Ex = nullptr;
        auto It = ScalarToEEs.find(Scalar);
        if (It != ScalarToEEs.end()) {
          // No need to emit many extracts, just move the only one in the
          // current block.
          auto EEIt = It->second.find(Builder.GetInsertBlock());
          if (EEIt != It->second.end()) {
            Instruction *I = EEIt->second;
            if (Builder.GetInsertPoint() != Builder.GetInsertBlock()->end() &&
                Builder.GetInsertPoint()->comesBefore(I))
              I->moveBefore(&*Builder.GetInsertPoint());
            Ex = I;
          }
        }
        if (!Ex) {
          // "Reuse" the existing extract to improve final codegen.
          if (auto *ES = dyn_cast<ExtractElementInst>(Scalar)) {
            Ex = Builder.CreateExtractElement(ES->getOperand(0),
                                              ES->getOperand(1));
          } else {
            Ex = Builder.CreateExtractElement(Vec, Lane);
          }
          if (auto *I = dyn_cast<Instruction>(Ex))
            ScalarToEEs[Scalar].try_emplace(Builder.GetInsertBlock(), I);
        }
        // The then branch of the previous if may produce constants, since 0
        // operand might be a constant.
        if (auto *ExI = dyn_cast<Instruction>(Ex)) {
          GatherShuffleExtractSeq.insert(ExI);
          CSEBlocks.insert(ExI->getParent());
        }
        // If necessary, sign-extend or zero-extend ScalarRoot
        // to the larger type.
        auto BWIt = MinBWs.find(ScalarRoot);
        if (BWIt == MinBWs.end())
          return Ex;
        if (BWIt->second.second)
          return Builder.CreateSExt(Ex, Scalar->getType());
        return Builder.CreateZExt(Ex, Scalar->getType());
      }
      assert(isa<FixedVectorType>(Scalar->getType()) &&
             isa<InsertElementInst>(Scalar) &&
             "In-tree scalar of vector type is not insertelement?");
      auto *IE = cast<InsertElementInst>(Scalar);
      VectorToInsertElement.try_emplace(Vec, IE);
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
        if (auto *PHI = dyn_cast<PHINode>(VecI))
          Builder.SetInsertPoint(PHI->getParent()->getFirstNonPHI());
        else
          Builder.SetInsertPoint(VecI->getParent(),
                                 std::next(VecI->getIterator()));
      } else {
        Builder.SetInsertPoint(&F->getEntryBlock().front());
      }
      Value *NewInst = ExtractAndExtendIfNeeded(Vec);
      // Required to update internally referenced instructions.
      Scalar->replaceAllUsesWith(NewInst);
      ReplacedExternals.emplace_back(Scalar, NewInst);
      continue;
    }

    if (auto *VU = dyn_cast<InsertElementInst>(User)) {
      // Skip if the scalar is another vector op or Vec is not an instruction.
      if (!Scalar->getType()->isVectorTy() && isa<Instruction>(Vec)) {
        if (auto *FTy = dyn_cast<FixedVectorType>(User->getType())) {
          std::optional<unsigned> InsertIdx = getInsertIndex(VU);
          if (InsertIdx) {
            // Need to use original vector, if the root is truncated.
            if (MinBWs.contains(Scalar) &&
                VectorizableTree[0]->VectorizedValue == Vec)
              Vec = VectorRoot;
            auto *It =
                find_if(ShuffledInserts, [VU](const ShuffledInsertData &Data) {
                  // Checks if 2 insertelements are from the same buildvector.
                  InsertElementInst *VecInsert = Data.InsertElements.front();
                  return areTwoInsertFromSameBuildVector(
                      VU, VecInsert,
                      [](InsertElementInst *II) { return II->getOperand(0); });
                });
            unsigned Idx = *InsertIdx;
            if (It == ShuffledInserts.end()) {
              (void)ShuffledInserts.emplace_back();
              It = std::next(ShuffledInserts.begin(),
                             ShuffledInserts.size() - 1);
              SmallVectorImpl<int> &Mask = It->ValueMasks[Vec];
              if (Mask.empty())
                Mask.assign(FTy->getNumElements(), PoisonMaskElem);
              // Find the insertvector, vectorized in tree, if any.
              Value *Base = VU;
              while (auto *IEBase = dyn_cast<InsertElementInst>(Base)) {
                if (IEBase != User &&
                    (!IEBase->hasOneUse() ||
                     getInsertIndex(IEBase).value_or(Idx) == Idx))
                  break;
                // Build the mask for the vectorized insertelement instructions.
                if (const TreeEntry *E = getTreeEntry(IEBase)) {
                  do {
                    IEBase = cast<InsertElementInst>(Base);
                    int IEIdx = *getInsertIndex(IEBase);
                    assert(Mask[Idx] == PoisonMaskElem &&
                           "InsertElementInstruction used already.");
                    Mask[IEIdx] = IEIdx;
                    Base = IEBase->getOperand(0);
                  } while (E == getTreeEntry(Base));
                  break;
                }
                Base = cast<InsertElementInst>(Base)->getOperand(0);
                // After the vectorization the def-use chain has changed, need
                // to look through original insertelement instructions, if they
                // get replaced by vector instructions.
                auto It = VectorToInsertElement.find(Base);
                if (It != VectorToInsertElement.end())
                  Base = It->second;
              }
            }
            SmallVectorImpl<int> &Mask = It->ValueMasks[Vec];
            if (Mask.empty())
              Mask.assign(FTy->getNumElements(), PoisonMaskElem);
            Mask[Idx] = ExternalUse.Lane;
            It->InsertElements.push_back(cast<InsertElementInst>(User));
            continue;
          }
        }
      }
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
            PH->setOperand(i, NewInst);
          }
        }
      } else {
        Builder.SetInsertPoint(cast<Instruction>(User));
        Value *NewInst = ExtractAndExtendIfNeeded(Vec);
        User->replaceUsesOfWith(Scalar, NewInst);
      }
    } else {
      Builder.SetInsertPoint(&F->getEntryBlock().front());
      Value *NewInst = ExtractAndExtendIfNeeded(Vec);
      User->replaceUsesOfWith(Scalar, NewInst);
    }

    LLVM_DEBUG(dbgs() << "SLP: Replaced:" << *User << ".\n");
  }

  auto CreateShuffle = [&](Value *V1, Value *V2, ArrayRef<int> Mask) {
    SmallVector<int> CombinedMask1(Mask.size(), PoisonMaskElem);
    SmallVector<int> CombinedMask2(Mask.size(), PoisonMaskElem);
    int VF = cast<FixedVectorType>(V1->getType())->getNumElements();
    for (int I = 0, E = Mask.size(); I < E; ++I) {
      if (Mask[I] < VF)
        CombinedMask1[I] = Mask[I];
      else
        CombinedMask2[I] = Mask[I] - VF;
    }
    ShuffleInstructionBuilder ShuffleBuilder(Builder, *this);
    ShuffleBuilder.add(V1, CombinedMask1);
    if (V2)
      ShuffleBuilder.add(V2, CombinedMask2);
    return ShuffleBuilder.finalize(std::nullopt);
  };

  auto &&ResizeToVF = [&CreateShuffle](Value *Vec, ArrayRef<int> Mask,
                                       bool ForSingleMask) {
    unsigned VF = Mask.size();
    unsigned VecVF = cast<FixedVectorType>(Vec->getType())->getNumElements();
    if (VF != VecVF) {
      if (any_of(Mask, [VF](int Idx) { return Idx >= static_cast<int>(VF); })) {
        Vec = CreateShuffle(Vec, nullptr, Mask);
        return std::make_pair(Vec, true);
      }
      if (!ForSingleMask) {
        SmallVector<int> ResizeMask(VF, PoisonMaskElem);
        for (unsigned I = 0; I < VF; ++I) {
          if (Mask[I] != PoisonMaskElem)
            ResizeMask[Mask[I]] = Mask[I];
        }
        Vec = CreateShuffle(Vec, nullptr, ResizeMask);
      }
    }

    return std::make_pair(Vec, false);
  };
  // Perform shuffling of the vectorize tree entries for better handling of
  // external extracts.
  for (int I = 0, E = ShuffledInserts.size(); I < E; ++I) {
    // Find the first and the last instruction in the list of insertelements.
    sort(ShuffledInserts[I].InsertElements, isFirstInsertElement);
    InsertElementInst *FirstInsert = ShuffledInserts[I].InsertElements.front();
    InsertElementInst *LastInsert = ShuffledInserts[I].InsertElements.back();
    Builder.SetInsertPoint(LastInsert);
    auto Vector = ShuffledInserts[I].ValueMasks.takeVector();
    Value *NewInst = performExtractsShuffleAction<Value>(
        MutableArrayRef(Vector.data(), Vector.size()),
        FirstInsert->getOperand(0),
        [](Value *Vec) {
          return cast<VectorType>(Vec->getType())
              ->getElementCount()
              .getKnownMinValue();
        },
        ResizeToVF,
        [FirstInsert, &CreateShuffle](ArrayRef<int> Mask,
                                      ArrayRef<Value *> Vals) {
          assert((Vals.size() == 1 || Vals.size() == 2) &&
                 "Expected exactly 1 or 2 input values.");
          if (Vals.size() == 1) {
            // Do not create shuffle if the mask is a simple identity
            // non-resizing mask.
            if (Mask.size() != cast<FixedVectorType>(Vals.front()->getType())
                                   ->getNumElements() ||
                !ShuffleVectorInst::isIdentityMask(Mask))
              return CreateShuffle(Vals.front(), nullptr, Mask);
            return Vals.front();
          }
          return CreateShuffle(Vals.front() ? Vals.front()
                                            : FirstInsert->getOperand(0),
                               Vals.back(), Mask);
        });
    auto It = ShuffledInserts[I].InsertElements.rbegin();
    // Rebuild buildvector chain.
    InsertElementInst *II = nullptr;
    if (It != ShuffledInserts[I].InsertElements.rend())
      II = *It;
    SmallVector<Instruction *> Inserts;
    while (It != ShuffledInserts[I].InsertElements.rend()) {
      assert(II && "Must be an insertelement instruction.");
      if (*It == II)
        ++It;
      else
        Inserts.push_back(cast<Instruction>(II));
      II = dyn_cast<InsertElementInst>(II->getOperand(0));
    }
    for (Instruction *II : reverse(Inserts)) {
      II->replaceUsesOfWith(II->getOperand(0), NewInst);
      if (auto *NewI = dyn_cast<Instruction>(NewInst))
        if (II->getParent() == NewI->getParent() && II->comesBefore(NewI))
          II->moveAfter(NewI);
      NewInst = II;
    }
    LastInsert->replaceAllUsesWith(NewInst);
    for (InsertElementInst *IE : reverse(ShuffledInserts[I].InsertElements)) {
      IE->replaceUsesOfWith(IE->getOperand(0),
                            PoisonValue::get(IE->getOperand(0)->getType()));
      IE->replaceUsesOfWith(IE->getOperand(1),
                            PoisonValue::get(IE->getOperand(1)->getType()));
      eraseInstruction(IE);
    }
    CSEBlocks.insert(LastInsert->getParent());
  }

  SmallVector<Instruction *> RemovedInsts;
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

      if (Entry->getOpcode() == Instruction::GetElementPtr &&
          !isa<GetElementPtrInst>(Scalar))
        continue;
#ifndef NDEBUG
      Type *Ty = Scalar->getType();
      if (!Ty->isVoidTy()) {
        for (User *U : Scalar->users()) {
          LLVM_DEBUG(dbgs() << "SLP: \tvalidating user:" << *U << ".\n");

          // It is legal to delete users in the ignorelist.
          assert((getTreeEntry(U) ||
                  (UserIgnoreList && UserIgnoreList->contains(U)) ||
                  (isa_and_nonnull<Instruction>(U) &&
                   isDeleted(cast<Instruction>(U)))) &&
                 "Deleting out-of-tree value");
        }
      }
#endif
      LLVM_DEBUG(dbgs() << "SLP: \tErasing scalar:" << *Scalar << ".\n");
      eraseInstruction(cast<Instruction>(Scalar));
      // Retain to-be-deleted instructions for some debug-info
      // bookkeeping. NOTE: eraseInstruction only marks the instruction for
      // deletion - instructions are not deleted until later.
      RemovedInsts.push_back(cast<Instruction>(Scalar));
    }
  }

  // Merge the DIAssignIDs from the about-to-be-deleted instructions into the
  // new vector instruction.
  if (auto *V = dyn_cast<Instruction>(VectorizableTree[0]->VectorizedValue))
    V->mergeDIAssignID(RemovedInsts);

  Builder.ClearInsertionPoint();
  InstrElementSize.clear();

  return VectorizableTree[0]->VectorizedValue;
}

void BoUpSLP::optimizeGatherSequence() {
  LLVM_DEBUG(dbgs() << "SLP: Optimizing " << GatherShuffleExtractSeq.size()
                    << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (Instruction *I : GatherShuffleExtractSeq) {
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
    CSEBlocks.insert(PreHeader);
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
      if (SM1[I] == PoisonMaskElem)
        ++LastUndefsCnt;
      else
        LastUndefsCnt = 0;
      if (NewMask[I] != PoisonMaskElem && SM1[I] != PoisonMaskElem &&
          NewMask[I] != SM1[I])
        return false;
      if (NewMask[I] == PoisonMaskElem)
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
      if (!isa<InsertElementInst, ExtractElementInst, ShuffleVectorInst>(&In) &&
          !GatherShuffleExtractSeq.contains(&In))
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
            GatherShuffleExtractSeq.contains(V) &&
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
  GatherShuffleExtractSeq.clear();
}

BoUpSLP::ScheduleData *
BoUpSLP::BlockScheduling::buildBundle(ArrayRef<Value *> VL) {
  ScheduleData *Bundle = nullptr;
  ScheduleData *PrevInBundle = nullptr;
  for (Value *V : VL) {
    if (doesNotNeedToBeScheduled(V))
      continue;
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
std::optional<BoUpSLP::ScheduleData *>
BoUpSLP::BlockScheduling::tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP,
                                            const InstructionsState &S) {
  // No need to schedule PHIs, insertelement, extractelement and extractvalue
  // instructions.
  if (isa<PHINode>(S.OpValue) || isVectorLikeInstWithConstOps(S.OpValue) ||
      doesNotNeedToSchedule(VL))
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
    if (doesNotNeedToBeScheduled(V))
      continue;
    if (!extendSchedulingRegion(V, S)) {
      // If the scheduling region got new instructions at the lower end (or it
      // is a new region for the first bundle). This makes it necessary to
      // recalculate all dependencies.
      // Otherwise the compiler may crash trying to incorrectly calculate
      // dependencies and emit instruction in the wrong order at the actual
      // scheduling.
      TryScheduleBundleImpl(/*ReSchedule=*/false, nullptr);
      return std::nullopt;
    }
  }

  bool ReSchedule = false;
  for (Value *V : VL) {
    if (doesNotNeedToBeScheduled(V))
      continue;
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
    return std::nullopt;
  }
  return Bundle;
}

void BoUpSLP::BlockScheduling::cancelScheduling(ArrayRef<Value *> VL,
                                                Value *OpValue) {
  if (isa<PHINode>(OpValue) || isVectorLikeInstWithConstOps(OpValue) ||
      doesNotNeedToSchedule(VL))
    return;

  if (doesNotNeedToBeScheduled(OpValue))
    OpValue = *find_if_not(VL, doesNotNeedToBeScheduled);
  ScheduleData *Bundle = getScheduleData(OpValue);
  LLVM_DEBUG(dbgs() << "SLP:  cancel scheduling of " << *Bundle << "\n");
  assert(!Bundle->IsScheduled &&
         "Can't cancel bundle which is already scheduled");
  assert(Bundle->isSchedulingEntity() &&
         (Bundle->isPartOfBundle() || needToScheduleSingleInstruction(VL)) &&
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
    BundleMember->TE = nullptr;
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
         !doesNotNeedToBeScheduled(I) &&
         "phi nodes/insertelements/extractelements/extractvalues don't need to "
         "be scheduled");
  auto &&CheckScheduleForI = [this, &S](Instruction *I) -> bool {
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
  if (CheckScheduleForI(I))
    return true;
  if (!ScheduleStart) {
    // It's the first instruction in the new region.
    initScheduleData(I, I->getNextNode(), nullptr, nullptr);
    ScheduleStart = I;
    ScheduleEnd = I->getNextNode();
    if (isOneOf(S, I) != I)
      CheckScheduleForI(I);
    assert(ScheduleEnd && "tried to vectorize a terminator?");
    LLVM_DEBUG(dbgs() << "SLP:  initialize schedule region to " << *I << "\n");
    return true;
  }
  // Search up and down at the same time, because we don't know if the new
  // instruction is above or below the existing scheduling region.
  // Ignore debug info (and other "AssumeLike" intrinsics) so that's not counted
  // against the budget. Otherwise debug info could affect codegen.
  BasicBlock::reverse_iterator UpIter =
      ++ScheduleStart->getIterator().getReverse();
  BasicBlock::reverse_iterator UpperEnd = BB->rend();
  BasicBlock::iterator DownIter = ScheduleEnd->getIterator();
  BasicBlock::iterator LowerEnd = BB->end();
  auto IsAssumeLikeIntr = [](const Instruction &I) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      return II->isAssumeLikeIntrinsic();
    return false;
  };
  UpIter = std::find_if_not(UpIter, UpperEnd, IsAssumeLikeIntr);
  DownIter = std::find_if_not(DownIter, LowerEnd, IsAssumeLikeIntr);
  while (UpIter != UpperEnd && DownIter != LowerEnd && &*UpIter != I &&
         &*DownIter != I) {
    if (++ScheduleRegionSize > ScheduleRegionSizeLimit) {
      LLVM_DEBUG(dbgs() << "SLP:  exceeded schedule region size limit\n");
      return false;
    }

    ++UpIter;
    ++DownIter;

    UpIter = std::find_if_not(UpIter, UpperEnd, IsAssumeLikeIntr);
    DownIter = std::find_if_not(DownIter, LowerEnd, IsAssumeLikeIntr);
  }
  if (DownIter == LowerEnd || (UpIter != UpperEnd && &*UpIter == I)) {
    assert(I->getParent() == ScheduleStart->getParent() &&
           "Instruction is in wrong basic block.");
    initScheduleData(I, ScheduleStart, nullptr, FirstLoadStoreInRegion);
    ScheduleStart = I;
    if (isOneOf(S, I) != I)
      CheckScheduleForI(I);
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
    CheckScheduleForI(I);
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
    // No need to allocate data for non-schedulable instructions.
    if (doesNotNeedToBeScheduled(I))
      continue;
    ScheduleData *SD = ScheduleDataMap.lookup(I);
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

    if (match(I, m_Intrinsic<Intrinsic::stacksave>()) ||
        match(I, m_Intrinsic<Intrinsic::stackrestore>()))
      RegionHasStackSave = true;
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

      auto makeControlDependent = [&](Instruction *I) {
        auto *DepDest = getScheduleData(I);
        assert(DepDest && "must be in schedule window");
        DepDest->ControlDependencies.push_back(BundleMember);
        BundleMember->Dependencies++;
        ScheduleData *DestBundle = DepDest->FirstInBundle;
        if (!DestBundle->IsScheduled)
          BundleMember->incrementUnscheduledDeps(1);
        if (!DestBundle->hasValidDependencies())
          WorkList.push_back(DestBundle);
      };

      // Any instruction which isn't safe to speculate at the beginning of the
      // block is control dependend on any early exit or non-willreturn call
      // which proceeds it.
      if (!isGuaranteedToTransferExecutionToSuccessor(BundleMember->Inst)) {
        for (Instruction *I = BundleMember->Inst->getNextNode();
             I != ScheduleEnd; I = I->getNextNode()) {
          if (isSafeToSpeculativelyExecute(I, &*BB->begin(), SLP->AC))
            continue;

          // Add the dependency
          makeControlDependent(I);

          if (!isGuaranteedToTransferExecutionToSuccessor(I))
            // Everything past here must be control dependent on I.
            break;
        }
      }

      if (RegionHasStackSave) {
        // If we have an inalloc alloca instruction, it needs to be scheduled
        // after any preceeding stacksave.  We also need to prevent any alloca
        // from reordering above a preceeding stackrestore.
        if (match(BundleMember->Inst, m_Intrinsic<Intrinsic::stacksave>()) ||
            match(BundleMember->Inst, m_Intrinsic<Intrinsic::stackrestore>())) {
          for (Instruction *I = BundleMember->Inst->getNextNode();
               I != ScheduleEnd; I = I->getNextNode()) {
            if (match(I, m_Intrinsic<Intrinsic::stacksave>()) ||
                match(I, m_Intrinsic<Intrinsic::stackrestore>()))
              // Any allocas past here must be control dependent on I, and I
              // must be memory dependend on BundleMember->Inst.
              break;

            if (!isa<AllocaInst>(I))
              continue;

            // Add the dependency
            makeControlDependent(I);
          }
        }

        // In addition to the cases handle just above, we need to prevent
        // allocas and loads/stores from moving below a stacksave or a
        // stackrestore. Avoiding moving allocas below stackrestore is currently
        // thought to be conservatism. Moving loads/stores below a stackrestore
        // can lead to incorrect code.
        if (isa<AllocaInst>(BundleMember->Inst) ||
            BundleMember->Inst->mayReadOrWriteMemory()) {
          for (Instruction *I = BundleMember->Inst->getNextNode();
               I != ScheduleEnd; I = I->getNextNode()) {
            if (!match(I, m_Intrinsic<Intrinsic::stacksave>()) &&
                !match(I, m_Intrinsic<Intrinsic::stackrestore>()))
              continue;

            // Add the dependency
            makeControlDependent(I);
            break;
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

      for (; DepDest; DepDest = DepDest->NextLoadStore) {
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

  // A key point - if we got here, pre-scheduling was able to find a valid
  // scheduling of the sub-graph of the scheduling window which consists
  // of all vector bundles and their transitive users.  As such, we do not
  // need to reschedule anything *outside of* that subgraph.

  BS->resetSchedule();

  // For the real scheduling we use a more sophisticated ready-list: it is
  // sorted by the original instruction location. This lets the final schedule
  // be as  close as possible to the original instruction order.
  // WARNING: If changing this order causes a correctness issue, that means
  // there is some missing dependence edge in the schedule data graph.
  struct ScheduleDataCompare {
    bool operator()(ScheduleData *SD1, ScheduleData *SD2) const {
      return SD2->SchedulingPriority < SD1->SchedulingPriority;
    }
  };
  std::set<ScheduleData *, ScheduleDataCompare> ReadyInsts;

  // Ensure that all dependency data is updated (for nodes in the sub-graph)
  // and fill the ready-list with initial instructions.
  int Idx = 0;
  for (auto *I = BS->ScheduleStart; I != BS->ScheduleEnd;
       I = I->getNextNode()) {
    BS->doForAllOpcodes(I, [this, &Idx, BS](ScheduleData *SD) {
      TreeEntry *SDTE = getTreeEntry(SD->Inst);
      (void)SDTE;
      assert((isVectorLikeInstWithConstOps(SD->Inst) ||
              SD->isPartOfBundle() ==
                  (SDTE && !doesNotNeedToSchedule(SDTE->Scalars))) &&
             "scheduler and vectorizer bundle mismatch");
      SD->FirstInBundle->SchedulingPriority = Idx++;

      if (SD->isSchedulingEntity() && SD->isPartOfBundle())
        BS->calculateDependencies(SD, false, this);
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
  }

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
  if (auto *Store = dyn_cast<StoreInst>(V))
    return DL->getTypeSizeInBits(Store->getValueOperand()->getType());

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
    if (isa<LoadInst, ExtractElementInst, ExtractValueInst>(I))
      Width = std::max<unsigned>(Width, DL->getTypeSizeInBits(Ty));

    // Otherwise, we need to visit the operands of the instruction. We only
    // handle the interesting cases from buildTree here. If an operand is an
    // instruction we haven't yet visited and from the same basic block as the
    // user or the use is a PHI node, we add it to the worklist.
    else if (isa<PHINode, CastInst, GetElementPtrInst, CmpInst, SelectInst,
                 BinaryOperator, UnaryOperator>(I)) {
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
    if (isa<ExtractElementInst, InsertElementInst>(I->getOperand(0)))
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
    MaxBitWidth = std::max<unsigned>(Mask.getBitWidth() - Mask.countl_zero(),
                                     MaxBitWidth);
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
  MaxBitWidth = llvm::bit_ceil(MaxBitWidth);

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
    MinBWs.try_emplace(Scalar, MaxBitWidth, !IsKnownPositive);
}

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
  for (auto *BB : post_order(&F.getEntryBlock())) {
    // Start new block - clear the list of reduction roots.
    R.clearReductionData();
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
                                            unsigned Idx, unsigned MinVF) {
  LLVM_DEBUG(dbgs() << "SLP: Analyzing a store chain of length " << Chain.size()
                    << "\n");
  const unsigned Sz = R.getVectorElementSize(Chain[0]);
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

  LLVM_DEBUG(dbgs() << "SLP: Found cost = " << Cost << " for VF=" << VF << "\n");
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

  // Stores the pair of stores (first_store, last_store) in a range, that were
  // already tried to be vectorized. Allows to skip the store ranges that were
  // already tried to be vectorized but the attempts were unsuccessful.
  DenseSet<std::pair<Value *, Value *>> TriedSequences;
  struct StoreDistCompare {
    bool operator()(const std::pair<unsigned, int> &Op1,
                    const std::pair<unsigned, int> &Op2) const {
      return Op1.second < Op2.second;
    }
  };
  // A set of pairs (index of store in Stores array ref, Distance of the store
  // address relative to base store address in units).
  using StoreIndexToDistSet =
      std::set<std::pair<unsigned, int>, StoreDistCompare>;
  auto TryToVectorize = [&](const StoreIndexToDistSet &Set) {
    int PrevDist = -1;
    BoUpSLP::ValueList Operands;
    // Collect the chain into a list.
    for (auto [Idx, Data] : enumerate(Set)) {
      if (Operands.empty() || Data.second - PrevDist == 1) {
        Operands.push_back(Stores[Data.first]);
        PrevDist = Data.second;
        if (Idx != Set.size() - 1)
          continue;
      }
      if (Operands.size() <= 1) {
        Operands.clear();
        Operands.push_back(Stores[Data.first]);
        PrevDist = Data.second;
        continue;
      }

      unsigned MaxVecRegSize = R.getMaxVecRegSize();
      unsigned EltSize = R.getVectorElementSize(Operands[0]);
      unsigned MaxElts = llvm::bit_floor(MaxVecRegSize / EltSize);

      unsigned MaxVF =
          std::min(R.getMaximumVF(EltSize, Instruction::Store), MaxElts);
      auto *Store = cast<StoreInst>(Operands[0]);
      Type *StoreTy = Store->getValueOperand()->getType();
      Type *ValueTy = StoreTy;
      if (auto *Trunc = dyn_cast<TruncInst>(Store->getValueOperand()))
        ValueTy = Trunc->getSrcTy();
      unsigned MinVF = TTI->getStoreMinimumVF(
          R.getMinVF(DL->getTypeSizeInBits(ValueTy)), StoreTy, ValueTy);

      if (MaxVF <= MinVF) {
        LLVM_DEBUG(dbgs() << "SLP: Vectorization infeasible as MaxVF (" << MaxVF
                          << ") <= "
                          << "MinVF (" << MinVF << ")\n");
      }

      // FIXME: Is division-by-2 the correct step? Should we assert that the
      // register size is a power-of-2?
      unsigned StartIdx = 0;
      for (unsigned Size = MaxVF; Size >= MinVF; Size /= 2) {
        for (unsigned Cnt = StartIdx, E = Operands.size(); Cnt + Size <= E;) {
          ArrayRef<Value *> Slice = ArrayRef(Operands).slice(Cnt, Size);
          assert(
              all_of(
                  Slice,
                  [&](Value *V) {
                    return cast<StoreInst>(V)->getValueOperand()->getType() ==
                           cast<StoreInst>(Slice.front())
                               ->getValueOperand()
                               ->getType();
                  }) &&
              "Expected all operands of same type.");
          if (!VectorizedStores.count(Slice.front()) &&
              !VectorizedStores.count(Slice.back()) &&
              TriedSequences.insert(std::make_pair(Slice.front(), Slice.back()))
                  .second &&
              vectorizeStoreChain(Slice, R, Cnt, MinVF)) {
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
      Operands.clear();
      Operands.push_back(Stores[Data.first]);
      PrevDist = Data.second;
    }
  };

  // Stores pair (first: index of the store into Stores array ref, address of
  // which taken as base, second: sorted set of pairs {index, dist}, which are
  // indices of stores in the set and their store location distances relative to
  // the base address).

  // Need to store the index of the very first store separately, since the set
  // may be reordered after the insertion and the first store may be moved. This
  // container allows to reduce number of calls of getPointersDiff() function.
  SmallVector<std::pair<unsigned, StoreIndexToDistSet>> SortedStores;
  // Inserts the specified store SI with the given index Idx to the set of the
  // stores. If the store with the same distance is found already - stop
  // insertion, try to vectorize already found stores. If some stores from this
  // sequence were not vectorized - try to vectorize them with the new store
  // later. But this logic is applied only to the stores, that come before the
  // previous store with the same distance.
  // Example:
  // 1. store x, %p
  // 2. store y, %p+1
  // 3. store z, %p+2
  // 4. store a, %p
  // 5. store b, %p+3
  // - Scan this from the last to first store. The very first bunch of stores is
  // {5, {{4, -3}, {2, -2}, {3, -1}, {5, 0}}} (the element in SortedStores
  // vector).
  // - The next store in the list - #1 - has the same distance from store #5 as
  // the store #4.
  // - Try to vectorize sequence of stores 4,2,3,5.
  // - If all these stores are vectorized - just drop them.
  // - If some of them are not vectorized (say, #3 and #5), do extra analysis.
  // - Start new stores sequence.
  // The new bunch of stores is {1, {1, 0}}.
  // - Add the stores from previous sequence, that were not vectorized.
  // Here we consider the stores in the reversed order, rather they are used in
  // the IR (Stores are reversed already, see vectorizeStoreChains() function).
  // Store #3 can be added -> comes after store #4 with the same distance as
  // store #1.
  // Store #5 cannot be added - comes before store #4.
  // This logic allows to improve the compile time, we assume that the stores
  // after previous store with the same distance most likely have memory
  // dependencies and no need to waste compile time to try to vectorize them.
  // - Try to vectorize the sequence {1, {1, 0}, {3, 2}}.
  auto FillStoresSet = [&](unsigned Idx, StoreInst *SI) {
    for (std::pair<unsigned, StoreIndexToDistSet> &Set : SortedStores) {
      std::optional<int> Diff = getPointersDiff(
          Stores[Set.first]->getValueOperand()->getType(),
          Stores[Set.first]->getPointerOperand(),
          SI->getValueOperand()->getType(), SI->getPointerOperand(), *DL, *SE,
          /*StrictCheck=*/true);
      if (!Diff)
        continue;
      auto It = Set.second.find(std::make_pair(Idx, *Diff));
      if (It == Set.second.end()) {
        Set.second.emplace(Idx, *Diff);
        return;
      }
      // Try to vectorize the first found set to avoid duplicate analysis.
      TryToVectorize(Set.second);
      StoreIndexToDistSet PrevSet;
      PrevSet.swap(Set.second);
      Set.first = Idx;
      Set.second.emplace(Idx, 0);
      // Insert stores that followed previous match to try to vectorize them
      // with this store.
      unsigned StartIdx = It->first + 1;
      SmallBitVector UsedStores(Idx - StartIdx);
      // Distances to previously found dup store (or this store, since they
      // store to the same addresses).
      SmallVector<int> Dists(Idx - StartIdx, 0);
      for (const std::pair<unsigned, int> &Pair : reverse(PrevSet)) {
        // Do not try to vectorize sequences, we already tried.
        if (Pair.first <= It->first ||
            VectorizedStores.contains(Stores[Pair.first]))
          break;
        unsigned BI = Pair.first - StartIdx;
        UsedStores.set(BI);
        Dists[BI] = Pair.second - It->second;
      }
      for (unsigned I = StartIdx; I < Idx; ++I) {
        unsigned BI = I - StartIdx;
        if (UsedStores.test(BI))
          Set.second.emplace(I, Dists[BI]);
      }
      return;
    }
    auto &Res = SortedStores.emplace_back();
    Res.first = Idx;
    Res.second.emplace(Idx, 0);
  };
  StoreInst *PrevStore = Stores.front();
  for (auto [I, SI] : enumerate(Stores)) {
    // Check that we do not try to vectorize stores of different types.
    if (PrevStore->getValueOperand()->getType() !=
        SI->getValueOperand()->getType()) {
      for (auto &Set : SortedStores)
        TryToVectorize(Set.second);
      SortedStores.clear();
      PrevStore = SI;
    }
    FillStoresSet(I, SI);
  }

  // Final vectorization attempt.
  for (auto &Set : SortedStores)
    TryToVectorize(Set.second);

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

bool SLPVectorizerPass::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                           bool MaxVFOnly) {
  if (VL.size() < 2)
    return false;

  LLVM_DEBUG(dbgs() << "SLP: Trying to vectorize a list of length = "
                    << VL.size() << ".\n");

  // Check that all of the parts are instructions of the same type,
  // we permit an alternate opcode via InstructionsState.
  InstructionsState S = getSameOpcode(VL, *TLI);
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
  unsigned MaxVF = std::max<unsigned>(llvm::bit_floor(VL.size()), MinVF);
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
      unsigned ActualVF = std::min(MaxInst - I, VF);

      if (!isPowerOf2_32(ActualVF))
        continue;

      if (MaxVFOnly && ActualVF < MaxVF)
        break;
      if ((VF > MinVF && ActualVF <= VF / 2) || (VF == MinVF && ActualVF < 2))
        break;

      ArrayRef<Value *> Ops = VL.slice(I, ActualVF);
      // Check that a previous iteration of this loop did not delete the Value.
      if (llvm::any_of(Ops, [&R](Value *V) {
            auto *I = dyn_cast<Instruction>(V);
            return I && R.isDeleted(I);
          }))
        continue;

      LLVM_DEBUG(dbgs() << "SLP: Analyzing " << ActualVF << " operations "
                        << "\n");

      R.buildTree(Ops);
      if (R.isTreeTinyAndNotFullyVectorizable())
        continue;
      R.reorderTopToBottom();
      R.reorderBottomToTop(
          /*IgnoreReorder=*/!isa<InsertElementInst>(Ops.front()) &&
          !R.doesRootHaveInTreeUses());
      R.buildExternalUses();

      R.computeMinimumValueSizes();
      InstructionCost Cost = R.getTreeCost();
      CandidateFound = true;
      MinCost = std::min(MinCost, Cost);

      LLVM_DEBUG(dbgs() << "SLP: Found cost = " << Cost
                        << " for VF=" << ActualVF << "\n");
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

  if (!isa<BinaryOperator, CmpInst>(I) || isa<VectorType>(I->getType()))
    return false;

  Value *P = I->getParent();

  // Vectorize in current basic block only.
  auto *Op0 = dyn_cast<Instruction>(I->getOperand(0));
  auto *Op1 = dyn_cast<Instruction>(I->getOperand(1));
  if (!Op0 || !Op1 || Op0->getParent() != P || Op1->getParent() != P)
    return false;

  // First collect all possible candidates
  SmallVector<std::pair<Value *, Value *>, 4> Candidates;
  Candidates.emplace_back(Op0, Op1);

  auto *A = dyn_cast<BinaryOperator>(Op0);
  auto *B = dyn_cast<BinaryOperator>(Op1);
  // Try to skip B.
  if (A && B && B->hasOneUse()) {
    auto *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    auto *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (B0 && B0->getParent() == P)
      Candidates.emplace_back(A, B0);
    if (B1 && B1->getParent() == P)
      Candidates.emplace_back(A, B1);
  }
  // Try to skip A.
  if (B && A && A->hasOneUse()) {
    auto *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    auto *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (A0 && A0->getParent() == P)
      Candidates.emplace_back(A0, B);
    if (A1 && A1->getParent() == P)
      Candidates.emplace_back(A1, B);
  }

  if (Candidates.size() == 1)
    return tryToVectorizeList({Op0, Op1}, R);

  // We have multiple options. Try to pick the single best.
  std::optional<int> BestCandidate = R.findBestRootPair(Candidates);
  if (!BestCandidate)
    return false;
  return tryToVectorizeList(
      {Candidates[*BestCandidate].first, Candidates[*BestCandidate].second}, R);
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
  /// List of possibly reduced values.
  SmallVector<SmallVector<Value *>> ReducedVals;
  /// Maps reduced value to the corresponding reduction operation.
  DenseMap<Value *, SmallVector<Instruction *>> ReducedValsToOps;
  // Use map vector to make stable output.
  MapVector<Instruction *, Value *> ExtraArgs;
  WeakTrackingVH ReductionRoot;
  /// The type of reduction operation.
  RecurKind RdxKind;
  /// Checks if the optimization of original scalar identity operations on
  /// matched horizontal reductions is enabled and allowed.
  bool IsSupportedHorRdxIdentityOp = false;

  static bool isCmpSelMinMax(Instruction *I) {
    return match(I, m_Select(m_Cmp(), m_Value(), m_Value())) &&
           RecurrenceDescriptor::isMinMaxRecurrenceKind(getRdxKind(I));
  }

  // And/or are potentially poison-safe logical patterns like:
  // select x, y, false
  // select x, true, y
  static bool isBoolLogicOp(Instruction *I) {
    return isa<SelectInst>(I) &&
           (match(I, m_LogicalAnd()) || match(I, m_LogicalOr()));
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

    if (Kind == RecurKind::FMaximum || Kind == RecurKind::FMinimum)
      return true;

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

  /// Creates reduction operation with the current opcode.
  static Value *createOp(IRBuilder<> &Builder, RecurKind Kind, Value *LHS,
                         Value *RHS, const Twine &Name, bool UseSelect) {
    unsigned RdxOpcode = RecurrenceDescriptor::getOpcode(Kind);
    bool IsConstant = isConstant(LHS) && isConstant(RHS);
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
      if (IsConstant)
        return ConstantFP::get(LHS->getType(),
                               maxnum(cast<ConstantFP>(LHS)->getValueAPF(),
                                      cast<ConstantFP>(RHS)->getValueAPF()));
      return Builder.CreateBinaryIntrinsic(Intrinsic::maxnum, LHS, RHS);
    case RecurKind::FMin:
      if (IsConstant)
        return ConstantFP::get(LHS->getType(),
                               minnum(cast<ConstantFP>(LHS)->getValueAPF(),
                                      cast<ConstantFP>(RHS)->getValueAPF()));
      return Builder.CreateBinaryIntrinsic(Intrinsic::minnum, LHS, RHS);
    case RecurKind::FMaximum:
      if (IsConstant)
        return ConstantFP::get(LHS->getType(),
                               maximum(cast<ConstantFP>(LHS)->getValueAPF(),
                                      cast<ConstantFP>(RHS)->getValueAPF()));
      return Builder.CreateBinaryIntrinsic(Intrinsic::maximum, LHS, RHS);
    case RecurKind::FMinimum:
      if (IsConstant)
        return ConstantFP::get(LHS->getType(),
                               minimum(cast<ConstantFP>(LHS)->getValueAPF(),
                                      cast<ConstantFP>(RHS)->getValueAPF()));
      return Builder.CreateBinaryIntrinsic(Intrinsic::minimum, LHS, RHS);
    case RecurKind::SMax:
      if (IsConstant || UseSelect) {
        Value *Cmp = Builder.CreateICmpSGT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::smax, LHS, RHS);
    case RecurKind::SMin:
      if (IsConstant || UseSelect) {
        Value *Cmp = Builder.CreateICmpSLT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::smin, LHS, RHS);
    case RecurKind::UMax:
      if (IsConstant || UseSelect) {
        Value *Cmp = Builder.CreateICmpUGT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::umax, LHS, RHS);
    case RecurKind::UMin:
      if (IsConstant || UseSelect) {
        Value *Cmp = Builder.CreateICmpULT(LHS, RHS, Name);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      }
      return Builder.CreateBinaryIntrinsic(Intrinsic::umin, LHS, RHS);
    default:
      llvm_unreachable("Unknown reduction operation.");
    }
  }

  /// Creates reduction operation with the current opcode with the IR flags
  /// from \p ReductionOps, dropping nuw/nsw flags.
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
        propagateIRFlags(Sel->getCondition(), ReductionOps[0], nullptr,
                         /*IncludeWrapFlags=*/false);
        propagateIRFlags(Op, ReductionOps[1], nullptr,
                         /*IncludeWrapFlags=*/false);
        return Op;
      }
    }
    propagateIRFlags(Op, ReductionOps[0], nullptr, /*IncludeWrapFlags=*/false);
    return Op;
  }

public:
  static RecurKind getRdxKind(Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return RecurKind::None;
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

    if (match(I, m_Intrinsic<Intrinsic::maximum>(m_Value(), m_Value())))
      return RecurKind::FMaximum;
    if (match(I, m_Intrinsic<Intrinsic::minimum>(m_Value(), m_Value())))
      return RecurKind::FMinimum;
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

private:
  /// Total number of operands in the reduction operation.
  static unsigned getNumberOfOperands(Instruction *I) {
    return isCmpSelMinMax(I) ? 3 : 2;
  }

  /// Checks if the instruction is in basic block \p BB.
  /// For a cmp+sel min/max reduction check that both ops are in \p BB.
  static bool hasSameParent(Instruction *I, BasicBlock *BB) {
    if (isCmpSelMinMax(I) || isBoolLogicOp(I)) {
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

  static bool isGoodForReduction(ArrayRef<Value *> Data) {
    int Sz = Data.size();
    auto *I = dyn_cast<Instruction>(Data.front());
    return Sz > 1 || isConstant(Data.front()) ||
           (I && !isa<LoadInst>(I) && isValidForAlternation(I->getOpcode()));
  }

public:
  HorizontalReduction() = default;

  /// Try to find a reduction tree.
  bool matchAssociativeReduction(BoUpSLP &R, Instruction *Root,
                                 ScalarEvolution &SE, const DataLayout &DL,
                                 const TargetLibraryInfo &TLI) {
    RdxKind = HorizontalReduction::getRdxKind(Root);
    if (!isVectorizable(RdxKind, Root))
      return false;

    // Analyze "regular" integer/FP types for reductions - no target-specific
    // types or pointers.
    Type *Ty = Root->getType();
    if (!isValidElementType(Ty) || Ty->isPointerTy())
      return false;

    // Though the ultimate reduction may have multiple uses, its condition must
    // have only single use.
    if (auto *Sel = dyn_cast<SelectInst>(Root))
      if (!Sel->getCondition()->hasOneUse())
        return false;

    ReductionRoot = Root;

    // Iterate through all the operands of the possible reduction tree and
    // gather all the reduced values, sorting them by their value id.
    BasicBlock *BB = Root->getParent();
    bool IsCmpSelMinMax = isCmpSelMinMax(Root);
    SmallVector<Instruction *> Worklist(1, Root);
    // Checks if the operands of the \p TreeN instruction are also reduction
    // operations or should be treated as reduced values or an extra argument,
    // which is not part of the reduction.
    auto CheckOperands = [&](Instruction *TreeN,
                             SmallVectorImpl<Value *> &ExtraArgs,
                             SmallVectorImpl<Value *> &PossibleReducedVals,
                             SmallVectorImpl<Instruction *> &ReductionOps) {
      for (int I = getFirstOperandIndex(TreeN),
               End = getNumberOfOperands(TreeN);
           I < End; ++I) {
        Value *EdgeVal = getRdxOperand(TreeN, I);
        ReducedValsToOps[EdgeVal].push_back(TreeN);
        auto *EdgeInst = dyn_cast<Instruction>(EdgeVal);
        // Edge has wrong parent - mark as an extra argument.
        if (EdgeInst && !isVectorLikeInstWithConstOps(EdgeInst) &&
            !hasSameParent(EdgeInst, BB)) {
          ExtraArgs.push_back(EdgeVal);
          continue;
        }
        // If the edge is not an instruction, or it is different from the main
        // reduction opcode or has too many uses - possible reduced value.
        // Also, do not try to reduce const values, if the operation is not
        // foldable.
        if (!EdgeInst || getRdxKind(EdgeInst) != RdxKind ||
            IsCmpSelMinMax != isCmpSelMinMax(EdgeInst) ||
            !hasRequiredNumberOfUses(IsCmpSelMinMax, EdgeInst) ||
            !isVectorizable(RdxKind, EdgeInst) ||
            (R.isAnalyzedReductionRoot(EdgeInst) &&
             all_of(EdgeInst->operands(), Constant::classof))) {
          PossibleReducedVals.push_back(EdgeVal);
          continue;
        }
        ReductionOps.push_back(EdgeInst);
      }
    };
    // Try to regroup reduced values so that it gets more profitable to try to
    // reduce them. Values are grouped by their value ids, instructions - by
    // instruction op id and/or alternate op id, plus do extra analysis for
    // loads (grouping them by the distabce between pointers) and cmp
    // instructions (grouping them by the predicate).
    MapVector<size_t, MapVector<size_t, MapVector<Value *, unsigned>>>
        PossibleReducedVals;
    initReductionOps(Root);
    DenseMap<Value *, SmallVector<LoadInst *>> LoadsMap;
    SmallSet<size_t, 2> LoadKeyUsed;
    SmallPtrSet<Value *, 4> DoNotReverseVals;

    auto GenerateLoadsSubkey = [&](size_t Key, LoadInst *LI) {
      Value *Ptr = getUnderlyingObject(LI->getPointerOperand());
      if (LoadKeyUsed.contains(Key)) {
        auto LIt = LoadsMap.find(Ptr);
        if (LIt != LoadsMap.end()) {
          for (LoadInst *RLI : LIt->second) {
            if (getPointersDiff(RLI->getType(), RLI->getPointerOperand(),
                                LI->getType(), LI->getPointerOperand(), DL, SE,
                                /*StrictCheck=*/true))
              return hash_value(RLI->getPointerOperand());
          }
          for (LoadInst *RLI : LIt->second) {
            if (arePointersCompatible(RLI->getPointerOperand(),
                                      LI->getPointerOperand(), TLI)) {
              hash_code SubKey = hash_value(RLI->getPointerOperand());
              DoNotReverseVals.insert(RLI);
              return SubKey;
            }
          }
          if (LIt->second.size() > 2) {
            hash_code SubKey =
                hash_value(LIt->second.back()->getPointerOperand());
            DoNotReverseVals.insert(LIt->second.back());
            return SubKey;
          }
        }
      }
      LoadKeyUsed.insert(Key);
      LoadsMap.try_emplace(Ptr).first->second.push_back(LI);
      return hash_value(LI->getPointerOperand());
    };

    while (!Worklist.empty()) {
      Instruction *TreeN = Worklist.pop_back_val();
      SmallVector<Value *> Args;
      SmallVector<Value *> PossibleRedVals;
      SmallVector<Instruction *> PossibleReductionOps;
      CheckOperands(TreeN, Args, PossibleRedVals, PossibleReductionOps);
      // If too many extra args - mark the instruction itself as a reduction
      // value, not a reduction operation.
      if (Args.size() < 2) {
        addReductionOps(TreeN);
        // Add extra args.
        if (!Args.empty()) {
          assert(Args.size() == 1 && "Expected only single argument.");
          ExtraArgs[TreeN] = Args.front();
        }
        // Add reduction values. The values are sorted for better vectorization
        // results.
        for (Value *V : PossibleRedVals) {
          size_t Key, Idx;
          std::tie(Key, Idx) = generateKeySubkey(V, &TLI, GenerateLoadsSubkey,
                                                 /*AllowAlternate=*/false);
          ++PossibleReducedVals[Key][Idx]
                .insert(std::make_pair(V, 0))
                .first->second;
        }
        Worklist.append(PossibleReductionOps.rbegin(),
                        PossibleReductionOps.rend());
      } else {
        size_t Key, Idx;
        std::tie(Key, Idx) = generateKeySubkey(TreeN, &TLI, GenerateLoadsSubkey,
                                               /*AllowAlternate=*/false);
        ++PossibleReducedVals[Key][Idx]
              .insert(std::make_pair(TreeN, 0))
              .first->second;
      }
    }
    auto PossibleReducedValsVect = PossibleReducedVals.takeVector();
    // Sort values by the total number of values kinds to start the reduction
    // from the longest possible reduced values sequences.
    for (auto &PossibleReducedVals : PossibleReducedValsVect) {
      auto PossibleRedVals = PossibleReducedVals.second.takeVector();
      SmallVector<SmallVector<Value *>> PossibleRedValsVect;
      for (auto It = PossibleRedVals.begin(), E = PossibleRedVals.end();
           It != E; ++It) {
        PossibleRedValsVect.emplace_back();
        auto RedValsVect = It->second.takeVector();
        stable_sort(RedValsVect, llvm::less_second());
        for (const std::pair<Value *, unsigned> &Data : RedValsVect)
          PossibleRedValsVect.back().append(Data.second, Data.first);
      }
      stable_sort(PossibleRedValsVect, [](const auto &P1, const auto &P2) {
        return P1.size() > P2.size();
      });
      int NewIdx = -1;
      for (ArrayRef<Value *> Data : PossibleRedValsVect) {
        if (isGoodForReduction(Data) ||
            (isa<LoadInst>(Data.front()) && NewIdx >= 0 &&
             isa<LoadInst>(ReducedVals[NewIdx].front()) &&
             getUnderlyingObject(
                 cast<LoadInst>(Data.front())->getPointerOperand()) ==
                 getUnderlyingObject(cast<LoadInst>(ReducedVals[NewIdx].front())
                                         ->getPointerOperand()))) {
          if (NewIdx < 0) {
            NewIdx = ReducedVals.size();
            ReducedVals.emplace_back();
          }
          if (DoNotReverseVals.contains(Data.front()))
            ReducedVals[NewIdx].append(Data.begin(), Data.end());
          else
            ReducedVals[NewIdx].append(Data.rbegin(), Data.rend());
        } else {
          ReducedVals.emplace_back().append(Data.rbegin(), Data.rend());
        }
      }
    }
    // Sort the reduced values by number of same/alternate opcode and/or pointer
    // operand.
    stable_sort(ReducedVals, [](ArrayRef<Value *> P1, ArrayRef<Value *> P2) {
      return P1.size() > P2.size();
    });
    return true;
  }

  /// Attempt to vectorize the tree found by matchAssociativeReduction.
  Value *tryToReduce(BoUpSLP &V, TargetTransformInfo *TTI,
                     const TargetLibraryInfo &TLI) {
    constexpr int ReductionLimit = 4;
    constexpr unsigned RegMaxNumber = 4;
    constexpr unsigned RedValsMaxNumber = 128;
    // If there are a sufficient number of reduction values, reduce
    // to a nearby power-of-2. We can safely generate oversized
    // vectors and rely on the backend to split them to legal sizes.
    unsigned NumReducedVals =
        std::accumulate(ReducedVals.begin(), ReducedVals.end(), 0,
                        [](unsigned Num, ArrayRef<Value *> Vals) -> unsigned {
                          if (!isGoodForReduction(Vals))
                            return Num;
                          return Num + Vals.size();
                        });
    if (NumReducedVals < ReductionLimit &&
        (!AllowHorRdxIdenityOptimization ||
         all_of(ReducedVals, [](ArrayRef<Value *> RedV) {
           return RedV.size() < 2 || !allConstant(RedV) || !isSplat(RedV);
         }))) {
      for (ReductionOpsType &RdxOps : ReductionOps)
        for (Value *RdxOp : RdxOps)
          V.analyzedReductionRoot(cast<Instruction>(RdxOp));
      return nullptr;
    }

    IRBuilder<> Builder(cast<Instruction>(ReductionRoot));

    // Track the reduced values in case if they are replaced by extractelement
    // because of the vectorization.
    DenseMap<Value *, WeakTrackingVH> TrackedVals(
        ReducedVals.size() * ReducedVals.front().size() + ExtraArgs.size());
    BoUpSLP::ExtraValueToDebugLocsMap ExternallyUsedValues;
    SmallVector<std::pair<Value *, Value *>> ReplacedExternals;
    ExternallyUsedValues.reserve(ExtraArgs.size() + 1);
    // The same extra argument may be used several times, so log each attempt
    // to use it.
    for (const std::pair<Instruction *, Value *> &Pair : ExtraArgs) {
      assert(Pair.first && "DebugLoc must be set.");
      ExternallyUsedValues[Pair.second].push_back(Pair.first);
      TrackedVals.try_emplace(Pair.second, Pair.second);
    }

    // The compare instruction of a min/max is the insertion point for new
    // instructions and may be replaced with a new compare instruction.
    auto &&GetCmpForMinMaxReduction = [](Instruction *RdxRootInst) {
      assert(isa<SelectInst>(RdxRootInst) &&
             "Expected min/max reduction to have select root instruction");
      Value *ScalarCond = cast<SelectInst>(RdxRootInst)->getCondition();
      assert(isa<Instruction>(ScalarCond) &&
             "Expected min/max reduction to have compare condition");
      return cast<Instruction>(ScalarCond);
    };

    // Return new VectorizedTree, based on previous value.
    auto GetNewVectorizedTree = [&](Value *VectorizedTree, Value *Res) {
      if (VectorizedTree) {
        // Update the final value in the reduction.
        Builder.SetCurrentDebugLocation(
            cast<Instruction>(ReductionOps.front().front())->getDebugLoc());
        return createOp(Builder, RdxKind, VectorizedTree, Res, "op.rdx",
                        ReductionOps);
      }
      // Initialize the final value in the reduction.
      return Res;
    };
    // The reduction root is used as the insertion point for new instructions,
    // so set it as externally used to prevent it from being deleted.
    ExternallyUsedValues[ReductionRoot];
    SmallDenseSet<Value *> IgnoreList(ReductionOps.size() *
                                      ReductionOps.front().size());
    for (ReductionOpsType &RdxOps : ReductionOps)
      for (Value *RdxOp : RdxOps) {
        if (!RdxOp)
          continue;
        IgnoreList.insert(RdxOp);
      }
    // Intersect the fast-math-flags from all reduction operations.
    FastMathFlags RdxFMF;
    RdxFMF.set();
    for (Value *U : IgnoreList)
      if (auto *FPMO = dyn_cast<FPMathOperator>(U))
        RdxFMF &= FPMO->getFastMathFlags();
    bool IsCmpSelMinMax = isCmpSelMinMax(cast<Instruction>(ReductionRoot));

    // Need to track reduced vals, they may be changed during vectorization of
    // subvectors.
    for (ArrayRef<Value *> Candidates : ReducedVals)
      for (Value *V : Candidates)
        TrackedVals.try_emplace(V, V);

    DenseMap<Value *, unsigned> VectorizedVals(ReducedVals.size());
    // List of the values that were reduced in other trees as part of gather
    // nodes and thus requiring extract if fully vectorized in other trees.
    SmallPtrSet<Value *, 4> RequiredExtract;
    Value *VectorizedTree = nullptr;
    bool CheckForReusedReductionOps = false;
    // Try to vectorize elements based on their type.
    for (unsigned I = 0, E = ReducedVals.size(); I < E; ++I) {
      ArrayRef<Value *> OrigReducedVals = ReducedVals[I];
      InstructionsState S = getSameOpcode(OrigReducedVals, TLI);
      SmallVector<Value *> Candidates;
      Candidates.reserve(2 * OrigReducedVals.size());
      DenseMap<Value *, Value *> TrackedToOrig(2 * OrigReducedVals.size());
      for (unsigned Cnt = 0, Sz = OrigReducedVals.size(); Cnt < Sz; ++Cnt) {
        Value *RdxVal = TrackedVals.find(OrigReducedVals[Cnt])->second;
        // Check if the reduction value was not overriden by the extractelement
        // instruction because of the vectorization and exclude it, if it is not
        // compatible with other values.
        // Also check if the instruction was folded to constant/other value.
        auto *Inst = dyn_cast<Instruction>(RdxVal);
        if ((Inst && isVectorLikeInstWithConstOps(Inst) &&
             (!S.getOpcode() || !S.isOpcodeOrAlt(Inst))) ||
            (S.getOpcode() && !Inst))
          continue;
        Candidates.push_back(RdxVal);
        TrackedToOrig.try_emplace(RdxVal, OrigReducedVals[Cnt]);
      }
      bool ShuffledExtracts = false;
      // Try to handle shuffled extractelements.
      if (S.getOpcode() == Instruction::ExtractElement && !S.isAltShuffle() &&
          I + 1 < E) {
        InstructionsState NextS = getSameOpcode(ReducedVals[I + 1], TLI);
        if (NextS.getOpcode() == Instruction::ExtractElement &&
            !NextS.isAltShuffle()) {
          SmallVector<Value *> CommonCandidates(Candidates);
          for (Value *RV : ReducedVals[I + 1]) {
            Value *RdxVal = TrackedVals.find(RV)->second;
            // Check if the reduction value was not overriden by the
            // extractelement instruction because of the vectorization and
            // exclude it, if it is not compatible with other values.
            if (auto *Inst = dyn_cast<Instruction>(RdxVal))
              if (!NextS.getOpcode() || !NextS.isOpcodeOrAlt(Inst))
                continue;
            CommonCandidates.push_back(RdxVal);
            TrackedToOrig.try_emplace(RdxVal, RV);
          }
          SmallVector<int> Mask;
          if (isFixedVectorShuffle(CommonCandidates, Mask)) {
            ++I;
            Candidates.swap(CommonCandidates);
            ShuffledExtracts = true;
          }
        }
      }

      // Emit code for constant values.
      if (AllowHorRdxIdenityOptimization && Candidates.size() > 1 &&
          allConstant(Candidates)) {
        Value *Res = Candidates.front();
        ++VectorizedVals.try_emplace(Candidates.front(), 0).first->getSecond();
        for (Value *VC : ArrayRef(Candidates).drop_front()) {
          Res = createOp(Builder, RdxKind, Res, VC, "const.rdx", ReductionOps);
          ++VectorizedVals.try_emplace(VC, 0).first->getSecond();
          if (auto *ResI = dyn_cast<Instruction>(Res))
            V.analyzedReductionRoot(ResI);
        }
        VectorizedTree = GetNewVectorizedTree(VectorizedTree, Res);
        continue;
      }

      unsigned NumReducedVals = Candidates.size();
      if (NumReducedVals < ReductionLimit &&
          (NumReducedVals < 2 || !AllowHorRdxIdenityOptimization ||
           !isSplat(Candidates)))
        continue;

      // Check if we support repeated scalar values processing (optimization of
      // original scalar identity operations on matched horizontal reductions).
      IsSupportedHorRdxIdentityOp =
          AllowHorRdxIdenityOptimization && RdxKind != RecurKind::Mul &&
          RdxKind != RecurKind::FMul && RdxKind != RecurKind::FMulAdd;
      // Gather same values.
      MapVector<Value *, unsigned> SameValuesCounter;
      if (IsSupportedHorRdxIdentityOp)
        for (Value *V : Candidates)
          ++SameValuesCounter.insert(std::make_pair(V, 0)).first->second;
      // Used to check if the reduced values used same number of times. In this
      // case the compiler may produce better code. E.g. if reduced values are
      // aabbccdd (8 x values), then the first node of the tree will have a node
      // for 4 x abcd + shuffle <4 x abcd>, <0, 0, 1, 1, 2, 2, 3, 3>.
      // Plus, the final reduction will be performed on <8 x aabbccdd>.
      // Instead compiler may build <4 x abcd> tree immediately, + reduction (4
      // x abcd) * 2.
      // Currently it only handles add/fadd/xor. and/or/min/max do not require
      // this analysis, other operations may require an extra estimation of
      // the profitability.
      bool SameScaleFactor = false;
      bool OptReusedScalars = IsSupportedHorRdxIdentityOp &&
                              SameValuesCounter.size() != Candidates.size();
      if (OptReusedScalars) {
        SameScaleFactor =
            (RdxKind == RecurKind::Add || RdxKind == RecurKind::FAdd ||
             RdxKind == RecurKind::Xor) &&
            all_of(drop_begin(SameValuesCounter),
                   [&SameValuesCounter](const std::pair<Value *, unsigned> &P) {
                     return P.second == SameValuesCounter.front().second;
                   });
        Candidates.resize(SameValuesCounter.size());
        transform(SameValuesCounter, Candidates.begin(),
                  [](const auto &P) { return P.first; });
        NumReducedVals = Candidates.size();
        // Have a reduction of the same element.
        if (NumReducedVals == 1) {
          Value *OrigV = TrackedToOrig.find(Candidates.front())->second;
          unsigned Cnt = SameValuesCounter.lookup(OrigV);
          Value *RedVal =
              emitScaleForReusedOps(Candidates.front(), Builder, Cnt);
          VectorizedTree = GetNewVectorizedTree(VectorizedTree, RedVal);
          VectorizedVals.try_emplace(OrigV, Cnt);
          continue;
        }
      }

      unsigned MaxVecRegSize = V.getMaxVecRegSize();
      unsigned EltSize = V.getVectorElementSize(Candidates[0]);
      unsigned MaxElts =
          RegMaxNumber * llvm::bit_floor(MaxVecRegSize / EltSize);

      unsigned ReduxWidth = std::min<unsigned>(
          llvm::bit_floor(NumReducedVals), std::max(RedValsMaxNumber, MaxElts));
      unsigned Start = 0;
      unsigned Pos = Start;
      // Restarts vectorization attempt with lower vector factor.
      unsigned PrevReduxWidth = ReduxWidth;
      bool CheckForReusedReductionOpsLocal = false;
      auto &&AdjustReducedVals = [&Pos, &Start, &ReduxWidth, NumReducedVals,
                                  &CheckForReusedReductionOpsLocal,
                                  &PrevReduxWidth, &V,
                                  &IgnoreList](bool IgnoreVL = false) {
        bool IsAnyRedOpGathered = !IgnoreVL && V.isAnyGathered(IgnoreList);
        if (!CheckForReusedReductionOpsLocal && PrevReduxWidth == ReduxWidth) {
          // Check if any of the reduction ops are gathered. If so, worth
          // trying again with less number of reduction ops.
          CheckForReusedReductionOpsLocal |= IsAnyRedOpGathered;
        }
        ++Pos;
        if (Pos < NumReducedVals - ReduxWidth + 1)
          return IsAnyRedOpGathered;
        Pos = Start;
        ReduxWidth /= 2;
        return IsAnyRedOpGathered;
      };
      bool AnyVectorized = false;
      while (Pos < NumReducedVals - ReduxWidth + 1 &&
             ReduxWidth >= ReductionLimit) {
        // Dependency in tree of the reduction ops - drop this attempt, try
        // later.
        if (CheckForReusedReductionOpsLocal && PrevReduxWidth != ReduxWidth &&
            Start == 0) {
          CheckForReusedReductionOps = true;
          break;
        }
        PrevReduxWidth = ReduxWidth;
        ArrayRef<Value *> VL(std::next(Candidates.begin(), Pos), ReduxWidth);
        // Beeing analyzed already - skip.
        if (V.areAnalyzedReductionVals(VL)) {
          (void)AdjustReducedVals(/*IgnoreVL=*/true);
          continue;
        }
        // Early exit if any of the reduction values were deleted during
        // previous vectorization attempts.
        if (any_of(VL, [&V](Value *RedVal) {
              auto *RedValI = dyn_cast<Instruction>(RedVal);
              if (!RedValI)
                return false;
              return V.isDeleted(RedValI);
            }))
          break;
        V.buildTree(VL, IgnoreList);
        if (V.isTreeTinyAndNotFullyVectorizable(/*ForReduction=*/true)) {
          if (!AdjustReducedVals())
            V.analyzedReductionVals(VL);
          continue;
        }
        if (V.isLoadCombineReductionCandidate(RdxKind)) {
          if (!AdjustReducedVals())
            V.analyzedReductionVals(VL);
          continue;
        }
        V.reorderTopToBottom();
        // No need to reorder the root node at all.
        V.reorderBottomToTop(/*IgnoreReorder=*/true);
        // Keep extracted other reduction values, if they are used in the
        // vectorization trees.
        BoUpSLP::ExtraValueToDebugLocsMap LocalExternallyUsedValues(
            ExternallyUsedValues);
        for (unsigned Cnt = 0, Sz = ReducedVals.size(); Cnt < Sz; ++Cnt) {
          if (Cnt == I || (ShuffledExtracts && Cnt == I - 1))
            continue;
          for_each(ReducedVals[Cnt],
                   [&LocalExternallyUsedValues, &TrackedVals](Value *V) {
                     if (isa<Instruction>(V))
                       LocalExternallyUsedValues[TrackedVals[V]];
                   });
        }
        if (!IsSupportedHorRdxIdentityOp) {
          // Number of uses of the candidates in the vector of values.
          assert(SameValuesCounter.empty() &&
                 "Reused values counter map is not empty");
          for (unsigned Cnt = 0; Cnt < NumReducedVals; ++Cnt) {
            if (Cnt >= Pos && Cnt < Pos + ReduxWidth)
              continue;
            Value *V = Candidates[Cnt];
            Value *OrigV = TrackedToOrig.find(V)->second;
            ++SameValuesCounter[OrigV];
          }
        }
        SmallPtrSet<Value *, 4> VLScalars(VL.begin(), VL.end());
        // Gather externally used values.
        SmallPtrSet<Value *, 4> Visited;
        for (unsigned Cnt = 0; Cnt < NumReducedVals; ++Cnt) {
          if (Cnt >= Pos && Cnt < Pos + ReduxWidth)
            continue;
          Value *RdxVal = Candidates[Cnt];
          if (!Visited.insert(RdxVal).second)
            continue;
          // Check if the scalar was vectorized as part of the vectorization
          // tree but not the top node.
          if (!VLScalars.contains(RdxVal) && V.isVectorized(RdxVal)) {
            LocalExternallyUsedValues[RdxVal];
            continue;
          }
          Value *OrigV = TrackedToOrig.find(RdxVal)->second;
          unsigned NumOps =
              VectorizedVals.lookup(RdxVal) + SameValuesCounter[OrigV];
          if (NumOps != ReducedValsToOps.find(OrigV)->second.size())
            LocalExternallyUsedValues[RdxVal];
        }
        // Do not need the list of reused scalars in regular mode anymore.
        if (!IsSupportedHorRdxIdentityOp)
          SameValuesCounter.clear();
        for (Value *RdxVal : VL)
          if (RequiredExtract.contains(RdxVal))
            LocalExternallyUsedValues[RdxVal];
        // Update LocalExternallyUsedValues for the scalar, replaced by
        // extractelement instructions.
        for (const std::pair<Value *, Value *> &Pair : ReplacedExternals) {
          auto *It = ExternallyUsedValues.find(Pair.first);
          if (It == ExternallyUsedValues.end())
            continue;
          LocalExternallyUsedValues[Pair.second].append(It->second);
        }
        V.buildExternalUses(LocalExternallyUsedValues);

        V.computeMinimumValueSizes();

        // Estimate cost.
        InstructionCost TreeCost = V.getTreeCost(VL);
        InstructionCost ReductionCost =
            getReductionCost(TTI, VL, IsCmpSelMinMax, ReduxWidth, RdxFMF);
        InstructionCost Cost = TreeCost + ReductionCost;
        LLVM_DEBUG(dbgs() << "SLP: Found cost = " << Cost
                          << " for reduction\n");
        if (!Cost.isValid())
          return nullptr;
        if (Cost >= -SLPCostThreshold) {
          V.getORE()->emit([&]() {
            return OptimizationRemarkMissed(
                       SV_NAME, "HorSLPNotBeneficial",
                       ReducedValsToOps.find(VL[0])->second.front())
                   << "Vectorizing horizontal reduction is possible "
                   << "but not beneficial with cost " << ore::NV("Cost", Cost)
                   << " and threshold "
                   << ore::NV("Threshold", -SLPCostThreshold);
          });
          if (!AdjustReducedVals())
            V.analyzedReductionVals(VL);
          continue;
        }

        LLVM_DEBUG(dbgs() << "SLP: Vectorizing horizontal reduction at cost:"
                          << Cost << ". (HorRdx)\n");
        V.getORE()->emit([&]() {
          return OptimizationRemark(
                     SV_NAME, "VectorizedHorizontalReduction",
                     ReducedValsToOps.find(VL[0])->second.front())
                 << "Vectorized horizontal reduction with cost "
                 << ore::NV("Cost", Cost) << " and with tree size "
                 << ore::NV("TreeSize", V.getTreeSize());
        });

        Builder.setFastMathFlags(RdxFMF);

        // Emit a reduction. If the root is a select (min/max idiom), the insert
        // point is the compare condition of that select.
        Instruction *RdxRootInst = cast<Instruction>(ReductionRoot);
        Instruction *InsertPt = RdxRootInst;
        if (IsCmpSelMinMax)
          InsertPt = GetCmpForMinMaxReduction(RdxRootInst);

        // Vectorize a tree.
        Value *VectorizedRoot = V.vectorizeTree(LocalExternallyUsedValues,
                                                ReplacedExternals, InsertPt);

        Builder.SetInsertPoint(InsertPt);

        // To prevent poison from leaking across what used to be sequential,
        // safe, scalar boolean logic operations, the reduction operand must be
        // frozen.
        if (isBoolLogicOp(RdxRootInst))
          VectorizedRoot = Builder.CreateFreeze(VectorizedRoot);

        // Emit code to correctly handle reused reduced values, if required.
        if (OptReusedScalars && !SameScaleFactor) {
          VectorizedRoot =
              emitReusedOps(VectorizedRoot, Builder, V.getRootNodeScalars(),
                            SameValuesCounter, TrackedToOrig);
        }

        Value *ReducedSubTree =
            emitReduction(VectorizedRoot, Builder, ReduxWidth, TTI);

        // Improved analysis for add/fadd/xor reductions with same scale factor
        // for all operands of reductions. We can emit scalar ops for them
        // instead.
        if (OptReusedScalars && SameScaleFactor)
          ReducedSubTree = emitScaleForReusedOps(
              ReducedSubTree, Builder, SameValuesCounter.front().second);

        VectorizedTree = GetNewVectorizedTree(VectorizedTree, ReducedSubTree);
        // Count vectorized reduced values to exclude them from final reduction.
        for (Value *RdxVal : VL) {
          Value *OrigV = TrackedToOrig.find(RdxVal)->second;
          if (IsSupportedHorRdxIdentityOp) {
            VectorizedVals.try_emplace(OrigV, SameValuesCounter[RdxVal]);
            continue;
          }
          ++VectorizedVals.try_emplace(OrigV, 0).first->getSecond();
          if (!V.isVectorized(RdxVal))
            RequiredExtract.insert(RdxVal);
        }
        Pos += ReduxWidth;
        Start = Pos;
        ReduxWidth = llvm::bit_floor(NumReducedVals - Pos);
        AnyVectorized = true;
      }
      if (OptReusedScalars && !AnyVectorized) {
        for (const std::pair<Value *, unsigned> &P : SameValuesCounter) {
          Value *RedVal = emitScaleForReusedOps(P.first, Builder, P.second);
          VectorizedTree = GetNewVectorizedTree(VectorizedTree, RedVal);
          Value *OrigV = TrackedToOrig.find(P.first)->second;
          VectorizedVals.try_emplace(OrigV, P.second);
        }
        continue;
      }
    }
    if (VectorizedTree) {
      // Reorder operands of bool logical op in the natural order to avoid
      // possible problem with poison propagation. If not possible to reorder
      // (both operands are originally RHS), emit an extra freeze instruction
      // for the LHS operand.
      // I.e., if we have original code like this:
      // RedOp1 = select i1 ?, i1 LHS, i1 false
      // RedOp2 = select i1 RHS, i1 ?, i1 false

      // Then, we swap LHS/RHS to create a new op that matches the poison
      // semantics of the original code.

      // If we have original code like this and both values could be poison:
      // RedOp1 = select i1 ?, i1 LHS, i1 false
      // RedOp2 = select i1 ?, i1 RHS, i1 false

      // Then, we must freeze LHS in the new op.
      auto &&FixBoolLogicalOps =
          [&Builder, VectorizedTree](Value *&LHS, Value *&RHS,
                                     Instruction *RedOp1, Instruction *RedOp2) {
            if (!isBoolLogicOp(RedOp1))
              return;
            if (LHS == VectorizedTree || getRdxOperand(RedOp1, 0) == LHS ||
                isGuaranteedNotToBePoison(LHS))
              return;
            if (!isBoolLogicOp(RedOp2))
              return;
            if (RHS == VectorizedTree || getRdxOperand(RedOp2, 0) == RHS ||
                isGuaranteedNotToBePoison(RHS)) {
              std::swap(LHS, RHS);
              return;
            }
            LHS = Builder.CreateFreeze(LHS);
          };
      // Finish the reduction.
      // Need to add extra arguments and not vectorized possible reduction
      // values.
      // Try to avoid dependencies between the scalar remainders after
      // reductions.
      auto &&FinalGen =
          [this, &Builder, &TrackedVals, &FixBoolLogicalOps](
              ArrayRef<std::pair<Instruction *, Value *>> InstVals) {
            unsigned Sz = InstVals.size();
            SmallVector<std::pair<Instruction *, Value *>> ExtraReds(Sz / 2 +
                                                                     Sz % 2);
            for (unsigned I = 0, E = (Sz / 2) * 2; I < E; I += 2) {
              Instruction *RedOp = InstVals[I + 1].first;
              Builder.SetCurrentDebugLocation(RedOp->getDebugLoc());
              Value *RdxVal1 = InstVals[I].second;
              Value *StableRdxVal1 = RdxVal1;
              auto It1 = TrackedVals.find(RdxVal1);
              if (It1 != TrackedVals.end())
                StableRdxVal1 = It1->second;
              Value *RdxVal2 = InstVals[I + 1].second;
              Value *StableRdxVal2 = RdxVal2;
              auto It2 = TrackedVals.find(RdxVal2);
              if (It2 != TrackedVals.end())
                StableRdxVal2 = It2->second;
              // To prevent poison from leaking across what used to be
              // sequential, safe, scalar boolean logic operations, the
              // reduction operand must be frozen.
              FixBoolLogicalOps(StableRdxVal1, StableRdxVal2, InstVals[I].first,
                                RedOp);
              Value *ExtraRed = createOp(Builder, RdxKind, StableRdxVal1,
                                         StableRdxVal2, "op.rdx", ReductionOps);
              ExtraReds[I / 2] = std::make_pair(InstVals[I].first, ExtraRed);
            }
            if (Sz % 2 == 1)
              ExtraReds[Sz / 2] = InstVals.back();
            return ExtraReds;
          };
      SmallVector<std::pair<Instruction *, Value *>> ExtraReductions;
      ExtraReductions.emplace_back(cast<Instruction>(ReductionRoot),
                                   VectorizedTree);
      SmallPtrSet<Value *, 8> Visited;
      for (ArrayRef<Value *> Candidates : ReducedVals) {
        for (Value *RdxVal : Candidates) {
          if (!Visited.insert(RdxVal).second)
            continue;
          unsigned NumOps = VectorizedVals.lookup(RdxVal);
          for (Instruction *RedOp :
               ArrayRef(ReducedValsToOps.find(RdxVal)->second)
                   .drop_back(NumOps))
            ExtraReductions.emplace_back(RedOp, RdxVal);
        }
      }
      for (auto &Pair : ExternallyUsedValues) {
        // Add each externally used value to the final reduction.
        for (auto *I : Pair.second)
          ExtraReductions.emplace_back(I, Pair.first);
      }
      // Iterate through all not-vectorized reduction values/extra arguments.
      while (ExtraReductions.size() > 1) {
        VectorizedTree = ExtraReductions.front().second;
        SmallVector<std::pair<Instruction *, Value *>> NewReds =
            FinalGen(ExtraReductions);
        ExtraReductions.swap(NewReds);
      }
      VectorizedTree = ExtraReductions.front().second;

      ReductionRoot->replaceAllUsesWith(VectorizedTree);

      // The original scalar reduction is expected to have no remaining
      // uses outside the reduction tree itself.  Assert that we got this
      // correct, replace internal uses with undef, and mark for eventual
      // deletion.
#ifndef NDEBUG
      SmallSet<Value *, 4> IgnoreSet;
      for (ArrayRef<Value *> RdxOps : ReductionOps)
        IgnoreSet.insert(RdxOps.begin(), RdxOps.end());
#endif
      for (ArrayRef<Value *> RdxOps : ReductionOps) {
        for (Value *Ignore : RdxOps) {
          if (!Ignore)
            continue;
#ifndef NDEBUG
          for (auto *U : Ignore->users()) {
            assert(IgnoreSet.count(U) &&
                   "All users must be either in the reduction ops list.");
          }
#endif
          if (!Ignore->use_empty()) {
            Value *Undef = UndefValue::get(Ignore->getType());
            Ignore->replaceAllUsesWith(Undef);
          }
          V.eraseInstruction(cast<Instruction>(Ignore));
        }
      }
    } else if (!CheckForReusedReductionOps) {
      for (ReductionOpsType &RdxOps : ReductionOps)
        for (Value *RdxOp : RdxOps)
          V.analyzedReductionRoot(cast<Instruction>(RdxOp));
    }
    return VectorizedTree;
  }

private:
  /// Calculate the cost of a reduction.
  InstructionCost getReductionCost(TargetTransformInfo *TTI,
                                   ArrayRef<Value *> ReducedVals,
                                   bool IsCmpSelMinMax, unsigned ReduxWidth,
                                   FastMathFlags FMF) {
    TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
    Type *ScalarTy = ReducedVals.front()->getType();
    FixedVectorType *VectorTy = FixedVectorType::get(ScalarTy, ReduxWidth);
    InstructionCost VectorCost = 0, ScalarCost;
    // If all of the reduced values are constant, the vector cost is 0, since
    // the reduction value can be calculated at the compile time.
    bool AllConsts = allConstant(ReducedVals);
    auto EvaluateScalarCost = [&](function_ref<InstructionCost()> GenCostFn) {
      InstructionCost Cost = 0;
      // Scalar cost is repeated for N-1 elements.
      int Cnt = ReducedVals.size();
      for (Value *RdxVal : ReducedVals) {
        if (Cnt == 1)
          break;
        --Cnt;
        if (RdxVal->hasNUsesOrMore(IsCmpSelMinMax ? 3 : 2)) {
          Cost += GenCostFn();
          continue;
        }
        InstructionCost ScalarCost = 0;
        for (User *U : RdxVal->users()) {
          auto *RdxOp = cast<Instruction>(U);
          if (hasRequiredNumberOfUses(IsCmpSelMinMax, RdxOp)) {
            ScalarCost += TTI->getInstructionCost(RdxOp, CostKind);
            continue;
          }
          ScalarCost = InstructionCost::getInvalid();
          break;
        }
        if (ScalarCost.isValid())
          Cost += ScalarCost;
        else
          Cost += GenCostFn();
      }
      return Cost;
    };
    switch (RdxKind) {
    case RecurKind::Add:
    case RecurKind::Mul:
    case RecurKind::Or:
    case RecurKind::And:
    case RecurKind::Xor:
    case RecurKind::FAdd:
    case RecurKind::FMul: {
      unsigned RdxOpcode = RecurrenceDescriptor::getOpcode(RdxKind);
      if (!AllConsts)
        VectorCost =
            TTI->getArithmeticReductionCost(RdxOpcode, VectorTy, FMF, CostKind);
      ScalarCost = EvaluateScalarCost([&]() {
        return TTI->getArithmeticInstrCost(RdxOpcode, ScalarTy, CostKind);
      });
      break;
    }
    case RecurKind::FMax:
    case RecurKind::FMin:
    case RecurKind::FMaximum:
    case RecurKind::FMinimum:
    case RecurKind::SMax:
    case RecurKind::SMin:
    case RecurKind::UMax:
    case RecurKind::UMin: {
      Intrinsic::ID Id = getMinMaxReductionIntrinsicOp(RdxKind);
      if (!AllConsts)
        VectorCost = TTI->getMinMaxReductionCost(Id, VectorTy, FMF, CostKind);
      ScalarCost = EvaluateScalarCost([&]() {
        IntrinsicCostAttributes ICA(Id, ScalarTy, {ScalarTy, ScalarTy}, FMF);
        return TTI->getIntrinsicInstrCost(ICA, CostKind);
      });
      break;
    }
    default:
      llvm_unreachable("Expected arithmetic or min/max reduction operation");
    }

    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << VectorCost - ScalarCost
                      << " for reduction of " << shortBundleName(ReducedVals)
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
    return createSimpleTargetReduction(Builder, VectorizedValue, RdxKind);
  }

  /// Emits optimized code for unique scalar value reused \p Cnt times.
  Value *emitScaleForReusedOps(Value *VectorizedValue, IRBuilderBase &Builder,
                               unsigned Cnt) {
    assert(IsSupportedHorRdxIdentityOp &&
           "The optimization of matched scalar identity horizontal reductions "
           "must be supported.");
    switch (RdxKind) {
    case RecurKind::Add: {
      // res = mul vv, n
      Value *Scale = ConstantInt::get(VectorizedValue->getType(), Cnt);
      LLVM_DEBUG(dbgs() << "SLP: Add (to-mul) " << Cnt << "of "
                        << VectorizedValue << ". (HorRdx)\n");
      return Builder.CreateMul(VectorizedValue, Scale);
    }
    case RecurKind::Xor: {
      // res = n % 2 ? 0 : vv
      LLVM_DEBUG(dbgs() << "SLP: Xor " << Cnt << "of " << VectorizedValue
                        << ". (HorRdx)\n");
      if (Cnt % 2 == 0)
        return Constant::getNullValue(VectorizedValue->getType());
      return VectorizedValue;
    }
    case RecurKind::FAdd: {
      // res = fmul v, n
      Value *Scale = ConstantFP::get(VectorizedValue->getType(), Cnt);
      LLVM_DEBUG(dbgs() << "SLP: FAdd (to-fmul) " << Cnt << "of "
                        << VectorizedValue << ". (HorRdx)\n");
      return Builder.CreateFMul(VectorizedValue, Scale);
    }
    case RecurKind::And:
    case RecurKind::Or:
    case RecurKind::SMax:
    case RecurKind::SMin:
    case RecurKind::UMax:
    case RecurKind::UMin:
    case RecurKind::FMax:
    case RecurKind::FMin:
    case RecurKind::FMaximum:
    case RecurKind::FMinimum:
      // res = vv
      return VectorizedValue;
    case RecurKind::Mul:
    case RecurKind::FMul:
    case RecurKind::FMulAdd:
    case RecurKind::IAnyOf:
    case RecurKind::FAnyOf:
    case RecurKind::None:
      llvm_unreachable("Unexpected reduction kind for repeated scalar.");
    }
    return nullptr;
  }

  /// Emits actual operation for the scalar identity values, found during
  /// horizontal reduction analysis.
  Value *emitReusedOps(Value *VectorizedValue, IRBuilderBase &Builder,
                       ArrayRef<Value *> VL,
                       const MapVector<Value *, unsigned> &SameValuesCounter,
                       const DenseMap<Value *, Value *> &TrackedToOrig) {
    assert(IsSupportedHorRdxIdentityOp &&
           "The optimization of matched scalar identity horizontal reductions "
           "must be supported.");
    switch (RdxKind) {
    case RecurKind::Add: {
      // root = mul prev_root, <1, 1, n, 1>
      SmallVector<Constant *> Vals;
      for (Value *V : VL) {
        unsigned Cnt = SameValuesCounter.lookup(TrackedToOrig.find(V)->second);
        Vals.push_back(ConstantInt::get(V->getType(), Cnt, /*IsSigned=*/false));
      }
      auto *Scale = ConstantVector::get(Vals);
      LLVM_DEBUG(dbgs() << "SLP: Add (to-mul) " << Scale << "of "
                        << VectorizedValue << ". (HorRdx)\n");
      return Builder.CreateMul(VectorizedValue, Scale);
    }
    case RecurKind::And:
    case RecurKind::Or:
      // No need for multiple or/and(s).
      LLVM_DEBUG(dbgs() << "SLP: And/or of same " << VectorizedValue
                        << ". (HorRdx)\n");
      return VectorizedValue;
    case RecurKind::SMax:
    case RecurKind::SMin:
    case RecurKind::UMax:
    case RecurKind::UMin:
    case RecurKind::FMax:
    case RecurKind::FMin:
    case RecurKind::FMaximum:
    case RecurKind::FMinimum:
      // No need for multiple min/max(s) of the same value.
      LLVM_DEBUG(dbgs() << "SLP: Max/min of same " << VectorizedValue
                        << ". (HorRdx)\n");
      return VectorizedValue;
    case RecurKind::Xor: {
      // Replace values with even number of repeats with 0, since
      // x xor x = 0.
      // root = shuffle prev_root, zeroinitalizer, <0, 1, 2, vf, 4, vf, 5, 6,
      // 7>, if elements 4th and 6th elements have even number of repeats.
      SmallVector<int> Mask(
          cast<FixedVectorType>(VectorizedValue->getType())->getNumElements(),
          PoisonMaskElem);
      std::iota(Mask.begin(), Mask.end(), 0);
      bool NeedShuffle = false;
      for (unsigned I = 0, VF = VL.size(); I < VF; ++I) {
        Value *V = VL[I];
        unsigned Cnt = SameValuesCounter.lookup(TrackedToOrig.find(V)->second);
        if (Cnt % 2 == 0) {
          Mask[I] = VF;
          NeedShuffle = true;
        }
      }
      LLVM_DEBUG(dbgs() << "SLP: Xor <"; for (int I
                                              : Mask) dbgs()
                                         << I << " ";
                 dbgs() << "> of " << VectorizedValue << ". (HorRdx)\n");
      if (NeedShuffle)
        VectorizedValue = Builder.CreateShuffleVector(
            VectorizedValue,
            ConstantVector::getNullValue(VectorizedValue->getType()), Mask);
      return VectorizedValue;
    }
    case RecurKind::FAdd: {
      // root = fmul prev_root, <1.0, 1.0, n.0, 1.0>
      SmallVector<Constant *> Vals;
      for (Value *V : VL) {
        unsigned Cnt = SameValuesCounter.lookup(TrackedToOrig.find(V)->second);
        Vals.push_back(ConstantFP::get(V->getType(), Cnt));
      }
      auto *Scale = ConstantVector::get(Vals);
      return Builder.CreateFMul(VectorizedValue, Scale);
    }
    case RecurKind::Mul:
    case RecurKind::FMul:
    case RecurKind::FMulAdd:
    case RecurKind::IAnyOf:
    case RecurKind::FAnyOf:
    case RecurKind::None:
      llvm_unreachable("Unexpected reduction kind for reused scalars.");
    }
    return nullptr;
  }
};
} // end anonymous namespace

static std::optional<unsigned> getAggregateSize(Instruction *InsertInst) {
  if (auto *IE = dyn_cast<InsertElementInst>(InsertInst))
    return cast<FixedVectorType>(IE->getType())->getNumElements();

  unsigned AggregateSize = 1;
  auto *IV = cast<InsertValueInst>(InsertInst);
  Type *CurrentType = IV->getType();
  do {
    if (auto *ST = dyn_cast<StructType>(CurrentType)) {
      for (auto *Elt : ST->elements())
        if (Elt != ST->getElementType(0)) // check homogeneity
          return std::nullopt;
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
      return std::nullopt;
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
    std::optional<unsigned> OperandIndex =
        getInsertIndex(LastInsertInst, OperandOffset);
    if (!OperandIndex)
      return;
    if (isa<InsertElementInst, InsertValueInst>(InsertedOperand)) {
      findBuildAggregate_rec(cast<Instruction>(InsertedOperand), TTI,
                             BuildVectorOpds, InsertElts, *OperandIndex);

    } else {
      BuildVectorOpds[*OperandIndex] = InsertedOperand;
      InsertElts[*OperandIndex] = LastInsertInst;
    }
    LastInsertInst = dyn_cast<Instruction>(LastInsertInst->getOperand(0));
  } while (LastInsertInst != nullptr &&
           isa<InsertValueInst, InsertElementInst>(LastInsertInst) &&
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

  std::optional<unsigned> AggregateSize = getAggregateSize(LastInsertInst);
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

/// Try and get a reduction instruction from a phi node.
///
/// Given a phi node \p P in a block \p ParentBB, consider possible reductions
/// if they come from either \p ParentBB or a containing loop latch.
///
/// \returns A candidate reduction value if possible, or \code nullptr \endcode
/// if not possible.
static Instruction *getReductionInstr(const DominatorTree *DT, PHINode *P,
                                      BasicBlock *ParentBB, LoopInfo *LI) {
  // There are situations where the reduction value is not dominated by the
  // reduction phi. Vectorizing such cases has been reported to cause
  // miscompiles. See PR25787.
  auto DominatedReduxValue = [&](Value *R) {
    return isa<Instruction>(R) &&
           DT->dominates(P->getParent(), cast<Instruction>(R)->getParent());
  };

  Instruction *Rdx = nullptr;

  // Return the incoming value if it comes from the same BB as the phi node.
  if (P->getIncomingBlock(0) == ParentBB) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(0));
  } else if (P->getIncomingBlock(1) == ParentBB) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(1));
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
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(0));
  } else if (P->getIncomingBlock(1) == BBLatch) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(1));
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
  if (match(I, m_Intrinsic<Intrinsic::maximum>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::minimum>(m_Value(V0), m_Value(V1))))
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

/// We could have an initial reduction that is not an add.
///  r *= v1 + v2 + v3 + v4
/// In such a case start looking for a tree rooted in the first '+'.
/// \Returns the new root if found, which may be nullptr if not an instruction.
static Instruction *tryGetSecondaryReductionRoot(PHINode *Phi,
                                                 Instruction *Root) {
  assert((isa<BinaryOperator>(Root) || isa<SelectInst>(Root) ||
          isa<IntrinsicInst>(Root)) &&
         "Expected binop, select, or intrinsic for reduction matching");
  Value *LHS =
      Root->getOperand(HorizontalReduction::getFirstOperandIndex(Root));
  Value *RHS =
      Root->getOperand(HorizontalReduction::getFirstOperandIndex(Root) + 1);
  if (LHS == Phi)
    return dyn_cast<Instruction>(RHS);
  if (RHS == Phi)
    return dyn_cast<Instruction>(LHS);
  return nullptr;
}

/// \p Returns the first operand of \p I that does not match \p Phi. If
/// operand is not an instruction it returns nullptr.
static Instruction *getNonPhiOperand(Instruction *I, PHINode *Phi) {
  Value *Op0 = nullptr;
  Value *Op1 = nullptr;
  if (!matchRdxBop(I, Op0, Op1))
    return nullptr;
  return dyn_cast<Instruction>(Op0 == Phi ? Op1 : Op0);
}

/// \Returns true if \p I is a candidate instruction for reduction vectorization.
static bool isReductionCandidate(Instruction *I) {
  bool IsSelect = match(I, m_Select(m_Value(), m_Value(), m_Value()));
  Value *B0 = nullptr, *B1 = nullptr;
  bool IsBinop = matchRdxBop(I, B0, B1);
  return IsBinop || IsSelect;
}

bool SLPVectorizerPass::vectorizeHorReduction(
    PHINode *P, Instruction *Root, BasicBlock *BB, BoUpSLP &R, TargetTransformInfo *TTI,
    SmallVectorImpl<WeakTrackingVH> &PostponedInsts) {
  if (!ShouldVectorizeHor)
    return false;
  bool TryOperandsAsNewSeeds = P && isa<BinaryOperator>(Root);

  if (Root->getParent() != BB || isa<PHINode>(Root))
    return false;

  // If we can find a secondary reduction root, use that instead.
  auto SelectRoot = [&]() {
    if (TryOperandsAsNewSeeds && isReductionCandidate(Root) &&
        HorizontalReduction::getRdxKind(Root) != RecurKind::None)
      if (Instruction *NewRoot = tryGetSecondaryReductionRoot(P, Root))
        return NewRoot;
    return Root;
  };

  // Start analysis starting from Root instruction. If horizontal reduction is
  // found, try to vectorize it. If it is not a horizontal reduction or
  // vectorization is not possible or not effective, and currently analyzed
  // instruction is a binary operation, try to vectorize the operands, using
  // pre-order DFS traversal order. If the operands were not vectorized, repeat
  // the same procedure considering each operand as a possible root of the
  // horizontal reduction.
  // Interrupt the process if the Root instruction itself was vectorized or all
  // sub-trees not higher that RecursionMaxDepth were analyzed/vectorized.
  // If a horizintal reduction was not matched or vectorized we collect
  // instructions for possible later attempts for vectorization.
  std::queue<std::pair<Instruction *, unsigned>> Stack;
  Stack.emplace(SelectRoot(), 0);
  SmallPtrSet<Value *, 8> VisitedInstrs;
  bool Res = false;
  auto &&TryToReduce = [this, TTI, &R](Instruction *Inst) -> Value * {
    if (R.isAnalyzedReductionRoot(Inst))
      return nullptr;
    if (!isReductionCandidate(Inst))
      return nullptr;
    HorizontalReduction HorRdx;
    if (!HorRdx.matchAssociativeReduction(R, Inst, *SE, *DL, *TLI))
      return nullptr;
    return HorRdx.tryToReduce(R, TTI, *TLI);
  };
  auto TryAppendToPostponedInsts = [&](Instruction *FutureSeed) {
    if (TryOperandsAsNewSeeds && FutureSeed == Root) {
      FutureSeed = getNonPhiOperand(Root, P);
      if (!FutureSeed)
        return false;
    }
    // Do not collect CmpInst or InsertElementInst/InsertValueInst as their
    // analysis is done separately.
    if (!isa<CmpInst, InsertElementInst, InsertValueInst>(FutureSeed))
      PostponedInsts.push_back(FutureSeed);
    return true;
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
    if (Value *VectorizedV = TryToReduce(Inst)) {
      Res = true;
      if (auto *I = dyn_cast<Instruction>(VectorizedV)) {
        // Try to find another reduction.
        Stack.emplace(I, Level);
        continue;
      }
    } else {
      // We could not vectorize `Inst` so try to use it as a future seed.
      if (!TryAppendToPostponedInsts(Inst)) {
        assert(Stack.empty() && "Expected empty stack");
        break;
      }
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
            if (!isa<PHINode, CmpInst, InsertElementInst, InsertValueInst>(I) &&
                !R.isDeleted(I) && I->getParent() == BB)
              Stack.emplace(I, Level);
  }
  return Res;
}

bool SLPVectorizerPass::vectorizeRootInstruction(PHINode *P, Instruction *Root,
                                                 BasicBlock *BB, BoUpSLP &R,
                                                 TargetTransformInfo *TTI) {
  SmallVector<WeakTrackingVH> PostponedInsts;
  bool Res = vectorizeHorReduction(P, Root, BB, R, TTI, PostponedInsts);
  Res |= tryToVectorize(PostponedInsts, R);
  return Res;
}

bool SLPVectorizerPass::tryToVectorize(ArrayRef<WeakTrackingVH> Insts,
                                       BoUpSLP &R) {
  bool Res = false;
  for (Value *V : Insts)
    if (auto *Inst = dyn_cast<Instruction>(V); Inst && !R.isDeleted(Inst))
      Res |= tryToVectorize(Inst, R);
  return Res;
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
static bool tryToVectorizeSequence(
    SmallVectorImpl<T *> &Incoming, function_ref<bool(T *, T *)> Comparator,
    function_ref<bool(T *, T *)> AreCompatible,
    function_ref<bool(ArrayRef<T *>, bool)> TryToVectorizeHelper,
    bool MaxVFOnly, BoUpSLP &R) {
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
        TryToVectorizeHelper(ArrayRef(IncIt, NumElts), MaxVFOnly)) {
      // Success start over because instructions might have been changed.
      Changed = true;
    } else {
      /// \Returns the minimum number of elements that we will attempt to
      /// vectorize.
      auto GetMinNumElements = [&R](Value *V) {
        unsigned EltSize = R.getVectorElementSize(V);
        return std::max(2U, R.getMaxVecRegSize() / EltSize);
      };
      if (NumElts < GetMinNumElements(*IncIt) &&
          (Candidates.empty() ||
           Candidates.front()->getType() == (*IncIt)->getType())) {
        Candidates.append(IncIt, std::next(IncIt, NumElts));
      }
    }
    // Final attempt to vectorize instructions with the same types.
    if (Candidates.size() > 1 &&
        (SameTypeIt == E || (*SameTypeIt)->getType() != (*IncIt)->getType())) {
      if (TryToVectorizeHelper(Candidates, /*MaxVFOnly=*/false)) {
        // Success start over because instructions might have been changed.
        Changed = true;
      } else if (MaxVFOnly) {
        // Try to vectorize using small vectors.
        for (auto *It = Candidates.begin(), *End = Candidates.end();
             It != End;) {
          auto *SameTypeIt = It;
          while (SameTypeIt != End && AreCompatible(*SameTypeIt, *It))
            ++SameTypeIt;
          unsigned NumElts = (SameTypeIt - It);
          if (NumElts > 1 && TryToVectorizeHelper(ArrayRef(It, NumElts),
                                                  /*MaxVFOnly=*/false))
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
static bool compareCmp(Value *V, Value *V2, TargetLibraryInfo &TLI,
                       const DominatorTree &DT) {
  assert(isValidElementType(V->getType()) &&
         isValidElementType(V2->getType()) &&
         "Expected valid element types only.");
  auto *CI1 = cast<CmpInst>(V);
  auto *CI2 = cast<CmpInst>(V2);
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
  bool CI1Preds = Pred1 == BasePred1;
  bool CI2Preds = Pred2 == BasePred1;
  for (int I = 0, E = CI1->getNumOperands(); I < E; ++I) {
    auto *Op1 = CI1->getOperand(CI1Preds ? I : E - I - 1);
    auto *Op2 = CI2->getOperand(CI2Preds ? I : E - I - 1);
    if (Op1->getValueID() < Op2->getValueID())
      return !IsCompatibility;
    if (Op1->getValueID() > Op2->getValueID())
      return false;
    if (auto *I1 = dyn_cast<Instruction>(Op1))
      if (auto *I2 = dyn_cast<Instruction>(Op2)) {
        if (IsCompatibility) {
          if (I1->getParent() != I2->getParent())
            return false;
        } else {
          // Try to compare nodes with same parent.
          DomTreeNodeBase<BasicBlock> *NodeI1 = DT.getNode(I1->getParent());
          DomTreeNodeBase<BasicBlock> *NodeI2 = DT.getNode(I2->getParent());
          if (!NodeI1)
            return NodeI2 != nullptr;
          if (!NodeI2)
            return false;
          assert((NodeI1 == NodeI2) ==
                     (NodeI1->getDFSNumIn() == NodeI2->getDFSNumIn()) &&
                 "Different nodes should have different DFS numbers");
          if (NodeI1 != NodeI2)
            return NodeI1->getDFSNumIn() < NodeI2->getDFSNumIn();
        }
        InstructionsState S = getSameOpcode({I1, I2}, TLI);
        if (S.getOpcode() && (IsCompatibility || !S.isAltShuffle()))
          continue;
        return !IsCompatibility && I1->getOpcode() < I2->getOpcode();
      }
  }
  return IsCompatibility;
}

template <typename ItT>
bool SLPVectorizerPass::vectorizeCmpInsts(iterator_range<ItT> CmpInsts,
                                          BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  // Try to find reductions first.
  for (CmpInst *I : CmpInsts) {
    if (R.isDeleted(I))
      continue;
    for (Value *Op : I->operands())
      if (auto *RootOp = dyn_cast<Instruction>(Op))
        Changed |= vectorizeRootInstruction(nullptr, RootOp, BB, R, TTI);
  }
  // Try to vectorize operands as vector bundles.
  for (CmpInst *I : CmpInsts) {
    if (R.isDeleted(I))
      continue;
    Changed |= tryToVectorize(I, R);
  }
  // Try to vectorize list of compares.
  // Sort by type, compare predicate, etc.
  auto CompareSorter = [&](Value *V, Value *V2) {
    if (V == V2)
      return false;
    return compareCmp<false>(V, V2, *TLI, *DT);
  };

  auto AreCompatibleCompares = [&](Value *V1, Value *V2) {
    if (V1 == V2)
      return true;
    return compareCmp<true>(V1, V2, *TLI, *DT);
  };

  SmallVector<Value *> Vals;
  for (Instruction *V : CmpInsts)
    if (!R.isDeleted(V) && isValidElementType(V->getType()))
      Vals.push_back(V);
  if (Vals.size() <= 1)
    return Changed;
  Changed |= tryToVectorizeSequence<Value>(
      Vals, CompareSorter, AreCompatibleCompares,
      [this, &R](ArrayRef<Value *> Candidates, bool MaxVFOnly) {
        // Exclude possible reductions from other blocks.
        bool ArePossiblyReducedInOtherBlock = any_of(Candidates, [](Value *V) {
          return any_of(V->users(), [V](User *U) {
            auto *Select = dyn_cast<SelectInst>(U);
            return Select &&
                   Select->getParent() != cast<Instruction>(V)->getParent();
          });
        });
        if (ArePossiblyReducedInOtherBlock)
          return false;
        return tryToVectorizeList(Candidates, R, MaxVFOnly);
      },
      /*MaxVFOnly=*/true, R);
  return Changed;
}

bool SLPVectorizerPass::vectorizeInserts(InstSetVector &Instructions,
                                         BasicBlock *BB, BoUpSLP &R) {
  assert(all_of(Instructions,
                [](auto *I) {
                  return isa<InsertElementInst, InsertValueInst>(I);
                }) &&
         "This function only accepts Insert instructions");
  bool OpsChanged = false;
  SmallVector<WeakTrackingVH> PostponedInsts;
  // pass1 - try to vectorize reductions only
  for (auto *I : reverse(Instructions)) {
    if (R.isDeleted(I))
      continue;
    OpsChanged |= vectorizeHorReduction(nullptr, I, BB, R, TTI, PostponedInsts);
  }
  // pass2 - try to match and vectorize a buildvector sequence.
  for (auto *I : reverse(Instructions)) {
    if (R.isDeleted(I) || isa<CmpInst>(I))
      continue;
    if (auto *LastInsertValue = dyn_cast<InsertValueInst>(I)) {
      OpsChanged |= vectorizeInsertValueInst(LastInsertValue, BB, R);
    } else if (auto *LastInsertElem = dyn_cast<InsertElementInst>(I)) {
      OpsChanged |= vectorizeInsertElementInst(LastInsertElem, BB, R);
    }
  }
  // Now try to vectorize postponed instructions.
  OpsChanged |= tryToVectorize(PostponedInsts, R);

  Instructions.clear();
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
    for (int I = 0, E = Opcodes1.size(); I < E; ++I) {
      // Undefs are compatible with any other value.
      if (isa<UndefValue>(Opcodes1[I]) || isa<UndefValue>(Opcodes2[I])) {
        if (isa<Instruction>(Opcodes1[I]))
          return true;
        if (isa<Instruction>(Opcodes2[I]))
          return false;
        if (isa<Constant>(Opcodes1[I]) && !isa<UndefValue>(Opcodes1[I]))
          return true;
        if (isa<Constant>(Opcodes2[I]) && !isa<UndefValue>(Opcodes2[I]))
          return false;
        if (isa<UndefValue>(Opcodes1[I]) && isa<UndefValue>(Opcodes2[I]))
          continue;
        return isa<UndefValue>(Opcodes2[I]);
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
          InstructionsState S = getSameOpcode({I1, I2}, *TLI);
          if (S.getOpcode() && !S.isAltShuffle())
            continue;
          return I1->getOpcode() < I2->getOpcode();
        }
      if (isa<Constant>(Opcodes1[I]) && isa<Constant>(Opcodes2[I]))
        return Opcodes1[I]->getValueID() < Opcodes2[I]->getValueID();
      if (isa<Instruction>(Opcodes1[I]))
        return true;
      if (isa<Instruction>(Opcodes2[I]))
        return false;
      if (isa<Constant>(Opcodes1[I]))
        return true;
      if (isa<Constant>(Opcodes2[I]))
        return false;
      if (Opcodes1[I]->getValueID() < Opcodes2[I]->getValueID())
        return true;
      if (Opcodes1[I]->getValueID() > Opcodes2[I]->getValueID())
        return false;
    }
    return false;
  };
  auto AreCompatiblePHIs = [&PHIToOpcodes, this](Value *V1, Value *V2) {
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
          InstructionsState S = getSameOpcode({I1, I2}, *TLI);
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

    if (Incoming.size() <= 1)
      break;

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
        Incoming, PHICompare, AreCompatiblePHIs,
        [this, &R](ArrayRef<Value *> Candidates, bool MaxVFOnly) {
          return tryToVectorizeList(Candidates, R, MaxVFOnly);
        },
        /*MaxVFOnly=*/true, R);
    Changed |= HaveVectorizedPhiNodes;
    VisitedInstrs.insert(Incoming.begin(), Incoming.end());
  } while (HaveVectorizedPhiNodes);

  VisitedInstrs.clear();

  InstSetVector PostProcessInserts;
  SmallSetVector<CmpInst *, 8> PostProcessCmps;
  // Vectorizes Inserts in `PostProcessInserts` and if `VecctorizeCmps` is true
  // also vectorizes `PostProcessCmps`.
  auto VectorizeInsertsAndCmps = [&](bool VectorizeCmps) {
    bool Changed = vectorizeInserts(PostProcessInserts, BB, R);
    if (VectorizeCmps) {
      Changed |= vectorizeCmpInsts(reverse(PostProcessCmps), BB, R);
      PostProcessCmps.clear();
    }
    PostProcessInserts.clear();
    return Changed;
  };
  // Returns true if `I` is in `PostProcessInserts` or `PostProcessCmps`.
  auto IsInPostProcessInstrs = [&](Instruction *I) {
    if (auto *Cmp = dyn_cast<CmpInst>(I))
      return PostProcessCmps.contains(Cmp);
    return isa<InsertElementInst, InsertValueInst>(I) &&
           PostProcessInserts.contains(I);
  };
  // Returns true if `I` is an instruction without users, like terminator, or
  // function call with ignored return value, store. Ignore unused instructions
  // (basing on instruction type, except for CallInst and InvokeInst).
  auto HasNoUsers = [](Instruction *I) {
    return I->use_empty() &&
           (I->getType()->isVoidTy() || isa<CallInst, InvokeInst>(I));
  };
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
      if (HasNoUsers(&*it) &&
          VectorizeInsertsAndCmps(/*VectorizeCmps=*/it->isTerminator())) {
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
        Instruction *Root = getReductionInstr(DT, P, BB, LI);
        if (Root && vectorizeRootInstruction(P, Root, BB, R, TTI)) {
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

        // Postponed instructions should not be vectorized here, delay their
        // vectorization.
        if (auto *PI = dyn_cast<Instruction>(P->getIncomingValue(I));
            PI && !IsInPostProcessInstrs(PI))
          Changed |= vectorizeRootInstruction(nullptr, PI,
                                              P->getIncomingBlock(I), R, TTI);
      }
      continue;
    }

    if (HasNoUsers(&*it)) {
      bool OpsChanged = false;
      auto *SI = dyn_cast<StoreInst>(it);
      bool TryToVectorizeRoot = ShouldStartVectorizeHorAtStore || !SI;
      if (SI) {
        auto I = Stores.find(getUnderlyingObject(SI->getPointerOperand()));
        // Try to vectorize chain in store, if this is the only store to the
        // address in the block.
        // TODO: This is just a temporarily solution to save compile time. Need
        // to investigate if we can safely turn on slp-vectorize-hor-store
        // instead to allow lookup for reduction chains in all non-vectorized
        // stores (need to check side effects and compile time).
        TryToVectorizeRoot = (I == Stores.end() || I->second.size() == 1) &&
                             SI->getValueOperand()->hasOneUse();
      }
      if (TryToVectorizeRoot) {
        for (auto *V : it->operand_values()) {
          // Postponed instructions should not be vectorized here, delay their
          // vectorization.
          if (auto *VI = dyn_cast<Instruction>(V);
              VI && !IsInPostProcessInstrs(VI))
            // Try to match and vectorize a horizontal reduction.
            OpsChanged |= vectorizeRootInstruction(nullptr, VI, BB, R, TTI);
        }
      }
      // Start vectorization of post-process list of instructions from the
      // top-tree instructions to try to vectorize as many instructions as
      // possible.
      OpsChanged |=
          VectorizeInsertsAndCmps(/*VectorizeCmps=*/it->isTerminator());
      if (OpsChanged) {
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }
    }

    if (isa<InsertElementInst, InsertValueInst>(it))
      PostProcessInserts.insert(&*it);
    else if (isa<CmpInst>(it))
      PostProcessCmps.insert(cast<CmpInst>(&*it));
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
    if (V->getValueOperand()->getType()->getTypeID() <
        V2->getValueOperand()->getType()->getTypeID())
      return true;
    if (V->getValueOperand()->getType()->getTypeID() >
        V2->getValueOperand()->getType()->getTypeID())
      return false;
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
        assert(NodeI2 && "Should only process reachable instructions");
        assert((NodeI1 == NodeI2) ==
                   (NodeI1->getDFSNumIn() == NodeI2->getDFSNumIn()) &&
               "Different nodes should have different DFS numbers");
        if (NodeI1 != NodeI2)
          return NodeI1->getDFSNumIn() < NodeI2->getDFSNumIn();
        InstructionsState S = getSameOpcode({I1, I2}, *TLI);
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

  auto &&AreCompatibleStores = [this](StoreInst *V1, StoreInst *V2) {
    if (V1 == V2)
      return true;
    if (V1->getValueOperand()->getType() != V2->getValueOperand()->getType())
      return false;
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
        InstructionsState S = getSameOpcode({I1, I2}, *TLI);
        return S.getOpcode() > 0;
      }
    if (isa<Constant>(V1->getValueOperand()) &&
        isa<Constant>(V2->getValueOperand()))
      return true;
    return V1->getValueOperand()->getValueID() ==
           V2->getValueOperand()->getValueID();
  };

  // Attempt to sort and vectorize each of the store-groups.
  for (auto &Pair : Stores) {
    if (Pair.second.size() < 2)
      continue;

    LLVM_DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
                      << Pair.second.size() << ".\n");

    if (!isValidElementType(Pair.second.front()->getValueOperand()->getType()))
      continue;

    // Reverse stores to do bottom-to-top analysis. This is important if the
    // values are stores to the same addresses several times, in this case need
    // to follow the stores order (reversed to meet the memory dependecies).
    SmallVector<StoreInst *> ReversedStores(Pair.second.rbegin(),
                                            Pair.second.rend());
    Changed |= tryToVectorizeSequence<StoreInst>(
        ReversedStores, StoreSorter, AreCompatibleStores,
        [this, &R](ArrayRef<StoreInst *> Candidates, bool) {
          return vectorizeStores(Candidates, R);
        },
        /*MaxVFOnly=*/false, R);
  }
  return Changed;
}
