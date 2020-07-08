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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
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
#include "llvm/Analysis/AssumptionCache.h"
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
#include "llvm/IR/PassManager.h"
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

// The Look-ahead heuristic goes through the users of the bundle to calculate
// the users cost in getExternalUsesCost(). To avoid compilation time increase
// we limit the number of users visited to this value.
static cl::opt<unsigned> LookAheadUsersBudget(
    "slp-look-ahead-users-budget", cl::init(2), cl::Hidden,
    cl::desc("The maximum number of users to visit while visiting the "
             "predecessors. This prevents compilation time increase."));

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

/// \returns true if all of the instructions in \p VL are in the same block or
/// false otherwise.
static bool allSameBlock(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return false;
  BasicBlock *BB = I0->getParent();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I)
      return false;

    if (BB != I->getParent())
      return false;
  }
  return true;
}

/// \returns True if all of the values in \p VL are constants (but not
/// globals/constant expressions).
static bool allConstant(ArrayRef<Value *> VL) {
  // Constant expressions and globals can't be vectorized like normal integer/FP
  // constants.
  for (Value *i : VL)
    if (!isa<Constant>(i) || isa<ConstantExpr>(i) || isa<GlobalValue>(i))
      return false;
  return true;
}

/// \returns True if all of the values in \p VL are identical.
static bool isSplat(ArrayRef<Value *> VL) {
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    if (VL[i] != VL[0])
      return false;
  return true;
}

/// \returns True if \p I is commutative, handles CmpInst as well as Instruction.
static bool isCommutative(Instruction *I) {
  if (auto *IC = dyn_cast<CmpInst>(I))
    return IC->isCommutative();
  return I->isCommutative();
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
/// %ins1 = insertelement <4 x i8> undef, i8 %x0x0, i32 0
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
/// %1 = insertelement <4 x i8> undef, i8 %x0, i32 0
/// %2 = insertelement <4 x i8> %1, i8 %x3, i32 1
/// %3 = insertelement <4 x i8> %2, i8 %y1, i32 2
/// %4 = insertelement <4 x i8> %3, i8 %y2, i32 3
/// %5 = mul <4 x i8> %4, %4
/// %6 = extractelement <4 x i8> %5, i32 0
/// %ins1 = insertelement <4 x i8> undef, i8 %6, i32 0
/// %7 = extractelement <4 x i8> %5, i32 1
/// %ins2 = insertelement <4 x i8> %ins1, i8 %7, i32 1
/// %8 = extractelement <4 x i8> %5, i32 2
/// %ins3 = insertelement <4 x i8> %ins2, i8 %8, i32 2
/// %9 = extractelement <4 x i8> %5, i32 3
/// %ins4 = insertelement <4 x i8> %ins3, i8 %9, i32 3
/// ret <4 x i8> %ins4
/// InstCombiner transforms this into a shuffle and vector mul
/// TODO: Can we split off and reuse the shuffle mask detection from
/// TargetTransformInfo::getInstructionThroughput?
static Optional<TargetTransformInfo::ShuffleKind>
isShuffle(ArrayRef<Value *> VL) {
  auto *EI0 = cast<ExtractElementInst>(VL[0]);
  unsigned Size = EI0->getVectorOperandType()->getNumElements();
  Value *Vec1 = nullptr;
  Value *Vec2 = nullptr;
  enum ShuffleMode { Unknown, Select, Permute };
  ShuffleMode CommonShuffleMode = Unknown;
  for (unsigned I = 0, E = VL.size(); I < E; ++I) {
    auto *EI = cast<ExtractElementInst>(VL[I]);
    auto *Vec = EI->getVectorOperand();
    // All vector operands must have the same number of vector elements.
    if (cast<VectorType>(Vec->getType())->getNumElements() != Size)
      return None;
    auto *Idx = dyn_cast<ConstantInt>(EI->getIndexOperand());
    if (!Idx)
      return None;
    // Undefined behavior if Idx is negative or >= Size.
    if (Idx->getValue().uge(Size))
      continue;
    unsigned IntIdx = Idx->getValue().getZExtValue();
    // We can extractelement from undef vector.
    if (isa<UndefValue>(Vec))
      continue;
    // For correct shuffling we have to have at most 2 different vector operands
    // in all extractelement instructions.
    if (!Vec1 || Vec1 == Vec)
      Vec1 = Vec;
    else if (!Vec2 || Vec2 == Vec)
      Vec2 = Vec;
    else
      return None;
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
  bool isAltShuffle() const { return getOpcode() != getAltOpcode(); }

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

/// \returns analysis of the Instructions in \p VL described in
/// InstructionsState, the Opcode that we suppose the whole list
/// could be vectorized even if its structure is diverse.
static InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                       unsigned BaseIndex = 0) {
  // Make sure these are all Instructions.
  if (llvm::any_of(VL, [](Value *V) { return !isa<Instruction>(V); }))
    return InstructionsState(VL[BaseIndex], nullptr, nullptr);

  bool IsCastOp = isa<CastInst>(VL[BaseIndex]);
  bool IsBinOp = isa<BinaryOperator>(VL[BaseIndex]);
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
    for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i) {
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
static MemoryLocation getLocation(Instruction *I, AliasAnalysis *AA) {
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

namespace llvm {

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

  BoUpSLP(Function *Func, ScalarEvolution *Se, TargetTransformInfo *Tti,
          TargetLibraryInfo *TLi, AliasAnalysis *Aa, LoopInfo *Li,
          DominatorTree *Dt, AssumptionCache *AC, DemandedBits *DB,
          const DataLayout *DL, OptimizationRemarkEmitter *ORE)
      : F(Func), SE(Se), TTI(Tti), TLI(TLi), AA(Aa), LI(Li), DT(Dt), AC(AC),
        DB(DB), DL(DL), ORE(ORE), Builder(Se->getContext()) {
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
      MaxVecRegSize = TTI->getRegisterBitWidth(true);

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
  int getSpillCost() const;

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost();

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst.
  void buildTree(ArrayRef<Value *> Roots,
                 ArrayRef<Value *> UserIgnoreLst = None);

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst taking
  /// into account (and updating it, if required) list of externally used
  /// values stored in \p ExternallyUsedValues.
  void buildTree(ArrayRef<Value *> Roots,
                 ExtraValueToDebugLocsMap &ExternallyUsedValues,
                 ArrayRef<Value *> UserIgnoreLst = None);

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    VectorizableTree.clear();
    ScalarToTreeEntry.clear();
    MustGather.clear();
    ExternalUses.clear();
    NumOpsWantToKeepOrder.clear();
    NumOpsWantToKeepOriginalOrder = 0;
    for (auto &Iter : BlocksSchedules) {
      BlockScheduling *BS = Iter.second.get();
      BS->clear();
    }
    MinBWs.clear();
  }

  unsigned getTreeSize() const { return VectorizableTree.size(); }

  /// Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  /// \returns The best order of instructions for vectorization.
  Optional<ArrayRef<unsigned>> bestOrder() const {
    auto I = std::max_element(
        NumOpsWantToKeepOrder.begin(), NumOpsWantToKeepOrder.end(),
        [](const decltype(NumOpsWantToKeepOrder)::value_type &D1,
           const decltype(NumOpsWantToKeepOrder)::value_type &D2) {
          return D1.second < D2.second;
        });
    if (I == NumOpsWantToKeepOrder.end() ||
        I->getSecond() <= NumOpsWantToKeepOriginalOrder)
      return None;

    return makeArrayRef(I->getFirst());
  }

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

  /// Check if homogeneous aggregate is isomorphic to some VectorType.
  /// Accepts homogeneous multidimensional aggregate of scalars/vectors like
  /// {[4 x i16], [4 x i16]}, { <2 x float>, <2 x float> },
  /// {{{i16, i16}, {i16, i16}}, {{i16, i16}, {i16, i16}}} and so on.
  ///
  /// \returns number of elements in vector if isomorphism exists, 0 otherwise.
  unsigned canMapToVector(Type *T, const DataLayout &DL) const;

  /// \returns True if the VectorizableTree is both tiny and not fully
  /// vectorizable. We do not vectorize such trees.
  bool isTreeTinyAndNotFullyVectorizable() const;

  /// Assume that a legal-sized 'or'-reduction of shifted/zexted loaded values
  /// can be load combined in the backend. Load combining may not be allowed in
  /// the IR optimizer, so we do not want to alter the pattern. For example,
  /// partially transforming a scalar bswap() pattern into vector code is
  /// effectively impossible for the backend to undo.
  /// TODO: If load combining is allowed in the IR optimizer, this analysis
  ///       may not be necessary.
  bool isLoadCombineReductionCandidate(unsigned ReductionOpcode) const;

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

    // The hard-coded scores listed here are not very important. When computing
    // the scores of matching one sub-tree with another, we are basically
    // counting the number of values that are matching. So even if all scores
    // are set to 1, we would still get a decent matching result.
    // However, sometimes we have to break ties. For example we may have to
    // choose between matching loads vs matching opcodes. This is what these
    // scores are helping us with: they provide the order of preference.

    /// Loads from consecutive memory addresses, e.g. load(A[i]), load(A[i+1]).
    static const int ScoreConsecutiveLoads = 3;
    /// ExtractElementInst from same vector and consecutive indexes.
    static const int ScoreConsecutiveExtracts = 3;
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
    /// User exteranl to the vectorized code.
    static const int ExternalUseCost = 1;
    /// The user is internal but in a different lane.
    static const int UserInDiffLaneCost = ExternalUseCost;

    /// \returns the score of placing \p V1 and \p V2 in consecutive lanes.
    static int getShallowScore(Value *V1, Value *V2, const DataLayout &DL,
                               ScalarEvolution &SE) {
      auto *LI1 = dyn_cast<LoadInst>(V1);
      auto *LI2 = dyn_cast<LoadInst>(V2);
      if (LI1 && LI2)
        return isConsecutiveAccess(LI1, LI2, DL, SE)
                   ? VLOperands::ScoreConsecutiveLoads
                   : VLOperands::ScoreFail;

      auto *C1 = dyn_cast<Constant>(V1);
      auto *C2 = dyn_cast<Constant>(V2);
      if (C1 && C2)
        return VLOperands::ScoreConstants;

      // Extracts from consecutive indexes of the same vector better score as
      // the extracts could be optimized away.
      Value *EV;
      ConstantInt *Ex1Idx, *Ex2Idx;
      if (match(V1, m_ExtractElt(m_Value(EV), m_ConstantInt(Ex1Idx))) &&
          match(V2, m_ExtractElt(m_Deferred(EV), m_ConstantInt(Ex2Idx))) &&
          Ex1Idx->getZExtValue() + 1 == Ex2Idx->getZExtValue())
        return VLOperands::ScoreConsecutiveExtracts;

      auto *I1 = dyn_cast<Instruction>(V1);
      auto *I2 = dyn_cast<Instruction>(V2);
      if (I1 && I2) {
        if (I1 == I2)
          return VLOperands::ScoreSplat;
        InstructionsState S = getSameOpcode({I1, I2});
        // Note: Only consider instructions with <= 2 operands to avoid
        // complexity explosion.
        if (S.getOpcode() && S.MainOp->getNumOperands() <= 2)
          return S.isAltShuffle() ? VLOperands::ScoreAltOpcodes
                                  : VLOperands::ScoreSameOpcode;
      }

      if (isa<UndefValue>(V2))
        return VLOperands::ScoreUndef;

      return VLOperands::ScoreFail;
    }

    /// Holds the values and their lane that are taking part in the look-ahead
    /// score calculation. This is used in the external uses cost calculation.
    SmallDenseMap<Value *, int> InLookAheadValues;

    /// \Returns the additinal cost due to uses of \p LHS and \p RHS that are
    /// either external to the vectorized code, or require shuffling.
    int getExternalUsesCost(const std::pair<Value *, int> &LHS,
                            const std::pair<Value *, int> &RHS) {
      int Cost = 0;
      std::array<std::pair<Value *, int>, 2> Values = {{LHS, RHS}};
      for (int Idx = 0, IdxE = Values.size(); Idx != IdxE; ++Idx) {
        Value *V = Values[Idx].first;
        // Calculate the absolute lane, using the minimum relative lane of LHS
        // and RHS as base and Idx as the offset.
        int Ln = std::min(LHS.second, RHS.second) + Idx;
        assert(Ln >= 0 && "Bad lane calculation");
        unsigned UsersBudget = LookAheadUsersBudget;
        for (User *U : V->users()) {
          if (const TreeEntry *UserTE = R.getTreeEntry(U)) {
            // The user is in the VectorizableTree. Check if we need to insert.
            auto It = llvm::find(UserTE->Scalars, U);
            assert(It != UserTE->Scalars.end() && "U is in UserTE");
            int UserLn = std::distance(UserTE->Scalars.begin(), It);
            assert(UserLn >= 0 && "Bad lane");
            if (UserLn != Ln)
              Cost += UserInDiffLaneCost;
          } else {
            // Check if the user is in the look-ahead code.
            auto It2 = InLookAheadValues.find(U);
            if (It2 != InLookAheadValues.end()) {
              // The user is in the look-ahead code. Check the lane.
              if (It2->second != Ln)
                Cost += UserInDiffLaneCost;
            } else {
              // The user is neither in SLP tree nor in the look-ahead code.
              Cost += ExternalUseCost;
            }
          }
          // Limit the number of visited uses to cap compilation time.
          if (--UsersBudget == 0)
            break;
        }
      }
      return Cost;
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
    int getScoreAtLevelRec(const std::pair<Value *, int> &LHS,
                           const std::pair<Value *, int> &RHS, int CurrLevel,
                           int MaxLevel) {

      Value *V1 = LHS.first;
      Value *V2 = RHS.first;
      // Get the shallow score of V1 and V2.
      int ShallowScoreAtThisLevel =
          std::max((int)ScoreFail, getShallowScore(V1, V2, DL, SE) -
                                       getExternalUsesCost(LHS, RHS));
      int Lane1 = LHS.second;
      int Lane2 = RHS.second;

      // If reached MaxLevel,
      //  or if V1 and V2 are not instructions,
      //  or if they are SPLAT,
      //  or if they are not consecutive, early return the current cost.
      auto *I1 = dyn_cast<Instruction>(V1);
      auto *I2 = dyn_cast<Instruction>(V2);
      if (CurrLevel == MaxLevel || !(I1 && I2) || I1 == I2 ||
          ShallowScoreAtThisLevel == VLOperands::ScoreFail ||
          (isa<LoadInst>(I1) && isa<LoadInst>(I2) && ShallowScoreAtThisLevel))
        return ShallowScoreAtThisLevel;
      assert(I1 && I2 && "Should have early exited.");

      // Keep track of in-tree values for determining the external-use cost.
      InLookAheadValues[V1] = Lane1;
      InLookAheadValues[V2] = Lane2;

      // Contains the I2 operand indexes that got matched with I1 operands.
      SmallSet<unsigned, 4> Op2Used;

      // Recursion towards the operands of I1 and I2. We are trying all possbile
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
          int TmpScore = getScoreAtLevelRec({I1->getOperand(OpIdx1), Lane1},
                                            {I2->getOperand(OpIdx2), Lane2},
                                            CurrLevel + 1, MaxLevel);
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

    /// \Returns the look-ahead score, which tells us how much the sub-trees
    /// rooted at \p LHS and \p RHS match, the more they match the higher the
    /// score. This helps break ties in an informed way when we cannot decide on
    /// the order of the operands by just considering the immediate
    /// predecessors.
    int getLookAheadScore(const std::pair<Value *, int> &LHS,
                          const std::pair<Value *, int> &RHS) {
      InLookAheadValues.clear();
      return getScoreAtLevelRec(LHS, RHS, 1, LookAheadMaxDepth);
    }

    // Search all operands in Ops[*][Lane] for the one that matches best
    // Ops[OpIdx][LastLane] and return its opreand index.
    // If no good match can be found, return None.
    Optional<unsigned>
    getBestOperand(unsigned OpIdx, int Lane, int LastLane,
                   ArrayRef<ReorderingMode> ReorderingModes) {
      unsigned NumOperands = getNumOperands();

      // The operand of the previous lane at OpIdx.
      Value *OpLastLane = getData(OpIdx, LastLane).V;

      // Our strategy mode for OpIdx.
      ReorderingMode RMode = ReorderingModes[OpIdx];

      // The linearized opcode of the operand at OpIdx, Lane.
      bool OpIdxAPO = getData(OpIdx, Lane).APO;

      // The best operand index and its score.
      // Sometimes we have more than one option (e.g., Opcode and Undefs), so we
      // are using the score to differentiate between the two.
      struct BestOpData {
        Optional<unsigned> Idx = None;
        unsigned Score = 0;
      } BestOp;

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
          unsigned Score =
              getLookAheadScore({OpLeft, LastLane}, {OpRight, Lane});
          if (Score > BestOp.Score) {
            BestOp.Idx = Idx;
            BestOp.Score = Score;
          }
          break;
        }
        case ReorderingMode::Splat:
          if (Op == OpLastLane)
            BestOp.Idx = Idx;
          break;
        case ReorderingMode::Failed:
          return None;
        }
      }

      if (BestOp.Idx) {
        getData(BestOp.Idx.getValue(), Lane).IsUsed = true;
        return BestOp.Idx;
      }
      // If we could not find a good match return None.
      return None;
    }

    /// Helper for reorderOperandVecs. \Returns the lane that we should start
    /// reordering from. This is the one which has the least number of operands
    /// that can freely move about.
    unsigned getBestLaneToStartReordering() const {
      unsigned BestLane = 0;
      unsigned Min = UINT_MAX;
      for (unsigned Lane = 0, NumLanes = getNumLanes(); Lane != NumLanes;
           ++Lane) {
        unsigned NumFreeOps = getMaxNumOperandsThatCanBeReordered(Lane);
        if (NumFreeOps < Min) {
          Min = NumFreeOps;
          BestLane = Lane;
        }
      }
      return BestLane;
    }

    /// \Returns the maximum number of operands that are allowed to be reordered
    /// for \p Lane. This is used as a heuristic for selecting the first lane to
    /// start operand reordering.
    unsigned getMaxNumOperandsThatCanBeReordered(unsigned Lane) const {
      unsigned CntTrue = 0;
      unsigned NumOperands = getNumOperands();
      // Operands with the same APO can be reordered. We therefore need to count
      // how many of them we have for each APO, like this: Cnt[APO] = x.
      // Since we only have two APOs, namely true and false, we can avoid using
      // a map. Instead we can simply count the number of operands that
      // correspond to one of them (in this case the 'true' APO), and calculate
      // the other by subtracting it from the total number of operands.
      for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx)
        if (getData(OpIdx, Lane).APO)
          ++CntTrue;
      unsigned CntFalse = NumOperands - CntTrue;
      return std::max(CntTrue, CntFalse);
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

      // If the initial strategy fails for any of the operand indexes, then we
      // perform reordering again in a second pass. This helps avoid assigning
      // high priority to the failed strategy, and should improve reordering for
      // the non-failed operand indexes.
      for (int Pass = 0; Pass != 2; ++Pass) {
        // Skip the second pass if the first pass did not fail.
        bool StrategyFailed = false;
        // Mark all operand data as free to use.
        clearUsed();
        // We keep the original operand order for the FirstLane, so reorder the
        // rest of the lanes. We are visiting the nodes in a circular fashion,
        // using FirstLane as the center point and increasing the radius
        // distance.
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
              Optional<unsigned> BestIdx =
                  getBestOperand(OpIdx, Lane, LastLane, ReorderingModes);
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
  /// Checks if all users of \p I are the part of the vectorization tree.
  bool areAllUsersVectorized(Instruction *I) const;

  /// \returns the cost of the vectorizable entry.
  int getEntryCost(TreeEntry *E);

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
  /// context means the creation of vectors from a group of scalars.
  int getGatherCost(VectorType *Ty,
                    const DenseSet<unsigned> &ShuffledIndices) const;

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getGatherCost(ArrayRef<Value *> VL) const;

  /// Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(TreeEntry *E);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Gather(ArrayRef<Value *> VL, VectorType *Ty);

  /// \returns whether the VectorizableTree is fully vectorizable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree() const;

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
      if (VL.size() == Scalars.size())
        return std::equal(VL.begin(), VL.end(), Scalars.begin());
      return VL.size() == ReuseShuffleIndices.size() &&
             std::equal(
                 VL.begin(), VL.end(), ReuseShuffleIndices.begin(),
                 [this](Value *V, int Idx) { return V == Scalars[Idx]; });
    }

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    Value *VectorizedValue = nullptr;

    /// Do we need to gather this sequence ?
    enum EntryState { Vectorize, NeedToGather };
    EntryState State;

    /// Does this sequence require some shuffling?
    SmallVector<int, 4> ReuseShuffleIndices;

    /// Does this entry require reordering?
    ArrayRef<unsigned> ReorderIndices;

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
      assert(Operands[OpIdx].size() == 0 && "Already resized?");
      Operands[OpIdx].resize(Scalars.size());
      for (unsigned Lane = 0, E = Scalars.size(); Lane != E; ++Lane)
        Operands[OpIdx][Lane] = OpVL[Lane];
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

    /// \returns the \p OpIdx operand of this TreeEntry.
    ValueList &getOperand(unsigned OpIdx) {
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
    bool isAltShuffle() const {
      return getOpcode() != getAltOpcode();
    }

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

    /// Update operations state of this entry if reorder occurred.
    bool updateStateIfReorder() {
      if (ReorderIndices.empty())
        return false;
      InstructionsState S = getSameOpcode(Scalars, ReorderIndices.front());
      setOperations(S);
      return true;
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
        dbgs() << "Emtpy";
      else
        for (unsigned ReuseIdx : ReuseShuffleIndices)
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

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, Optional<ScheduleData *> Bundle,
                          const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<unsigned> ReuseShuffleIndices = None,
                          ArrayRef<unsigned> ReorderIndices = None) {
    bool Vectorized = (bool)Bundle;
    VectorizableTree.push_back(std::make_unique<TreeEntry>(VectorizableTree));
    TreeEntry *Last = VectorizableTree.back().get();
    Last->Idx = VectorizableTree.size() - 1;
    Last->Scalars.insert(Last->Scalars.begin(), VL.begin(), VL.end());
    Last->State = Vectorized ? TreeEntry::Vectorize : TreeEntry::NeedToGather;
    Last->ReuseShuffleIndices.append(ReuseShuffleIndices.begin(),
                                     ReuseShuffleIndices.end());
    Last->ReorderIndices = ReorderIndices;
    Last->setOperations(S);
    if (Vectorized) {
      for (int i = 0, e = VL.size(); i != e; ++i) {
        assert(!getTreeEntry(VL[i]) && "Scalar already in tree!");
        ScalarToTreeEntry[VL[i]] = Last;
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

  TreeEntry *getTreeEntry(Value *V) {
    auto I = ScalarToTreeEntry.find(V);
    if (I != ScalarToTreeEntry.end())
      return I->second;
    return nullptr;
  }

  const TreeEntry *getTreeEntry(Value *V) const {
    auto I = ScalarToTreeEntry.find(V);
    if (I != ScalarToTreeEntry.end())
      return I->second;
    return nullptr;
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
    MemoryLocation Loc2 = getLocation(Inst2, AA);
    bool aliased = true;
    if (Loc1.Ptr && Loc2.Ptr && isSimple(Inst1) && isSimple(Inst2)) {
      // Do the alias check.
      aliased = AA->alias(Loc1, Loc2);
    }
    // Store the result in the cache.
    result = aliased;
    return aliased;
  }

  using AliasCacheKey = std::pair<Instruction *, Instruction *>;

  /// Cache for alias results.
  /// TODO: consider moving this to the AliasAnalysis itself.
  DenseMap<AliasCacheKey, Optional<bool>> AliasCache;

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
  SetVector<Instruction *> GatherSeq;

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
      UnscheduledDepsInBundle = UnscheduledDeps;
      clearDependencies();
      OpValue = OpVal;
      TE = nullptr;
      Lane = -1;
    }

    /// Returns true if the dependency information has been calculated.
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
      return UnscheduledDepsInBundle == 0 && !IsScheduled;
    }

    /// Modifies the number of unscheduled dependencies, also updating it for
    /// the whole bundle.
    int incrementUnscheduledDeps(int Incr) {
      UnscheduledDeps += Incr;
      return FirstInBundle->UnscheduledDepsInBundle += Incr;
    }

    /// Sets the number of unscheduled dependencies to the number of
    /// dependencies.
    void resetUnscheduledDeps() {
      incrementUnscheduledDeps(Dependencies - UnscheduledDeps);
    }

    /// Clears all dependency information.
    void clearDependencies() {
      Dependencies = InvalidDeps;
      resetUnscheduledDeps();
      MemoryDependencies.clear();
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

    /// The sum of UnscheduledDeps in a bundle. Equals to UnscheduledDeps for
    /// single instructions.
    int UnscheduledDepsInBundle = InvalidDeps;

    /// True if this instruction is scheduled (or considered as scheduled in the
    /// dry-run).
    bool IsScheduled = false;

    /// Opcode of the current instruction in the schedule data.
    Value *OpValue = nullptr;

    /// The TreeEntry that this instruction corresponds to.
    TreeEntry *TE = nullptr;

    /// The lane of this node in the TreeEntry.
    int Lane = -1;
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

    ScheduleData *getScheduleData(Value *V) {
      ScheduleData *SD = ScheduleDataMap[V];
      if (SD && SD->SchedulingRegionID == SchedulingRegionID)
        return SD;
      return nullptr;
    }

    ScheduleData *getScheduleData(Value *V, Value *Key) {
      if (V == Key)
        return getScheduleData(V);
      auto I = ExtraScheduleDataMap.find(V);
      if (I != ExtraScheduleDataMap.end()) {
        ScheduleData *SD = I->second[Key];
        if (SD && SD->SchedulingRegionID == SchedulingRegionID)
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

      ScheduleData *BundleMember = SD;
      while (BundleMember) {
        if (BundleMember->Inst != BundleMember->OpValue) {
          BundleMember = BundleMember->NextInBundle;
          continue;
        }
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
        // reordered duiring buildTree(). We therefore need to get its operands
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
        BundleMember = BundleMember->NextInBundle;
      }
    }

    void doForAllOpcodes(Value *V,
                         function_ref<void(ScheduleData *SD)> Action) {
      if (ScheduleData *SD = getScheduleData(V))
        Action(SD);
      auto I = ExtraScheduleDataMap.find(V);
      if (I != ExtraScheduleDataMap.end())
        for (auto &P : I->second)
          if (P.second->SchedulingRegionID == SchedulingRegionID)
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
                       << "SLP:    initially in ready list: " << *I << "\n");
          }
        });
      }
    }

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
    DenseMap<Value *, ScheduleData *> ScheduleDataMap;

    /// Attaches ScheduleData to Instruction with the leading key.
    DenseMap<Value *, SmallDenseMap<Value *, ScheduleData *>>
        ExtraScheduleDataMap;

    struct ReadyList : SmallVector<ScheduleData *, 8> {
      void insert(ScheduleData *SD) { push_back(SD); }
    };

    /// The ready-list for scheduling (only used for the dry-run).
    ReadyList ReadyInsts;

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
    // Make sure that the initial SchedulingRegionID is greater than the
    // initial SchedulingRegionID in ScheduleData (which is 0).
    int SchedulingRegionID = 1;
  };

  /// Attaches the BlockScheduling structures to basic blocks.
  MapVector<BasicBlock *, std::unique_ptr<BlockScheduling>> BlocksSchedules;

  /// Performs the "real" scheduling. Done before vectorization is actually
  /// performed in a basic block.
  void scheduleBlock(BlockScheduling *BS);

  /// List of users to ignore during scheduling and that don't need extracting.
  ArrayRef<Value *> UserIgnoreList;

  using OrdersType = SmallVector<unsigned, 4>;
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

  /// Contains orders of operations along with the number of bundles that have
  /// operations in this order. It stores only those orders that require
  /// reordering, if reordering is not required it is counted using \a
  /// NumOpsWantToKeepOriginalOrder.
  DenseMap<OrdersType, unsigned, OrdersTypeDenseMapInfo> NumOpsWantToKeepOrder;
  /// Number of bundles that do not require reordering.
  unsigned NumOpsWantToKeepOriginalOrder = 0;

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
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
    if (isSplat(Entry->Scalars)) {
      OS << "<splat> " << *Entry->Scalars[0];
      return Str;
    }
    for (auto V : Entry->Scalars) {
      OS << *V;
      if (std::any_of(
              R->ExternalUses.begin(), R->ExternalUses.end(),
              [&](const BoUpSLP::ExternalUser &EU) { return EU.Scalar == V; }))
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
  assert(!verifyFunction(*F, &dbgs()));
}

void BoUpSLP::eraseInstructions(ArrayRef<Value *> AV) {
  for (auto *V : AV) {
    if (auto *I = dyn_cast<Instruction>(V))
      eraseInstruction(I, /*ReplaceWithUndef=*/true);
  };
}

void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        ArrayRef<Value *> UserIgnoreLst) {
  ExtraValueToDebugLocsMap ExternallyUsedValues;
  buildTree(Roots, ExternallyUsedValues, UserIgnoreLst);
}

void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        ExtraValueToDebugLocsMap &ExternallyUsedValues,
                        ArrayRef<Value *> UserIgnoreLst) {
  deleteTree();
  UserIgnoreList = UserIgnoreLst;
  if (!allSameType(Roots))
    return;
  buildTree_rec(Roots, 0, EdgeInfo());

  // Collect the values that we need to extract from the tree.
  for (auto &TEPtr : VectorizableTree) {
    TreeEntry *Entry = TEPtr.get();

    // No need to handle users of gathered values.
    if (Entry->State == TreeEntry::NeedToGather)
      continue;

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];
      int FoundLane = Lane;
      if (!Entry->ReuseShuffleIndices.empty()) {
        FoundLane =
            std::distance(Entry->ReuseShuffleIndices.begin(),
                          llvm::find(Entry->ReuseShuffleIndices, FoundLane));
      }

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

        // Skip in-tree scalars that become vectors
        if (TreeEntry *UseEntry = getTreeEntry(U)) {
          Value *UseScalar = UseEntry->Scalars[0];
          // Some in-tree scalars will remain as scalar in vectorized
          // instructions. If that is the case, the one in Lane 0 will
          // be used.
          if (UseScalar != U ||
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

void BoUpSLP::buildTree_rec(ArrayRef<Value *> VL, unsigned Depth,
                            const EdgeInfo &UserTreeIdx) {
  assert((allConstant(VL) || allSameType(VL)) && "Invalid types!");

  InstructionsState S = getSameOpcode(VL);
  if (Depth == RecursionMaxDepth) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  // Don't handle vectors.
  if (S.OpValue->getType()->isVectorTy()) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
    return;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(S.OpValue))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
      return;
    }

  // If all of the operands are identical or constant we have a simple solution.
  if (allConstant(VL) || isSplat(VL) || !allSameBlock(VL) || !S.getOpcode()) {
    LLVM_DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O. \n");
    newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
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
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
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
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
      return;
    }
  }

  // If any of the scalars is marked as a value that needs to stay scalar, then
  // we need to gather the scalars.
  // The reduction nodes (stored in UserIgnoreList) also should stay scalar.
  for (Value *V : VL) {
    if (MustGather.count(V) || is_contained(UserIgnoreList, V)) {
      LLVM_DEBUG(dbgs() << "SLP: Gathering due to gathered scalar.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
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
  SmallVector<unsigned, 4> ReuseShuffleIndicies;
  SmallVector<Value *, 4> UniqueValues;
  DenseMap<Value *, unsigned> UniquePositions;
  for (Value *V : VL) {
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
        !llvm::isPowerOf2_32(NumUniqueScalarValues)) {
      LLVM_DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx);
      return;
    }
    VL = UniqueValues;
  }

  auto &BSRef = BlocksSchedules[BB];
  if (!BSRef)
    BSRef = std::make_unique<BlockScheduling>(BB);

  BlockScheduling &BS = *BSRef.get();

  Optional<ScheduleData *> Bundle = BS.tryScheduleBundle(VL, this, S);
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
      for (unsigned j = 0; j < VL.size(); ++j)
        for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
          Instruction *Term = dyn_cast<Instruction>(
              cast<PHINode>(VL[j])->getIncomingValueForBlock(
                  PH->getIncomingBlock(i)));
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
      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<PHINode>(j)->getIncomingValueForBlock(
              PH->getIncomingBlock(i)));
        TE->setOperand(i, Operands);
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
        ++NumOpsWantToKeepOriginalOrder;
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
        // Insert new order with initial value 0, if it does not exist,
        // otherwise return the iterator to the existing one.
        auto StoredCurrentOrderAndNum =
            NumOpsWantToKeepOrder.try_emplace(CurrentOrder).first;
        ++StoredCurrentOrderAndNum->getSecond();
        newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies,
                     StoredCurrentOrderAndNum->getFirst());
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
    case Instruction::Load: {
      // Check that a vectorized load would load the same memory as a scalar
      // load. For example, we don't want to vectorize loads that are smaller
      // than 8-bit. Even though we have a packed struct {<i2, i2, i2, i2>} LLVM
      // treats loading/storing it as an i8 struct. If we vectorize loads/stores
      // from such a struct, we read/write packed bits disagreeing with the
      // unvectorized version.
      Type *ScalarTy = VL0->getType();

      if (DL->getTypeSizeInBits(ScalarTy) !=
          DL->getTypeAllocSizeInBits(ScalarTy)) {
        BS.cancelScheduling(VL, VL0);
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
        LLVM_DEBUG(dbgs() << "SLP: Gathering loads of non-packed type.\n");
        return;
      }

      // Make sure all loads in the bundle are simple - we can't vectorize
      // atomic or volatile loads.
      SmallVector<Value *, 4> PointerOps(VL.size());
      auto POIter = PointerOps.begin();
      for (Value *V : VL) {
        auto *L = cast<LoadInst>(V);
        if (!L->isSimple()) {
          BS.cancelScheduling(VL, VL0);
          newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                       ReuseShuffleIndicies);
          LLVM_DEBUG(dbgs() << "SLP: Gathering non-simple loads.\n");
          return;
        }
        *POIter = L->getPointerOperand();
        ++POIter;
      }

      OrdersType CurrentOrder;
      // Check the order of pointer operands.
      if (llvm::sortPtrAccesses(PointerOps, *DL, *SE, CurrentOrder)) {
        Value *Ptr0;
        Value *PtrN;
        if (CurrentOrder.empty()) {
          Ptr0 = PointerOps.front();
          PtrN = PointerOps.back();
        } else {
          Ptr0 = PointerOps[CurrentOrder.front()];
          PtrN = PointerOps[CurrentOrder.back()];
        }
        const SCEV *Scev0 = SE->getSCEV(Ptr0);
        const SCEV *ScevN = SE->getSCEV(PtrN);
        const auto *Diff =
            dyn_cast<SCEVConstant>(SE->getMinusSCEV(ScevN, Scev0));
        uint64_t Size = DL->getTypeAllocSize(ScalarTy);
        // Check that the sorted loads are consecutive.
        if (Diff && Diff->getAPInt() == (VL.size() - 1) * Size) {
          if (CurrentOrder.empty()) {
            // Original loads are consecutive and does not require reordering.
            ++NumOpsWantToKeepOriginalOrder;
            TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S,
                                         UserTreeIdx, ReuseShuffleIndicies);
            TE->setOperandsInOrder();
            LLVM_DEBUG(dbgs() << "SLP: added a vector of loads.\n");
          } else {
            // Need to reorder.
            auto I = NumOpsWantToKeepOrder.try_emplace(CurrentOrder).first;
            ++I->getSecond();
            TreeEntry *TE =
                newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                             ReuseShuffleIndicies, I->getFirst());
            TE->setOperandsInOrder();
            LLVM_DEBUG(dbgs() << "SLP: added a vector of jumbled loads.\n");
          }
          return;
        }
      }

      LLVM_DEBUG(dbgs() << "SLP: Gathering non-consecutive loads.\n");
      BS.cancelScheduling(VL, VL0);
      newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                   ReuseShuffleIndicies);
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
        for (Value *j : VL)
          Operands.push_back(cast<Instruction>(j)->getOperand(i));

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
      Type *Ty0 = VL0->getOperand(0)->getType();
      for (Value *V : VL) {
        Type *CurTy = cast<Instruction>(V)->getOperand(0)->getType();
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
      TE->setOperandsInOrder();
      for (unsigned i = 0, e = 2; i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *V : VL)
          Operands.push_back(cast<Instruction>(V)->getOperand(i));

        buildTree_rec(Operands, Depth + 1, {TE, i});
      }
      return;
    }
    case Instruction::Store: {
      // Check if the stores are consecutive or if we need to swizzle them.
      llvm::Type *ScalarTy = cast<StoreInst>(VL0)->getValueOperand()->getType();
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
      if (llvm::sortPtrAccesses(PointerOps, *DL, *SE, CurrentOrder)) {
        Value *Ptr0;
        Value *PtrN;
        if (CurrentOrder.empty()) {
          Ptr0 = PointerOps.front();
          PtrN = PointerOps.back();
        } else {
          Ptr0 = PointerOps[CurrentOrder.front()];
          PtrN = PointerOps[CurrentOrder.back()];
        }
        const SCEV *Scev0 = SE->getSCEV(Ptr0);
        const SCEV *ScevN = SE->getSCEV(PtrN);
        const auto *Diff =
            dyn_cast<SCEVConstant>(SE->getMinusSCEV(ScevN, Scev0));
        uint64_t Size = DL->getTypeAllocSize(ScalarTy);
        // Check that the sorted pointer operands are consecutive.
        if (Diff && Diff->getAPInt() == (VL.size() - 1) * Size) {
          if (CurrentOrder.empty()) {
            // Original stores are consecutive and does not require reordering.
            ++NumOpsWantToKeepOriginalOrder;
            TreeEntry *TE = newTreeEntry(VL, Bundle /*vectorized*/, S,
                                         UserTreeIdx, ReuseShuffleIndicies);
            TE->setOperandsInOrder();
            buildTree_rec(Operands, Depth + 1, {TE, 0});
            LLVM_DEBUG(dbgs() << "SLP: added a vector of stores.\n");
          } else {
            // Need to reorder.
            auto I = NumOpsWantToKeepOrder.try_emplace(CurrentOrder).first;
            ++(I->getSecond());
            TreeEntry *TE =
                newTreeEntry(VL, Bundle /*vectorized*/, S, UserTreeIdx,
                             ReuseShuffleIndicies, I->getFirst());
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
      // Check if the calls are all to the same vectorizable intrinsic.
      CallInst *CI = cast<CallInst>(VL0);
      // Check if this is an Intrinsic call or something that can be
      // represented by an intrinsic call
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
      if (!isTriviallyVectorizable(ID)) {
        BS.cancelScheduling(VL, VL0);
        newTreeEntry(VL, None /*not vectorized*/, S, UserTreeIdx,
                     ReuseShuffleIndicies);
        LLVM_DEBUG(dbgs() << "SLP: Non-vectorizable call.\n");
        return;
      }
      Function *Int = CI->getCalledFunction();
      unsigned NumArgs = CI->getNumArgOperands();
      SmallVector<Value*, 4> ScalarArgs(NumArgs, nullptr);
      for (unsigned j = 0; j != NumArgs; ++j)
        if (hasVectorInstrinsicScalarOpd(ID, j))
          ScalarArgs[j] = CI->getArgOperand(j);
      for (Value *V : VL) {
        CallInst *CI2 = dyn_cast<CallInst>(V);
        if (!CI2 || CI2->getCalledFunction() != Int ||
            getVectorIntrinsicIDForCall(CI2, TLI) != ID ||
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
      for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i) {
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
      if (isa<BinaryOperator>(VL0)) {
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
      auto *VT = cast<VectorType>(EltTy);
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
  Instruction *E0 = cast<Instruction>(OpValue);
  assert(E0->getOpcode() == Instruction::ExtractElement ||
         E0->getOpcode() == Instruction::ExtractValue);
  assert(E0->getOpcode() == getSameOpcode(VL).getOpcode() && "Invalid opcode");
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
    NElts = cast<VectorType>(Vec->getType())->getNumElements();
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
  CurrentOrder.assign(E, E + 1);
  unsigned I = 0;
  for (; I < E; ++I) {
    auto *Inst = cast<Instruction>(VL[I]);
    if (Inst->getOperand(0) != Vec)
      break;
    Optional<unsigned> Idx = getExtractIndex(Inst);
    if (!Idx)
      break;
    const unsigned ExtIdx = *Idx;
    if (ExtIdx != I) {
      if (ExtIdx >= E || CurrentOrder[ExtIdx] != E + 1)
        break;
      ShouldKeepOrder = false;
      CurrentOrder[ExtIdx] = I;
    } else {
      if (CurrentOrder[I] != E + 1)
        break;
      CurrentOrder[I] = I;
    }
  }
  if (I < E) {
    CurrentOrder.clear();
    return false;
  }

  return ShouldKeepOrder;
}

bool BoUpSLP::areAllUsersVectorized(Instruction *I) const {
  return I->hasOneUse() ||
         std::all_of(I->user_begin(), I->user_end(), [this](User *U) {
           return ScalarToTreeEntry.count(U) > 0;
         });
}

static std::pair<unsigned, unsigned>
getVectorCallCosts(CallInst *CI, VectorType *VecTy, TargetTransformInfo *TTI,
                   TargetLibraryInfo *TLI) {
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

  // Calculate the cost of the scalar and vector calls.
  IntrinsicCostAttributes CostAttrs(ID, *CI, VecTy->getNumElements());
  int IntrinsicCost =
    TTI->getIntrinsicInstrCost(CostAttrs, TTI::TCK_RecipThroughput);

  auto Shape =
      VFShape::get(*CI, {static_cast<unsigned>(VecTy->getNumElements()), false},
                   false /*HasGlobalPred*/);
  Function *VecFunc = VFDatabase(*CI).getVectorizedFunction(Shape);
  int LibCost = IntrinsicCost;
  if (!CI->isNoBuiltin() && VecFunc) {
    // Calculate the cost of the vector library call.
    SmallVector<Type *, 4> VecTys;
    for (Use &Arg : CI->args())
      VecTys.push_back(
          FixedVectorType::get(Arg->getType(), VecTy->getNumElements()));

    // If the corresponding vector call is cheaper, return its cost.
    LibCost = TTI->getCallInstrCost(nullptr, VecTy, VecTys,
                                    TTI::TCK_RecipThroughput);
  }
  return {IntrinsicCost, LibCost};
}

int BoUpSLP::getEntryCost(TreeEntry *E) {
  ArrayRef<Value*> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  else if (CmpInst *CI = dyn_cast<CmpInst>(VL[0]))
    ScalarTy = CI->getOperand(0)->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  // If we have computed a smaller type for the expression, update VecTy so
  // that the costs will be accurate.
  if (MinBWs.count(VL[0]))
    VecTy = FixedVectorType::get(
        IntegerType::get(F->getContext(), MinBWs[VL[0]].first), VL.size());

  unsigned ReuseShuffleNumbers = E->ReuseShuffleIndices.size();
  bool NeedToShuffleReuses = !E->ReuseShuffleIndices.empty();
  int ReuseShuffleCost = 0;
  if (NeedToShuffleReuses) {
    ReuseShuffleCost =
        TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, VecTy);
  }
  if (E->State == TreeEntry::NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isSplat(VL)) {
      return ReuseShuffleCost +
             TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);
    }
    if (E->getOpcode() == Instruction::ExtractElement &&
        allSameType(VL) && allSameBlock(VL)) {
      Optional<TargetTransformInfo::ShuffleKind> ShuffleKind = isShuffle(VL);
      if (ShuffleKind.hasValue()) {
        int Cost = TTI->getShuffleCost(ShuffleKind.getValue(), VecTy);
        for (auto *V : VL) {
          // If all users of instruction are going to be vectorized and this
          // instruction itself is not going to be vectorized, consider this
          // instruction as dead and remove its cost from the final cost of the
          // vectorized tree.
          if (areAllUsersVectorized(cast<Instruction>(V)) &&
              !ScalarToTreeEntry.count(V)) {
            auto *IO = cast<ConstantInt>(
                cast<ExtractElementInst>(V)->getIndexOperand());
            Cost -= TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy,
                                            IO->getZExtValue());
          }
        }
        return ReuseShuffleCost + Cost;
      }
    }
    return ReuseShuffleCost + getGatherCost(VL);
  }
  assert(E->State == TreeEntry::Vectorize && "Unhandled state");
  assert(E->getOpcode() && allSameType(VL) && allSameBlock(VL) && "Invalid VL");
  Instruction *VL0 = E->getMainOp();
  unsigned ShuffleOrOp =
      E->isAltShuffle() ? (unsigned)Instruction::ShuffleVector : E->getOpcode();
  switch (ShuffleOrOp) {
    case Instruction::PHI:
      return 0;

    case Instruction::ExtractValue:
    case Instruction::ExtractElement: {
      if (NeedToShuffleReuses) {
        unsigned Idx = 0;
        for (unsigned I : E->ReuseShuffleIndices) {
          if (ShuffleOrOp == Instruction::ExtractElement) {
            auto *IO = cast<ConstantInt>(
                cast<ExtractElementInst>(VL[I])->getIndexOperand());
            Idx = IO->getZExtValue();
            ReuseShuffleCost -= TTI->getVectorInstrCost(
                Instruction::ExtractElement, VecTy, Idx);
          } else {
            ReuseShuffleCost -= TTI->getVectorInstrCost(
                Instruction::ExtractElement, VecTy, Idx);
            ++Idx;
          }
        }
        Idx = ReuseShuffleNumbers;
        for (Value *V : VL) {
          if (ShuffleOrOp == Instruction::ExtractElement) {
            auto *IO = cast<ConstantInt>(
                cast<ExtractElementInst>(V)->getIndexOperand());
            Idx = IO->getZExtValue();
          } else {
            --Idx;
          }
          ReuseShuffleCost +=
              TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, Idx);
        }
      }
      int DeadCost = ReuseShuffleCost;
      if (!E->ReorderIndices.empty()) {
        // TODO: Merge this shuffle with the ReuseShuffleCost.
        DeadCost += TTI->getShuffleCost(
            TargetTransformInfo::SK_PermuteSingleSrc, VecTy);
      }
      for (unsigned i = 0, e = VL.size(); i < e; ++i) {
        Instruction *E = cast<Instruction>(VL[i]);
        // If all users are going to be vectorized, instruction can be
        // considered as dead.
        // The same, if have only one user, it will be vectorized for sure.
        if (areAllUsersVectorized(E)) {
          // Take credit for instruction that will become dead.
          if (E->hasOneUse()) {
            Instruction *Ext = E->user_back();
            if ((isa<SExtInst>(Ext) || isa<ZExtInst>(Ext)) &&
                all_of(Ext->users(),
                       [](User *U) { return isa<GetElementPtrInst>(U); })) {
              // Use getExtractWithExtendCost() to calculate the cost of
              // extractelement/ext pair.
              DeadCost -= TTI->getExtractWithExtendCost(
                  Ext->getOpcode(), Ext->getType(), VecTy, i);
              // Add back the cost of s|zext which is subtracted separately.
              DeadCost += TTI->getCastInstrCost(
                  Ext->getOpcode(), Ext->getType(), E->getType(), CostKind,
                  Ext);
              continue;
            }
          }
          DeadCost -=
              TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, i);
        }
      }
      return DeadCost;
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
      int ScalarEltCost =
          TTI->getCastInstrCost(E->getOpcode(), ScalarTy, SrcTy, CostKind,
                                VL0);
      if (NeedToShuffleReuses) {
        ReuseShuffleCost -= (ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      }

      // Calculate the cost of this instruction.
      int ScalarCost = VL.size() * ScalarEltCost;

      auto *SrcVecTy = FixedVectorType::get(SrcTy, VL.size());
      int VecCost = 0;
      // Check if the values are candidates to demote.
      if (!MinBWs.count(VL0) || VecTy != SrcVecTy) {
        VecCost = ReuseShuffleCost +
                  TTI->getCastInstrCost(E->getOpcode(), VecTy, SrcVecTy,
                                        CostKind, VL0);
      }
      return VecCost - ScalarCost;
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select: {
      // Calculate the cost of this instruction.
      int ScalarEltCost = TTI->getCmpSelInstrCost(E->getOpcode(), ScalarTy,
                                                  Builder.getInt1Ty(),
                                                  CostKind, VL0);
      if (NeedToShuffleReuses) {
        ReuseShuffleCost -= (ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      }
      auto *MaskTy = FixedVectorType::get(Builder.getInt1Ty(), VL.size());
      int ScalarCost = VecTy->getNumElements() * ScalarEltCost;
      int VecCost = TTI->getCmpSelInstrCost(E->getOpcode(), VecTy, MaskTy,
                                            CostKind, VL0);
      return ReuseShuffleCost + VecCost - ScalarCost;
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
      int ScalarEltCost = TTI->getArithmeticInstrCost(
          E->getOpcode(), ScalarTy, CostKind, Op1VK, Op2VK, Op1VP, Op2VP,
          Operands, VL0);
      if (NeedToShuffleReuses) {
        ReuseShuffleCost -= (ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      }
      int ScalarCost = VecTy->getNumElements() * ScalarEltCost;
      int VecCost = TTI->getArithmeticInstrCost(
          E->getOpcode(), VecTy, CostKind, Op1VK, Op2VK, Op1VP, Op2VP,
          Operands, VL0);
      return ReuseShuffleCost + VecCost - ScalarCost;
    }
    case Instruction::GetElementPtr: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;

      int ScalarEltCost =
          TTI->getArithmeticInstrCost(Instruction::Add, ScalarTy, CostKind,
                                      Op1VK, Op2VK);
      if (NeedToShuffleReuses) {
        ReuseShuffleCost -= (ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      }
      int ScalarCost = VecTy->getNumElements() * ScalarEltCost;
      int VecCost =
          TTI->getArithmeticInstrCost(Instruction::Add, VecTy, CostKind,
                                      Op1VK, Op2VK);
      return ReuseShuffleCost + VecCost - ScalarCost;
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      Align alignment = cast<LoadInst>(VL0)->getAlign();
      int ScalarEltCost =
          TTI->getMemoryOpCost(Instruction::Load, ScalarTy, alignment, 0,
                               CostKind, VL0);
      if (NeedToShuffleReuses) {
        ReuseShuffleCost -= (ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      }
      int ScalarLdCost = VecTy->getNumElements() * ScalarEltCost;
      int VecLdCost =
          TTI->getMemoryOpCost(Instruction::Load, VecTy, alignment, 0,
                               CostKind, VL0);
      if (!E->ReorderIndices.empty()) {
        // TODO: Merge this shuffle with the ReuseShuffleCost.
        VecLdCost += TTI->getShuffleCost(
            TargetTransformInfo::SK_PermuteSingleSrc, VecTy);
      }
      return ReuseShuffleCost + VecLdCost - ScalarLdCost;
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      bool IsReorder = !E->ReorderIndices.empty();
      auto *SI =
          cast<StoreInst>(IsReorder ? VL[E->ReorderIndices.front()] : VL0);
      Align Alignment = SI->getAlign();
      int ScalarEltCost =
          TTI->getMemoryOpCost(Instruction::Store, ScalarTy, Alignment, 0,
                               CostKind, VL0);
      if (NeedToShuffleReuses)
        ReuseShuffleCost = -(ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      int ScalarStCost = VecTy->getNumElements() * ScalarEltCost;
      int VecStCost = TTI->getMemoryOpCost(Instruction::Store,
                                           VecTy, Alignment, 0, CostKind, VL0);
      if (IsReorder) {
        // TODO: Merge this shuffle with the ReuseShuffleCost.
        VecStCost += TTI->getShuffleCost(
            TargetTransformInfo::SK_PermuteSingleSrc, VecTy);
      }
      return ReuseShuffleCost + VecStCost - ScalarStCost;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      // Calculate the cost of the scalar and vector calls.
      IntrinsicCostAttributes CostAttrs(ID, *CI, 1, 1);
      int ScalarEltCost = TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
      if (NeedToShuffleReuses) {
        ReuseShuffleCost -= (ReuseShuffleNumbers - VL.size()) * ScalarEltCost;
      }
      int ScalarCallCost = VecTy->getNumElements() * ScalarEltCost;

      auto VecCallCosts = getVectorCallCosts(CI, VecTy, TTI, TLI);
      int VecCallCost = std::min(VecCallCosts.first, VecCallCosts.second);

      LLVM_DEBUG(dbgs() << "SLP: Call cost " << VecCallCost - ScalarCallCost
                        << " (" << VecCallCost << "-" << ScalarCallCost << ")"
                        << " for " << *CI << "\n");

      return ReuseShuffleCost + VecCallCost - ScalarCallCost;
    }
    case Instruction::ShuffleVector: {
      assert(E->isAltShuffle() &&
             ((Instruction::isBinaryOp(E->getOpcode()) &&
               Instruction::isBinaryOp(E->getAltOpcode())) ||
              (Instruction::isCast(E->getOpcode()) &&
               Instruction::isCast(E->getAltOpcode()))) &&
             "Invalid Shuffle Vector Operand");
      int ScalarCost = 0;
      if (NeedToShuffleReuses) {
        for (unsigned Idx : E->ReuseShuffleIndices) {
          Instruction *I = cast<Instruction>(VL[Idx]);
          ReuseShuffleCost -= TTI->getInstructionCost(I, CostKind);
        }
        for (Value *V : VL) {
          Instruction *I = cast<Instruction>(V);
          ReuseShuffleCost += TTI->getInstructionCost(I, CostKind);
        }
      }
      for (Value *V : VL) {
        Instruction *I = cast<Instruction>(V);
        assert(E->isOpcodeOrAlt(I) && "Unexpected main/alternate opcode");
        ScalarCost += TTI->getInstructionCost(I, CostKind);
      }
      // VecCost is equal to sum of the cost of creating 2 vectors
      // and the cost of creating shuffle.
      int VecCost = 0;
      if (Instruction::isBinaryOp(E->getOpcode())) {
        VecCost = TTI->getArithmeticInstrCost(E->getOpcode(), VecTy, CostKind);
        VecCost += TTI->getArithmeticInstrCost(E->getAltOpcode(), VecTy,
                                               CostKind);
      } else {
        Type *Src0SclTy = E->getMainOp()->getOperand(0)->getType();
        Type *Src1SclTy = E->getAltOp()->getOperand(0)->getType();
        auto *Src0Ty = FixedVectorType::get(Src0SclTy, VL.size());
        auto *Src1Ty = FixedVectorType::get(Src1SclTy, VL.size());
        VecCost = TTI->getCastInstrCost(E->getOpcode(), VecTy, Src0Ty,
                                        CostKind);
        VecCost += TTI->getCastInstrCost(E->getAltOpcode(), VecTy, Src1Ty,
                                         CostKind);
      }
      VecCost += TTI->getShuffleCost(TargetTransformInfo::SK_Select, VecTy, 0);
      return ReuseShuffleCost + VecCost - ScalarCost;
    }
    default:
      llvm_unreachable("Unknown instruction");
  }
}

bool BoUpSLP::isFullyVectorizableTinyTree() const {
  LLVM_DEBUG(dbgs() << "SLP: Check whether the tree with height "
                    << VectorizableTree.size() << " is fully vectorizable .\n");

  // We only handle trees of heights 1 and 2.
  if (VectorizableTree.size() == 1 &&
      VectorizableTree[0]->State == TreeEntry::Vectorize)
    return true;

  if (VectorizableTree.size() != 2)
    return false;

  // Handle splat and all-constants stores.
  if (VectorizableTree[0]->State == TreeEntry::Vectorize &&
      (allConstant(VectorizableTree[1]->Scalars) ||
       isSplat(VectorizableTree[1]->Scalars)))
    return true;

  // Gathering cost would be too much for tiny trees.
  if (VectorizableTree[0]->State == TreeEntry::NeedToGather ||
      VectorizableTree[1]->State == TreeEntry::NeedToGather)
    return false;

  return true;
}

static bool isLoadCombineCandidateImpl(Value *Root, unsigned NumElts,
                                       TargetTransformInfo *TTI) {
  // Look past the root to find a source value. Arbitrarily follow the
  // path through operand 0 of any 'or'. Also, peek through optional
  // shift-left-by-constant.
  Value *ZextLoad = Root;
  while (!isa<ConstantExpr>(ZextLoad) &&
         (match(ZextLoad, m_Or(m_Value(), m_Value())) ||
          match(ZextLoad, m_Shl(m_Value(), m_Constant()))))
    ZextLoad = cast<BinaryOperator>(ZextLoad)->getOperand(0);

  // Check if the input is an extended load of the required or/shift expression.
  Value *LoadPtr;
  if (ZextLoad == Root || !match(ZextLoad, m_ZExt(m_Load(m_Value(LoadPtr)))))
    return false;

  // Require that the total load bit width is a legal integer type.
  // For example, <8 x i8> --> i64 is a legal integer on a 64-bit target.
  // But <16 x i8> --> i128 is not, so the backend probably can't reduce it.
  Type *SrcTy = LoadPtr->getType()->getPointerElementType();
  unsigned LoadBitWidth = SrcTy->getIntegerBitWidth() * NumElts;
  if (!TTI->isTypeLegal(IntegerType::get(Root->getContext(), LoadBitWidth)))
    return false;

  // Everything matched - assume that we can fold the whole sequence using
  // load combining.
  LLVM_DEBUG(dbgs() << "SLP: Assume load combining for tree starting at "
             << *(cast<Instruction>(Root)) << "\n");

  return true;
}

bool BoUpSLP::isLoadCombineReductionCandidate(unsigned RdxOpcode) const {
  if (RdxOpcode != Instruction::Or)
    return false;

  unsigned NumElts = VectorizableTree[0]->Scalars.size();
  Value *FirstReduced = VectorizableTree[0]->Scalars[0];
  return isLoadCombineCandidateImpl(FirstReduced, NumElts, TTI);
}

bool BoUpSLP::isLoadCombineCandidate() const {
  // Peek through a final sequence of stores and check if all operations are
  // likely to be load-combined.
  unsigned NumElts = VectorizableTree[0]->Scalars.size();
  for (Value *Scalar : VectorizableTree[0]->Scalars) {
    Value *X;
    if (!match(Scalar, m_Store(m_Value(X), m_Value())) ||
        !isLoadCombineCandidateImpl(X, NumElts, TTI))
      return false;
  }
  return true;
}

bool BoUpSLP::isTreeTinyAndNotFullyVectorizable() const {
  // We can vectorize the tree if its size is greater than or equal to the
  // minimum size specified by the MinTreeSize command line option.
  if (VectorizableTree.size() >= MinTreeSize)
    return false;

  // If we have a tiny tree (a tree whose size is less than MinTreeSize), we
  // can vectorize it if we can prove it fully vectorizable.
  if (isFullyVectorizableTinyTree())
    return false;

  assert(VectorizableTree.empty()
             ? ExternalUses.empty()
             : true && "We shouldn't have any external users");

  // Otherwise, we can't vectorize the tree. It is both tiny and not fully
  // vectorizable.
  return true;
}

int BoUpSLP::getSpillCost() const {
  // Walk from the bottom of the tree to the top, tracking which values are
  // live. When we see a call instruction that is not part of our tree,
  // query TTI to see if there is a cost to keeping values live over it
  // (for example, if spills and fills are required).
  unsigned BundleWidth = VectorizableTree.front()->Scalars.size();
  int Cost = 0;

  SmallPtrSet<Instruction*, 4> LiveValues;
  Instruction *PrevInst = nullptr;

  for (const auto &TEPtr : VectorizableTree) {
    Instruction *Inst = dyn_cast<Instruction>(TEPtr->Scalars[0]);
    if (!Inst)
      continue;

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
      for (auto *II : LiveValues)
        V.push_back(FixedVectorType::get(II->getType(), BundleWidth));
      Cost += NumCalls * TTI->getCostOfKeepingLiveOverCall(V);
    }

    PrevInst = Inst;
  }

  return Cost;
}

int BoUpSLP::getTreeCost() {
  int Cost = 0;
  LLVM_DEBUG(dbgs() << "SLP: Calculating cost for tree of size "
                    << VectorizableTree.size() << ".\n");

  unsigned BundleWidth = VectorizableTree[0]->Scalars.size();

  for (unsigned I = 0, E = VectorizableTree.size(); I < E; ++I) {
    TreeEntry &TE = *VectorizableTree[I].get();

    // We create duplicate tree entries for gather sequences that have multiple
    // uses. However, we should not compute the cost of duplicate sequences.
    // For example, if we have a build vector (i.e., insertelement sequence)
    // that is used by more than one vector instruction, we only need to
    // compute the cost of the insertelement instructions once. The redundant
    // instructions will be eliminated by CSE.
    //
    // We should consider not creating duplicate tree entries for gather
    // sequences, and instead add additional edges to the tree representing
    // their uses. Since such an approach results in fewer total entries,
    // existing heuristics based on tree size may yield different results.
    //
    if (TE.State == TreeEntry::NeedToGather &&
        std::any_of(std::next(VectorizableTree.begin(), I + 1),
                    VectorizableTree.end(),
                    [TE](const std::unique_ptr<TreeEntry> &EntryPtr) {
                      return EntryPtr->State == TreeEntry::NeedToGather &&
                             EntryPtr->isSame(TE.Scalars);
                    }))
      continue;

    int C = getEntryCost(&TE);
    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << C
                      << " for bundle that starts with " << *TE.Scalars[0]
                      << ".\n");
    Cost += C;
  }

  SmallPtrSet<Value *, 16> ExtractCostCalculated;
  int ExtractCost = 0;
  for (ExternalUser &EU : ExternalUses) {
    // We only add extract cost once for the same scalar.
    if (!ExtractCostCalculated.insert(EU.Scalar).second)
      continue;

    // Uses by ephemeral values are free (because the ephemeral value will be
    // removed prior to code generation, and so the extraction will be
    // removed as well).
    if (EphValues.count(EU.User))
      continue;

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

  int SpillCost = getSpillCost();
  Cost += SpillCost + ExtractCost;

  std::string Str;
  {
    raw_string_ostream OS(Str);
    OS << "SLP: Spill Cost = " << SpillCost << ".\n"
       << "SLP: Extract Cost = " << ExtractCost << ".\n"
       << "SLP: Total Cost = " << Cost << ".\n";
  }
  LLVM_DEBUG(dbgs() << Str);

  if (ViewSLPTree)
    ViewGraph(this, "SLP" + F->getName(), false, Str);

  return Cost;
}

int BoUpSLP::getGatherCost(VectorType *Ty,
                           const DenseSet<unsigned> &ShuffledIndices) const {
  unsigned NumElts = Ty->getNumElements();
  APInt DemandedElts = APInt::getNullValue(NumElts);
  for (unsigned i = 0; i < NumElts; ++i)
    if (!ShuffledIndices.count(i))
      DemandedElts.setBit(i);
  int Cost = TTI->getScalarizationOverhead(Ty, DemandedElts, /*Insert*/ true,
                                           /*Extract*/ false);
  if (!ShuffledIndices.empty())
    Cost += TTI->getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, Ty);
  return Cost;
}

int BoUpSLP::getGatherCost(ArrayRef<Value *> VL) const {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());
  // Find the cost of inserting/extracting values from the vector.
  // Check if the same elements are inserted several times and count them as
  // shuffle candidates.
  DenseSet<unsigned> ShuffledElements;
  DenseSet<Value *> UniqueElements;
  // Iterate in reverse order to consider insert elements with the high cost.
  for (unsigned I = VL.size(); I > 0; --I) {
    unsigned Idx = I - 1;
    if (!UniqueElements.insert(VL[Idx]).second)
      ShuffledElements.insert(Idx);
  }
  return getGatherCost(VecTy, ShuffledElements);
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

void BoUpSLP::setInsertPointAfterBundle(TreeEntry *E) {
  // Get the basic block this bundle is in. All instructions in the bundle
  // should be in this block.
  auto *Front = E->getMainOp();
  auto *BB = Front->getParent();
  assert(llvm::all_of(make_range(E->Scalars.begin(), E->Scalars.end()),
                      [=](Value *V) -> bool {
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

Value *BoUpSLP::Gather(ArrayRef<Value *> VL, VectorType *Ty) {
  Value *Vec = UndefValue::get(Ty);
  // Generate the 'InsertElement' instruction.
  for (unsigned i = 0; i < Ty->getNumElements(); ++i) {
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    if (auto *Insrt = dyn_cast<InsertElementInst>(Vec)) {
      GatherSeq.insert(Insrt);
      CSEBlocks.insert(Insrt->getParent());

      // Add to our 'need-to-extract' list.
      if (TreeEntry *E = getTreeEntry(VL[i])) {
        // Find which lane we need to extract.
        int FoundLane = -1;
        for (unsigned Lane = 0, LE = E->Scalars.size(); Lane != LE; ++Lane) {
          // Is this the lane of the scalar that we are looking for ?
          if (E->Scalars[Lane] == VL[i]) {
            FoundLane = Lane;
            break;
          }
        }
        assert(FoundLane >= 0 && "Could not find the correct lane");
        if (!E->ReuseShuffleIndices.empty()) {
          FoundLane =
              std::distance(E->ReuseShuffleIndices.begin(),
                            llvm::find(E->ReuseShuffleIndices, FoundLane));
        }
        ExternalUses.push_back(ExternalUser(VL[i], Insrt, FoundLane));
      }
    }
  }

  return Vec;
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL) {
  InstructionsState S = getSameOpcode(VL);
  if (S.getOpcode()) {
    if (TreeEntry *E = getTreeEntry(S.OpValue)) {
      if (E->isSame(VL)) {
        Value *V = vectorizeTree(E);
        if (VL.size() == E->Scalars.size() && !E->ReuseShuffleIndices.empty()) {
          // We need to get the vectorized value but without shuffle.
          if (auto *SV = dyn_cast<ShuffleVectorInst>(V)) {
            V = SV->getOperand(0);
          } else {
            // Reshuffle to get only unique values.
            SmallVector<int, 4> UniqueIdxs;
            SmallSet<int, 4> UsedIdxs;
            for (int Idx : E->ReuseShuffleIndices)
              if (UsedIdxs.insert(Idx).second)
                UniqueIdxs.emplace_back(Idx);
            V = Builder.CreateShuffleVector(V, UndefValue::get(V->getType()),
                                            UniqueIdxs);
          }
        }
        return V;
      }
    }
  }

  Type *ScalarTy = S.OpValue->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(S.OpValue))
    ScalarTy = SI->getValueOperand()->getType();

  // Check that every instruction appears once in this bundle.
  SmallVector<int, 4> ReuseShuffleIndicies;
  SmallVector<Value *, 4> UniqueValues;
  if (VL.size() > 2) {
    DenseMap<Value *, unsigned> UniquePositions;
    for (Value *V : VL) {
      auto Res = UniquePositions.try_emplace(V, UniqueValues.size());
      ReuseShuffleIndicies.emplace_back(Res.first->second);
      if (Res.second || isa<Constant>(V))
        UniqueValues.emplace_back(V);
    }
    // Do not shuffle single element or if number of unique values is not power
    // of 2.
    if (UniqueValues.size() == VL.size() || UniqueValues.size() <= 1 ||
        !llvm::isPowerOf2_32(UniqueValues.size()))
      ReuseShuffleIndicies.clear();
    else
      VL = UniqueValues;
  }
  auto *VecTy = FixedVectorType::get(ScalarTy, VL.size());

  Value *V = Gather(VL, VecTy);
  if (!ReuseShuffleIndicies.empty()) {
    V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                    ReuseShuffleIndicies, "shuffle");
    if (auto *I = dyn_cast<Instruction>(V)) {
      GatherSeq.insert(I);
      CSEBlocks.insert(I->getParent());
    }
  }
  return V;
}

static void inversePermutation(ArrayRef<unsigned> Indices,
                               SmallVectorImpl<int> &Mask) {
  Mask.clear();
  const unsigned E = Indices.size();
  Mask.resize(E);
  for (unsigned I = 0; I < E; ++I)
    Mask[Indices[I]] = I;
}

Value *BoUpSLP::vectorizeTree(TreeEntry *E) {
  IRBuilder<>::InsertPointGuard Guard(Builder);

  if (E->VectorizedValue) {
    LLVM_DEBUG(dbgs() << "SLP: Diamond merged for " << *E->Scalars[0] << ".\n");
    return E->VectorizedValue;
  }

  Instruction *VL0 = E->getMainOp();
  Type *ScalarTy = VL0->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL0))
    ScalarTy = SI->getValueOperand()->getType();
  auto *VecTy = FixedVectorType::get(ScalarTy, E->Scalars.size());

  bool NeedToShuffleReuses = !E->ReuseShuffleIndices.empty();

  if (E->State == TreeEntry::NeedToGather) {
    setInsertPointAfterBundle(E);
    auto *V = Gather(E->Scalars, VecTy);
    if (NeedToShuffleReuses) {
      V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                      E->ReuseShuffleIndices, "shuffle");
      if (auto *I = dyn_cast<Instruction>(V)) {
        GatherSeq.insert(I);
        CSEBlocks.insert(I->getParent());
      }
    }
    E->VectorizedValue = V;
    return V;
  }

  assert(E->State == TreeEntry::Vectorize && "Unhandled state");
  unsigned ShuffleOrOp =
      E->isAltShuffle() ? (unsigned)Instruction::ShuffleVector : E->getOpcode();
  switch (ShuffleOrOp) {
    case Instruction::PHI: {
      auto *PH = cast<PHINode>(VL0);
      Builder.SetInsertPoint(PH->getParent()->getFirstNonPHI());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());
      PHINode *NewPhi = Builder.CreatePHI(VecTy, PH->getNumIncomingValues());
      Value *V = NewPhi;
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
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
      if (!E->ReorderIndices.empty()) {
        SmallVector<int, 4> Mask;
        inversePermutation(E->ReorderIndices, Mask);
        Builder.SetInsertPoint(VL0);
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy), Mask,
                                        "reorder_shuffle");
      }
      if (NeedToShuffleReuses) {
        // TODO: Merge this shuffle with the ReorderShuffleMask.
        if (E->ReorderIndices.empty())
          Builder.SetInsertPoint(VL0);
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::ExtractValue: {
      LoadInst *LI = cast<LoadInst>(E->getSingleOperand(0));
      Builder.SetInsertPoint(LI);
      PointerType *PtrTy =
          PointerType::get(VecTy, LI->getPointerAddressSpace());
      Value *Ptr = Builder.CreateBitCast(LI->getOperand(0), PtrTy);
      LoadInst *V = Builder.CreateAlignedLoad(VecTy, Ptr, LI->getAlign());
      Value *NewV = propagateMetadata(V, E->Scalars);
      if (!E->ReorderIndices.empty()) {
        SmallVector<int, 4> Mask;
        inversePermutation(E->ReorderIndices, Mask);
        NewV = Builder.CreateShuffleVector(NewV, UndefValue::get(VecTy), Mask,
                                           "reorder_shuffle");
      }
      if (NeedToShuffleReuses) {
        // TODO: Merge this shuffle with the ReorderShuffleMask.
        NewV = Builder.CreateShuffleVector(NewV, UndefValue::get(VecTy),
                                           E->ReuseShuffleIndices, "shuffle");
      }
      E->VectorizedValue = NewV;
      return NewV;
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
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
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
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
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
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
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

      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
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

      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
      E->VectorizedValue = V;
      ++NumVectorInstructions;

      return V;
    }
    case Instruction::Load: {
      // Loads are inserted at the head of the tree because we don't want to
      // sink them all the way down past store instructions.
      bool IsReorder = E->updateStateIfReorder();
      if (IsReorder)
        VL0 = E->getMainOp();
      setInsertPointAfterBundle(E);

      LoadInst *LI = cast<LoadInst>(VL0);
      unsigned AS = LI->getPointerAddressSpace();

      Value *VecPtr = Builder.CreateBitCast(LI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));

      // The pointer operand uses an in-tree scalar so we add the new BitCast to
      // ExternalUses list to make sure that an extract will be generated in the
      // future.
      Value *PO = LI->getPointerOperand();
      if (getTreeEntry(PO))
        ExternalUses.push_back(ExternalUser(PO, cast<User>(VecPtr), 0));

      LI = Builder.CreateAlignedLoad(VecTy, VecPtr, LI->getAlign());
      Value *V = propagateMetadata(LI, E->Scalars);
      if (IsReorder) {
        SmallVector<int, 4> Mask;
        inversePermutation(E->ReorderIndices, Mask);
        V = Builder.CreateShuffleVector(V, UndefValue::get(V->getType()),
                                        Mask, "reorder_shuffle");
      }
      if (NeedToShuffleReuses) {
        // TODO: Merge this shuffle with the ReorderShuffleMask.
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Store: {
      bool IsReorder = !E->ReorderIndices.empty();
      auto *SI = cast<StoreInst>(
          IsReorder ? E->Scalars[E->ReorderIndices.front()] : VL0);
      unsigned AS = SI->getPointerAddressSpace();

      setInsertPointAfterBundle(E);

      Value *VecValue = vectorizeTree(E->getOperand(0));
      if (IsReorder) {
        SmallVector<int, 4> Mask(E->ReorderIndices.begin(),
                                 E->ReorderIndices.end());
        VecValue = Builder.CreateShuffleVector(
            VecValue, UndefValue::get(VecValue->getType()), Mask,
            "reorder_shuffle");
      }
      Value *ScalarPtr = SI->getPointerOperand();
      Value *VecPtr = Builder.CreateBitCast(
          ScalarPtr, VecValue->getType()->getPointerTo(AS));
      StoreInst *ST = Builder.CreateAlignedStore(VecValue, VecPtr,
                                                 SI->getAlign());

      // The pointer operand uses an in-tree scalar, so add the new BitCast to
      // ExternalUses to make sure that an extract will be generated in the
      // future.
      if (getTreeEntry(ScalarPtr))
        ExternalUses.push_back(ExternalUser(ScalarPtr, cast<User>(VecPtr), 0));

      Value *V = propagateMetadata(ST, E->Scalars);
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::GetElementPtr: {
      setInsertPointAfterBundle(E);

      Value *Op0 = vectorizeTree(E->getOperand(0));

      std::vector<Value *> OpVecs;
      for (int j = 1, e = cast<GetElementPtrInst>(VL0)->getNumOperands(); j < e;
           ++j) {
        ValueList &VL = E->getOperand(j);
        // Need to cast all elements to the same type before vectorization to
        // avoid crash.
        Type *VL0Ty = VL0->getOperand(j)->getType();
        Type *Ty = llvm::all_of(
                       VL, [VL0Ty](Value *V) { return VL0Ty == V->getType(); })
                       ? VL0Ty
                       : DL->getIndexType(cast<GetElementPtrInst>(VL0)
                                              ->getPointerOperandType()
                                              ->getScalarType());
        for (Value *&V : VL) {
          auto *CI = cast<ConstantInt>(V);
          V = ConstantExpr::getIntegerCast(CI, Ty,
                                           CI->getValue().isSignBitSet());
        }
        Value *OpVec = vectorizeTree(VL);
        OpVecs.push_back(OpVec);
      }

      Value *V = Builder.CreateGEP(
          cast<GetElementPtrInst>(VL0)->getSourceElementType(), Op0, OpVecs);
      if (Instruction *I = dyn_cast<Instruction>(V))
        V = propagateMetadata(I, E->Scalars);

      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
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
      bool UseIntrinsic = VecCallCosts.first <= VecCallCosts.second;

      Value *ScalarArg = nullptr;
      std::vector<Value *> OpVecs;
      for (int j = 0, e = CI->getNumArgOperands(); j < e; ++j) {
        ValueList OpVL;
        // Some intrinsics have scalar arguments. This argument should not be
        // vectorized.
        if (UseIntrinsic && hasVectorInstrinsicScalarOpd(IID, j)) {
          CallInst *CEI = cast<CallInst>(VL0);
          ScalarArg = CEI->getArgOperand(j);
          OpVecs.push_back(CEI->getArgOperand(j));
          continue;
        }

        Value *OpVec = vectorizeTree(E->getOperand(j));
        LLVM_DEBUG(dbgs() << "SLP: OpVec[" << j << "]: " << *OpVec << "\n");
        OpVecs.push_back(OpVec);
      }

      Module *M = F->getParent();
      Type *Tys[] = {FixedVectorType::get(CI->getType(), E->Scalars.size())};
      Function *CF = Intrinsic::getDeclaration(M, ID, Tys);

      if (!UseIntrinsic) {
        VFShape Shape = VFShape::get(
            *CI, {static_cast<unsigned>(VecTy->getNumElements()), false},
            false /*HasGlobalPred*/);
        CF = VFDatabase(*CI).getVectorizedFunction(Shape);
      }

      SmallVector<OperandBundleDef, 1> OpBundles;
      CI->getOperandBundlesAsDefs(OpBundles);
      Value *V = Builder.CreateCall(CF, OpVecs, OpBundles);

      // The scalar argument uses an in-tree scalar so we add the new vectorized
      // call to ExternalUses list to make sure that an extract will be
      // generated in the future.
      if (ScalarArg && getTreeEntry(ScalarArg))
        ExternalUses.push_back(ExternalUser(ScalarArg, cast<User>(V), 0));

      propagateIRFlags(V, E->Scalars, VL0);
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
      }
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::ShuffleVector: {
      assert(E->isAltShuffle() &&
             ((Instruction::isBinaryOp(E->getOpcode()) &&
               Instruction::isBinaryOp(E->getAltOpcode())) ||
              (Instruction::isCast(E->getOpcode()) &&
               Instruction::isCast(E->getAltOpcode()))) &&
             "Invalid Shuffle Vector Operand");

      Value *LHS = nullptr, *RHS = nullptr;
      if (Instruction::isBinaryOp(E->getOpcode())) {
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
      } else {
        V0 = Builder.CreateCast(
            static_cast<Instruction::CastOps>(E->getOpcode()), LHS, VecTy);
        V1 = Builder.CreateCast(
            static_cast<Instruction::CastOps>(E->getAltOpcode()), LHS, VecTy);
      }

      // Create shuffle to take alternate operations from the vector.
      // Also, gather up main and alt scalar ops to propagate IR flags to
      // each vector operation.
      ValueList OpScalars, AltScalars;
      unsigned e = E->Scalars.size();
      SmallVector<int, 8> Mask(e);
      for (unsigned i = 0; i < e; ++i) {
        auto *OpInst = cast<Instruction>(E->Scalars[i]);
        assert(E->isOpcodeOrAlt(OpInst) && "Unexpected main/alternate opcode");
        if (OpInst->getOpcode() == E->getAltOpcode()) {
          Mask[i] = e + i;
          AltScalars.push_back(E->Scalars[i]);
        } else {
          Mask[i] = i;
          OpScalars.push_back(E->Scalars[i]);
        }
      }

      propagateIRFlags(V0, OpScalars);
      propagateIRFlags(V1, AltScalars);

      Value *V = Builder.CreateShuffleVector(V0, V1, Mask);
      if (Instruction *I = dyn_cast<Instruction>(V))
        V = propagateMetadata(I, E->Scalars);
      if (NeedToShuffleReuses) {
        V = Builder.CreateShuffleVector(V, UndefValue::get(VecTy),
                                        E->ReuseShuffleIndices, "shuffle");
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
    if (auto *I = dyn_cast<Instruction>(VectorRoot))
      Builder.SetInsertPoint(&*++BasicBlock::iterator(I));
    auto BundleWidth = VectorizableTree[0]->Scalars.size();
    auto *MinTy = IntegerType::get(F->getContext(), MinBWs[ScalarRoot].first);
    auto *VecTy = FixedVectorType::get(MinTy, BundleWidth);
    auto *Trunc = Builder.CreateTrunc(VectorRoot, VecTy);
    VectorizableTree[0]->VectorizedValue = Trunc;
  }

  LLVM_DEBUG(dbgs() << "SLP: Extracting " << ExternalUses.size()
                    << " values .\n");

  // If necessary, sign-extend or zero-extend ScalarRoot to the larger type
  // specified by ScalarType.
  auto extend = [&](Value *ScalarRoot, Value *Ex, Type *ScalarType) {
    if (!MinBWs.count(ScalarRoot))
      return Ex;
    if (MinBWs[ScalarRoot].second)
      return Builder.CreateSExt(Ex, ScalarType);
    return Builder.CreateZExt(Ex, ScalarType);
  };

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
    assert(E->State == TreeEntry::Vectorize && "Extracting from a gather list");

    Value *Vec = E->VectorizedValue;
    assert(Vec && "Can't find vectorizable value");

    Value *Lane = Builder.getInt32(ExternalUse.Lane);
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
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      Ex = extend(ScalarRoot, Ex, Scalar->getType());
      CSEBlocks.insert(cast<Instruction>(Scalar)->getParent());
      auto &Locs = ExternallyUsedValues[Scalar];
      ExternallyUsedValues.insert({Ex, Locs});
      ExternallyUsedValues.erase(Scalar);
      // Required to update internally referenced instructions.
      Scalar->replaceAllUsesWith(Ex);
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
            Value *Ex = Builder.CreateExtractElement(Vec, Lane);
            Ex = extend(ScalarRoot, Ex, Scalar->getType());
            CSEBlocks.insert(PH->getIncomingBlock(i));
            PH->setOperand(i, Ex);
          }
        }
      } else {
        Builder.SetInsertPoint(cast<Instruction>(User));
        Value *Ex = Builder.CreateExtractElement(Vec, Lane);
        Ex = extend(ScalarRoot, Ex, Scalar->getType());
        CSEBlocks.insert(cast<Instruction>(User)->getParent());
        User->replaceUsesOfWith(Scalar, Ex);
      }
    } else {
      Builder.SetInsertPoint(&F->getEntryBlock().front());
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      Ex = extend(ScalarRoot, Ex, Scalar->getType());
      CSEBlocks.insert(&F->getEntryBlock());
      User->replaceUsesOfWith(Scalar, Ex);
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
          assert((getTreeEntry(U) || is_contained(UserIgnoreList, U)) &&
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
  LLVM_DEBUG(dbgs() << "SLP: Optimizing " << GatherSeq.size()
                    << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (Instruction *I : GatherSeq) {
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
    auto *Op0 = dyn_cast<Instruction>(I->getOperand(0));
    auto *Op1 = dyn_cast<Instruction>(I->getOperand(1));
    if (Op0 && L->contains(Op0))
      continue;
    if (Op1 && L->contains(Op1))
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
  llvm::stable_sort(CSEWorkList,
                    [this](const DomTreeNode *A, const DomTreeNode *B) {
                      return DT->properlyDominates(A, B);
                    });

  // Perform O(N^2) search over the gather sequences and merge identical
  // instructions. TODO: We can further optimize this scan if we split the
  // instructions into different buckets based on the insert lane.
  SmallVector<Instruction *, 16> Visited;
  for (auto I = CSEWorkList.begin(), E = CSEWorkList.end(); I != E; ++I) {
    assert((I == CSEWorkList.begin() || !DT->dominates(*I, *std::prev(I))) &&
           "Worklist not sorted properly!");
    BasicBlock *BB = (*I)->getBlock();
    // For all instructions in blocks containing gather sequences:
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e;) {
      Instruction *In = &*it++;
      if (isDeleted(In))
        continue;
      if (!isa<InsertElementInst>(In) && !isa<ExtractElementInst>(In))
        continue;

      // Check if we can replace this instruction with any of the
      // visited instructions.
      for (Instruction *v : Visited) {
        if (In->isIdenticalTo(v) &&
            DT->dominates(v->getParent(), In->getParent())) {
          In->replaceAllUsesWith(v);
          eraseInstruction(In);
          In = nullptr;
          break;
        }
      }
      if (In) {
        assert(!is_contained(Visited, In));
        Visited.push_back(In);
      }
    }
  }
  CSEBlocks.clear();
  GatherSeq.clear();
}

// Groups the instructions to a bundle (which is then a single scheduling entity)
// and schedules instructions until the bundle gets ready.
Optional<BoUpSLP::ScheduleData *>
BoUpSLP::BlockScheduling::tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP,
                                            const InstructionsState &S) {
  if (isa<PHINode>(S.OpValue))
    return nullptr;

  // Initialize the instruction bundle.
  Instruction *OldScheduleEnd = ScheduleEnd;
  ScheduleData *PrevInBundle = nullptr;
  ScheduleData *Bundle = nullptr;
  bool ReSchedule = false;
  LLVM_DEBUG(dbgs() << "SLP:  bundle: " << *S.OpValue << "\n");

  // Make sure that the scheduling region contains all
  // instructions of the bundle.
  for (Value *V : VL) {
    if (!extendSchedulingRegion(V, S))
      return None;
  }

  for (Value *V : VL) {
    ScheduleData *BundleMember = getScheduleData(V);
    assert(BundleMember &&
           "no ScheduleData for bundle member (maybe not in same basic block)");
    if (BundleMember->IsScheduled) {
      // A bundle member was scheduled as single instruction before and now
      // needs to be scheduled as part of the bundle. We just get rid of the
      // existing schedule.
      LLVM_DEBUG(dbgs() << "SLP:  reset schedule because " << *BundleMember
                        << " was already scheduled\n");
      ReSchedule = true;
    }
    assert(BundleMember->isSchedulingEntity() &&
           "bundle member already part of other bundle");
    if (PrevInBundle) {
      PrevInBundle->NextInBundle = BundleMember;
    } else {
      Bundle = BundleMember;
    }
    BundleMember->UnscheduledDepsInBundle = 0;
    Bundle->UnscheduledDepsInBundle += BundleMember->UnscheduledDeps;

    // Group the instructions to a bundle.
    BundleMember->FirstInBundle = Bundle;
    PrevInBundle = BundleMember;
  }
  if (ScheduleEnd != OldScheduleEnd) {
    // The scheduling region got new instructions at the lower end (or it is a
    // new region for the first bundle). This makes it necessary to
    // recalculate all dependencies.
    // It is seldom that this needs to be done a second time after adding the
    // initial bundle to the region.
    for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
      doForAllOpcodes(I, [](ScheduleData *SD) {
        SD->clearDependencies();
      });
    }
    ReSchedule = true;
  }
  if (ReSchedule) {
    resetSchedule();
    initialFillReadyList(ReadyInsts);
  }
  assert(Bundle && "Failed to find schedule bundle");

  LLVM_DEBUG(dbgs() << "SLP: try schedule bundle " << *Bundle << " in block "
                    << BB->getName() << "\n");

  calculateDependencies(Bundle, true, SLP);

  // Now try to schedule the new bundle. As soon as the bundle is "ready" it
  // means that there are no cyclic dependencies and we can schedule it.
  // Note that's important that we don't "schedule" the bundle yet (see
  // cancelScheduling).
  while (!Bundle->isReady() && !ReadyInsts.empty()) {

    ScheduleData *pickedSD = ReadyInsts.back();
    ReadyInsts.pop_back();

    if (pickedSD->isSchedulingEntity() && pickedSD->isReady()) {
      schedule(pickedSD, ReadyInsts);
    }
  }
  if (!Bundle->isReady()) {
    cancelScheduling(VL, S.OpValue);
    return None;
  }
  return Bundle;
}

void BoUpSLP::BlockScheduling::cancelScheduling(ArrayRef<Value *> VL,
                                                Value *OpValue) {
  if (isa<PHINode>(OpValue))
    return;

  ScheduleData *Bundle = getScheduleData(OpValue);
  LLVM_DEBUG(dbgs() << "SLP:  cancel scheduling of " << *Bundle << "\n");
  assert(!Bundle->IsScheduled &&
         "Can't cancel bundle which is already scheduled");
  assert(Bundle->isSchedulingEntity() && Bundle->isPartOfBundle() &&
         "tried to unbundle something which is not a bundle");

  // Un-bundle: make single instructions out of the bundle.
  ScheduleData *BundleMember = Bundle;
  while (BundleMember) {
    assert(BundleMember->FirstInBundle == Bundle && "corrupt bundle links");
    BundleMember->FirstInBundle = BundleMember;
    ScheduleData *Next = BundleMember->NextInBundle;
    BundleMember->NextInBundle = nullptr;
    BundleMember->UnscheduledDepsInBundle = BundleMember->UnscheduledDeps;
    if (BundleMember->UnscheduledDepsInBundle == 0) {
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
  assert(!isa<PHINode>(I) && "phi nodes don't need to be scheduled");
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
  while (true) {
    if (++ScheduleRegionSize > ScheduleRegionSizeLimit) {
      LLVM_DEBUG(dbgs() << "SLP:  exceeded schedule region size limit\n");
      return false;
    }

    if (UpIter != UpperEnd) {
      if (&*UpIter == I) {
        initScheduleData(I, ScheduleStart, nullptr, FirstLoadStoreInRegion);
        ScheduleStart = I;
        if (isOneOf(S, I) != I)
          CheckSheduleForI(I);
        LLVM_DEBUG(dbgs() << "SLP:  extend schedule region start to " << *I
                          << "\n");
        return true;
      }
      ++UpIter;
    }
    if (DownIter != LowerEnd) {
      if (&*DownIter == I) {
        initScheduleData(ScheduleEnd, I->getNextNode(), LastLoadStoreInRegion,
                         nullptr);
        ScheduleEnd = I->getNextNode();
        if (isOneOf(S, I) != I)
          CheckSheduleForI(I);
        assert(ScheduleEnd && "tried to vectorize a terminator?");
        LLVM_DEBUG(dbgs() << "SLP:  extend schedule region end to " << *I
                          << "\n");
        return true;
      }
      ++DownIter;
    }
    assert((UpIter != UpperEnd || DownIter != LowerEnd) &&
           "instruction not found in block");
  }
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
         cast<IntrinsicInst>(I)->getIntrinsicID() != Intrinsic::sideeffect)) {
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
    ScheduleData *SD = WorkList.back();
    WorkList.pop_back();

    ScheduleData *BundleMember = SD;
    while (BundleMember) {
      assert(isInSchedulingRegion(BundleMember));
      if (!BundleMember->hasValidDependencies()) {

        LLVM_DEBUG(dbgs() << "SLP:       update deps of " << *BundleMember
                          << "\n");
        BundleMember->Dependencies = 0;
        BundleMember->resetUnscheduledDeps();

        // Handle def-use chain dependencies.
        if (BundleMember->OpValue != BundleMember->Inst) {
          ScheduleData *UseSD = getScheduleData(BundleMember->Inst);
          if (UseSD && isInSchedulingRegion(UseSD->FirstInBundle)) {
            BundleMember->Dependencies++;
            ScheduleData *DestBundle = UseSD->FirstInBundle;
            if (!DestBundle->IsScheduled)
              BundleMember->incrementUnscheduledDeps(1);
            if (!DestBundle->hasValidDependencies())
              WorkList.push_back(DestBundle);
          }
        } else {
          for (User *U : BundleMember->Inst->users()) {
            if (isa<Instruction>(U)) {
              ScheduleData *UseSD = getScheduleData(U);
              if (UseSD && isInSchedulingRegion(UseSD->FirstInBundle)) {
                BundleMember->Dependencies++;
                ScheduleData *DestBundle = UseSD->FirstInBundle;
                if (!DestBundle->IsScheduled)
                  BundleMember->incrementUnscheduledDeps(1);
                if (!DestBundle->hasValidDependencies())
                  WorkList.push_back(DestBundle);
              }
            } else {
              // I'm not sure if this can ever happen. But we need to be safe.
              // This lets the instruction/bundle never be scheduled and
              // eventually disable vectorization.
              BundleMember->Dependencies++;
              BundleMember->incrementUnscheduledDeps(1);
            }
          }
        }

        // Handle the memory dependencies.
        ScheduleData *DepDest = BundleMember->NextLoadStore;
        if (DepDest) {
          Instruction *SrcInst = BundleMember->Inst;
          MemoryLocation SrcLoc = getLocation(SrcInst, SLP->AA);
          bool SrcMayWrite = BundleMember->Inst->mayWriteToMemory();
          unsigned numAliased = 0;
          unsigned DistToSrc = 1;

          while (DepDest) {
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
            DepDest = DepDest->NextLoadStore;

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
      }
      BundleMember = BundleMember->NextInBundle;
    }
    if (InsertInReadyList && SD->isReady()) {
      ReadyInsts.push_back(SD);
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
      assert(SD->isPartOfBundle() ==
                 (getTreeEntry(SD->Inst) != nullptr) &&
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
    ScheduleData *BundleMember = picked;
    while (BundleMember) {
      Instruction *pickedInst = BundleMember->Inst;
      if (LastScheduledInst->getNextNode() != pickedInst) {
        BS->BB->getInstList().remove(pickedInst);
        BS->BB->getInstList().insert(LastScheduledInst->getIterator(),
                                     pickedInst);
      }
      LastScheduledInst = pickedInst;
      BundleMember = BundleMember->NextInBundle;
    }

    BS->schedule(picked, ReadyInsts);
    NumToSchedule--;
  }
  assert(NumToSchedule == 0 && "could not schedule all instructions");

  // Avoid duplicate scheduling of the block.
  BS->ScheduleStart = nullptr;
}

unsigned BoUpSLP::getVectorElementSize(Value *V) {
  // If V is a store, just return the width of the stored value without
  // traversing the expression tree. This is the common case.
  if (auto *Store = dyn_cast<StoreInst>(V))
    return DL->getTypeSizeInBits(Store->getValueOperand()->getType());

  auto E = InstrElementSize.find(V);
  if (E != InstrElementSize.end())
    return E->second;

  // If V is not a store, we can traverse the expression tree to find loads
  // that feed it. The type of the loaded value may indicate a more suitable
  // width than V's type. We want to base the vector element size on the width
  // of memory operations where possible.
  SmallVector<Instruction *, 16> Worklist;
  SmallPtrSet<Instruction *, 16> Visited;
  if (auto *I = dyn_cast<Instruction>(V)) {
    Worklist.push_back(I);
    Visited.insert(I);
  }

  // Traverse the expression tree in bottom-up order looking for loads. If we
  // encounter an instruction we don't yet handle, we give up.
  auto MaxWidth = 0u;
  auto FoundUnknownInst = false;
  while (!Worklist.empty() && !FoundUnknownInst) {
    auto *I = Worklist.pop_back_val();

    // We should only be looking at scalar instructions here. If the current
    // instruction has a vector type, give up.
    auto *Ty = I->getType();
    if (isa<VectorType>(Ty))
      FoundUnknownInst = true;

    // If the current instruction is a load, update MaxWidth to reflect the
    // width of the loaded value.
    else if (isa<LoadInst>(I))
      MaxWidth = std::max<unsigned>(MaxWidth, DL->getTypeSizeInBits(Ty));

    // Otherwise, we need to visit the operands of the instruction. We only
    // handle the interesting cases from buildTree here. If an operand is an
    // instruction we haven't yet visited, we add it to the worklist.
    else if (isa<PHINode>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I) ||
             isa<CmpInst>(I) || isa<SelectInst>(I) || isa<BinaryOperator>(I)) {
      for (Use &U : I->operands())
        if (auto *J = dyn_cast<Instruction>(U.get()))
          if (Visited.insert(J).second)
            Worklist.push_back(J);
    }

    // If we don't yet handle the instruction, give up.
    else
      FoundUnknownInst = true;
  }

  int Width = MaxWidth;
  // If we didn't encounter a memory access in the expression tree, or if we
  // gave up for some reason, just return the width of V. Otherwise, return the
  // maximum width we found.
  if (!MaxWidth || FoundUnknownInst)
    Width = DL->getTypeSizeInBits(V->getType());

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

  bool doInitialization(Module &M) override {
    return false;
  }

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
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
  return PA;
}

bool SLPVectorizerPass::runImpl(Function &F, ScalarEvolution *SE_,
                                TargetTransformInfo *TTI_,
                                TargetLibraryInfo *TLI_, AliasAnalysis *AA_,
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
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true)))
    return false;

  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  LLVM_DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

  // Use the bottom up slp vectorizer to construct chains that start with
  // store instructions.
  BoUpSLP R(&F, SE, TTI, TLI, AA, LI, DT, AC, DB, DL, ORE_);

  // A general note: the vectorizer must use BoUpSLP::eraseInstruction() to
  // delete instructions.

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
  Optional<ArrayRef<unsigned>> Order = R.bestOrder();
  // TODO: Handle orders of size less than number of elements in the vector.
  if (Order && Order->size() == Chain.size()) {
    // TODO: reorder tree nodes without tree rebuilding.
    SmallVector<Value *, 4> ReorderedOps(Chain.rbegin(), Chain.rend());
    llvm::transform(*Order, ReorderedOps.begin(),
                    [Chain](const unsigned Idx) { return Chain[Idx]; });
    R.buildTree(ReorderedOps);
  }
  if (R.isTreeTinyAndNotFullyVectorizable())
    return false;
  if (R.isLoadCombineCandidate())
    return false;

  R.computeMinimumValueSizes();

  int Cost = R.getTreeCost();

  LLVM_DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
  if (Cost < -SLPCostThreshold) {
    LLVM_DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");

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
  SmallVector<int, 16> ConsecutiveChain(E, E + 1);
  int MaxIter = MaxStoreLookup.getValue();
  int IterCnt;
  auto &&FindConsecutiveAccess = [this, &Stores, &Tails, &IterCnt, MaxIter,
                                  &ConsecutiveChain](int K, int Idx) {
    if (IterCnt >= MaxIter)
      return true;
    ++IterCnt;
    if (!isConsecutiveAccess(Stores[K], Stores[Idx], *DL, *SE))
      return false;

    Tails.set(Idx);
    ConsecutiveChain[K] = Idx;
    return true;
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

  // For stores that start but don't end a link in the chain:
  for (int Cnt = E; Cnt > 0; --Cnt) {
    int I = Cnt - 1;
    if (ConsecutiveChain[I] == E + 1 || Tails.test(I))
      continue;
    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    BoUpSLP::ValueList Operands;
    // Collect the chain into a list.
    while (I != E + 1 && !VectorizedStores.count(Stores[I])) {
      Operands.push_back(Stores[I]);
      // Move to the next value in the chain.
      I = ConsecutiveChain[I];
    }

    // If a vector register can't hold 1 element, we are done.
    unsigned MaxVecRegSize = R.getMaxVecRegSize();
    unsigned EltSize = R.getVectorElementSize(Stores[0]);
    if (MaxVecRegSize % EltSize != 0)
      continue;

    unsigned MaxElts = MaxVecRegSize / EltSize;
    // FIXME: Is division-by-2 the correct step? Should we assert that the
    // register size is a power-of-2?
    unsigned StartIdx = 0;
    for (unsigned Size = llvm::PowerOf2Ceil(MaxElts); Size >= 2; Size /= 2) {
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
      Stores[GetUnderlyingObject(SI->getPointerOperand(), *DL)].push_back(SI);
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
  Value *VL[] = {A, B};
  return tryToVectorizeList(VL, R, /*AllowReorder=*/true);
}

bool SLPVectorizerPass::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                           bool AllowReorder,
                                           ArrayRef<Value *> InsertUses) {
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
    if (!isValidElementType(Ty)) {
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
  unsigned MinVF = std::max(2U, R.getMinVecRegSize() / Sz);
  unsigned MaxVF = std::max<unsigned>(PowerOf2Floor(VL.size()), MinVF);
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
  int MinCost = SLPCostThreshold;

  bool CompensateUseCost =
      !InsertUses.empty() && llvm::all_of(InsertUses, [](const Value *V) {
        return V && isa<InsertElementInst>(V);
      });
  assert((!CompensateUseCost || InsertUses.size() == VL.size()) &&
         "Each scalar expected to have an associated InsertElement user.");

  unsigned NextInst = 0, MaxInst = VL.size();
  for (unsigned VF = MaxVF; NextInst + 1 < MaxInst && VF >= MinVF; VF /= 2) {
    // No actual vectorization should happen, if number of parts is the same as
    // provided vectorization factor (i.e. the scalar type is used for vector
    // code during codegen).
    auto *VecTy = FixedVectorType::get(VL[0]->getType(), VF);
    if (TTI->getNumberOfParts(VecTy) == VF)
      continue;
    for (unsigned I = NextInst; I < MaxInst; ++I) {
      unsigned OpsWidth = 0;

      if (I + VF > MaxInst)
        OpsWidth = MaxInst - I;
      else
        OpsWidth = VF;

      if (!isPowerOf2_32(OpsWidth) || OpsWidth < 2)
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
      Optional<ArrayRef<unsigned>> Order = R.bestOrder();
      // TODO: check if we can allow reordering for more cases.
      if (AllowReorder && Order) {
        // TODO: reorder tree nodes without tree rebuilding.
        // Conceptually, there is nothing actually preventing us from trying to
        // reorder a larger list. In fact, we do exactly this when vectorizing
        // reductions. However, at this point, we only expect to get here when
        // there are exactly two operations.
        assert(Ops.size() == 2);
        Value *ReorderedOps[] = {Ops[1], Ops[0]};
        R.buildTree(ReorderedOps, None);
      }
      if (R.isTreeTinyAndNotFullyVectorizable())
        continue;

      R.computeMinimumValueSizes();
      int Cost = R.getTreeCost();
      CandidateFound = true;
      if (CompensateUseCost) {
        // TODO: Use TTI's getScalarizationOverhead for sequence of inserts
        // rather than sum of single inserts as the latter may overestimate
        // cost. This work should imply improving cost estimation for extracts
        // that added in for external (for vectorization tree) users,i.e. that
        // part should also switch to same interface.
        // For example, the following case is projected code after SLP:
        //  %4 = extractelement <4 x i64> %3, i32 0
        //  %v0 = insertelement <4 x i64> undef, i64 %4, i32 0
        //  %5 = extractelement <4 x i64> %3, i32 1
        //  %v1 = insertelement <4 x i64> %v0, i64 %5, i32 1
        //  %6 = extractelement <4 x i64> %3, i32 2
        //  %v2 = insertelement <4 x i64> %v1, i64 %6, i32 2
        //  %7 = extractelement <4 x i64> %3, i32 3
        //  %v3 = insertelement <4 x i64> %v2, i64 %7, i32 3
        //
        // Extracts here added by SLP in order to feed users (the inserts) of
        // original scalars and contribute to "ExtractCost" at cost evaluation.
        // The inserts in turn form sequence to build an aggregate that
        // detected by findBuildAggregate routine.
        // SLP makes an assumption that such sequence will be optimized away
        // later (instcombine) so it tries to compensate ExctractCost with
        // cost of insert sequence.
        // Current per element cost calculation approach is not quite accurate
        // and tends to create bias toward favoring vectorization.
        // Switching to the TTI interface might help a bit.
        // Alternative solution could be pattern-match to detect a no-op or
        // shuffle.
        unsigned UserCost = 0;
        for (unsigned Lane = 0; Lane < OpsWidth; Lane++) {
          auto *IE = cast<InsertElementInst>(InsertUses[I + Lane]);
          if (auto *CI = dyn_cast<ConstantInt>(IE->getOperand(2)))
            UserCost += TTI->getVectorInstrCost(
                Instruction::InsertElement, IE->getType(), CI->getZExtValue());
        }
        LLVM_DEBUG(dbgs() << "SLP: Compensate cost of users by: " << UserCost
                          << ".\n");
        Cost -= UserCost;
      }

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

/// Generate a shuffle mask to be used in a reduction tree.
///
/// \param VecLen The length of the vector to be reduced.
/// \param NumEltsToRdx The number of elements that should be reduced in the
///        vector.
/// \param IsPairwise Whether the reduction is a pairwise or splitting
///        reduction. A pairwise reduction will generate a mask of
///        <0,2,...> or <1,3,..> while a splitting reduction will generate
///        <2,3, undef,undef> for a vector of 4 and NumElts = 2.
/// \param IsLeft True will generate a mask of even elements, odd otherwise.
static SmallVector<int, 32> createRdxShuffleMask(unsigned VecLen,
                                                 unsigned NumEltsToRdx,
                                                 bool IsPairwise, bool IsLeft) {
  assert((IsPairwise || !IsLeft) && "Don't support a <0,1,undef,...> mask");

  SmallVector<int, 32> ShuffleMask(VecLen, -1);

  if (IsPairwise)
    // Build a mask of 0, 2, ... (left) or 1, 3, ... (right).
    for (unsigned i = 0; i != NumEltsToRdx; ++i)
      ShuffleMask[i] = 2 * i + !IsLeft;
  else
    // Move the upper half of the vector to the lower half.
    for (unsigned i = 0; i != NumEltsToRdx; ++i)
      ShuffleMask[i] = NumEltsToRdx + i;

  return ShuffleMask;
}

namespace {

/// Model horizontal reductions.
///
/// A horizontal reduction is a tree of reduction operations (currently add and
/// fadd) that has operations that can be put into a vector as its leaf.
/// For example, this tree:
///
/// mul mul mul mul
///  \  /    \  /
///   +       +
///    \     /
///       +
/// This tree has "mul" as its reduced values and "+" as its reduction
/// operations. A reduction might be feeding into a store or a binary operation
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
  ReductionOpsListType  ReductionOps;
  SmallVector<Value *, 32> ReducedVals;
  // Use map vector to make stable output.
  MapVector<Instruction *, Value *> ExtraArgs;

  /// Kind of the reduction data.
  enum ReductionKind {
    RK_None,       /// Not a reduction.
    RK_Arithmetic, /// Binary reduction data.
    RK_Min,        /// Minimum reduction data.
    RK_UMin,       /// Unsigned minimum reduction data.
    RK_Max,        /// Maximum reduction data.
    RK_UMax,       /// Unsigned maximum reduction data.
  };

  /// Contains info about operation, like its opcode, left and right operands.
  class OperationData {
    /// Opcode of the instruction.
    unsigned Opcode = 0;

    /// Left operand of the reduction operation.
    Value *LHS = nullptr;

    /// Right operand of the reduction operation.
    Value *RHS = nullptr;

    /// Kind of the reduction operation.
    ReductionKind Kind = RK_None;

    /// True if float point min/max reduction has no NaNs.
    bool NoNaN = false;

    /// Checks if the reduction operation can be vectorized.
    bool isVectorizable() const {
      return LHS && RHS &&
             // We currently only support add/mul/logical && min/max reductions.
             ((Kind == RK_Arithmetic &&
               (Opcode == Instruction::Add || Opcode == Instruction::FAdd ||
                Opcode == Instruction::Mul || Opcode == Instruction::FMul ||
                Opcode == Instruction::And || Opcode == Instruction::Or ||
                Opcode == Instruction::Xor)) ||
              ((Opcode == Instruction::ICmp || Opcode == Instruction::FCmp) &&
               (Kind == RK_Min || Kind == RK_Max)) ||
              (Opcode == Instruction::ICmp &&
               (Kind == RK_UMin || Kind == RK_UMax)));
    }

    /// Creates reduction operation with the current opcode.
    Value *createOp(IRBuilder<> &Builder, const Twine &Name) const {
      assert(isVectorizable() &&
             "Expected add|fadd or min/max reduction operation.");
      Value *Cmp = nullptr;
      switch (Kind) {
      case RK_Arithmetic:
        return Builder.CreateBinOp((Instruction::BinaryOps)Opcode, LHS, RHS,
                                   Name);
      case RK_Min:
        Cmp = Opcode == Instruction::ICmp ? Builder.CreateICmpSLT(LHS, RHS)
                                          : Builder.CreateFCmpOLT(LHS, RHS);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      case RK_Max:
        Cmp = Opcode == Instruction::ICmp ? Builder.CreateICmpSGT(LHS, RHS)
                                          : Builder.CreateFCmpOGT(LHS, RHS);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      case RK_UMin:
        assert(Opcode == Instruction::ICmp && "Expected integer types.");
        Cmp = Builder.CreateICmpULT(LHS, RHS);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      case RK_UMax:
        assert(Opcode == Instruction::ICmp && "Expected integer types.");
        Cmp = Builder.CreateICmpUGT(LHS, RHS);
        return Builder.CreateSelect(Cmp, LHS, RHS, Name);
      case RK_None:
        break;
      }
      llvm_unreachable("Unknown reduction operation.");
    }

  public:
    explicit OperationData() = default;

    /// Construction for reduced values. They are identified by opcode only and
    /// don't have associated LHS/RHS values.
    explicit OperationData(Value *V) {
      if (auto *I = dyn_cast<Instruction>(V))
        Opcode = I->getOpcode();
    }

    /// Constructor for reduction operations with opcode and its left and
    /// right operands.
    OperationData(unsigned Opcode, Value *LHS, Value *RHS, ReductionKind Kind,
                  bool NoNaN = false)
        : Opcode(Opcode), LHS(LHS), RHS(RHS), Kind(Kind), NoNaN(NoNaN) {
      assert(Kind != RK_None && "One of the reduction operations is expected.");
    }

    explicit operator bool() const { return Opcode; }

    /// Return true if this operation is any kind of minimum or maximum.
    bool isMinMax() const {
      switch (Kind) {
      case RK_Arithmetic:
        return false;
      case RK_Min:
      case RK_Max:
      case RK_UMin:
      case RK_UMax:
        return true;
      case RK_None:
        break;
      }
      llvm_unreachable("Reduction kind is not set");
    }

    /// Get the index of the first operand.
    unsigned getFirstOperandIndex() const {
      assert(!!*this && "The opcode is not set.");
      // We allow calling this before 'Kind' is set, so handle that specially.
      if (Kind == RK_None)
        return 0;
      return isMinMax() ? 1 : 0;
    }

    /// Total number of operands in the reduction operation.
    unsigned getNumberOfOperands() const {
      assert(Kind != RK_None && !!*this && LHS && RHS &&
             "Expected reduction operation.");
      return isMinMax() ? 3 : 2;
    }

    /// Checks if the operation has the same parent as \p P.
    bool hasSameParent(Instruction *I, Value *P, bool IsRedOp) const {
      assert(Kind != RK_None && !!*this && LHS && RHS &&
             "Expected reduction operation.");
      if (!IsRedOp)
        return I->getParent() == P;
      if (isMinMax()) {
        // SelectInst must be used twice while the condition op must have single
        // use only.
        auto *Cmp = cast<Instruction>(cast<SelectInst>(I)->getCondition());
        return I->getParent() == P && Cmp && Cmp->getParent() == P;
      }
      // Arithmetic reduction operation must be used once only.
      return I->getParent() == P;
    }

    /// Expected number of uses for reduction operations/reduced values.
    bool hasRequiredNumberOfUses(Instruction *I, bool IsReductionOp) const {
      assert(Kind != RK_None && !!*this && LHS && RHS &&
             "Expected reduction operation.");
      if (isMinMax())
        return I->hasNUses(2) &&
               (!IsReductionOp ||
                cast<SelectInst>(I)->getCondition()->hasOneUse());
      return I->hasOneUse();
    }

    /// Initializes the list of reduction operations.
    void initReductionOps(ReductionOpsListType &ReductionOps) {
      assert(Kind != RK_None && !!*this && LHS && RHS &&
             "Expected reduction operation.");
      if (isMinMax())
        ReductionOps.assign(2, ReductionOpsType());
      else
        ReductionOps.assign(1, ReductionOpsType());
    }

    /// Add all reduction operations for the reduction instruction \p I.
    void addReductionOps(Instruction *I, ReductionOpsListType &ReductionOps) {
      assert(Kind != RK_None && !!*this && LHS && RHS &&
             "Expected reduction operation.");
      if (isMinMax()) {
        ReductionOps[0].emplace_back(cast<SelectInst>(I)->getCondition());
        ReductionOps[1].emplace_back(I);
      } else {
        ReductionOps[0].emplace_back(I);
      }
    }

    /// Checks if instruction is associative and can be vectorized.
    bool isAssociative(Instruction *I) const {
      assert(Kind != RK_None && *this && LHS && RHS &&
             "Expected reduction operation.");
      switch (Kind) {
      case RK_Arithmetic:
        return I->isAssociative();
      case RK_Min:
      case RK_Max:
        return Opcode == Instruction::ICmp ||
               cast<Instruction>(I->getOperand(0))->isFast();
      case RK_UMin:
      case RK_UMax:
        assert(Opcode == Instruction::ICmp &&
               "Only integer compare operation is expected.");
        return true;
      case RK_None:
        break;
      }
      llvm_unreachable("Reduction kind is not set");
    }

    /// Checks if the reduction operation can be vectorized.
    bool isVectorizable(Instruction *I) const {
      return isVectorizable() && isAssociative(I);
    }

    /// Checks if two operation data are both a reduction op or both a reduced
    /// value.
    bool operator==(const OperationData &OD) const {
      assert(((Kind != OD.Kind) || ((!LHS == !OD.LHS) && (!RHS == !OD.RHS))) &&
             "One of the comparing operations is incorrect.");
      return this == &OD || (Kind == OD.Kind && Opcode == OD.Opcode);
    }
    bool operator!=(const OperationData &OD) const { return !(*this == OD); }
    void clear() {
      Opcode = 0;
      LHS = nullptr;
      RHS = nullptr;
      Kind = RK_None;
      NoNaN = false;
    }

    /// Get the opcode of the reduction operation.
    unsigned getOpcode() const {
      assert(isVectorizable() && "Expected vectorizable operation.");
      return Opcode;
    }

    /// Get kind of reduction data.
    ReductionKind getKind() const { return Kind; }
    Value *getLHS() const { return LHS; }
    Value *getRHS() const { return RHS; }
    Type *getConditionType() const {
      return isMinMax() ? CmpInst::makeCmpResultType(LHS->getType()) : nullptr;
    }

    /// Creates reduction operation with the current opcode with the IR flags
    /// from \p ReductionOps.
    Value *createOp(IRBuilder<> &Builder, const Twine &Name,
                    const ReductionOpsListType &ReductionOps) const {
      assert(isVectorizable() &&
             "Expected add|fadd or min/max reduction operation.");
      auto *Op = createOp(Builder, Name);
      switch (Kind) {
      case RK_Arithmetic:
        propagateIRFlags(Op, ReductionOps[0]);
        return Op;
      case RK_Min:
      case RK_Max:
      case RK_UMin:
      case RK_UMax:
        if (auto *SI = dyn_cast<SelectInst>(Op))
          propagateIRFlags(SI->getCondition(), ReductionOps[0]);
        propagateIRFlags(Op, ReductionOps[1]);
        return Op;
      case RK_None:
        break;
      }
      llvm_unreachable("Unknown reduction operation.");
    }
    /// Creates reduction operation with the current opcode with the IR flags
    /// from \p I.
    Value *createOp(IRBuilder<> &Builder, const Twine &Name,
                    Instruction *I) const {
      assert(isVectorizable() &&
             "Expected add|fadd or min/max reduction operation.");
      auto *Op = createOp(Builder, Name);
      switch (Kind) {
      case RK_Arithmetic:
        propagateIRFlags(Op, I);
        return Op;
      case RK_Min:
      case RK_Max:
      case RK_UMin:
      case RK_UMax:
        if (auto *SI = dyn_cast<SelectInst>(Op)) {
          propagateIRFlags(SI->getCondition(),
                           cast<SelectInst>(I)->getCondition());
        }
        propagateIRFlags(Op, I);
        return Op;
      case RK_None:
        break;
      }
      llvm_unreachable("Unknown reduction operation.");
    }

    TargetTransformInfo::ReductionFlags getFlags() const {
      TargetTransformInfo::ReductionFlags Flags;
      Flags.NoNaN = NoNaN;
      switch (Kind) {
      case RK_Arithmetic:
        break;
      case RK_Min:
        Flags.IsSigned = Opcode == Instruction::ICmp;
        Flags.IsMaxOp = false;
        break;
      case RK_Max:
        Flags.IsSigned = Opcode == Instruction::ICmp;
        Flags.IsMaxOp = true;
        break;
      case RK_UMin:
        Flags.IsSigned = false;
        Flags.IsMaxOp = false;
        break;
      case RK_UMax:
        Flags.IsSigned = false;
        Flags.IsMaxOp = true;
        break;
      case RK_None:
        llvm_unreachable("Reduction kind is not set");
      }
      return Flags;
    }
  };

  WeakTrackingVH ReductionRoot;

  /// The operation data of the reduction operation.
  OperationData ReductionData;

  /// The operation data of the values we perform a reduction on.
  OperationData ReducedValueData;

  /// Should we model this reduction as a pairwise reduction tree or a tree that
  /// splits the vector in halves and adds those halves.
  bool IsPairwiseReduction = false;

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
      ParentStackElem.second = ParentStackElem.first->getNumOperands();
    } else {
      // We ran into something like:
      // ParentStackElem.first += ... + ExtraArg + ...
      ExtraArgs[ParentStackElem.first] = ExtraArg;
    }
  }

  static OperationData getOperationData(Value *V) {
    if (!V)
      return OperationData();

    Value *LHS;
    Value *RHS;
    if (m_BinOp(m_Value(LHS), m_Value(RHS)).match(V)) {
      return OperationData(cast<BinaryOperator>(V)->getOpcode(), LHS, RHS,
                           RK_Arithmetic);
    }
    if (auto *Select = dyn_cast<SelectInst>(V)) {
      // Look for a min/max pattern.
      if (m_UMin(m_Value(LHS), m_Value(RHS)).match(Select)) {
        return OperationData(Instruction::ICmp, LHS, RHS, RK_UMin);
      } else if (m_SMin(m_Value(LHS), m_Value(RHS)).match(Select)) {
        return OperationData(Instruction::ICmp, LHS, RHS, RK_Min);
      } else if (m_OrdFMin(m_Value(LHS), m_Value(RHS)).match(Select) ||
                 m_UnordFMin(m_Value(LHS), m_Value(RHS)).match(Select)) {
        return OperationData(
            Instruction::FCmp, LHS, RHS, RK_Min,
            cast<Instruction>(Select->getCondition())->hasNoNaNs());
      } else if (m_UMax(m_Value(LHS), m_Value(RHS)).match(Select)) {
        return OperationData(Instruction::ICmp, LHS, RHS, RK_UMax);
      } else if (m_SMax(m_Value(LHS), m_Value(RHS)).match(Select)) {
        return OperationData(Instruction::ICmp, LHS, RHS, RK_Max);
      } else if (m_OrdFMax(m_Value(LHS), m_Value(RHS)).match(Select) ||
                 m_UnordFMax(m_Value(LHS), m_Value(RHS)).match(Select)) {
        return OperationData(
            Instruction::FCmp, LHS, RHS, RK_Max,
            cast<Instruction>(Select->getCondition())->hasNoNaNs());
      } else {
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

        LHS = Select->getTrueValue();
        RHS = Select->getFalseValue();
        Value *Cond = Select->getCondition();

        // TODO: Support inverse predicates.
        if (match(Cond, m_Cmp(Pred, m_Specific(LHS), m_Instruction(L2)))) {
          if (!isa<ExtractElementInst>(RHS) ||
              !L2->isIdenticalTo(cast<Instruction>(RHS)))
            return OperationData(V);
        } else if (match(Cond, m_Cmp(Pred, m_Instruction(L1), m_Specific(RHS)))) {
          if (!isa<ExtractElementInst>(LHS) ||
              !L1->isIdenticalTo(cast<Instruction>(LHS)))
            return OperationData(V);
        } else {
          if (!isa<ExtractElementInst>(LHS) || !isa<ExtractElementInst>(RHS))
            return OperationData(V);
          if (!match(Cond, m_Cmp(Pred, m_Instruction(L1), m_Instruction(L2))) ||
              !L1->isIdenticalTo(cast<Instruction>(LHS)) ||
              !L2->isIdenticalTo(cast<Instruction>(RHS)))
            return OperationData(V);
        }
        switch (Pred) {
        default:
          return OperationData(V);

        case CmpInst::ICMP_ULT:
        case CmpInst::ICMP_ULE:
          return OperationData(Instruction::ICmp, LHS, RHS, RK_UMin);

        case CmpInst::ICMP_SLT:
        case CmpInst::ICMP_SLE:
          return OperationData(Instruction::ICmp, LHS, RHS, RK_Min);

        case CmpInst::FCMP_OLT:
        case CmpInst::FCMP_OLE:
        case CmpInst::FCMP_ULT:
        case CmpInst::FCMP_ULE:
          return OperationData(Instruction::FCmp, LHS, RHS, RK_Min,
                               cast<Instruction>(Cond)->hasNoNaNs());

        case CmpInst::ICMP_UGT:
        case CmpInst::ICMP_UGE:
          return OperationData(Instruction::ICmp, LHS, RHS, RK_UMax);

        case CmpInst::ICMP_SGT:
        case CmpInst::ICMP_SGE:
          return OperationData(Instruction::ICmp, LHS, RHS, RK_Max);

        case CmpInst::FCMP_OGT:
        case CmpInst::FCMP_OGE:
        case CmpInst::FCMP_UGT:
        case CmpInst::FCMP_UGE:
          return OperationData(Instruction::FCmp, LHS, RHS, RK_Max,
                               cast<Instruction>(Cond)->hasNoNaNs());
        }
      }
    }
    return OperationData(V);
  }

public:
  HorizontalReduction() = default;

  /// Try to find a reduction tree.
  bool matchAssociativeReduction(PHINode *Phi, Instruction *B) {
    assert((!Phi || is_contained(Phi->operands(), B)) &&
           "Thi phi needs to use the binary operator");

    ReductionData = getOperationData(B);

    // We could have a initial reductions that is not an add.
    //  r *= v1 + v2 + v3 + v4
    // In such a case start looking for a tree rooted in the first '+'.
    if (Phi) {
      if (ReductionData.getLHS() == Phi) {
        Phi = nullptr;
        B = dyn_cast<Instruction>(ReductionData.getRHS());
        ReductionData = getOperationData(B);
      } else if (ReductionData.getRHS() == Phi) {
        Phi = nullptr;
        B = dyn_cast<Instruction>(ReductionData.getLHS());
        ReductionData = getOperationData(B);
      }
    }

    if (!ReductionData.isVectorizable(B))
      return false;

    Type *Ty = B->getType();
    if (!isValidElementType(Ty))
      return false;
    if (!Ty->isIntOrIntVectorTy() && !Ty->isFPOrFPVectorTy())
      return false;

    ReducedValueData.clear();
    ReductionRoot = B;

    // Post order traverse the reduction tree starting at B. We only handle true
    // trees containing only binary operators.
    SmallVector<std::pair<Instruction *, unsigned>, 32> Stack;
    Stack.push_back(std::make_pair(B, ReductionData.getFirstOperandIndex()));
    ReductionData.initReductionOps(ReductionOps);
    while (!Stack.empty()) {
      Instruction *TreeN = Stack.back().first;
      unsigned EdgeToVist = Stack.back().second++;
      OperationData OpData = getOperationData(TreeN);
      bool IsReducedValue = OpData != ReductionData;

      // Postorder vist.
      if (IsReducedValue || EdgeToVist == OpData.getNumberOfOperands()) {
        if (IsReducedValue)
          ReducedVals.push_back(TreeN);
        else {
          auto I = ExtraArgs.find(TreeN);
          if (I != ExtraArgs.end() && !I->second) {
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
            ReductionData.addReductionOps(TreeN, ReductionOps);
        }
        // Retract.
        Stack.pop_back();
        continue;
      }

      // Visit left or right.
      Value *NextV = TreeN->getOperand(EdgeToVist);
      if (NextV != Phi) {
        auto *I = dyn_cast<Instruction>(NextV);
        OpData = getOperationData(I);
        // Continue analysis if the next operand is a reduction operation or
        // (possibly) a reduced value. If the reduced value opcode is not set,
        // the first met operation != reduction operation is considered as the
        // reduced value class.
        if (I && (!ReducedValueData || OpData == ReducedValueData ||
                  OpData == ReductionData)) {
          const bool IsReductionOperation = OpData == ReductionData;
          // Only handle trees in the current basic block.
          if (!ReductionData.hasSameParent(I, B->getParent(),
                                           IsReductionOperation)) {
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), I);
            continue;
          }

          // Each tree node needs to have minimal number of users except for the
          // ultimate reduction.
          if (!ReductionData.hasRequiredNumberOfUses(I,
                                                     OpData == ReductionData) &&
              I != B) {
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), I);
            continue;
          }

          if (IsReductionOperation) {
            // We need to be able to reassociate the reduction operations.
            if (!OpData.isAssociative(I)) {
              // I is an extra argument for TreeN (its parent operation).
              markExtraArg(Stack.back(), I);
              continue;
            }
          } else if (ReducedValueData &&
                     ReducedValueData != OpData) {
            // Make sure that the opcodes of the operations that we are going to
            // reduce match.
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), I);
            continue;
          } else if (!ReducedValueData)
            ReducedValueData = OpData;

          Stack.push_back(std::make_pair(I, OpData.getFirstOperandIndex()));
          continue;
        }
      }
      // NextV is an extra argument for TreeN (its parent operation).
      markExtraArg(Stack.back(), NextV);
    }
    return true;
  }

  /// Attempt to vectorize the tree found by
  /// matchAssociativeReduction.
  bool tryToReduce(BoUpSLP &V, TargetTransformInfo *TTI) {
    if (ReducedVals.empty())
      return false;

    // If there is a sufficient number of reduction values, reduce
    // to a nearby power-of-2. Can safely generate oversized
    // vectors and rely on the backend to split them to legal sizes.
    unsigned NumReducedVals = ReducedVals.size();
    if (NumReducedVals < 4)
      return false;

    unsigned ReduxWidth = PowerOf2Floor(NumReducedVals);

    Value *VectorizedTree = nullptr;

    // FIXME: Fast-math-flags should be set based on the instructions in the
    //        reduction (not all of 'fast' are required).
    IRBuilder<> Builder(cast<Instruction>(ReductionRoot));
    FastMathFlags Unsafe;
    Unsafe.setFast();
    Builder.setFastMathFlags(Unsafe);
    unsigned i = 0;

    BoUpSLP::ExtraValueToDebugLocsMap ExternallyUsedValues;
    // The same extra argument may be used several time, so log each attempt
    // to use it.
    for (auto &Pair : ExtraArgs) {
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
    for (auto &V : ReductionOps)
      IgnoreList.append(V.begin(), V.end());
    while (i < NumReducedVals - ReduxWidth + 1 && ReduxWidth > 2) {
      auto VL = makeArrayRef(&ReducedVals[i], ReduxWidth);
      V.buildTree(VL, ExternallyUsedValues, IgnoreList);
      Optional<ArrayRef<unsigned>> Order = V.bestOrder();
      // TODO: Handle orders of size less than number of elements in the vector.
      if (Order && Order->size() == VL.size()) {
        // TODO: reorder tree nodes without tree rebuilding.
        SmallVector<Value *, 4> ReorderedOps(VL.size());
        llvm::transform(*Order, ReorderedOps.begin(),
                        [VL](const unsigned Idx) { return VL[Idx]; });
        V.buildTree(ReorderedOps, ExternallyUsedValues, IgnoreList);
      }
      if (V.isTreeTinyAndNotFullyVectorizable())
        break;
      if (V.isLoadCombineReductionCandidate(ReductionData.getOpcode()))
        break;

      V.computeMinimumValueSizes();

      // Estimate cost.
      int TreeCost = V.getTreeCost();
      int ReductionCost = getReductionCost(TTI, ReducedVals[i], ReduxWidth);
      int Cost = TreeCost + ReductionCost;
      if (Cost >= -SLPCostThreshold) {
          V.getORE()->emit([&]() {
              return OptimizationRemarkMissed(
                         SV_NAME, "HorSLPNotBeneficial", cast<Instruction>(VL[0]))
                     << "Vectorizing horizontal reduction is possible"
                     << "but not beneficial with cost "
                     << ore::NV("Cost", Cost) << " and threshold "
                     << ore::NV("Threshold", -SLPCostThreshold);
          });
          break;
      }

      LLVM_DEBUG(dbgs() << "SLP: Vectorizing horizontal reduction at cost:"
                        << Cost << ". (HorRdx)\n");
      V.getORE()->emit([&]() {
          return OptimizationRemark(
                     SV_NAME, "VectorizedHorizontalReduction", cast<Instruction>(VL[0]))
          << "Vectorized horizontal reduction with cost "
          << ore::NV("Cost", Cost) << " and with tree size "
          << ore::NV("TreeSize", V.getTreeSize());
      });

      // Vectorize a tree.
      DebugLoc Loc = cast<Instruction>(ReducedVals[i])->getDebugLoc();
      Value *VectorizedRoot = V.vectorizeTree(ExternallyUsedValues);

      // Emit a reduction. For min/max, the root is a select, but the insertion
      // point is the compare condition of that select.
      Instruction *RdxRootInst = cast<Instruction>(ReductionRoot);
      if (ReductionData.isMinMax())
        Builder.SetInsertPoint(getCmpForMinMaxReduction(RdxRootInst));
      else
        Builder.SetInsertPoint(RdxRootInst);

      Value *ReducedSubTree =
          emitReduction(VectorizedRoot, Builder, ReduxWidth, TTI);
      if (VectorizedTree) {
        Builder.SetCurrentDebugLocation(Loc);
        OperationData VectReductionData(ReductionData.getOpcode(),
                                        VectorizedTree, ReducedSubTree,
                                        ReductionData.getKind());
        VectorizedTree =
            VectReductionData.createOp(Builder, "op.rdx", ReductionOps);
      } else
        VectorizedTree = ReducedSubTree;
      i += ReduxWidth;
      ReduxWidth = PowerOf2Floor(NumReducedVals - i);
    }

    if (VectorizedTree) {
      // Finish the reduction.
      for (; i < NumReducedVals; ++i) {
        auto *I = cast<Instruction>(ReducedVals[i]);
        Builder.SetCurrentDebugLocation(I->getDebugLoc());
        OperationData VectReductionData(ReductionData.getOpcode(),
                                        VectorizedTree, I,
                                        ReductionData.getKind());
        VectorizedTree = VectReductionData.createOp(Builder, "", ReductionOps);
      }
      for (auto &Pair : ExternallyUsedValues) {
        // Add each externally used value to the final reduction.
        for (auto *I : Pair.second) {
          Builder.SetCurrentDebugLocation(I->getDebugLoc());
          OperationData VectReductionData(ReductionData.getOpcode(),
                                          VectorizedTree, Pair.first,
                                          ReductionData.getKind());
          VectorizedTree = VectReductionData.createOp(Builder, "op.extra", I);
        }
      }

      // Update users. For a min/max reduction that ends with a compare and
      // select, we also have to RAUW for the compare instruction feeding the
      // reduction root. That's because the original compare may have extra uses
      // besides the final select of the reduction.
      if (ReductionData.isMinMax()) {
        if (auto *VecSelect = dyn_cast<SelectInst>(VectorizedTree)) {
          Instruction *ScalarCmp =
              getCmpForMinMaxReduction(cast<Instruction>(ReductionRoot));
          ScalarCmp->replaceAllUsesWith(VecSelect->getCondition());
        }
      }
      ReductionRoot->replaceAllUsesWith(VectorizedTree);

      // Mark all scalar reduction ops for deletion, they are replaced by the
      // vector reductions.
      V.eraseInstructions(IgnoreList);
    }
    return VectorizedTree != nullptr;
  }

  unsigned numReductionValues() const {
    return ReducedVals.size();
  }

private:
  /// Calculate the cost of a reduction.
  int getReductionCost(TargetTransformInfo *TTI, Value *FirstReducedVal,
                       unsigned ReduxWidth) {
    Type *ScalarTy = FirstReducedVal->getType();
    auto *VecTy = FixedVectorType::get(ScalarTy, ReduxWidth);

    int PairwiseRdxCost;
    int SplittingRdxCost;
    switch (ReductionData.getKind()) {
    case RK_Arithmetic:
      PairwiseRdxCost =
          TTI->getArithmeticReductionCost(ReductionData.getOpcode(), VecTy,
                                          /*IsPairwiseForm=*/true);
      SplittingRdxCost =
          TTI->getArithmeticReductionCost(ReductionData.getOpcode(), VecTy,
                                          /*IsPairwiseForm=*/false);
      break;
    case RK_Min:
    case RK_Max:
    case RK_UMin:
    case RK_UMax: {
      auto *VecCondTy = cast<VectorType>(CmpInst::makeCmpResultType(VecTy));
      bool IsUnsigned = ReductionData.getKind() == RK_UMin ||
                        ReductionData.getKind() == RK_UMax;
      PairwiseRdxCost =
          TTI->getMinMaxReductionCost(VecTy, VecCondTy,
                                      /*IsPairwiseForm=*/true, IsUnsigned);
      SplittingRdxCost =
          TTI->getMinMaxReductionCost(VecTy, VecCondTy,
                                      /*IsPairwiseForm=*/false, IsUnsigned);
      break;
    }
    case RK_None:
      llvm_unreachable("Expected arithmetic or min/max reduction operation");
    }

    IsPairwiseReduction = PairwiseRdxCost < SplittingRdxCost;
    int VecReduxCost = IsPairwiseReduction ? PairwiseRdxCost : SplittingRdxCost;

    int ScalarReduxCost = 0;
    switch (ReductionData.getKind()) {
    case RK_Arithmetic:
      ScalarReduxCost =
          TTI->getArithmeticInstrCost(ReductionData.getOpcode(), ScalarTy);
      break;
    case RK_Min:
    case RK_Max:
    case RK_UMin:
    case RK_UMax:
      ScalarReduxCost =
          TTI->getCmpSelInstrCost(ReductionData.getOpcode(), ScalarTy) +
          TTI->getCmpSelInstrCost(Instruction::Select, ScalarTy,
                                  CmpInst::makeCmpResultType(ScalarTy));
      break;
    case RK_None:
      llvm_unreachable("Expected arithmetic or min/max reduction operation");
    }
    ScalarReduxCost *= (ReduxWidth - 1);

    LLVM_DEBUG(dbgs() << "SLP: Adding cost " << VecReduxCost - ScalarReduxCost
                      << " for reduction that starts with " << *FirstReducedVal
                      << " (It is a "
                      << (IsPairwiseReduction ? "pairwise" : "splitting")
                      << " reduction)\n");

    return VecReduxCost - ScalarReduxCost;
  }

  /// Emit a horizontal reduction of the vectorized value.
  Value *emitReduction(Value *VectorizedValue, IRBuilder<> &Builder,
                       unsigned ReduxWidth, const TargetTransformInfo *TTI) {
    assert(VectorizedValue && "Need to have a vectorized tree node");
    assert(isPowerOf2_32(ReduxWidth) &&
           "We only handle power-of-two reductions for now");

    if (!IsPairwiseReduction) {
      // FIXME: The builder should use an FMF guard. It should not be hard-coded
      //        to 'fast'.
      assert(Builder.getFastMathFlags().isFast() && "Expected 'fast' FMF");
      return createSimpleTargetReduction(
          Builder, TTI, ReductionData.getOpcode(), VectorizedValue,
          ReductionData.getFlags(), ReductionOps.back());
    }

    Value *TmpVec = VectorizedValue;
    for (unsigned i = ReduxWidth / 2; i != 0; i >>= 1) {
      auto LeftMask = createRdxShuffleMask(ReduxWidth, i, true, true);
      auto RightMask = createRdxShuffleMask(ReduxWidth, i, true, false);

      Value *LeftShuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), LeftMask, "rdx.shuf.l");
      Value *RightShuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), (RightMask),
          "rdx.shuf.r");
      OperationData VectReductionData(ReductionData.getOpcode(), LeftShuf,
                                      RightShuf, ReductionData.getKind());
      TmpVec = VectReductionData.createOp(Builder, "op.rdx", ReductionOps);
    }

    // The result is in the first element of the vector.
    return Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));
  }
};

} // end anonymous namespace

/// Recognize construction of vectors like
///  %ra = insertelement <4 x float> undef, float %s0, i32 0
///  %rb = insertelement <4 x float> %ra, float %s1, i32 1
///  %rc = insertelement <4 x float> %rb, float %s2, i32 2
///  %rd = insertelement <4 x float> %rc, float %s3, i32 3
///  starting from the last insertelement or insertvalue instruction.
///
/// Also recognize aggregates like {<2 x float>, <2 x float>},
/// {{float, float}, {float, float}}, [2 x {float, float}] and so on.
/// See llvm/test/Transforms/SLPVectorizer/X86/pr42022.ll for examples.
///
/// Assume LastInsertInst is of InsertElementInst or InsertValueInst type.
///
/// \return true if it matches.
static bool findBuildAggregate(Value *LastInsertInst, TargetTransformInfo *TTI,
                               SmallVectorImpl<Value *> &BuildVectorOpds,
                               SmallVectorImpl<Value *> &InsertElts) {
  assert((isa<InsertElementInst>(LastInsertInst) ||
          isa<InsertValueInst>(LastInsertInst)) &&
         "Expected insertelement or insertvalue instruction!");
  do {
    Value *InsertedOperand;
    auto *IE = dyn_cast<InsertElementInst>(LastInsertInst);
    if (IE) {
      InsertedOperand = IE->getOperand(1);
      LastInsertInst = IE->getOperand(0);
    } else {
      auto *IV = cast<InsertValueInst>(LastInsertInst);
      InsertedOperand = IV->getInsertedValueOperand();
      LastInsertInst = IV->getAggregateOperand();
    }
    if (isa<InsertElementInst>(InsertedOperand) ||
        isa<InsertValueInst>(InsertedOperand)) {
      SmallVector<Value *, 8> TmpBuildVectorOpds;
      SmallVector<Value *, 8> TmpInsertElts;
      if (!findBuildAggregate(InsertedOperand, TTI, TmpBuildVectorOpds,
                              TmpInsertElts))
        return false;
      BuildVectorOpds.append(TmpBuildVectorOpds.rbegin(),
                             TmpBuildVectorOpds.rend());
      InsertElts.append(TmpInsertElts.rbegin(), TmpInsertElts.rend());
    } else {
      BuildVectorOpds.push_back(InsertedOperand);
      InsertElts.push_back(IE);
    }
    if (isa<UndefValue>(LastInsertInst))
      break;
    if ((!isa<InsertValueInst>(LastInsertInst) &&
         !isa<InsertElementInst>(LastInsertInst)) ||
        !LastInsertInst->hasOneUse())
      return false;
  } while (true);
  std::reverse(BuildVectorOpds.begin(), BuildVectorOpds.end());
  std::reverse(InsertElts.begin(), InsertElts.end());
  return true;
}

static bool PhiTypeSorterFunc(Value *V, Value *V2) {
  return V->getType() < V2->getType();
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
  SmallVector<std::pair<Instruction *, unsigned>, 8> Stack(1, {Root, 0});
  SmallPtrSet<Value *, 8> VisitedInstrs;
  bool Res = false;
  while (!Stack.empty()) {
    Instruction *Inst;
    unsigned Level;
    std::tie(Inst, Level) = Stack.pop_back_val();
    auto *BI = dyn_cast<BinaryOperator>(Inst);
    auto *SI = dyn_cast<SelectInst>(Inst);
    if (BI || SI) {
      HorizontalReduction HorRdx;
      if (HorRdx.matchAssociativeReduction(P, Inst)) {
        if (HorRdx.tryToReduce(R, TTI)) {
          Res = true;
          // Set P to nullptr to avoid re-analysis of phi node in
          // matchAssociativeReduction function unless this is the root node.
          P = nullptr;
          continue;
        }
      }
      if (P && BI) {
        Inst = dyn_cast<Instruction>(BI->getOperand(0));
        if (Inst == P)
          Inst = dyn_cast<Instruction>(BI->getOperand(1));
        if (!Inst) {
          // Set P to nullptr to avoid re-analysis of phi node in
          // matchAssociativeReduction function unless this is the root node.
          P = nullptr;
          continue;
        }
      }
    }
    // Set P to nullptr to avoid re-analysis of phi node in
    // matchAssociativeReduction function unless this is the root node.
    P = nullptr;
    if (Vectorize(Inst, R)) {
      Res = true;
      continue;
    }

    // Try to vectorize operands.
    // Continue analysis for the instruction from the same basic block only to
    // save compile time.
    if (++Level < RecursionMaxDepth)
      for (auto *Op : Inst->operand_values())
        if (VisitedInstrs.insert(Op).second)
          if (auto *I = dyn_cast<Instruction>(Op))
            if (!isa<PHINode>(I) && !R.isDeleted(I) && I->getParent() == BB)
              Stack.emplace_back(I, Level);
  }
  return Res;
}

bool SLPVectorizerPass::vectorizeRootInstruction(PHINode *P, Value *V,
                                                 BasicBlock *BB, BoUpSLP &R,
                                                 TargetTransformInfo *TTI) {
  if (!V)
    return false;
  auto *I = dyn_cast<Instruction>(V);
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
  if (!findBuildAggregate(IVI, TTI, BuildVectorOpds, BuildVectorInsts) ||
      BuildVectorOpds.size() < 2)
    return false;

  LLVM_DEBUG(dbgs() << "SLP: array mappable to vector: " << *IVI << "\n");
  // Aggregate value is unlikely to be processed in vector register, we need to
  // extract scalars into scalar registers, so NeedExtraction is set true.
  return tryToVectorizeList(BuildVectorOpds, R, /*AllowReorder=*/false,
                            BuildVectorInsts);
}

bool SLPVectorizerPass::vectorizeInsertElementInst(InsertElementInst *IEI,
                                                   BasicBlock *BB, BoUpSLP &R) {
  SmallVector<Value *, 16> BuildVectorInsts;
  SmallVector<Value *, 16> BuildVectorOpds;
  if (!findBuildAggregate(IEI, TTI, BuildVectorOpds, BuildVectorInsts) ||
      BuildVectorOpds.size() < 2 ||
      (llvm::all_of(BuildVectorOpds,
                    [](Value *V) { return isa<ExtractElementInst>(V); }) &&
       isShuffle(BuildVectorOpds)))
    return false;

  // Vectorize starting with the build vector operands ignoring the BuildVector
  // instructions for the purpose of scheduling and user extraction.
  return tryToVectorizeList(BuildVectorOpds, R, /*AllowReorder=*/false,
                            BuildVectorInsts);
}

bool SLPVectorizerPass::vectorizeCmpInst(CmpInst *CI, BasicBlock *BB,
                                         BoUpSLP &R) {
  if (tryToVectorizePair(CI->getOperand(0), CI->getOperand(1), R))
    return true;

  bool OpsChanged = false;
  for (int Idx = 0; Idx < 2; ++Idx) {
    OpsChanged |=
        vectorizeRootInstruction(nullptr, CI->getOperand(Idx), BB, R, TTI);
  }
  return OpsChanged;
}

bool SLPVectorizerPass::vectorizeSimpleInstructions(
    SmallVectorImpl<Instruction *> &Instructions, BasicBlock *BB, BoUpSLP &R) {
  bool OpsChanged = false;
  for (auto *I : reverse(Instructions)) {
    if (R.isDeleted(I))
      continue;
    if (auto *LastInsertValue = dyn_cast<InsertValueInst>(I))
      OpsChanged |= vectorizeInsertValueInst(LastInsertValue, BB, R);
    else if (auto *LastInsertElem = dyn_cast<InsertElementInst>(I))
      OpsChanged |= vectorizeInsertElementInst(LastInsertElem, BB, R);
    else if (auto *CI = dyn_cast<CmpInst>(I))
      OpsChanged |= vectorizeCmpInst(CI, BB, R);
  }
  Instructions.clear();
  return OpsChanged;
}

bool SLPVectorizerPass::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  SmallVector<Value *, 4> Incoming;
  SmallPtrSet<Value *, 16> VisitedInstrs;

  bool HaveVectorizedPhiNodes = true;
  while (HaveVectorizedPhiNodes) {
    HaveVectorizedPhiNodes = false;

    // Collect the incoming values from the PHIs.
    Incoming.clear();
    for (Instruction &I : *BB) {
      PHINode *P = dyn_cast<PHINode>(&I);
      if (!P)
        break;

      if (!VisitedInstrs.count(P) && !R.isDeleted(P))
        Incoming.push_back(P);
    }

    // Sort by type.
    llvm::stable_sort(Incoming, PhiTypeSorterFunc);

    // Try to vectorize elements base on their type.
    for (SmallVector<Value *, 4>::iterator IncIt = Incoming.begin(),
                                           E = Incoming.end();
         IncIt != E;) {

      // Look for the next elements with the same type.
      SmallVector<Value *, 4>::iterator SameTypeIt = IncIt;
      while (SameTypeIt != E &&
             (*SameTypeIt)->getType() == (*IncIt)->getType()) {
        VisitedInstrs.insert(*SameTypeIt);
        ++SameTypeIt;
      }

      // Try to vectorize them.
      unsigned NumElts = (SameTypeIt - IncIt);
      LLVM_DEBUG(dbgs() << "SLP: Trying to vectorize starting at PHIs ("
                        << NumElts << ")\n");
      // The order in which the phi nodes appear in the program does not matter.
      // So allow tryToVectorizeList to reorder them if it is beneficial. This
      // is done when there are exactly two elements since tryToVectorizeList
      // asserts that there are only two values when AllowReorder is true.
      bool AllowReorder = NumElts == 2;
      if (NumElts > 1 &&
          tryToVectorizeList(makeArrayRef(IncIt, NumElts), R, AllowReorder)) {
        // Success start over because instructions might have been changed.
        HaveVectorizedPhiNodes = true;
        Changed = true;
        break;
      }

      // Start over at the next instruction of a different type (or the end).
      IncIt = SameTypeIt;
    }
  }

  VisitedInstrs.clear();

  SmallVector<Instruction *, 8> PostProcessInstructions;
  SmallDenseSet<Instruction *, 4> KeyNodes;
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    // Skip instructions marked for the deletion.
    if (R.isDeleted(&*it))
      continue;
    // We may go through BB multiple times so skip the one we have checked.
    if (!VisitedInstrs.insert(&*it).second) {
      if (it->use_empty() && KeyNodes.count(&*it) > 0 &&
          vectorizeSimpleInstructions(PostProcessInstructions, BB, R)) {
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
      if (P->getNumIncomingValues() != 2)
        return Changed;

      // Try to match and vectorize a horizontal reduction.
      if (vectorizeRootInstruction(P, getReductionValue(DT, P, BB, LI), BB, R,
                                   TTI)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
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
      OpsChanged |= vectorizeSimpleInstructions(PostProcessInstructions, BB, R);
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
      auto GEPList = makeArrayRef(&Entry.second[BI], Len);

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
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = Stores.begin(), e = Stores.end(); it != e;
       ++it) {
    if (it->second.size() < 2)
      continue;

    LLVM_DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
                      << it->second.size() << ".\n");

    Changed |= vectorizeStores(it->second, R);
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
