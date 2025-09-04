//===------- VectorCombine.cpp - Optimize partial vector operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes scalar/vector interactions using target cost models. The
// transforms implemented here may not fit in traditional loop-based or SLP
// vectorization passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/VectorCombine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <numeric>
#include <optional>
#include <queue>
#include <set>

#define DEBUG_TYPE "vector-combine"
#include "llvm/Transforms/Utils/InstructionWorklist.h"

using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumVecLoad, "Number of vector loads formed");
STATISTIC(NumVecCmp, "Number of vector compares formed");
STATISTIC(NumVecBO, "Number of vector binops formed");
STATISTIC(NumVecCmpBO, "Number of vector compare + binop formed");
STATISTIC(NumShufOfBitcast, "Number of shuffles moved after bitcast");
STATISTIC(NumScalarOps, "Number of scalar unary + binary ops formed");
STATISTIC(NumScalarCmp, "Number of scalar compares formed");
STATISTIC(NumScalarIntrinsic, "Number of scalar intrinsic calls formed");

static cl::opt<bool> DisableVectorCombine(
    "disable-vector-combine", cl::init(false), cl::Hidden,
    cl::desc("Disable all vector combine transforms"));

static cl::opt<bool> DisableBinopExtractShuffle(
    "disable-binop-extract-shuffle", cl::init(false), cl::Hidden,
    cl::desc("Disable binop extract to shuffle transforms"));

static cl::opt<unsigned> MaxInstrsToScan(
    "vector-combine-max-scan-instrs", cl::init(30), cl::Hidden,
    cl::desc("Max number of instructions to scan for vector combining."));

static const unsigned InvalidIndex = std::numeric_limits<unsigned>::max();

namespace {
class VectorCombine {
public:
  VectorCombine(Function &F, const TargetTransformInfo &TTI,
                const DominatorTree &DT, AAResults &AA, AssumptionCache &AC,
                const DataLayout *DL, TTI::TargetCostKind CostKind,
                bool TryEarlyFoldsOnly)
      : F(F), Builder(F.getContext(), InstSimplifyFolder(*DL)), TTI(TTI),
        DT(DT), AA(AA), AC(AC), DL(DL), CostKind(CostKind), SQ(*DL),
        TryEarlyFoldsOnly(TryEarlyFoldsOnly) {}

  bool run();

private:
  Function &F;
  IRBuilder<InstSimplifyFolder> Builder;
  const TargetTransformInfo &TTI;
  const DominatorTree &DT;
  AAResults &AA;
  AssumptionCache &AC;
  const DataLayout *DL;
  TTI::TargetCostKind CostKind;
  const SimplifyQuery SQ;

  /// If true, only perform beneficial early IR transforms. Do not introduce new
  /// vector operations.
  bool TryEarlyFoldsOnly;

  InstructionWorklist Worklist;

  /// Next instruction to iterate. It will be updated when it is erased by
  /// RecursivelyDeleteTriviallyDeadInstructions.
  Instruction *NextInst;

  // TODO: Direct calls from the top-level "run" loop use a plain "Instruction"
  //       parameter. That should be updated to specific sub-classes because the
  //       run loop was changed to dispatch on opcode.
  bool vectorizeLoadInsert(Instruction &I);
  bool widenSubvectorLoad(Instruction &I);
  ExtractElementInst *getShuffleExtract(ExtractElementInst *Ext0,
                                        ExtractElementInst *Ext1,
                                        unsigned PreferredExtractIndex) const;
  bool isExtractExtractCheap(ExtractElementInst *Ext0, ExtractElementInst *Ext1,
                             const Instruction &I,
                             ExtractElementInst *&ConvertToShuffle,
                             unsigned PreferredExtractIndex);
  Value *foldExtExtCmp(Value *V0, Value *V1, Value *ExtIndex, Instruction &I);
  Value *foldExtExtBinop(Value *V0, Value *V1, Value *ExtIndex, Instruction &I);
  bool foldExtractExtract(Instruction &I);
  bool foldInsExtFNeg(Instruction &I);
  bool foldInsExtBinop(Instruction &I);
  bool foldInsExtVectorToShuffle(Instruction &I);
  bool foldBitOpOfCastops(Instruction &I);
  bool foldBitOpOfCastConstant(Instruction &I);
  bool foldBitcastShuffle(Instruction &I);
  bool scalarizeOpOrCmp(Instruction &I);
  bool scalarizeVPIntrinsic(Instruction &I);
  bool foldExtractedCmps(Instruction &I);
  bool foldBinopOfReductions(Instruction &I);
  bool foldSingleElementStore(Instruction &I);
  bool scalarizeLoadExtract(Instruction &I);
  bool scalarizeExtExtract(Instruction &I);
  bool foldConcatOfBoolMasks(Instruction &I);
  bool foldPermuteOfBinops(Instruction &I);
  bool foldShuffleOfBinops(Instruction &I);
  bool foldShuffleOfSelects(Instruction &I);
  bool foldShuffleOfCastops(Instruction &I);
  bool foldShuffleOfShuffles(Instruction &I);
  bool foldShuffleOfIntrinsics(Instruction &I);
  bool foldShuffleToIdentity(Instruction &I);
  bool foldShuffleFromReductions(Instruction &I);
  bool foldShuffleChainsToReduce(Instruction &I);
  bool foldCastFromReductions(Instruction &I);
  bool foldSelectShuffle(Instruction &I, bool FromReduction = false);
  bool foldInterleaveIntrinsics(Instruction &I);
  bool shrinkType(Instruction &I);
  bool shrinkLoadForShuffles(Instruction &I);
  bool shrinkPhiOfShuffles(Instruction &I);

  void replaceValue(Instruction &Old, Value &New, bool Erase = true) {
    LLVM_DEBUG(dbgs() << "VC: Replacing: " << Old << '\n');
    LLVM_DEBUG(dbgs() << "         With: " << New << '\n');
    Old.replaceAllUsesWith(&New);
    if (auto *NewI = dyn_cast<Instruction>(&New)) {
      New.takeName(&Old);
      Worklist.pushUsersToWorkList(*NewI);
      Worklist.pushValue(NewI);
    }
    if (Erase && isInstructionTriviallyDead(&Old)) {
      eraseInstruction(Old);
    } else {
      Worklist.push(&Old);
    }
  }

  void eraseInstruction(Instruction &I) {
    LLVM_DEBUG(dbgs() << "VC: Erasing: " << I << '\n');
    SmallVector<Value *> Ops(I.operands());
    Worklist.remove(&I);
    I.eraseFromParent();

    // Push remaining users of the operands and then the operand itself - allows
    // further folds that were hindered by OneUse limits.
    SmallPtrSet<Value *, 4> Visited;
    for (Value *Op : Ops) {
      if (!Visited.contains(Op)) {
        if (auto *OpI = dyn_cast<Instruction>(Op)) {
          if (RecursivelyDeleteTriviallyDeadInstructions(
                  OpI, nullptr, nullptr, [&](Value *V) {
                    if (auto *I = dyn_cast<Instruction>(V)) {
                      LLVM_DEBUG(dbgs() << "VC: Erased: " << *I << '\n');
                      Worklist.remove(I);
                      if (I == NextInst)
                        NextInst = NextInst->getNextNode();
                      Visited.insert(I);
                    }
                  }))
            continue;
          Worklist.pushUsersToWorkList(*OpI);
          Worklist.pushValue(OpI);
        }
      }
    }
  }
};
} // namespace

/// Return the source operand of a potentially bitcasted value. If there is no
/// bitcast, return the input value itself.
static Value *peekThroughBitcasts(Value *V) {
  while (auto *BitCast = dyn_cast<BitCastInst>(V))
    V = BitCast->getOperand(0);
  return V;
}

static bool canWidenLoad(LoadInst *Load, const TargetTransformInfo &TTI) {
  // Do not widen load if atomic/volatile or under asan/hwasan/memtag/tsan.
  // The widened load may load data from dirty regions or create data races
  // non-existent in the source.
  if (!Load || !Load->isSimple() || !Load->hasOneUse() ||
      Load->getFunction()->hasFnAttribute(Attribute::SanitizeMemTag) ||
      mustSuppressSpeculation(*Load))
    return false;

  // We are potentially transforming byte-sized (8-bit) memory accesses, so make
  // sure we have all of our type-based constraints in place for this target.
  Type *ScalarTy = Load->getType()->getScalarType();
  uint64_t ScalarSize = ScalarTy->getPrimitiveSizeInBits();
  unsigned MinVectorSize = TTI.getMinVectorRegisterBitWidth();
  if (!ScalarSize || !MinVectorSize || MinVectorSize % ScalarSize != 0 ||
      ScalarSize % 8 != 0)
    return false;

  return true;
}

bool VectorCombine::vectorizeLoadInsert(Instruction &I) {
  // Match insert into fixed vector of scalar value.
  // TODO: Handle non-zero insert index.
  Value *Scalar;
  if (!match(&I,
             m_InsertElt(m_Poison(), m_OneUse(m_Value(Scalar)), m_ZeroInt())))
    return false;

  // Optionally match an extract from another vector.
  Value *X;
  bool HasExtract = match(Scalar, m_ExtractElt(m_Value(X), m_ZeroInt()));
  if (!HasExtract)
    X = Scalar;

  auto *Load = dyn_cast<LoadInst>(X);
  if (!canWidenLoad(Load, TTI))
    return false;

  Type *ScalarTy = Scalar->getType();
  uint64_t ScalarSize = ScalarTy->getPrimitiveSizeInBits();
  unsigned MinVectorSize = TTI.getMinVectorRegisterBitWidth();

  // Check safety of replacing the scalar load with a larger vector load.
  // We use minimal alignment (maximum flexibility) because we only care about
  // the dereferenceable region. When calculating cost and creating a new op,
  // we may use a larger value based on alignment attributes.
  Value *SrcPtr = Load->getPointerOperand()->stripPointerCasts();
  assert(isa<PointerType>(SrcPtr->getType()) && "Expected a pointer type");

  unsigned MinVecNumElts = MinVectorSize / ScalarSize;
  auto *MinVecTy = VectorType::get(ScalarTy, MinVecNumElts, false);
  unsigned OffsetEltIndex = 0;
  Align Alignment = Load->getAlign();
  if (!isSafeToLoadUnconditionally(SrcPtr, MinVecTy, Align(1), *DL, Load, &AC,
                                   &DT)) {
    // It is not safe to load directly from the pointer, but we can still peek
    // through gep offsets and check if it safe to load from a base address with
    // updated alignment. If it is, we can shuffle the element(s) into place
    // after loading.
    unsigned OffsetBitWidth = DL->getIndexTypeSizeInBits(SrcPtr->getType());
    APInt Offset(OffsetBitWidth, 0);
    SrcPtr = SrcPtr->stripAndAccumulateInBoundsConstantOffsets(*DL, Offset);

    // We want to shuffle the result down from a high element of a vector, so
    // the offset must be positive.
    if (Offset.isNegative())
      return false;

    // The offset must be a multiple of the scalar element to shuffle cleanly
    // in the element's size.
    uint64_t ScalarSizeInBytes = ScalarSize / 8;
    if (Offset.urem(ScalarSizeInBytes) != 0)
      return false;

    // If we load MinVecNumElts, will our target element still be loaded?
    OffsetEltIndex = Offset.udiv(ScalarSizeInBytes).getZExtValue();
    if (OffsetEltIndex >= MinVecNumElts)
      return false;

    if (!isSafeToLoadUnconditionally(SrcPtr, MinVecTy, Align(1), *DL, Load, &AC,
                                     &DT))
      return false;

    // Update alignment with offset value. Note that the offset could be negated
    // to more accurately represent "(new) SrcPtr - Offset = (old) SrcPtr", but
    // negation does not change the result of the alignment calculation.
    Alignment = commonAlignment(Alignment, Offset.getZExtValue());
  }

  // Original pattern: insertelt undef, load [free casts of] PtrOp, 0
  // Use the greater of the alignment on the load or its source pointer.
  Alignment = std::max(SrcPtr->getPointerAlignment(*DL), Alignment);
  Type *LoadTy = Load->getType();
  unsigned AS = Load->getPointerAddressSpace();
  InstructionCost OldCost =
      TTI.getMemoryOpCost(Instruction::Load, LoadTy, Alignment, AS, CostKind);
  APInt DemandedElts = APInt::getOneBitSet(MinVecNumElts, 0);
  OldCost +=
      TTI.getScalarizationOverhead(MinVecTy, DemandedElts,
                                   /* Insert */ true, HasExtract, CostKind);

  // New pattern: load VecPtr
  InstructionCost NewCost =
      TTI.getMemoryOpCost(Instruction::Load, MinVecTy, Alignment, AS, CostKind);
  // Optionally, we are shuffling the loaded vector element(s) into place.
  // For the mask set everything but element 0 to undef to prevent poison from
  // propagating from the extra loaded memory. This will also optionally
  // shrink/grow the vector from the loaded size to the output size.
  // We assume this operation has no cost in codegen if there was no offset.
  // Note that we could use freeze to avoid poison problems, but then we might
  // still need a shuffle to change the vector size.
  auto *Ty = cast<FixedVectorType>(I.getType());
  unsigned OutputNumElts = Ty->getNumElements();
  SmallVector<int, 16> Mask(OutputNumElts, PoisonMaskElem);
  assert(OffsetEltIndex < MinVecNumElts && "Address offset too big");
  Mask[0] = OffsetEltIndex;
  if (OffsetEltIndex)
    NewCost += TTI.getShuffleCost(TTI::SK_PermuteSingleSrc, Ty, MinVecTy, Mask,
                                  CostKind);

  // We can aggressively convert to the vector form because the backend can
  // invert this transform if it does not result in a performance win.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // It is safe and potentially profitable to load a vector directly:
  // inselt undef, load Scalar, 0 --> load VecPtr
  IRBuilder<> Builder(Load);
  Value *CastedPtr =
      Builder.CreatePointerBitCastOrAddrSpaceCast(SrcPtr, Builder.getPtrTy(AS));
  Value *VecLd = Builder.CreateAlignedLoad(MinVecTy, CastedPtr, Alignment);
  VecLd = Builder.CreateShuffleVector(VecLd, Mask);

  replaceValue(I, *VecLd);
  ++NumVecLoad;
  return true;
}

/// If we are loading a vector and then inserting it into a larger vector with
/// undefined elements, try to load the larger vector and eliminate the insert.
/// This removes a shuffle in IR and may allow combining of other loaded values.
bool VectorCombine::widenSubvectorLoad(Instruction &I) {
  // Match subvector insert of fixed vector.
  auto *Shuf = cast<ShuffleVectorInst>(&I);
  if (!Shuf->isIdentityWithPadding())
    return false;

  // Allow a non-canonical shuffle mask that is choosing elements from op1.
  unsigned NumOpElts =
      cast<FixedVectorType>(Shuf->getOperand(0)->getType())->getNumElements();
  unsigned OpIndex = any_of(Shuf->getShuffleMask(), [&NumOpElts](int M) {
    return M >= (int)(NumOpElts);
  });

  auto *Load = dyn_cast<LoadInst>(Shuf->getOperand(OpIndex));
  if (!canWidenLoad(Load, TTI))
    return false;

  // We use minimal alignment (maximum flexibility) because we only care about
  // the dereferenceable region. When calculating cost and creating a new op,
  // we may use a larger value based on alignment attributes.
  auto *Ty = cast<FixedVectorType>(I.getType());
  Value *SrcPtr = Load->getPointerOperand()->stripPointerCasts();
  assert(isa<PointerType>(SrcPtr->getType()) && "Expected a pointer type");
  Align Alignment = Load->getAlign();
  if (!isSafeToLoadUnconditionally(SrcPtr, Ty, Align(1), *DL, Load, &AC, &DT))
    return false;

  Alignment = std::max(SrcPtr->getPointerAlignment(*DL), Alignment);
  Type *LoadTy = Load->getType();
  unsigned AS = Load->getPointerAddressSpace();

  // Original pattern: insert_subvector (load PtrOp)
  // This conservatively assumes that the cost of a subvector insert into an
  // undef value is 0. We could add that cost if the cost model accurately
  // reflects the real cost of that operation.
  InstructionCost OldCost =
      TTI.getMemoryOpCost(Instruction::Load, LoadTy, Alignment, AS, CostKind);

  // New pattern: load PtrOp
  InstructionCost NewCost =
      TTI.getMemoryOpCost(Instruction::Load, Ty, Alignment, AS, CostKind);

  // We can aggressively convert to the vector form because the backend can
  // invert this transform if it does not result in a performance win.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  IRBuilder<> Builder(Load);
  Value *CastedPtr =
      Builder.CreatePointerBitCastOrAddrSpaceCast(SrcPtr, Builder.getPtrTy(AS));
  Value *VecLd = Builder.CreateAlignedLoad(Ty, CastedPtr, Alignment);
  replaceValue(I, *VecLd);
  ++NumVecLoad;
  return true;
}

/// Determine which, if any, of the inputs should be replaced by a shuffle
/// followed by extract from a different index.
ExtractElementInst *VectorCombine::getShuffleExtract(
    ExtractElementInst *Ext0, ExtractElementInst *Ext1,
    unsigned PreferredExtractIndex = InvalidIndex) const {
  auto *Index0C = dyn_cast<ConstantInt>(Ext0->getIndexOperand());
  auto *Index1C = dyn_cast<ConstantInt>(Ext1->getIndexOperand());
  assert(Index0C && Index1C && "Expected constant extract indexes");

  unsigned Index0 = Index0C->getZExtValue();
  unsigned Index1 = Index1C->getZExtValue();

  // If the extract indexes are identical, no shuffle is needed.
  if (Index0 == Index1)
    return nullptr;

  Type *VecTy = Ext0->getVectorOperand()->getType();
  assert(VecTy == Ext1->getVectorOperand()->getType() && "Need matching types");
  InstructionCost Cost0 =
      TTI.getVectorInstrCost(*Ext0, VecTy, CostKind, Index0);
  InstructionCost Cost1 =
      TTI.getVectorInstrCost(*Ext1, VecTy, CostKind, Index1);

  // If both costs are invalid no shuffle is needed
  if (!Cost0.isValid() && !Cost1.isValid())
    return nullptr;

  // We are extracting from 2 different indexes, so one operand must be shuffled
  // before performing a vector operation and/or extract. The more expensive
  // extract will be replaced by a shuffle.
  if (Cost0 > Cost1)
    return Ext0;
  if (Cost1 > Cost0)
    return Ext1;

  // If the costs are equal and there is a preferred extract index, shuffle the
  // opposite operand.
  if (PreferredExtractIndex == Index0)
    return Ext1;
  if (PreferredExtractIndex == Index1)
    return Ext0;

  // Otherwise, replace the extract with the higher index.
  return Index0 > Index1 ? Ext0 : Ext1;
}

/// Compare the relative costs of 2 extracts followed by scalar operation vs.
/// vector operation(s) followed by extract. Return true if the existing
/// instructions are cheaper than a vector alternative. Otherwise, return false
/// and if one of the extracts should be transformed to a shufflevector, set
/// \p ConvertToShuffle to that extract instruction.
bool VectorCombine::isExtractExtractCheap(ExtractElementInst *Ext0,
                                          ExtractElementInst *Ext1,
                                          const Instruction &I,
                                          ExtractElementInst *&ConvertToShuffle,
                                          unsigned PreferredExtractIndex) {
  auto *Ext0IndexC = dyn_cast<ConstantInt>(Ext0->getIndexOperand());
  auto *Ext1IndexC = dyn_cast<ConstantInt>(Ext1->getIndexOperand());
  assert(Ext0IndexC && Ext1IndexC && "Expected constant extract indexes");

  unsigned Opcode = I.getOpcode();
  Value *Ext0Src = Ext0->getVectorOperand();
  Value *Ext1Src = Ext1->getVectorOperand();
  Type *ScalarTy = Ext0->getType();
  auto *VecTy = cast<VectorType>(Ext0Src->getType());
  InstructionCost ScalarOpCost, VectorOpCost;

  // Get cost estimates for scalar and vector versions of the operation.
  bool IsBinOp = Instruction::isBinaryOp(Opcode);
  if (IsBinOp) {
    ScalarOpCost = TTI.getArithmeticInstrCost(Opcode, ScalarTy, CostKind);
    VectorOpCost = TTI.getArithmeticInstrCost(Opcode, VecTy, CostKind);
  } else {
    assert((Opcode == Instruction::ICmp || Opcode == Instruction::FCmp) &&
           "Expected a compare");
    CmpInst::Predicate Pred = cast<CmpInst>(I).getPredicate();
    ScalarOpCost = TTI.getCmpSelInstrCost(
        Opcode, ScalarTy, CmpInst::makeCmpResultType(ScalarTy), Pred, CostKind);
    VectorOpCost = TTI.getCmpSelInstrCost(
        Opcode, VecTy, CmpInst::makeCmpResultType(VecTy), Pred, CostKind);
  }

  // Get cost estimates for the extract elements. These costs will factor into
  // both sequences.
  unsigned Ext0Index = Ext0IndexC->getZExtValue();
  unsigned Ext1Index = Ext1IndexC->getZExtValue();

  InstructionCost Extract0Cost =
      TTI.getVectorInstrCost(*Ext0, VecTy, CostKind, Ext0Index);
  InstructionCost Extract1Cost =
      TTI.getVectorInstrCost(*Ext1, VecTy, CostKind, Ext1Index);

  // A more expensive extract will always be replaced by a splat shuffle.
  // For example, if Ext0 is more expensive:
  // opcode (extelt V0, Ext0), (ext V1, Ext1) -->
  // extelt (opcode (splat V0, Ext0), V1), Ext1
  // TODO: Evaluate whether that always results in lowest cost. Alternatively,
  //       check the cost of creating a broadcast shuffle and shuffling both
  //       operands to element 0.
  unsigned BestExtIndex = Extract0Cost > Extract1Cost ? Ext0Index : Ext1Index;
  unsigned BestInsIndex = Extract0Cost > Extract1Cost ? Ext1Index : Ext0Index;
  InstructionCost CheapExtractCost = std::min(Extract0Cost, Extract1Cost);

  // Extra uses of the extracts mean that we include those costs in the
  // vector total because those instructions will not be eliminated.
  InstructionCost OldCost, NewCost;
  if (Ext0Src == Ext1Src && Ext0Index == Ext1Index) {
    // Handle a special case. If the 2 extracts are identical, adjust the
    // formulas to account for that. The extra use charge allows for either the
    // CSE'd pattern or an unoptimized form with identical values:
    // opcode (extelt V, C), (extelt V, C) --> extelt (opcode V, V), C
    bool HasUseTax = Ext0 == Ext1 ? !Ext0->hasNUses(2)
                                  : !Ext0->hasOneUse() || !Ext1->hasOneUse();
    OldCost = CheapExtractCost + ScalarOpCost;
    NewCost = VectorOpCost + CheapExtractCost + HasUseTax * CheapExtractCost;
  } else {
    // Handle the general case. Each extract is actually a different value:
    // opcode (extelt V0, C0), (extelt V1, C1) --> extelt (opcode V0, V1), C
    OldCost = Extract0Cost + Extract1Cost + ScalarOpCost;
    NewCost = VectorOpCost + CheapExtractCost +
              !Ext0->hasOneUse() * Extract0Cost +
              !Ext1->hasOneUse() * Extract1Cost;
  }

  ConvertToShuffle = getShuffleExtract(Ext0, Ext1, PreferredExtractIndex);
  if (ConvertToShuffle) {
    if (IsBinOp && DisableBinopExtractShuffle)
      return true;

    // If we are extracting from 2 different indexes, then one operand must be
    // shuffled before performing the vector operation. The shuffle mask is
    // poison except for 1 lane that is being translated to the remaining
    // extraction lane. Therefore, it is a splat shuffle. Ex:
    // ShufMask = { poison, poison, 0, poison }
    // TODO: The cost model has an option for a "broadcast" shuffle
    //       (splat-from-element-0), but no option for a more general splat.
    if (auto *FixedVecTy = dyn_cast<FixedVectorType>(VecTy)) {
      SmallVector<int> ShuffleMask(FixedVecTy->getNumElements(),
                                   PoisonMaskElem);
      ShuffleMask[BestInsIndex] = BestExtIndex;
      NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                    VecTy, VecTy, ShuffleMask, CostKind, 0,
                                    nullptr, {ConvertToShuffle});
    } else {
      NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                    VecTy, VecTy, {}, CostKind, 0, nullptr,
                                    {ConvertToShuffle});
    }
  }

  // Aggressively form a vector op if the cost is equal because the transform
  // may enable further optimization.
  // Codegen can reverse this transform (scalarize) if it was not profitable.
  return OldCost < NewCost;
}

/// Create a shuffle that translates (shifts) 1 element from the input vector
/// to a new element location.
static Value *createShiftShuffle(Value *Vec, unsigned OldIndex,
                                 unsigned NewIndex, IRBuilderBase &Builder) {
  // The shuffle mask is poison except for 1 lane that is being translated
  // to the new element index. Example for OldIndex == 2 and NewIndex == 0:
  // ShufMask = { 2, poison, poison, poison }
  auto *VecTy = cast<FixedVectorType>(Vec->getType());
  SmallVector<int, 32> ShufMask(VecTy->getNumElements(), PoisonMaskElem);
  ShufMask[NewIndex] = OldIndex;
  return Builder.CreateShuffleVector(Vec, ShufMask, "shift");
}

/// Given an extract element instruction with constant index operand, shuffle
/// the source vector (shift the scalar element) to a NewIndex for extraction.
/// Return null if the input can be constant folded, so that we are not creating
/// unnecessary instructions.
static Value *translateExtract(ExtractElementInst *ExtElt, unsigned NewIndex,
                               IRBuilderBase &Builder) {
  // Shufflevectors can only be created for fixed-width vectors.
  Value *X = ExtElt->getVectorOperand();
  if (!isa<FixedVectorType>(X->getType()))
    return nullptr;

  // If the extract can be constant-folded, this code is unsimplified. Defer
  // to other passes to handle that.
  Value *C = ExtElt->getIndexOperand();
  assert(isa<ConstantInt>(C) && "Expected a constant index operand");
  if (isa<Constant>(X))
    return nullptr;

  Value *Shuf = createShiftShuffle(X, cast<ConstantInt>(C)->getZExtValue(),
                                   NewIndex, Builder);
  return Shuf;
}

/// Try to reduce extract element costs by converting scalar compares to vector
/// compares followed by extract.
/// cmp (ext0 V0, ExtIndex), (ext1 V1, ExtIndex)
Value *VectorCombine::foldExtExtCmp(Value *V0, Value *V1, Value *ExtIndex,
                                    Instruction &I) {
  assert(isa<CmpInst>(&I) && "Expected a compare");

  // cmp Pred (extelt V0, ExtIndex), (extelt V1, ExtIndex)
  //   --> extelt (cmp Pred V0, V1), ExtIndex
  ++NumVecCmp;
  CmpInst::Predicate Pred = cast<CmpInst>(&I)->getPredicate();
  Value *VecCmp = Builder.CreateCmp(Pred, V0, V1);
  return Builder.CreateExtractElement(VecCmp, ExtIndex, "foldExtExtCmp");
}

/// Try to reduce extract element costs by converting scalar binops to vector
/// binops followed by extract.
/// bo (ext0 V0, ExtIndex), (ext1 V1, ExtIndex)
Value *VectorCombine::foldExtExtBinop(Value *V0, Value *V1, Value *ExtIndex,
                                      Instruction &I) {
  assert(isa<BinaryOperator>(&I) && "Expected a binary operator");

  // bo (extelt V0, ExtIndex), (extelt V1, ExtIndex)
  //   --> extelt (bo V0, V1), ExtIndex
  ++NumVecBO;
  Value *VecBO = Builder.CreateBinOp(cast<BinaryOperator>(&I)->getOpcode(), V0,
                                     V1, "foldExtExtBinop");

  // All IR flags are safe to back-propagate because any potential poison
  // created in unused vector elements is discarded by the extract.
  if (auto *VecBOInst = dyn_cast<Instruction>(VecBO))
    VecBOInst->copyIRFlags(&I);

  return Builder.CreateExtractElement(VecBO, ExtIndex, "foldExtExtBinop");
}

/// Match an instruction with extracted vector operands.
bool VectorCombine::foldExtractExtract(Instruction &I) {
  // It is not safe to transform things like div, urem, etc. because we may
  // create undefined behavior when executing those on unknown vector elements.
  if (!isSafeToSpeculativelyExecute(&I))
    return false;

  Instruction *I0, *I1;
  CmpPredicate Pred = CmpInst::BAD_ICMP_PREDICATE;
  if (!match(&I, m_Cmp(Pred, m_Instruction(I0), m_Instruction(I1))) &&
      !match(&I, m_BinOp(m_Instruction(I0), m_Instruction(I1))))
    return false;

  Value *V0, *V1;
  uint64_t C0, C1;
  if (!match(I0, m_ExtractElt(m_Value(V0), m_ConstantInt(C0))) ||
      !match(I1, m_ExtractElt(m_Value(V1), m_ConstantInt(C1))) ||
      V0->getType() != V1->getType())
    return false;

  // If the scalar value 'I' is going to be re-inserted into a vector, then try
  // to create an extract to that same element. The extract/insert can be
  // reduced to a "select shuffle".
  // TODO: If we add a larger pattern match that starts from an insert, this
  //       probably becomes unnecessary.
  auto *Ext0 = cast<ExtractElementInst>(I0);
  auto *Ext1 = cast<ExtractElementInst>(I1);
  uint64_t InsertIndex = InvalidIndex;
  if (I.hasOneUse())
    match(I.user_back(),
          m_InsertElt(m_Value(), m_Value(), m_ConstantInt(InsertIndex)));

  ExtractElementInst *ExtractToChange;
  if (isExtractExtractCheap(Ext0, Ext1, I, ExtractToChange, InsertIndex))
    return false;

  Value *ExtOp0 = Ext0->getVectorOperand();
  Value *ExtOp1 = Ext1->getVectorOperand();

  if (ExtractToChange) {
    unsigned CheapExtractIdx = ExtractToChange == Ext0 ? C1 : C0;
    Value *NewExtOp =
        translateExtract(ExtractToChange, CheapExtractIdx, Builder);
    if (!NewExtOp)
      return false;
    if (ExtractToChange == Ext0)
      ExtOp0 = NewExtOp;
    else
      ExtOp1 = NewExtOp;
  }

  Value *ExtIndex = ExtractToChange == Ext0 ? Ext1->getIndexOperand()
                                            : Ext0->getIndexOperand();
  Value *NewExt = Pred != CmpInst::BAD_ICMP_PREDICATE
                      ? foldExtExtCmp(ExtOp0, ExtOp1, ExtIndex, I)
                      : foldExtExtBinop(ExtOp0, ExtOp1, ExtIndex, I);
  Worklist.push(Ext0);
  Worklist.push(Ext1);
  replaceValue(I, *NewExt);
  return true;
}

/// Try to replace an extract + scalar fneg + insert with a vector fneg +
/// shuffle.
bool VectorCombine::foldInsExtFNeg(Instruction &I) {
  // Match an insert (op (extract)) pattern.
  Value *DestVec;
  uint64_t Index;
  Instruction *FNeg;
  if (!match(&I, m_InsertElt(m_Value(DestVec), m_OneUse(m_Instruction(FNeg)),
                             m_ConstantInt(Index))))
    return false;

  // Note: This handles the canonical fneg instruction and "fsub -0.0, X".
  Value *SrcVec;
  Instruction *Extract;
  if (!match(FNeg, m_FNeg(m_CombineAnd(
                       m_Instruction(Extract),
                       m_ExtractElt(m_Value(SrcVec), m_SpecificInt(Index))))))
    return false;

  auto *VecTy = cast<FixedVectorType>(I.getType());
  auto *ScalarTy = VecTy->getScalarType();
  auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcVec->getType());
  if (!SrcVecTy || ScalarTy != SrcVecTy->getScalarType())
    return false;

  // Ignore bogus insert/extract index.
  unsigned NumElts = VecTy->getNumElements();
  if (Index >= NumElts)
    return false;

  // We are inserting the negated element into the same lane that we extracted
  // from. This is equivalent to a select-shuffle that chooses all but the
  // negated element from the destination vector.
  SmallVector<int> Mask(NumElts);
  std::iota(Mask.begin(), Mask.end(), 0);
  Mask[Index] = Index + NumElts;
  InstructionCost OldCost =
      TTI.getArithmeticInstrCost(Instruction::FNeg, ScalarTy, CostKind) +
      TTI.getVectorInstrCost(I, VecTy, CostKind, Index);

  // If the extract has one use, it will be eliminated, so count it in the
  // original cost. If it has more than one use, ignore the cost because it will
  // be the same before/after.
  if (Extract->hasOneUse())
    OldCost += TTI.getVectorInstrCost(*Extract, VecTy, CostKind, Index);

  InstructionCost NewCost =
      TTI.getArithmeticInstrCost(Instruction::FNeg, VecTy, CostKind) +
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, VecTy, VecTy,
                         Mask, CostKind);

  bool NeedLenChg = SrcVecTy->getNumElements() != NumElts;
  // If the lengths of the two vectors are not equal,
  // we need to add a length-change vector. Add this cost.
  SmallVector<int> SrcMask;
  if (NeedLenChg) {
    SrcMask.assign(NumElts, PoisonMaskElem);
    SrcMask[Index] = Index;
    NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                  VecTy, SrcVecTy, SrcMask, CostKind);
  }

  if (NewCost > OldCost)
    return false;

  Value *NewShuf;
  // insertelt DestVec, (fneg (extractelt SrcVec, Index)), Index
  Value *VecFNeg = Builder.CreateFNegFMF(SrcVec, FNeg);
  if (NeedLenChg) {
    // shuffle DestVec, (shuffle (fneg SrcVec), poison, SrcMask), Mask
    Value *LenChgShuf = Builder.CreateShuffleVector(VecFNeg, SrcMask);
    NewShuf = Builder.CreateShuffleVector(DestVec, LenChgShuf, Mask);
  } else {
    // shuffle DestVec, (fneg SrcVec), Mask
    NewShuf = Builder.CreateShuffleVector(DestVec, VecFNeg, Mask);
  }

  replaceValue(I, *NewShuf);
  return true;
}

/// Try to fold insert(binop(x,y),binop(a,b),idx)
///         --> binop(insert(x,a,idx),insert(y,b,idx))
bool VectorCombine::foldInsExtBinop(Instruction &I) {
  BinaryOperator *VecBinOp, *SclBinOp;
  uint64_t Index;
  if (!match(&I,
             m_InsertElt(m_OneUse(m_BinOp(VecBinOp)),
                         m_OneUse(m_BinOp(SclBinOp)), m_ConstantInt(Index))))
    return false;

  // TODO: Add support for addlike etc.
  Instruction::BinaryOps BinOpcode = VecBinOp->getOpcode();
  if (BinOpcode != SclBinOp->getOpcode())
    return false;

  auto *ResultTy = dyn_cast<FixedVectorType>(I.getType());
  if (!ResultTy)
    return false;

  // TODO: Attempt to detect m_ExtractElt for scalar operands and convert to
  // shuffle?

  InstructionCost OldCost = TTI.getInstructionCost(&I, CostKind) +
                            TTI.getInstructionCost(VecBinOp, CostKind) +
                            TTI.getInstructionCost(SclBinOp, CostKind);
  InstructionCost NewCost =
      TTI.getArithmeticInstrCost(BinOpcode, ResultTy, CostKind) +
      TTI.getVectorInstrCost(Instruction::InsertElement, ResultTy, CostKind,
                             Index, VecBinOp->getOperand(0),
                             SclBinOp->getOperand(0)) +
      TTI.getVectorInstrCost(Instruction::InsertElement, ResultTy, CostKind,
                             Index, VecBinOp->getOperand(1),
                             SclBinOp->getOperand(1));

  LLVM_DEBUG(dbgs() << "Found an insertion of two binops: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");
  if (NewCost > OldCost)
    return false;

  Value *NewIns0 = Builder.CreateInsertElement(VecBinOp->getOperand(0),
                                               SclBinOp->getOperand(0), Index);
  Value *NewIns1 = Builder.CreateInsertElement(VecBinOp->getOperand(1),
                                               SclBinOp->getOperand(1), Index);
  Value *NewBO = Builder.CreateBinOp(BinOpcode, NewIns0, NewIns1);

  // Intersect flags from the old binops.
  if (auto *NewInst = dyn_cast<Instruction>(NewBO)) {
    NewInst->copyIRFlags(VecBinOp);
    NewInst->andIRFlags(SclBinOp);
  }

  Worklist.pushValue(NewIns0);
  Worklist.pushValue(NewIns1);
  replaceValue(I, *NewBO);
  return true;
}

/// Match: bitop(castop(x), castop(y)) -> castop(bitop(x, y))
/// Supports: bitcast, trunc, sext, zext
bool VectorCombine::foldBitOpOfCastops(Instruction &I) {
  // Check if this is a bitwise logic operation
  auto *BinOp = dyn_cast<BinaryOperator>(&I);
  if (!BinOp || !BinOp->isBitwiseLogicOp())
    return false;

  // Get the cast instructions
  auto *LHSCast = dyn_cast<CastInst>(BinOp->getOperand(0));
  auto *RHSCast = dyn_cast<CastInst>(BinOp->getOperand(1));
  if (!LHSCast || !RHSCast) {
    LLVM_DEBUG(dbgs() << "  One or both operands are not cast instructions\n");
    return false;
  }

  // Both casts must be the same type
  Instruction::CastOps CastOpcode = LHSCast->getOpcode();
  if (CastOpcode != RHSCast->getOpcode())
    return false;

  // Only handle supported cast operations
  switch (CastOpcode) {
  case Instruction::BitCast:
  case Instruction::Trunc:
  case Instruction::SExt:
  case Instruction::ZExt:
    break;
  default:
    return false;
  }

  Value *LHSSrc = LHSCast->getOperand(0);
  Value *RHSSrc = RHSCast->getOperand(0);

  // Source types must match
  if (LHSSrc->getType() != RHSSrc->getType())
    return false;

  // Only handle vector types with integer elements
  auto *SrcVecTy = dyn_cast<FixedVectorType>(LHSSrc->getType());
  auto *DstVecTy = dyn_cast<FixedVectorType>(I.getType());
  if (!SrcVecTy || !DstVecTy)
    return false;

  if (!SrcVecTy->getScalarType()->isIntegerTy() ||
      !DstVecTy->getScalarType()->isIntegerTy())
    return false;

  // Cost Check :
  // OldCost = bitlogic + 2*casts
  // NewCost = bitlogic + cast

  // Calculate specific costs for each cast with instruction context
  InstructionCost LHSCastCost =
      TTI.getCastInstrCost(CastOpcode, DstVecTy, SrcVecTy,
                           TTI::CastContextHint::None, CostKind, LHSCast);
  InstructionCost RHSCastCost =
      TTI.getCastInstrCost(CastOpcode, DstVecTy, SrcVecTy,
                           TTI::CastContextHint::None, CostKind, RHSCast);

  InstructionCost OldCost =
      TTI.getArithmeticInstrCost(BinOp->getOpcode(), DstVecTy, CostKind) +
      LHSCastCost + RHSCastCost;

  // For new cost, we can't provide an instruction (it doesn't exist yet)
  InstructionCost GenericCastCost = TTI.getCastInstrCost(
      CastOpcode, DstVecTy, SrcVecTy, TTI::CastContextHint::None, CostKind);

  InstructionCost NewCost =
      TTI.getArithmeticInstrCost(BinOp->getOpcode(), SrcVecTy, CostKind) +
      GenericCastCost;

  // Account for multi-use casts using specific costs
  if (!LHSCast->hasOneUse())
    NewCost += LHSCastCost;
  if (!RHSCast->hasOneUse())
    NewCost += RHSCastCost;

  LLVM_DEBUG(dbgs() << "foldBitOpOfCastops: OldCost=" << OldCost
                    << " NewCost=" << NewCost << "\n");

  if (NewCost > OldCost)
    return false;

  // Create the operation on the source type
  Value *NewOp = Builder.CreateBinOp(BinOp->getOpcode(), LHSSrc, RHSSrc,
                                     BinOp->getName() + ".inner");
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NewOp))
    NewBinOp->copyIRFlags(BinOp);

  Worklist.pushValue(NewOp);

  // Create the cast operation directly to ensure we get a new instruction
  Instruction *NewCast = CastInst::Create(CastOpcode, NewOp, I.getType());

  // Preserve cast instruction flags
  NewCast->copyIRFlags(LHSCast);
  NewCast->andIRFlags(RHSCast);

  // Insert the new instruction
  Value *Result = Builder.Insert(NewCast);

  replaceValue(I, *Result);
  return true;
}

struct PreservedCastFlags {
  bool NNeg = false;
  bool NUW = false;
  bool NSW = false;
};

// Try to cast C to InvC losslessly, satisfying CastOp(InvC) == C.
// Will try best to preserve the flags.
static Constant *getLosslessInvCast(Constant *C, Type *InvCastTo,
                                    Instruction::CastOps CastOp,
                                    const DataLayout &DL,
                                    PreservedCastFlags &Flags) {
  switch (CastOp) {
  case Instruction::BitCast:
    // Bitcast is always lossless.
    return ConstantFoldCastOperand(Instruction::BitCast, C, InvCastTo, DL);
  case Instruction::Trunc: {
    auto *ZExtC = ConstantFoldCastOperand(Instruction::ZExt, C, InvCastTo, DL);
    auto *SExtC = ConstantFoldCastOperand(Instruction::SExt, C, InvCastTo, DL);
    // Truncation back on ZExt value is always NUW.
    Flags.NUW = true;
    // Test positivity of C.
    Flags.NSW = ZExtC == SExtC;
    return ZExtC;
  }
  case Instruction::SExt:
  case Instruction::ZExt: {
    auto *InvC = ConstantExpr::getTrunc(C, InvCastTo);
    auto *CastInvC = ConstantFoldCastOperand(CastOp, InvC, C->getType(), DL);
    // Must satisfy CastOp(InvC) == C.
    if (!CastInvC || CastInvC != C)
      return nullptr;
    if (CastOp == Instruction::ZExt) {
      auto *SExtInvC =
          ConstantFoldCastOperand(Instruction::SExt, InvC, C->getType(), DL);
      // Test positivity of InvC.
      Flags.NNeg = CastInvC == SExtInvC;
    }
    return InvC;
  }
  default:
    return nullptr;
  }
}

/// Match:
// bitop(castop(x), C) ->
// bitop(castop(x), castop(InvC)) ->
// castop(bitop(x, InvC))
// Supports: bitcast
bool VectorCombine::foldBitOpOfCastConstant(Instruction &I) {
  Instruction *LHS;
  Constant *C;

  // Check if this is a bitwise logic operation
  if (!match(&I, m_c_BitwiseLogic(m_Instruction(LHS), m_Constant(C))))
    return false;

  // Get the cast instructions
  auto *LHSCast = dyn_cast<CastInst>(LHS);
  if (!LHSCast)
    return false;

  Instruction::CastOps CastOpcode = LHSCast->getOpcode();

  // Only handle supported cast operations
  switch (CastOpcode) {
  case Instruction::BitCast:
    break;
  default:
    return false;
  }

  Value *LHSSrc = LHSCast->getOperand(0);

  // Only handle vector types with integer elements
  auto *SrcVecTy = dyn_cast<FixedVectorType>(LHSSrc->getType());
  auto *DstVecTy = dyn_cast<FixedVectorType>(I.getType());
  if (!SrcVecTy || !DstVecTy)
    return false;

  if (!SrcVecTy->getScalarType()->isIntegerTy() ||
      !DstVecTy->getScalarType()->isIntegerTy())
    return false;

  // Find the constant InvC, such that castop(InvC) equals to C.
  PreservedCastFlags RHSFlags;
  Constant *InvC = getLosslessInvCast(C, SrcVecTy, CastOpcode, *DL, RHSFlags);
  if (!InvC)
    return false;

  // Cost Check :
  // OldCost = bitlogic + cast
  // NewCost = bitlogic + cast

  // Calculate specific costs for each cast with instruction context
  InstructionCost LHSCastCost =
      TTI.getCastInstrCost(CastOpcode, DstVecTy, SrcVecTy,
                           TTI::CastContextHint::None, CostKind, LHSCast);

  InstructionCost OldCost =
      TTI.getArithmeticInstrCost(I.getOpcode(), DstVecTy, CostKind) +
      LHSCastCost;

  // For new cost, we can't provide an instruction (it doesn't exist yet)
  InstructionCost GenericCastCost = TTI.getCastInstrCost(
      CastOpcode, DstVecTy, SrcVecTy, TTI::CastContextHint::None, CostKind);

  InstructionCost NewCost =
      TTI.getArithmeticInstrCost(I.getOpcode(), SrcVecTy, CostKind) +
      GenericCastCost;

  // Account for multi-use casts using specific costs
  if (!LHSCast->hasOneUse())
    NewCost += LHSCastCost;

  LLVM_DEBUG(dbgs() << "foldBitOpOfCastConstant: OldCost=" << OldCost
                    << " NewCost=" << NewCost << "\n");

  if (NewCost > OldCost)
    return false;

  // Create the operation on the source type
  Value *NewOp = Builder.CreateBinOp((Instruction::BinaryOps)I.getOpcode(),
                                     LHSSrc, InvC, I.getName() + ".inner");
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NewOp))
    NewBinOp->copyIRFlags(&I);

  Worklist.pushValue(NewOp);

  // Create the cast operation directly to ensure we get a new instruction
  Instruction *NewCast = CastInst::Create(CastOpcode, NewOp, I.getType());

  // Insert the new instruction
  Value *Result = Builder.Insert(NewCast);

  replaceValue(I, *Result);
  return true;
}

/// If this is a bitcast of a shuffle, try to bitcast the source vector to the
/// destination type followed by shuffle. This can enable further transforms by
/// moving bitcasts or shuffles together.
bool VectorCombine::foldBitcastShuffle(Instruction &I) {
  Value *V0, *V1;
  ArrayRef<int> Mask;
  if (!match(&I, m_BitCast(m_OneUse(
                     m_Shuffle(m_Value(V0), m_Value(V1), m_Mask(Mask))))))
    return false;

  // 1) Do not fold bitcast shuffle for scalable type. First, shuffle cost for
  // scalable type is unknown; Second, we cannot reason if the narrowed shuffle
  // mask for scalable type is a splat or not.
  // 2) Disallow non-vector casts.
  // TODO: We could allow any shuffle.
  auto *DestTy = dyn_cast<FixedVectorType>(I.getType());
  auto *SrcTy = dyn_cast<FixedVectorType>(V0->getType());
  if (!DestTy || !SrcTy)
    return false;

  unsigned DestEltSize = DestTy->getScalarSizeInBits();
  unsigned SrcEltSize = SrcTy->getScalarSizeInBits();
  if (SrcTy->getPrimitiveSizeInBits() % DestEltSize != 0)
    return false;

  bool IsUnary = isa<UndefValue>(V1);

  // For binary shuffles, only fold bitcast(shuffle(X,Y))
  // if it won't increase the number of bitcasts.
  if (!IsUnary) {
    auto *BCTy0 = dyn_cast<FixedVectorType>(peekThroughBitcasts(V0)->getType());
    auto *BCTy1 = dyn_cast<FixedVectorType>(peekThroughBitcasts(V1)->getType());
    if (!(BCTy0 && BCTy0->getElementType() == DestTy->getElementType()) &&
        !(BCTy1 && BCTy1->getElementType() == DestTy->getElementType()))
      return false;
  }

  SmallVector<int, 16> NewMask;
  if (DestEltSize <= SrcEltSize) {
    // The bitcast is from wide to narrow/equal elements. The shuffle mask can
    // always be expanded to the equivalent form choosing narrower elements.
    assert(SrcEltSize % DestEltSize == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = SrcEltSize / DestEltSize;
    narrowShuffleMaskElts(ScaleFactor, Mask, NewMask);
  } else {
    // The bitcast is from narrow elements to wide elements. The shuffle mask
    // must choose consecutive elements to allow casting first.
    assert(DestEltSize % SrcEltSize == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = DestEltSize / SrcEltSize;
    if (!widenShuffleMaskElts(ScaleFactor, Mask, NewMask))
      return false;
  }

  // Bitcast the shuffle src - keep its original width but using the destination
  // scalar type.
  unsigned NumSrcElts = SrcTy->getPrimitiveSizeInBits() / DestEltSize;
  auto *NewShuffleTy =
      FixedVectorType::get(DestTy->getScalarType(), NumSrcElts);
  auto *OldShuffleTy =
      FixedVectorType::get(SrcTy->getScalarType(), Mask.size());
  unsigned NumOps = IsUnary ? 1 : 2;

  // The new shuffle must not cost more than the old shuffle.
  TargetTransformInfo::ShuffleKind SK =
      IsUnary ? TargetTransformInfo::SK_PermuteSingleSrc
              : TargetTransformInfo::SK_PermuteTwoSrc;

  InstructionCost NewCost =
      TTI.getShuffleCost(SK, DestTy, NewShuffleTy, NewMask, CostKind) +
      (NumOps * TTI.getCastInstrCost(Instruction::BitCast, NewShuffleTy, SrcTy,
                                     TargetTransformInfo::CastContextHint::None,
                                     CostKind));
  InstructionCost OldCost =
      TTI.getShuffleCost(SK, OldShuffleTy, SrcTy, Mask, CostKind) +
      TTI.getCastInstrCost(Instruction::BitCast, DestTy, OldShuffleTy,
                           TargetTransformInfo::CastContextHint::None,
                           CostKind);

  LLVM_DEBUG(dbgs() << "Found a bitcasted shuffle: " << I << "\n  OldCost: "
                    << OldCost << " vs NewCost: " << NewCost << "\n");

  if (NewCost > OldCost || !NewCost.isValid())
    return false;

  // bitcast (shuf V0, V1, MaskC) --> shuf (bitcast V0), (bitcast V1), MaskC'
  ++NumShufOfBitcast;
  Value *CastV0 = Builder.CreateBitCast(peekThroughBitcasts(V0), NewShuffleTy);
  Value *CastV1 = Builder.CreateBitCast(peekThroughBitcasts(V1), NewShuffleTy);
  Value *Shuf = Builder.CreateShuffleVector(CastV0, CastV1, NewMask);
  replaceValue(I, *Shuf);
  return true;
}

/// VP Intrinsics whose vector operands are both splat values may be simplified
/// into the scalar version of the operation and the result splatted. This
/// can lead to scalarization down the line.
bool VectorCombine::scalarizeVPIntrinsic(Instruction &I) {
  if (!isa<VPIntrinsic>(I))
    return false;
  VPIntrinsic &VPI = cast<VPIntrinsic>(I);
  Value *Op0 = VPI.getArgOperand(0);
  Value *Op1 = VPI.getArgOperand(1);

  if (!isSplatValue(Op0) || !isSplatValue(Op1))
    return false;

  // Check getSplatValue early in this function, to avoid doing unnecessary
  // work.
  Value *ScalarOp0 = getSplatValue(Op0);
  Value *ScalarOp1 = getSplatValue(Op1);
  if (!ScalarOp0 || !ScalarOp1)
    return false;

  // For the binary VP intrinsics supported here, the result on disabled lanes
  // is a poison value. For now, only do this simplification if all lanes
  // are active.
  // TODO: Relax the condition that all lanes are active by using insertelement
  // on inactive lanes.
  auto IsAllTrueMask = [](Value *MaskVal) {
    if (Value *SplattedVal = getSplatValue(MaskVal))
      if (auto *ConstValue = dyn_cast<Constant>(SplattedVal))
        return ConstValue->isAllOnesValue();
    return false;
  };
  if (!IsAllTrueMask(VPI.getArgOperand(2)))
    return false;

  // Check to make sure we support scalarization of the intrinsic
  Intrinsic::ID IntrID = VPI.getIntrinsicID();
  if (!VPBinOpIntrinsic::isVPBinOp(IntrID))
    return false;

  // Calculate cost of splatting both operands into vectors and the vector
  // intrinsic
  VectorType *VecTy = cast<VectorType>(VPI.getType());
  SmallVector<int> Mask;
  if (auto *FVTy = dyn_cast<FixedVectorType>(VecTy))
    Mask.resize(FVTy->getNumElements(), 0);
  InstructionCost SplatCost =
      TTI.getVectorInstrCost(Instruction::InsertElement, VecTy, CostKind, 0) +
      TTI.getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, VecTy, Mask,
                         CostKind);

  // Calculate the cost of the VP Intrinsic
  SmallVector<Type *, 4> Args;
  for (Value *V : VPI.args())
    Args.push_back(V->getType());
  IntrinsicCostAttributes Attrs(IntrID, VecTy, Args);
  InstructionCost VectorOpCost = TTI.getIntrinsicInstrCost(Attrs, CostKind);
  InstructionCost OldCost = 2 * SplatCost + VectorOpCost;

  // Determine scalar opcode
  std::optional<unsigned> FunctionalOpcode =
      VPI.getFunctionalOpcode();
  std::optional<Intrinsic::ID> ScalarIntrID = std::nullopt;
  if (!FunctionalOpcode) {
    ScalarIntrID = VPI.getFunctionalIntrinsicID();
    if (!ScalarIntrID)
      return false;
  }

  // Calculate cost of scalarizing
  InstructionCost ScalarOpCost = 0;
  if (ScalarIntrID) {
    IntrinsicCostAttributes Attrs(*ScalarIntrID, VecTy->getScalarType(), Args);
    ScalarOpCost = TTI.getIntrinsicInstrCost(Attrs, CostKind);
  } else {
    ScalarOpCost = TTI.getArithmeticInstrCost(*FunctionalOpcode,
                                              VecTy->getScalarType(), CostKind);
  }

  // The existing splats may be kept around if other instructions use them.
  InstructionCost CostToKeepSplats =
      (SplatCost * !Op0->hasOneUse()) + (SplatCost * !Op1->hasOneUse());
  InstructionCost NewCost = ScalarOpCost + SplatCost + CostToKeepSplats;

  LLVM_DEBUG(dbgs() << "Found a VP Intrinsic to scalarize: " << VPI
                    << "\n");
  LLVM_DEBUG(dbgs() << "Cost of Intrinsic: " << OldCost
                    << ", Cost of scalarizing:" << NewCost << "\n");

  // We want to scalarize unless the vector variant actually has lower cost.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // Scalarize the intrinsic
  ElementCount EC = cast<VectorType>(Op0->getType())->getElementCount();
  Value *EVL = VPI.getArgOperand(3);

  // If the VP op might introduce UB or poison, we can scalarize it provided
  // that we know the EVL > 0: If the EVL is zero, then the original VP op
  // becomes a no-op and thus won't be UB, so make sure we don't introduce UB by
  // scalarizing it.
  bool SafeToSpeculate;
  if (ScalarIntrID)
    SafeToSpeculate = Intrinsic::getFnAttributes(I.getContext(), *ScalarIntrID)
                          .hasAttribute(Attribute::AttrKind::Speculatable);
  else
    SafeToSpeculate = isSafeToSpeculativelyExecuteWithOpcode(
        *FunctionalOpcode, &VPI, nullptr, &AC, &DT);
  if (!SafeToSpeculate &&
      !isKnownNonZero(EVL, SimplifyQuery(*DL, &DT, &AC, &VPI)))
    return false;

  Value *ScalarVal =
      ScalarIntrID
          ? Builder.CreateIntrinsic(VecTy->getScalarType(), *ScalarIntrID,
                                    {ScalarOp0, ScalarOp1})
          : Builder.CreateBinOp((Instruction::BinaryOps)(*FunctionalOpcode),
                                ScalarOp0, ScalarOp1);

  replaceValue(VPI, *Builder.CreateVectorSplat(EC, ScalarVal));
  return true;
}

/// Match a vector op/compare/intrinsic with at least one
/// inserted scalar operand and convert to scalar op/cmp/intrinsic followed
/// by insertelement.
bool VectorCombine::scalarizeOpOrCmp(Instruction &I) {
  auto *UO = dyn_cast<UnaryOperator>(&I);
  auto *BO = dyn_cast<BinaryOperator>(&I);
  auto *CI = dyn_cast<CmpInst>(&I);
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!UO && !BO && !CI && !II)
    return false;

  // TODO: Allow intrinsics with different argument types
  if (II) {
    if (!isTriviallyVectorizable(II->getIntrinsicID()))
      return false;
    for (auto [Idx, Arg] : enumerate(II->args()))
      if (Arg->getType() != II->getType() &&
          !isVectorIntrinsicWithScalarOpAtArg(II->getIntrinsicID(), Idx, &TTI))
        return false;
  }

  // Do not convert the vector condition of a vector select into a scalar
  // condition. That may cause problems for codegen because of differences in
  // boolean formats and register-file transfers.
  // TODO: Can we account for that in the cost model?
  if (CI)
    for (User *U : I.users())
      if (match(U, m_Select(m_Specific(&I), m_Value(), m_Value())))
        return false;

  // Match constant vectors or scalars being inserted into constant vectors:
  // vec_op [VecC0 | (inselt VecC0, V0, Index)], ...
  SmallVector<Value *> VecCs, ScalarOps;
  std::optional<uint64_t> Index;

  auto Ops = II ? II->args() : I.operands();
  for (auto [OpNum, Op] : enumerate(Ops)) {
    Constant *VecC;
    Value *V;
    uint64_t InsIdx = 0;
    if (match(Op.get(), m_InsertElt(m_Constant(VecC), m_Value(V),
                                    m_ConstantInt(InsIdx)))) {
      // Bail if any inserts are out of bounds.
      VectorType *OpTy = cast<VectorType>(Op->getType());
      if (OpTy->getElementCount().getKnownMinValue() <= InsIdx)
        return false;
      // All inserts must have the same index.
      // TODO: Deal with mismatched index constants and variable indexes?
      if (!Index)
        Index = InsIdx;
      else if (InsIdx != *Index)
        return false;
      VecCs.push_back(VecC);
      ScalarOps.push_back(V);
    } else if (II && isVectorIntrinsicWithScalarOpAtArg(II->getIntrinsicID(),
                                                        OpNum, &TTI)) {
      VecCs.push_back(Op.get());
      ScalarOps.push_back(Op.get());
    } else if (match(Op.get(), m_Constant(VecC))) {
      VecCs.push_back(VecC);
      ScalarOps.push_back(nullptr);
    } else {
      return false;
    }
  }

  // Bail if all operands are constant.
  if (!Index.has_value())
    return false;

  VectorType *VecTy = cast<VectorType>(I.getType());
  Type *ScalarTy = VecTy->getScalarType();
  assert(VecTy->isVectorTy() &&
         (ScalarTy->isIntegerTy() || ScalarTy->isFloatingPointTy() ||
          ScalarTy->isPointerTy()) &&
         "Unexpected types for insert element into binop or cmp");

  unsigned Opcode = I.getOpcode();
  InstructionCost ScalarOpCost, VectorOpCost;
  if (CI) {
    CmpInst::Predicate Pred = CI->getPredicate();
    ScalarOpCost = TTI.getCmpSelInstrCost(
        Opcode, ScalarTy, CmpInst::makeCmpResultType(ScalarTy), Pred, CostKind);
    VectorOpCost = TTI.getCmpSelInstrCost(
        Opcode, VecTy, CmpInst::makeCmpResultType(VecTy), Pred, CostKind);
  } else if (UO || BO) {
    ScalarOpCost = TTI.getArithmeticInstrCost(Opcode, ScalarTy, CostKind);
    VectorOpCost = TTI.getArithmeticInstrCost(Opcode, VecTy, CostKind);
  } else {
    IntrinsicCostAttributes ScalarICA(
        II->getIntrinsicID(), ScalarTy,
        SmallVector<Type *>(II->arg_size(), ScalarTy));
    ScalarOpCost = TTI.getIntrinsicInstrCost(ScalarICA, CostKind);
    IntrinsicCostAttributes VectorICA(
        II->getIntrinsicID(), VecTy,
        SmallVector<Type *>(II->arg_size(), VecTy));
    VectorOpCost = TTI.getIntrinsicInstrCost(VectorICA, CostKind);
  }

  // Fold the vector constants in the original vectors into a new base vector to
  // get more accurate cost modelling.
  Value *NewVecC = nullptr;
  if (CI)
    NewVecC = simplifyCmpInst(CI->getPredicate(), VecCs[0], VecCs[1], SQ);
  else if (UO)
    NewVecC =
        simplifyUnOp(UO->getOpcode(), VecCs[0], UO->getFastMathFlags(), SQ);
  else if (BO)
    NewVecC = simplifyBinOp(BO->getOpcode(), VecCs[0], VecCs[1], SQ);
  else if (II)
    NewVecC = simplifyCall(II, II->getCalledOperand(), VecCs, SQ);

  if (!NewVecC)
    return false;

  // Get cost estimate for the insert element. This cost will factor into
  // both sequences.
  InstructionCost OldCost = VectorOpCost;
  InstructionCost NewCost =
      ScalarOpCost + TTI.getVectorInstrCost(Instruction::InsertElement, VecTy,
                                            CostKind, *Index, NewVecC);

  for (auto [Idx, Op, VecC, Scalar] : enumerate(Ops, VecCs, ScalarOps)) {
    if (!Scalar || (II && isVectorIntrinsicWithScalarOpAtArg(
                              II->getIntrinsicID(), Idx, &TTI)))
      continue;
    InstructionCost InsertCost = TTI.getVectorInstrCost(
        Instruction::InsertElement, VecTy, CostKind, *Index, VecC, Scalar);
    OldCost += InsertCost;
    NewCost += !Op->hasOneUse() * InsertCost;
  }

  // We want to scalarize unless the vector variant actually has lower cost.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // vec_op (inselt VecC0, V0, Index), (inselt VecC1, V1, Index) -->
  // inselt NewVecC, (scalar_op V0, V1), Index
  if (CI)
    ++NumScalarCmp;
  else if (UO || BO)
    ++NumScalarOps;
  else
    ++NumScalarIntrinsic;

  // For constant cases, extract the scalar element, this should constant fold.
  for (auto [OpIdx, Scalar, VecC] : enumerate(ScalarOps, VecCs))
    if (!Scalar)
      ScalarOps[OpIdx] = ConstantExpr::getExtractElement(
          cast<Constant>(VecC), Builder.getInt64(*Index));

  Value *Scalar;
  if (CI)
    Scalar = Builder.CreateCmp(CI->getPredicate(), ScalarOps[0], ScalarOps[1]);
  else if (UO || BO)
    Scalar = Builder.CreateNAryOp(Opcode, ScalarOps);
  else
    Scalar = Builder.CreateIntrinsic(ScalarTy, II->getIntrinsicID(), ScalarOps);

  Scalar->setName(I.getName() + ".scalar");

  // All IR flags are safe to back-propagate. There is no potential for extra
  // poison to be created by the scalar instruction.
  if (auto *ScalarInst = dyn_cast<Instruction>(Scalar))
    ScalarInst->copyIRFlags(&I);

  Value *Insert = Builder.CreateInsertElement(NewVecC, Scalar, *Index);
  replaceValue(I, *Insert);
  return true;
}

/// Try to combine a scalar binop + 2 scalar compares of extracted elements of
/// a vector into vector operations followed by extract. Note: The SLP pass
/// may miss this pattern because of implementation problems.
bool VectorCombine::foldExtractedCmps(Instruction &I) {
  auto *BI = dyn_cast<BinaryOperator>(&I);

  // We are looking for a scalar binop of booleans.
  // binop i1 (cmp Pred I0, C0), (cmp Pred I1, C1)
  if (!BI || !I.getType()->isIntegerTy(1))
    return false;

  // The compare predicates should match, and each compare should have a
  // constant operand.
  Value *B0 = I.getOperand(0), *B1 = I.getOperand(1);
  Instruction *I0, *I1;
  Constant *C0, *C1;
  CmpPredicate P0, P1;
  if (!match(B0, m_Cmp(P0, m_Instruction(I0), m_Constant(C0))) ||
      !match(B1, m_Cmp(P1, m_Instruction(I1), m_Constant(C1))))
    return false;

  auto MatchingPred = CmpPredicate::getMatching(P0, P1);
  if (!MatchingPred)
    return false;

  // The compare operands must be extracts of the same vector with constant
  // extract indexes.
  Value *X;
  uint64_t Index0, Index1;
  if (!match(I0, m_ExtractElt(m_Value(X), m_ConstantInt(Index0))) ||
      !match(I1, m_ExtractElt(m_Specific(X), m_ConstantInt(Index1))))
    return false;

  auto *Ext0 = cast<ExtractElementInst>(I0);
  auto *Ext1 = cast<ExtractElementInst>(I1);
  ExtractElementInst *ConvertToShuf = getShuffleExtract(Ext0, Ext1, CostKind);
  if (!ConvertToShuf)
    return false;
  assert((ConvertToShuf == Ext0 || ConvertToShuf == Ext1) &&
         "Unknown ExtractElementInst");

  // The original scalar pattern is:
  // binop i1 (cmp Pred (ext X, Index0), C0), (cmp Pred (ext X, Index1), C1)
  CmpInst::Predicate Pred = *MatchingPred;
  unsigned CmpOpcode =
      CmpInst::isFPPredicate(Pred) ? Instruction::FCmp : Instruction::ICmp;
  auto *VecTy = dyn_cast<FixedVectorType>(X->getType());
  if (!VecTy)
    return false;

  InstructionCost Ext0Cost =
      TTI.getVectorInstrCost(*Ext0, VecTy, CostKind, Index0);
  InstructionCost Ext1Cost =
      TTI.getVectorInstrCost(*Ext1, VecTy, CostKind, Index1);
  InstructionCost CmpCost = TTI.getCmpSelInstrCost(
      CmpOpcode, I0->getType(), CmpInst::makeCmpResultType(I0->getType()), Pred,
      CostKind);

  InstructionCost OldCost =
      Ext0Cost + Ext1Cost + CmpCost * 2 +
      TTI.getArithmeticInstrCost(I.getOpcode(), I.getType(), CostKind);

  // The proposed vector pattern is:
  // vcmp = cmp Pred X, VecC
  // ext (binop vNi1 vcmp, (shuffle vcmp, Index1)), Index0
  int CheapIndex = ConvertToShuf == Ext0 ? Index1 : Index0;
  int ExpensiveIndex = ConvertToShuf == Ext0 ? Index0 : Index1;
  auto *CmpTy = cast<FixedVectorType>(CmpInst::makeCmpResultType(VecTy));
  InstructionCost NewCost = TTI.getCmpSelInstrCost(
      CmpOpcode, VecTy, CmpInst::makeCmpResultType(VecTy), Pred, CostKind);
  SmallVector<int, 32> ShufMask(VecTy->getNumElements(), PoisonMaskElem);
  ShufMask[CheapIndex] = ExpensiveIndex;
  NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, CmpTy,
                                CmpTy, ShufMask, CostKind);
  NewCost += TTI.getArithmeticInstrCost(I.getOpcode(), CmpTy, CostKind);
  NewCost += TTI.getVectorInstrCost(*Ext0, CmpTy, CostKind, CheapIndex);
  NewCost += Ext0->hasOneUse() ? 0 : Ext0Cost;
  NewCost += Ext1->hasOneUse() ? 0 : Ext1Cost;

  // Aggressively form vector ops if the cost is equal because the transform
  // may enable further optimization.
  // Codegen can reverse this transform (scalarize) if it was not profitable.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // Create a vector constant from the 2 scalar constants.
  SmallVector<Constant *, 32> CmpC(VecTy->getNumElements(),
                                   PoisonValue::get(VecTy->getElementType()));
  CmpC[Index0] = C0;
  CmpC[Index1] = C1;
  Value *VCmp = Builder.CreateCmp(Pred, X, ConstantVector::get(CmpC));
  Value *Shuf = createShiftShuffle(VCmp, ExpensiveIndex, CheapIndex, Builder);
  Value *LHS = ConvertToShuf == Ext0 ? Shuf : VCmp;
  Value *RHS = ConvertToShuf == Ext0 ? VCmp : Shuf;
  Value *VecLogic = Builder.CreateBinOp(BI->getOpcode(), LHS, RHS);
  Value *NewExt = Builder.CreateExtractElement(VecLogic, CheapIndex);
  replaceValue(I, *NewExt);
  ++NumVecCmpBO;
  return true;
}

static void analyzeCostOfVecReduction(const IntrinsicInst &II,
                                      TTI::TargetCostKind CostKind,
                                      const TargetTransformInfo &TTI,
                                      InstructionCost &CostBeforeReduction,
                                      InstructionCost &CostAfterReduction) {
  Instruction *Op0, *Op1;
  auto *RedOp = dyn_cast<Instruction>(II.getOperand(0));
  auto *VecRedTy = cast<VectorType>(II.getOperand(0)->getType());
  unsigned ReductionOpc =
      getArithmeticReductionInstruction(II.getIntrinsicID());
  if (RedOp && match(RedOp, m_ZExtOrSExt(m_Value()))) {
    bool IsUnsigned = isa<ZExtInst>(RedOp);
    auto *ExtType = cast<VectorType>(RedOp->getOperand(0)->getType());

    CostBeforeReduction =
        TTI.getCastInstrCost(RedOp->getOpcode(), VecRedTy, ExtType,
                             TTI::CastContextHint::None, CostKind, RedOp);
    CostAfterReduction =
        TTI.getExtendedReductionCost(ReductionOpc, IsUnsigned, II.getType(),
                                     ExtType, FastMathFlags(), CostKind);
    return;
  }
  if (RedOp && II.getIntrinsicID() == Intrinsic::vector_reduce_add &&
      match(RedOp,
            m_ZExtOrSExt(m_Mul(m_Instruction(Op0), m_Instruction(Op1)))) &&
      match(Op0, m_ZExtOrSExt(m_Value())) &&
      Op0->getOpcode() == Op1->getOpcode() &&
      Op0->getOperand(0)->getType() == Op1->getOperand(0)->getType() &&
      (Op0->getOpcode() == RedOp->getOpcode() || Op0 == Op1)) {
    // Matched reduce.add(ext(mul(ext(A), ext(B)))
    bool IsUnsigned = isa<ZExtInst>(Op0);
    auto *ExtType = cast<VectorType>(Op0->getOperand(0)->getType());
    VectorType *MulType = VectorType::get(Op0->getType(), VecRedTy);

    InstructionCost ExtCost =
        TTI.getCastInstrCost(Op0->getOpcode(), MulType, ExtType,
                             TTI::CastContextHint::None, CostKind, Op0);
    InstructionCost MulCost =
        TTI.getArithmeticInstrCost(Instruction::Mul, MulType, CostKind);
    InstructionCost Ext2Cost =
        TTI.getCastInstrCost(RedOp->getOpcode(), VecRedTy, MulType,
                             TTI::CastContextHint::None, CostKind, RedOp);

    CostBeforeReduction = ExtCost * 2 + MulCost + Ext2Cost;
    CostAfterReduction = TTI.getMulAccReductionCost(
        IsUnsigned, ReductionOpc, II.getType(), ExtType, CostKind);
    return;
  }
  CostAfterReduction = TTI.getArithmeticReductionCost(ReductionOpc, VecRedTy,
                                                      std::nullopt, CostKind);
}

bool VectorCombine::foldBinopOfReductions(Instruction &I) {
  Instruction::BinaryOps BinOpOpc = cast<BinaryOperator>(&I)->getOpcode();
  Intrinsic::ID ReductionIID = getReductionForBinop(BinOpOpc);
  if (BinOpOpc == Instruction::Sub)
    ReductionIID = Intrinsic::vector_reduce_add;
  if (ReductionIID == Intrinsic::not_intrinsic)
    return false;

  auto checkIntrinsicAndGetItsArgument = [](Value *V,
                                            Intrinsic::ID IID) -> Value * {
    auto *II = dyn_cast<IntrinsicInst>(V);
    if (!II)
      return nullptr;
    if (II->getIntrinsicID() == IID && II->hasOneUse())
      return II->getArgOperand(0);
    return nullptr;
  };

  Value *V0 = checkIntrinsicAndGetItsArgument(I.getOperand(0), ReductionIID);
  if (!V0)
    return false;
  Value *V1 = checkIntrinsicAndGetItsArgument(I.getOperand(1), ReductionIID);
  if (!V1)
    return false;

  auto *VTy = cast<VectorType>(V0->getType());
  if (V1->getType() != VTy)
    return false;
  const auto &II0 = *cast<IntrinsicInst>(I.getOperand(0));
  const auto &II1 = *cast<IntrinsicInst>(I.getOperand(1));
  unsigned ReductionOpc =
      getArithmeticReductionInstruction(II0.getIntrinsicID());

  InstructionCost OldCost = 0;
  InstructionCost NewCost = 0;
  InstructionCost CostOfRedOperand0 = 0;
  InstructionCost CostOfRed0 = 0;
  InstructionCost CostOfRedOperand1 = 0;
  InstructionCost CostOfRed1 = 0;
  analyzeCostOfVecReduction(II0, CostKind, TTI, CostOfRedOperand0, CostOfRed0);
  analyzeCostOfVecReduction(II1, CostKind, TTI, CostOfRedOperand1, CostOfRed1);
  OldCost = CostOfRed0 + CostOfRed1 + TTI.getInstructionCost(&I, CostKind);
  NewCost =
      CostOfRedOperand0 + CostOfRedOperand1 +
      TTI.getArithmeticInstrCost(BinOpOpc, VTy, CostKind) +
      TTI.getArithmeticReductionCost(ReductionOpc, VTy, std::nullopt, CostKind);
  if (NewCost >= OldCost || !NewCost.isValid())
    return false;

  LLVM_DEBUG(dbgs() << "Found two mergeable reductions: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");
  Value *VectorBO;
  if (BinOpOpc == Instruction::Or)
    VectorBO = Builder.CreateOr(V0, V1, "",
                                cast<PossiblyDisjointInst>(I).isDisjoint());
  else
    VectorBO = Builder.CreateBinOp(BinOpOpc, V0, V1);

  Instruction *Rdx = Builder.CreateIntrinsic(ReductionIID, {VTy}, {VectorBO});
  replaceValue(I, *Rdx);
  return true;
}

// Check if memory loc modified between two instrs in the same BB
static bool isMemModifiedBetween(BasicBlock::iterator Begin,
                                 BasicBlock::iterator End,
                                 const MemoryLocation &Loc, AAResults &AA) {
  unsigned NumScanned = 0;
  return std::any_of(Begin, End, [&](const Instruction &Instr) {
    return isModSet(AA.getModRefInfo(&Instr, Loc)) ||
           ++NumScanned > MaxInstrsToScan;
  });
}

namespace {
/// Helper class to indicate whether a vector index can be safely scalarized and
/// if a freeze needs to be inserted.
class ScalarizationResult {
  enum class StatusTy { Unsafe, Safe, SafeWithFreeze };

  StatusTy Status;
  Value *ToFreeze;

  ScalarizationResult(StatusTy Status, Value *ToFreeze = nullptr)
      : Status(Status), ToFreeze(ToFreeze) {}

public:
  ScalarizationResult(const ScalarizationResult &Other) = default;
  ~ScalarizationResult() {
    assert(!ToFreeze && "freeze() not called with ToFreeze being set");
  }

  static ScalarizationResult unsafe() { return {StatusTy::Unsafe}; }
  static ScalarizationResult safe() { return {StatusTy::Safe}; }
  static ScalarizationResult safeWithFreeze(Value *ToFreeze) {
    return {StatusTy::SafeWithFreeze, ToFreeze};
  }

  /// Returns true if the index can be scalarize without requiring a freeze.
  bool isSafe() const { return Status == StatusTy::Safe; }
  /// Returns true if the index cannot be scalarized.
  bool isUnsafe() const { return Status == StatusTy::Unsafe; }
  /// Returns true if the index can be scalarize, but requires inserting a
  /// freeze.
  bool isSafeWithFreeze() const { return Status == StatusTy::SafeWithFreeze; }

  /// Reset the state of Unsafe and clear ToFreze if set.
  void discard() {
    ToFreeze = nullptr;
    Status = StatusTy::Unsafe;
  }

  /// Freeze the ToFreeze and update the use in \p User to use it.
  void freeze(IRBuilderBase &Builder, Instruction &UserI) {
    assert(isSafeWithFreeze() &&
           "should only be used when freezing is required");
    assert(is_contained(ToFreeze->users(), &UserI) &&
           "UserI must be a user of ToFreeze");
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.SetInsertPoint(cast<Instruction>(&UserI));
    Value *Frozen =
        Builder.CreateFreeze(ToFreeze, ToFreeze->getName() + ".frozen");
    for (Use &U : make_early_inc_range((UserI.operands())))
      if (U.get() == ToFreeze)
        U.set(Frozen);

    ToFreeze = nullptr;
  }
};
} // namespace

/// Check if it is legal to scalarize a memory access to \p VecTy at index \p
/// Idx. \p Idx must access a valid vector element.
static ScalarizationResult canScalarizeAccess(VectorType *VecTy, Value *Idx,
                                              Instruction *CtxI,
                                              AssumptionCache &AC,
                                              const DominatorTree &DT) {
  // We do checks for both fixed vector types and scalable vector types.
  // This is the number of elements of fixed vector types,
  // or the minimum number of elements of scalable vector types.
  uint64_t NumElements = VecTy->getElementCount().getKnownMinValue();
  unsigned IntWidth = Idx->getType()->getScalarSizeInBits();

  if (auto *C = dyn_cast<ConstantInt>(Idx)) {
    if (C->getValue().ult(NumElements))
      return ScalarizationResult::safe();
    return ScalarizationResult::unsafe();
  }

  // Always unsafe if the index type can't handle all inbound values.
  if (!llvm::isUIntN(IntWidth, NumElements))
    return ScalarizationResult::unsafe();

  APInt Zero(IntWidth, 0);
  APInt MaxElts(IntWidth, NumElements);
  ConstantRange ValidIndices(Zero, MaxElts);
  ConstantRange IdxRange(IntWidth, true);

  if (isGuaranteedNotToBePoison(Idx, &AC)) {
    if (ValidIndices.contains(computeConstantRange(Idx, /* ForSigned */ false,
                                                   true, &AC, CtxI, &DT)))
      return ScalarizationResult::safe();
    return ScalarizationResult::unsafe();
  }

  // If the index may be poison, check if we can insert a freeze before the
  // range of the index is restricted.
  Value *IdxBase;
  ConstantInt *CI;
  if (match(Idx, m_And(m_Value(IdxBase), m_ConstantInt(CI)))) {
    IdxRange = IdxRange.binaryAnd(CI->getValue());
  } else if (match(Idx, m_URem(m_Value(IdxBase), m_ConstantInt(CI)))) {
    IdxRange = IdxRange.urem(CI->getValue());
  }

  if (ValidIndices.contains(IdxRange))
    return ScalarizationResult::safeWithFreeze(IdxBase);
  return ScalarizationResult::unsafe();
}

/// The memory operation on a vector of \p ScalarType had alignment of
/// \p VectorAlignment. Compute the maximal, but conservatively correct,
/// alignment that will be valid for the memory operation on a single scalar
/// element of the same type with index \p Idx.
static Align computeAlignmentAfterScalarization(Align VectorAlignment,
                                                Type *ScalarType, Value *Idx,
                                                const DataLayout &DL) {
  if (auto *C = dyn_cast<ConstantInt>(Idx))
    return commonAlignment(VectorAlignment,
                           C->getZExtValue() * DL.getTypeStoreSize(ScalarType));
  return commonAlignment(VectorAlignment, DL.getTypeStoreSize(ScalarType));
}

// Combine patterns like:
//   %0 = load <4 x i32>, <4 x i32>* %a
//   %1 = insertelement <4 x i32> %0, i32 %b, i32 1
//   store <4 x i32> %1, <4 x i32>* %a
// to:
//   %0 = bitcast <4 x i32>* %a to i32*
//   %1 = getelementptr inbounds i32, i32* %0, i64 0, i64 1
//   store i32 %b, i32* %1
bool VectorCombine::foldSingleElementStore(Instruction &I) {
  if (!TTI.allowVectorElementIndexingUsingGEP())
    return false;
  auto *SI = cast<StoreInst>(&I);
  if (!SI->isSimple() || !isa<VectorType>(SI->getValueOperand()->getType()))
    return false;

  // TODO: Combine more complicated patterns (multiple insert) by referencing
  // TargetTransformInfo.
  Instruction *Source;
  Value *NewElement;
  Value *Idx;
  if (!match(SI->getValueOperand(),
             m_InsertElt(m_Instruction(Source), m_Value(NewElement),
                         m_Value(Idx))))
    return false;

  if (auto *Load = dyn_cast<LoadInst>(Source)) {
    auto VecTy = cast<VectorType>(SI->getValueOperand()->getType());
    Value *SrcAddr = Load->getPointerOperand()->stripPointerCasts();
    // Don't optimize for atomic/volatile load or store. Ensure memory is not
    // modified between, vector type matches store size, and index is inbounds.
    if (!Load->isSimple() || Load->getParent() != SI->getParent() ||
        !DL->typeSizeEqualsStoreSize(Load->getType()->getScalarType()) ||
        SrcAddr != SI->getPointerOperand()->stripPointerCasts())
      return false;

    auto ScalarizableIdx = canScalarizeAccess(VecTy, Idx, Load, AC, DT);
    if (ScalarizableIdx.isUnsafe() ||
        isMemModifiedBetween(Load->getIterator(), SI->getIterator(),
                             MemoryLocation::get(SI), AA))
      return false;

    // Ensure we add the load back to the worklist BEFORE its users so they can
    // erased in the correct order.
    Worklist.push(Load);

    if (ScalarizableIdx.isSafeWithFreeze())
      ScalarizableIdx.freeze(Builder, *cast<Instruction>(Idx));
    Value *GEP = Builder.CreateInBoundsGEP(
        SI->getValueOperand()->getType(), SI->getPointerOperand(),
        {ConstantInt::get(Idx->getType(), 0), Idx});
    StoreInst *NSI = Builder.CreateStore(NewElement, GEP);
    NSI->copyMetadata(*SI);
    Align ScalarOpAlignment = computeAlignmentAfterScalarization(
        std::max(SI->getAlign(), Load->getAlign()), NewElement->getType(), Idx,
        *DL);
    NSI->setAlignment(ScalarOpAlignment);
    replaceValue(I, *NSI);
    eraseInstruction(I);
    return true;
  }

  return false;
}

/// Try to scalarize vector loads feeding extractelement instructions.
bool VectorCombine::scalarizeLoadExtract(Instruction &I) {
  if (!TTI.allowVectorElementIndexingUsingGEP())
    return false;

  Value *Ptr;
  if (!match(&I, m_Load(m_Value(Ptr))))
    return false;

  auto *LI = cast<LoadInst>(&I);
  auto *VecTy = cast<VectorType>(LI->getType());
  if (LI->isVolatile() || !DL->typeSizeEqualsStoreSize(VecTy->getScalarType()))
    return false;

  InstructionCost OriginalCost =
      TTI.getMemoryOpCost(Instruction::Load, VecTy, LI->getAlign(),
                          LI->getPointerAddressSpace(), CostKind);
  InstructionCost ScalarizedCost = 0;

  Instruction *LastCheckedInst = LI;
  unsigned NumInstChecked = 0;
  DenseMap<ExtractElementInst *, ScalarizationResult> NeedFreeze;
  auto FailureGuard = make_scope_exit([&]() {
    // If the transform is aborted, discard the ScalarizationResults.
    for (auto &Pair : NeedFreeze)
      Pair.second.discard();
  });

  // Check if all users of the load are extracts with no memory modifications
  // between the load and the extract. Compute the cost of both the original
  // code and the scalarized version.
  for (User *U : LI->users()) {
    auto *UI = dyn_cast<ExtractElementInst>(U);
    if (!UI || UI->getParent() != LI->getParent())
      return false;

    // If any extract is waiting to be erased, then bail out as this will
    // distort the cost calculation and possibly lead to infinite loops.
    if (UI->use_empty())
      return false;

    // Check if any instruction between the load and the extract may modify
    // memory.
    if (LastCheckedInst->comesBefore(UI)) {
      for (Instruction &I :
           make_range(std::next(LI->getIterator()), UI->getIterator())) {
        // Bail out if we reached the check limit or the instruction may write
        // to memory.
        if (NumInstChecked == MaxInstrsToScan || I.mayWriteToMemory())
          return false;
        NumInstChecked++;
      }
      LastCheckedInst = UI;
    }

    auto ScalarIdx =
        canScalarizeAccess(VecTy, UI->getIndexOperand(), LI, AC, DT);
    if (ScalarIdx.isUnsafe())
      return false;
    if (ScalarIdx.isSafeWithFreeze()) {
      NeedFreeze.try_emplace(UI, ScalarIdx);
      ScalarIdx.discard();
    }

    auto *Index = dyn_cast<ConstantInt>(UI->getIndexOperand());
    OriginalCost +=
        TTI.getVectorInstrCost(Instruction::ExtractElement, VecTy, CostKind,
                               Index ? Index->getZExtValue() : -1);
    ScalarizedCost +=
        TTI.getMemoryOpCost(Instruction::Load, VecTy->getElementType(),
                            Align(1), LI->getPointerAddressSpace(), CostKind);
    ScalarizedCost += TTI.getAddressComputationCost(LI->getPointerOperandType(),
                                                    nullptr, nullptr, CostKind);
  }

  LLVM_DEBUG(dbgs() << "Found all extractions of a vector load: " << I
                    << "\n  LoadExtractCost: " << OriginalCost
                    << " vs ScalarizedCost: " << ScalarizedCost << "\n");

  if (ScalarizedCost >= OriginalCost)
    return false;

  // Ensure we add the load back to the worklist BEFORE its users so they can
  // erased in the correct order.
  Worklist.push(LI);

  Type *ElemType = VecTy->getElementType();

  // Replace extracts with narrow scalar loads.
  for (User *U : LI->users()) {
    auto *EI = cast<ExtractElementInst>(U);
    Value *Idx = EI->getIndexOperand();

    // Insert 'freeze' for poison indexes.
    auto It = NeedFreeze.find(EI);
    if (It != NeedFreeze.end())
      It->second.freeze(Builder, *cast<Instruction>(Idx));

    Builder.SetInsertPoint(EI);
    Value *GEP =
        Builder.CreateInBoundsGEP(VecTy, Ptr, {Builder.getInt32(0), Idx});
    auto *NewLoad = cast<LoadInst>(
        Builder.CreateLoad(ElemType, GEP, EI->getName() + ".scalar"));

    Align ScalarOpAlignment =
        computeAlignmentAfterScalarization(LI->getAlign(), ElemType, Idx, *DL);
    NewLoad->setAlignment(ScalarOpAlignment);

    if (auto *ConstIdx = dyn_cast<ConstantInt>(Idx)) {
      size_t Offset = ConstIdx->getZExtValue() * DL->getTypeStoreSize(ElemType);
      AAMDNodes OldAAMD = LI->getAAMetadata();
      NewLoad->setAAMetadata(OldAAMD.adjustForAccess(Offset, ElemType, *DL));
    }

    replaceValue(*EI, *NewLoad, false);
  }

  FailureGuard.release();
  return true;
}

bool VectorCombine::scalarizeExtExtract(Instruction &I) {
  if (!TTI.allowVectorElementIndexingUsingGEP())
    return false;
  auto *Ext = dyn_cast<ZExtInst>(&I);
  if (!Ext)
    return false;

  // Try to convert a vector zext feeding only extracts to a set of scalar
  //   (Src << ExtIdx *Size) & (Size -1)
  // if profitable   .
  auto *SrcTy = dyn_cast<FixedVectorType>(Ext->getOperand(0)->getType());
  if (!SrcTy)
    return false;
  auto *DstTy = cast<FixedVectorType>(Ext->getType());

  Type *ScalarDstTy = DstTy->getElementType();
  if (DL->getTypeSizeInBits(SrcTy) != DL->getTypeSizeInBits(ScalarDstTy))
    return false;

  InstructionCost VectorCost =
      TTI.getCastInstrCost(Instruction::ZExt, DstTy, SrcTy,
                           TTI::CastContextHint::None, CostKind, Ext);
  unsigned ExtCnt = 0;
  bool ExtLane0 = false;
  for (User *U : Ext->users()) {
    uint64_t Idx;
    if (!match(U, m_ExtractElt(m_Value(), m_ConstantInt(Idx))))
      return false;
    if (cast<Instruction>(U)->use_empty())
      continue;
    ExtCnt += 1;
    ExtLane0 |= !Idx;
    VectorCost += TTI.getVectorInstrCost(Instruction::ExtractElement, DstTy,
                                         CostKind, Idx, U);
  }

  InstructionCost ScalarCost =
      ExtCnt * TTI.getArithmeticInstrCost(
                   Instruction::And, ScalarDstTy, CostKind,
                   {TTI::OK_AnyValue, TTI::OP_None},
                   {TTI::OK_NonUniformConstantValue, TTI::OP_None}) +
      (ExtCnt - ExtLane0) *
          TTI.getArithmeticInstrCost(
              Instruction::LShr, ScalarDstTy, CostKind,
              {TTI::OK_AnyValue, TTI::OP_None},
              {TTI::OK_NonUniformConstantValue, TTI::OP_None});
  if (ScalarCost > VectorCost)
    return false;

  Value *ScalarV = Ext->getOperand(0);
  if (!isGuaranteedNotToBePoison(ScalarV, &AC, dyn_cast<Instruction>(ScalarV),
                                 &DT))
    ScalarV = Builder.CreateFreeze(ScalarV);
  ScalarV = Builder.CreateBitCast(
      ScalarV,
      IntegerType::get(SrcTy->getContext(), DL->getTypeSizeInBits(SrcTy)));
  uint64_t SrcEltSizeInBits = DL->getTypeSizeInBits(SrcTy->getElementType());
  uint64_t EltBitMask = (1ull << SrcEltSizeInBits) - 1;
  for (User *U : Ext->users()) {
    auto *Extract = cast<ExtractElementInst>(U);
    uint64_t Idx =
        cast<ConstantInt>(Extract->getIndexOperand())->getZExtValue();
    Value *LShr = Builder.CreateLShr(ScalarV, Idx * SrcEltSizeInBits);
    Value *And = Builder.CreateAnd(LShr, EltBitMask);
    U->replaceAllUsesWith(And);
  }
  return true;
}

/// Try to fold "(or (zext (bitcast X)), (shl (zext (bitcast Y)), C))"
/// to "(bitcast (concat X, Y))"
/// where X/Y are bitcasted from i1 mask vectors.
bool VectorCombine::foldConcatOfBoolMasks(Instruction &I) {
  Type *Ty = I.getType();
  if (!Ty->isIntegerTy())
    return false;

  // TODO: Add big endian test coverage
  if (DL->isBigEndian())
    return false;

  // Restrict to disjoint cases so the mask vectors aren't overlapping.
  Instruction *X, *Y;
  if (!match(&I, m_DisjointOr(m_Instruction(X), m_Instruction(Y))))
    return false;

  // Allow both sources to contain shl, to handle more generic pattern:
  // "(or (shl (zext (bitcast X)), C1), (shl (zext (bitcast Y)), C2))"
  Value *SrcX;
  uint64_t ShAmtX = 0;
  if (!match(X, m_OneUse(m_ZExt(m_OneUse(m_BitCast(m_Value(SrcX)))))) &&
      !match(X, m_OneUse(
                    m_Shl(m_OneUse(m_ZExt(m_OneUse(m_BitCast(m_Value(SrcX))))),
                          m_ConstantInt(ShAmtX)))))
    return false;

  Value *SrcY;
  uint64_t ShAmtY = 0;
  if (!match(Y, m_OneUse(m_ZExt(m_OneUse(m_BitCast(m_Value(SrcY)))))) &&
      !match(Y, m_OneUse(
                    m_Shl(m_OneUse(m_ZExt(m_OneUse(m_BitCast(m_Value(SrcY))))),
                          m_ConstantInt(ShAmtY)))))
    return false;

  // Canonicalize larger shift to the RHS.
  if (ShAmtX > ShAmtY) {
    std::swap(X, Y);
    std::swap(SrcX, SrcY);
    std::swap(ShAmtX, ShAmtY);
  }

  // Ensure both sources are matching vXi1 bool mask types, and that the shift
  // difference is the mask width so they can be easily concatenated together.
  uint64_t ShAmtDiff = ShAmtY - ShAmtX;
  unsigned NumSHL = (ShAmtX > 0) + (ShAmtY > 0);
  unsigned BitWidth = Ty->getPrimitiveSizeInBits();
  auto *MaskTy = dyn_cast<FixedVectorType>(SrcX->getType());
  if (!MaskTy || SrcX->getType() != SrcY->getType() ||
      !MaskTy->getElementType()->isIntegerTy(1) ||
      MaskTy->getNumElements() != ShAmtDiff ||
      MaskTy->getNumElements() > (BitWidth / 2))
    return false;

  auto *ConcatTy = FixedVectorType::getDoubleElementsVectorType(MaskTy);
  auto *ConcatIntTy =
      Type::getIntNTy(Ty->getContext(), ConcatTy->getNumElements());
  auto *MaskIntTy = Type::getIntNTy(Ty->getContext(), ShAmtDiff);

  SmallVector<int, 32> ConcatMask(ConcatTy->getNumElements());
  std::iota(ConcatMask.begin(), ConcatMask.end(), 0);

  // TODO: Is it worth supporting multi use cases?
  InstructionCost OldCost = 0;
  OldCost += TTI.getArithmeticInstrCost(Instruction::Or, Ty, CostKind);
  OldCost +=
      NumSHL * TTI.getArithmeticInstrCost(Instruction::Shl, Ty, CostKind);
  OldCost += 2 * TTI.getCastInstrCost(Instruction::ZExt, Ty, MaskIntTy,
                                      TTI::CastContextHint::None, CostKind);
  OldCost += 2 * TTI.getCastInstrCost(Instruction::BitCast, MaskIntTy, MaskTy,
                                      TTI::CastContextHint::None, CostKind);

  InstructionCost NewCost = 0;
  NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, ConcatTy,
                                MaskTy, ConcatMask, CostKind);
  NewCost += TTI.getCastInstrCost(Instruction::BitCast, ConcatIntTy, ConcatTy,
                                  TTI::CastContextHint::None, CostKind);
  if (Ty != ConcatIntTy)
    NewCost += TTI.getCastInstrCost(Instruction::ZExt, Ty, ConcatIntTy,
                                    TTI::CastContextHint::None, CostKind);
  if (ShAmtX > 0)
    NewCost += TTI.getArithmeticInstrCost(Instruction::Shl, Ty, CostKind);

  LLVM_DEBUG(dbgs() << "Found a concatenation of bitcasted bool masks: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");

  if (NewCost > OldCost)
    return false;

  // Build bool mask concatenation, bitcast back to scalar integer, and perform
  // any residual zero-extension or shifting.
  Value *Concat = Builder.CreateShuffleVector(SrcX, SrcY, ConcatMask);
  Worklist.pushValue(Concat);

  Value *Result = Builder.CreateBitCast(Concat, ConcatIntTy);

  if (Ty != ConcatIntTy) {
    Worklist.pushValue(Result);
    Result = Builder.CreateZExt(Result, Ty);
  }

  if (ShAmtX > 0) {
    Worklist.pushValue(Result);
    Result = Builder.CreateShl(Result, ShAmtX);
  }

  replaceValue(I, *Result);
  return true;
}

/// Try to convert "shuffle (binop (shuffle, shuffle)), undef"
///           -->  "binop (shuffle), (shuffle)".
bool VectorCombine::foldPermuteOfBinops(Instruction &I) {
  BinaryOperator *BinOp;
  ArrayRef<int> OuterMask;
  if (!match(&I,
             m_Shuffle(m_OneUse(m_BinOp(BinOp)), m_Undef(), m_Mask(OuterMask))))
    return false;

  // Don't introduce poison into div/rem.
  if (BinOp->isIntDivRem() && llvm::is_contained(OuterMask, PoisonMaskElem))
    return false;

  Value *Op00, *Op01, *Op10, *Op11;
  ArrayRef<int> Mask0, Mask1;
  bool Match0 =
      match(BinOp->getOperand(0),
            m_OneUse(m_Shuffle(m_Value(Op00), m_Value(Op01), m_Mask(Mask0))));
  bool Match1 =
      match(BinOp->getOperand(1),
            m_OneUse(m_Shuffle(m_Value(Op10), m_Value(Op11), m_Mask(Mask1))));
  if (!Match0 && !Match1)
    return false;

  Op00 = Match0 ? Op00 : BinOp->getOperand(0);
  Op01 = Match0 ? Op01 : BinOp->getOperand(0);
  Op10 = Match1 ? Op10 : BinOp->getOperand(1);
  Op11 = Match1 ? Op11 : BinOp->getOperand(1);

  Instruction::BinaryOps Opcode = BinOp->getOpcode();
  auto *ShuffleDstTy = dyn_cast<FixedVectorType>(I.getType());
  auto *BinOpTy = dyn_cast<FixedVectorType>(BinOp->getType());
  auto *Op0Ty = dyn_cast<FixedVectorType>(Op00->getType());
  auto *Op1Ty = dyn_cast<FixedVectorType>(Op10->getType());
  if (!ShuffleDstTy || !BinOpTy || !Op0Ty || !Op1Ty)
    return false;

  unsigned NumSrcElts = BinOpTy->getNumElements();

  // Don't accept shuffles that reference the second operand in
  // div/rem or if its an undef arg.
  if ((BinOp->isIntDivRem() || !isa<PoisonValue>(I.getOperand(1))) &&
      any_of(OuterMask, [NumSrcElts](int M) { return M >= (int)NumSrcElts; }))
    return false;

  // Merge outer / inner (or identity if no match) shuffles.
  SmallVector<int> NewMask0, NewMask1;
  for (int M : OuterMask) {
    if (M < 0 || M >= (int)NumSrcElts) {
      NewMask0.push_back(PoisonMaskElem);
      NewMask1.push_back(PoisonMaskElem);
    } else {
      NewMask0.push_back(Match0 ? Mask0[M] : M);
      NewMask1.push_back(Match1 ? Mask1[M] : M);
    }
  }

  unsigned NumOpElts = Op0Ty->getNumElements();
  bool IsIdentity0 = ShuffleDstTy == Op0Ty &&
      all_of(NewMask0, [NumOpElts](int M) { return M < (int)NumOpElts; }) &&
      ShuffleVectorInst::isIdentityMask(NewMask0, NumOpElts);
  bool IsIdentity1 = ShuffleDstTy == Op1Ty &&
      all_of(NewMask1, [NumOpElts](int M) { return M < (int)NumOpElts; }) &&
      ShuffleVectorInst::isIdentityMask(NewMask1, NumOpElts);

  // Try to merge shuffles across the binop if the new shuffles are not costly.
  InstructionCost OldCost =
      TTI.getArithmeticInstrCost(Opcode, BinOpTy, CostKind) +
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, ShuffleDstTy,
                         BinOpTy, OuterMask, CostKind, 0, nullptr, {BinOp}, &I);
  if (Match0)
    OldCost += TTI.getShuffleCost(
        TargetTransformInfo::SK_PermuteTwoSrc, BinOpTy, Op0Ty, Mask0, CostKind,
        0, nullptr, {Op00, Op01}, cast<Instruction>(BinOp->getOperand(0)));
  if (Match1)
    OldCost += TTI.getShuffleCost(
        TargetTransformInfo::SK_PermuteTwoSrc, BinOpTy, Op1Ty, Mask1, CostKind,
        0, nullptr, {Op10, Op11}, cast<Instruction>(BinOp->getOperand(1)));

  InstructionCost NewCost =
      TTI.getArithmeticInstrCost(Opcode, ShuffleDstTy, CostKind);

  if (!IsIdentity0)
    NewCost +=
        TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, ShuffleDstTy,
                           Op0Ty, NewMask0, CostKind, 0, nullptr, {Op00, Op01});
  if (!IsIdentity1)
    NewCost +=
        TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, ShuffleDstTy,
                           Op1Ty, NewMask1, CostKind, 0, nullptr, {Op10, Op11});

  LLVM_DEBUG(dbgs() << "Found a shuffle feeding a shuffled binop: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");

  // If costs are equal, still fold as we reduce instruction count.
  if (NewCost > OldCost)
    return false;

  Value *LHS =
      IsIdentity0 ? Op00 : Builder.CreateShuffleVector(Op00, Op01, NewMask0);
  Value *RHS =
      IsIdentity1 ? Op10 : Builder.CreateShuffleVector(Op10, Op11, NewMask1);
  Value *NewBO = Builder.CreateBinOp(Opcode, LHS, RHS);

  // Intersect flags from the old binops.
  if (auto *NewInst = dyn_cast<Instruction>(NewBO))
    NewInst->copyIRFlags(BinOp);

  Worklist.pushValue(LHS);
  Worklist.pushValue(RHS);
  replaceValue(I, *NewBO);
  return true;
}

/// Try to convert "shuffle (binop), (binop)" into "binop (shuffle), (shuffle)".
/// Try to convert "shuffle (cmpop), (cmpop)" into "cmpop (shuffle), (shuffle)".
bool VectorCombine::foldShuffleOfBinops(Instruction &I) {
  ArrayRef<int> OldMask;
  Instruction *LHS, *RHS;
  if (!match(&I, m_Shuffle(m_OneUse(m_Instruction(LHS)),
                           m_OneUse(m_Instruction(RHS)), m_Mask(OldMask))))
    return false;

  // TODO: Add support for addlike etc.
  if (LHS->getOpcode() != RHS->getOpcode())
    return false;

  Value *X, *Y, *Z, *W;
  bool IsCommutative = false;
  CmpPredicate PredLHS = CmpInst::BAD_ICMP_PREDICATE;
  CmpPredicate PredRHS = CmpInst::BAD_ICMP_PREDICATE;
  if (match(LHS, m_BinOp(m_Value(X), m_Value(Y))) &&
      match(RHS, m_BinOp(m_Value(Z), m_Value(W)))) {
    auto *BO = cast<BinaryOperator>(LHS);
    // Don't introduce poison into div/rem.
    if (llvm::is_contained(OldMask, PoisonMaskElem) && BO->isIntDivRem())
      return false;
    IsCommutative = BinaryOperator::isCommutative(BO->getOpcode());
  } else if (match(LHS, m_Cmp(PredLHS, m_Value(X), m_Value(Y))) &&
             match(RHS, m_Cmp(PredRHS, m_Value(Z), m_Value(W))) &&
             (CmpInst::Predicate)PredLHS == (CmpInst::Predicate)PredRHS) {
    IsCommutative = cast<CmpInst>(LHS)->isCommutative();
  } else
    return false;

  auto *ShuffleDstTy = dyn_cast<FixedVectorType>(I.getType());
  auto *BinResTy = dyn_cast<FixedVectorType>(LHS->getType());
  auto *BinOpTy = dyn_cast<FixedVectorType>(X->getType());
  if (!ShuffleDstTy || !BinResTy || !BinOpTy || X->getType() != Z->getType())
    return false;

  unsigned NumSrcElts = BinOpTy->getNumElements();

  // If we have something like "add X, Y" and "add Z, X", swap ops to match.
  if (IsCommutative && X != Z && Y != W && (X == W || Y == Z))
    std::swap(X, Y);

  auto ConvertToUnary = [NumSrcElts](int &M) {
    if (M >= (int)NumSrcElts)
      M -= NumSrcElts;
  };

  SmallVector<int> NewMask0(OldMask);
  TargetTransformInfo::ShuffleKind SK0 = TargetTransformInfo::SK_PermuteTwoSrc;
  if (X == Z) {
    llvm::for_each(NewMask0, ConvertToUnary);
    SK0 = TargetTransformInfo::SK_PermuteSingleSrc;
    Z = PoisonValue::get(BinOpTy);
  }

  SmallVector<int> NewMask1(OldMask);
  TargetTransformInfo::ShuffleKind SK1 = TargetTransformInfo::SK_PermuteTwoSrc;
  if (Y == W) {
    llvm::for_each(NewMask1, ConvertToUnary);
    SK1 = TargetTransformInfo::SK_PermuteSingleSrc;
    W = PoisonValue::get(BinOpTy);
  }

  // Try to replace a binop with a shuffle if the shuffle is not costly.
  InstructionCost OldCost =
      TTI.getInstructionCost(LHS, CostKind) +
      TTI.getInstructionCost(RHS, CostKind) +
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, ShuffleDstTy,
                         BinResTy, OldMask, CostKind, 0, nullptr, {LHS, RHS},
                         &I);

  // Handle shuffle(binop(shuffle(x),y),binop(z,shuffle(w))) style patterns
  // where one use shuffles have gotten split across the binop/cmp. These
  // often allow a major reduction in total cost that wouldn't happen as
  // individual folds.
  auto MergeInner = [&](Value *&Op, int Offset, MutableArrayRef<int> Mask,
                        TTI::TargetCostKind CostKind) -> bool {
    Value *InnerOp;
    ArrayRef<int> InnerMask;
    if (match(Op, m_OneUse(m_Shuffle(m_Value(InnerOp), m_Undef(),
                                     m_Mask(InnerMask)))) &&
        InnerOp->getType() == Op->getType() &&
        all_of(InnerMask,
               [NumSrcElts](int M) { return M < (int)NumSrcElts; })) {
      for (int &M : Mask)
        if (Offset <= M && M < (int)(Offset + NumSrcElts)) {
          M = InnerMask[M - Offset];
          M = 0 <= M ? M + Offset : M;
        }
      OldCost += TTI.getInstructionCost(cast<Instruction>(Op), CostKind);
      Op = InnerOp;
      return true;
    }
    return false;
  };
  bool ReducedInstCount = false;
  ReducedInstCount |= MergeInner(X, 0, NewMask0, CostKind);
  ReducedInstCount |= MergeInner(Y, 0, NewMask1, CostKind);
  ReducedInstCount |= MergeInner(Z, NumSrcElts, NewMask0, CostKind);
  ReducedInstCount |= MergeInner(W, NumSrcElts, NewMask1, CostKind);

  auto *ShuffleCmpTy =
      FixedVectorType::get(BinOpTy->getElementType(), ShuffleDstTy);
  InstructionCost NewCost =
      TTI.getShuffleCost(SK0, ShuffleCmpTy, BinOpTy, NewMask0, CostKind, 0,
                         nullptr, {X, Z}) +
      TTI.getShuffleCost(SK1, ShuffleCmpTy, BinOpTy, NewMask1, CostKind, 0,
                         nullptr, {Y, W});

  if (PredLHS == CmpInst::BAD_ICMP_PREDICATE) {
    NewCost +=
        TTI.getArithmeticInstrCost(LHS->getOpcode(), ShuffleDstTy, CostKind);
  } else {
    NewCost += TTI.getCmpSelInstrCost(LHS->getOpcode(), ShuffleCmpTy,
                                      ShuffleDstTy, PredLHS, CostKind);
  }

  LLVM_DEBUG(dbgs() << "Found a shuffle feeding two binops: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");

  // If either shuffle will constant fold away, then fold for the same cost as
  // we will reduce the instruction count.
  ReducedInstCount |= (isa<Constant>(X) && isa<Constant>(Z)) ||
                      (isa<Constant>(Y) && isa<Constant>(W));
  if (ReducedInstCount ? (NewCost > OldCost) : (NewCost >= OldCost))
    return false;

  Value *Shuf0 = Builder.CreateShuffleVector(X, Z, NewMask0);
  Value *Shuf1 = Builder.CreateShuffleVector(Y, W, NewMask1);
  Value *NewBO = PredLHS == CmpInst::BAD_ICMP_PREDICATE
                     ? Builder.CreateBinOp(
                           cast<BinaryOperator>(LHS)->getOpcode(), Shuf0, Shuf1)
                     : Builder.CreateCmp(PredLHS, Shuf0, Shuf1);

  // Intersect flags from the old binops.
  if (auto *NewInst = dyn_cast<Instruction>(NewBO)) {
    NewInst->copyIRFlags(LHS);
    NewInst->andIRFlags(RHS);
  }

  Worklist.pushValue(Shuf0);
  Worklist.pushValue(Shuf1);
  replaceValue(I, *NewBO);
  return true;
}

/// Try to convert,
/// (shuffle(select(c1,t1,f1)), (select(c2,t2,f2)), m) into
/// (select (shuffle c1,c2,m), (shuffle t1,t2,m), (shuffle f1,f2,m))
bool VectorCombine::foldShuffleOfSelects(Instruction &I) {
  ArrayRef<int> Mask;
  Value *C1, *T1, *F1, *C2, *T2, *F2;
  if (!match(&I, m_Shuffle(
                     m_OneUse(m_Select(m_Value(C1), m_Value(T1), m_Value(F1))),
                     m_OneUse(m_Select(m_Value(C2), m_Value(T2), m_Value(F2))),
                     m_Mask(Mask))))
    return false;

  auto *C1VecTy = dyn_cast<FixedVectorType>(C1->getType());
  auto *C2VecTy = dyn_cast<FixedVectorType>(C2->getType());
  if (!C1VecTy || !C2VecTy || C1VecTy != C2VecTy)
    return false;

  auto *SI0FOp = dyn_cast<FPMathOperator>(I.getOperand(0));
  auto *SI1FOp = dyn_cast<FPMathOperator>(I.getOperand(1));
  // SelectInsts must have the same FMF.
  if (((SI0FOp == nullptr) != (SI1FOp == nullptr)) ||
      ((SI0FOp != nullptr) &&
       (SI0FOp->getFastMathFlags() != SI1FOp->getFastMathFlags())))
    return false;

  auto *SrcVecTy = cast<FixedVectorType>(T1->getType());
  auto *DstVecTy = cast<FixedVectorType>(I.getType());
  auto SK = TargetTransformInfo::SK_PermuteTwoSrc;
  auto SelOp = Instruction::Select;
  InstructionCost OldCost = TTI.getCmpSelInstrCost(
      SelOp, SrcVecTy, C1VecTy, CmpInst::BAD_ICMP_PREDICATE, CostKind);
  OldCost += TTI.getCmpSelInstrCost(SelOp, SrcVecTy, C2VecTy,
                                    CmpInst::BAD_ICMP_PREDICATE, CostKind);
  OldCost +=
      TTI.getShuffleCost(SK, DstVecTy, SrcVecTy, Mask, CostKind, 0, nullptr,
                         {I.getOperand(0), I.getOperand(1)}, &I);

  InstructionCost NewCost = TTI.getShuffleCost(
      SK, FixedVectorType::get(C1VecTy->getScalarType(), Mask.size()), C1VecTy,
      Mask, CostKind, 0, nullptr, {C1, C2});
  NewCost += TTI.getShuffleCost(SK, DstVecTy, SrcVecTy, Mask, CostKind, 0,
                                nullptr, {T1, T2});
  NewCost += TTI.getShuffleCost(SK, DstVecTy, SrcVecTy, Mask, CostKind, 0,
                                nullptr, {F1, F2});
  auto *C1C2ShuffledVecTy = cast<FixedVectorType>(
      toVectorTy(Type::getInt1Ty(I.getContext()), DstVecTy->getNumElements()));
  NewCost += TTI.getCmpSelInstrCost(SelOp, DstVecTy, C1C2ShuffledVecTy,
                                    CmpInst::BAD_ICMP_PREDICATE, CostKind);

  LLVM_DEBUG(dbgs() << "Found a shuffle feeding two selects: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");
  if (NewCost > OldCost)
    return false;

  Value *ShuffleCmp = Builder.CreateShuffleVector(C1, C2, Mask);
  Value *ShuffleTrue = Builder.CreateShuffleVector(T1, T2, Mask);
  Value *ShuffleFalse = Builder.CreateShuffleVector(F1, F2, Mask);
  Value *NewSel;
  // We presuppose that the SelectInsts have the same FMF.
  if (SI0FOp)
    NewSel = Builder.CreateSelectFMF(ShuffleCmp, ShuffleTrue, ShuffleFalse,
                                     SI0FOp->getFastMathFlags());
  else
    NewSel = Builder.CreateSelect(ShuffleCmp, ShuffleTrue, ShuffleFalse);

  Worklist.pushValue(ShuffleCmp);
  Worklist.pushValue(ShuffleTrue);
  Worklist.pushValue(ShuffleFalse);
  replaceValue(I, *NewSel);
  return true;
}

/// Try to convert "shuffle (castop), (castop)" with a shared castop operand
/// into "castop (shuffle)".
bool VectorCombine::foldShuffleOfCastops(Instruction &I) {
  Value *V0, *V1;
  ArrayRef<int> OldMask;
  if (!match(&I, m_Shuffle(m_Value(V0), m_Value(V1), m_Mask(OldMask))))
    return false;

  auto *C0 = dyn_cast<CastInst>(V0);
  auto *C1 = dyn_cast<CastInst>(V1);
  if (!C0 || !C1)
    return false;

  Instruction::CastOps Opcode = C0->getOpcode();
  if (C0->getSrcTy() != C1->getSrcTy())
    return false;

  // Handle shuffle(zext_nneg(x), sext(y)) -> sext(shuffle(x,y)) folds.
  if (Opcode != C1->getOpcode()) {
    if (match(C0, m_SExtLike(m_Value())) && match(C1, m_SExtLike(m_Value())))
      Opcode = Instruction::SExt;
    else
      return false;
  }

  auto *ShuffleDstTy = dyn_cast<FixedVectorType>(I.getType());
  auto *CastDstTy = dyn_cast<FixedVectorType>(C0->getDestTy());
  auto *CastSrcTy = dyn_cast<FixedVectorType>(C0->getSrcTy());
  if (!ShuffleDstTy || !CastDstTy || !CastSrcTy)
    return false;

  unsigned NumSrcElts = CastSrcTy->getNumElements();
  unsigned NumDstElts = CastDstTy->getNumElements();
  assert((NumDstElts == NumSrcElts || Opcode == Instruction::BitCast) &&
         "Only bitcasts expected to alter src/dst element counts");

  // Check for bitcasting of unscalable vector types.
  // e.g. <32 x i40> -> <40 x i32>
  if (NumDstElts != NumSrcElts && (NumSrcElts % NumDstElts) != 0 &&
      (NumDstElts % NumSrcElts) != 0)
    return false;

  SmallVector<int, 16> NewMask;
  if (NumSrcElts >= NumDstElts) {
    // The bitcast is from wide to narrow/equal elements. The shuffle mask can
    // always be expanded to the equivalent form choosing narrower elements.
    assert(NumSrcElts % NumDstElts == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = NumSrcElts / NumDstElts;
    narrowShuffleMaskElts(ScaleFactor, OldMask, NewMask);
  } else {
    // The bitcast is from narrow elements to wide elements. The shuffle mask
    // must choose consecutive elements to allow casting first.
    assert(NumDstElts % NumSrcElts == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = NumDstElts / NumSrcElts;
    if (!widenShuffleMaskElts(ScaleFactor, OldMask, NewMask))
      return false;
  }

  auto *NewShuffleDstTy =
      FixedVectorType::get(CastSrcTy->getScalarType(), NewMask.size());

  // Try to replace a castop with a shuffle if the shuffle is not costly.
  InstructionCost CostC0 =
      TTI.getCastInstrCost(C0->getOpcode(), CastDstTy, CastSrcTy,
                           TTI::CastContextHint::None, CostKind);
  InstructionCost CostC1 =
      TTI.getCastInstrCost(C1->getOpcode(), CastDstTy, CastSrcTy,
                           TTI::CastContextHint::None, CostKind);
  InstructionCost OldCost = CostC0 + CostC1;
  OldCost +=
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, ShuffleDstTy,
                         CastDstTy, OldMask, CostKind, 0, nullptr, {}, &I);

  InstructionCost NewCost =
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, NewShuffleDstTy,
                         CastSrcTy, NewMask, CostKind);
  NewCost += TTI.getCastInstrCost(Opcode, ShuffleDstTy, NewShuffleDstTy,
                                  TTI::CastContextHint::None, CostKind);
  if (!C0->hasOneUse())
    NewCost += CostC0;
  if (!C1->hasOneUse())
    NewCost += CostC1;

  LLVM_DEBUG(dbgs() << "Found a shuffle feeding two casts: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");
  if (NewCost > OldCost)
    return false;

  Value *Shuf = Builder.CreateShuffleVector(C0->getOperand(0),
                                            C1->getOperand(0), NewMask);
  Value *Cast = Builder.CreateCast(Opcode, Shuf, ShuffleDstTy);

  // Intersect flags from the old casts.
  if (auto *NewInst = dyn_cast<Instruction>(Cast)) {
    NewInst->copyIRFlags(C0);
    NewInst->andIRFlags(C1);
  }

  Worklist.pushValue(Shuf);
  replaceValue(I, *Cast);
  return true;
}

/// Try to convert any of:
/// "shuffle (shuffle x, y), (shuffle y, x)"
/// "shuffle (shuffle x, undef), (shuffle y, undef)"
/// "shuffle (shuffle x, undef), y"
/// "shuffle x, (shuffle y, undef)"
/// into "shuffle x, y".
bool VectorCombine::foldShuffleOfShuffles(Instruction &I) {
  ArrayRef<int> OuterMask;
  Value *OuterV0, *OuterV1;
  if (!match(&I,
             m_Shuffle(m_Value(OuterV0), m_Value(OuterV1), m_Mask(OuterMask))))
    return false;

  ArrayRef<int> InnerMask0, InnerMask1;
  Value *X0, *X1, *Y0, *Y1;
  bool Match0 =
      match(OuterV0, m_Shuffle(m_Value(X0), m_Value(Y0), m_Mask(InnerMask0)));
  bool Match1 =
      match(OuterV1, m_Shuffle(m_Value(X1), m_Value(Y1), m_Mask(InnerMask1)));
  if (!Match0 && !Match1)
    return false;

  // If the outer shuffle is a permute, then create a fake inner all-poison
  // shuffle. This is easier than accounting for length-changing shuffles below.
  SmallVector<int, 16> PoisonMask1;
  if (!Match1 && isa<PoisonValue>(OuterV1)) {
    X1 = X0;
    Y1 = Y0;
    PoisonMask1.append(InnerMask0.size(), PoisonMaskElem);
    InnerMask1 = PoisonMask1;
    Match1 = true; // fake match
  }

  X0 = Match0 ? X0 : OuterV0;
  Y0 = Match0 ? Y0 : OuterV0;
  X1 = Match1 ? X1 : OuterV1;
  Y1 = Match1 ? Y1 : OuterV1;
  auto *ShuffleDstTy = dyn_cast<FixedVectorType>(I.getType());
  auto *ShuffleSrcTy = dyn_cast<FixedVectorType>(X0->getType());
  auto *ShuffleImmTy = dyn_cast<FixedVectorType>(OuterV0->getType());
  if (!ShuffleDstTy || !ShuffleSrcTy || !ShuffleImmTy ||
      X0->getType() != X1->getType())
    return false;

  unsigned NumSrcElts = ShuffleSrcTy->getNumElements();
  unsigned NumImmElts = ShuffleImmTy->getNumElements();

  // Attempt to merge shuffles, matching upto 2 source operands.
  // Replace index to a poison arg with PoisonMaskElem.
  // Bail if either inner masks reference an undef arg.
  SmallVector<int, 16> NewMask(OuterMask);
  Value *NewX = nullptr, *NewY = nullptr;
  for (int &M : NewMask) {
    Value *Src = nullptr;
    if (0 <= M && M < (int)NumImmElts) {
      Src = OuterV0;
      if (Match0) {
        M = InnerMask0[M];
        Src = M >= (int)NumSrcElts ? Y0 : X0;
        M = M >= (int)NumSrcElts ? (M - NumSrcElts) : M;
      }
    } else if (M >= (int)NumImmElts) {
      Src = OuterV1;
      M -= NumImmElts;
      if (Match1) {
        M = InnerMask1[M];
        Src = M >= (int)NumSrcElts ? Y1 : X1;
        M = M >= (int)NumSrcElts ? (M - NumSrcElts) : M;
      }
    }
    if (Src && M != PoisonMaskElem) {
      assert(0 <= M && M < (int)NumSrcElts && "Unexpected shuffle mask index");
      if (isa<UndefValue>(Src)) {
        // We've referenced an undef element - if its poison, update the shuffle
        // mask, else bail.
        if (!isa<PoisonValue>(Src))
          return false;
        M = PoisonMaskElem;
        continue;
      }
      if (!NewX || NewX == Src) {
        NewX = Src;
        continue;
      }
      if (!NewY || NewY == Src) {
        M += NumSrcElts;
        NewY = Src;
        continue;
      }
      return false;
    }
  }

  if (!NewX)
    return PoisonValue::get(ShuffleDstTy);
  if (!NewY)
    NewY = PoisonValue::get(ShuffleSrcTy);

  // Have we folded to an Identity shuffle?
  if (ShuffleVectorInst::isIdentityMask(NewMask, NumSrcElts)) {
    replaceValue(I, *NewX);
    return true;
  }

  // Try to merge the shuffles if the new shuffle is not costly.
  InstructionCost InnerCost0 = 0;
  if (Match0)
    InnerCost0 = TTI.getInstructionCost(cast<User>(OuterV0), CostKind);

  InstructionCost InnerCost1 = 0;
  if (Match1)
    InnerCost1 = TTI.getInstructionCost(cast<User>(OuterV1), CostKind);

  InstructionCost OuterCost = TTI.getInstructionCost(&I, CostKind);

  InstructionCost OldCost = InnerCost0 + InnerCost1 + OuterCost;

  bool IsUnary = all_of(NewMask, [&](int M) { return M < (int)NumSrcElts; });
  TargetTransformInfo::ShuffleKind SK =
      IsUnary ? TargetTransformInfo::SK_PermuteSingleSrc
              : TargetTransformInfo::SK_PermuteTwoSrc;
  InstructionCost NewCost =
      TTI.getShuffleCost(SK, ShuffleDstTy, ShuffleSrcTy, NewMask, CostKind, 0,
                         nullptr, {NewX, NewY});
  if (!OuterV0->hasOneUse())
    NewCost += InnerCost0;
  if (!OuterV1->hasOneUse())
    NewCost += InnerCost1;

  LLVM_DEBUG(dbgs() << "Found a shuffle feeding two shuffles: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");
  if (NewCost > OldCost)
    return false;

  Value *Shuf = Builder.CreateShuffleVector(NewX, NewY, NewMask);
  replaceValue(I, *Shuf);
  return true;
}

/// Try to convert
/// "shuffle (intrinsic), (intrinsic)" into "intrinsic (shuffle), (shuffle)".
bool VectorCombine::foldShuffleOfIntrinsics(Instruction &I) {
  Value *V0, *V1;
  ArrayRef<int> OldMask;
  if (!match(&I, m_Shuffle(m_OneUse(m_Value(V0)), m_OneUse(m_Value(V1)),
                           m_Mask(OldMask))))
    return false;

  auto *II0 = dyn_cast<IntrinsicInst>(V0);
  auto *II1 = dyn_cast<IntrinsicInst>(V1);
  if (!II0 || !II1)
    return false;

  Intrinsic::ID IID = II0->getIntrinsicID();
  if (IID != II1->getIntrinsicID())
    return false;

  auto *ShuffleDstTy = dyn_cast<FixedVectorType>(I.getType());
  auto *II0Ty = dyn_cast<FixedVectorType>(II0->getType());
  if (!ShuffleDstTy || !II0Ty)
    return false;

  if (!isTriviallyVectorizable(IID))
    return false;

  for (unsigned I = 0, E = II0->arg_size(); I != E; ++I)
    if (isVectorIntrinsicWithScalarOpAtArg(IID, I, &TTI) &&
        II0->getArgOperand(I) != II1->getArgOperand(I))
      return false;

  InstructionCost OldCost =
      TTI.getIntrinsicInstrCost(IntrinsicCostAttributes(IID, *II0), CostKind) +
      TTI.getIntrinsicInstrCost(IntrinsicCostAttributes(IID, *II1), CostKind) +
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc, ShuffleDstTy,
                         II0Ty, OldMask, CostKind, 0, nullptr, {II0, II1}, &I);

  SmallVector<Type *> NewArgsTy;
  InstructionCost NewCost = 0;
  for (unsigned I = 0, E = II0->arg_size(); I != E; ++I) {
    if (isVectorIntrinsicWithScalarOpAtArg(IID, I, &TTI)) {
      NewArgsTy.push_back(II0->getArgOperand(I)->getType());
    } else {
      auto *VecTy = cast<FixedVectorType>(II0->getArgOperand(I)->getType());
      auto *ArgTy = FixedVectorType::get(VecTy->getElementType(),
                                         ShuffleDstTy->getNumElements());
      NewArgsTy.push_back(ArgTy);
      NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteTwoSrc,
                                    ArgTy, VecTy, OldMask, CostKind);
    }
  }
  IntrinsicCostAttributes NewAttr(IID, ShuffleDstTy, NewArgsTy);
  NewCost += TTI.getIntrinsicInstrCost(NewAttr, CostKind);

  LLVM_DEBUG(dbgs() << "Found a shuffle feeding two intrinsics: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");

  if (NewCost > OldCost)
    return false;

  SmallVector<Value *> NewArgs;
  for (unsigned I = 0, E = II0->arg_size(); I != E; ++I)
    if (isVectorIntrinsicWithScalarOpAtArg(IID, I, &TTI)) {
      NewArgs.push_back(II0->getArgOperand(I));
    } else {
      Value *Shuf = Builder.CreateShuffleVector(II0->getArgOperand(I),
                                                II1->getArgOperand(I), OldMask);
      NewArgs.push_back(Shuf);
      Worklist.pushValue(Shuf);
    }
  Value *NewIntrinsic = Builder.CreateIntrinsic(ShuffleDstTy, IID, NewArgs);

  // Intersect flags from the old intrinsics.
  if (auto *NewInst = dyn_cast<Instruction>(NewIntrinsic)) {
    NewInst->copyIRFlags(II0);
    NewInst->andIRFlags(II1);
  }

  replaceValue(I, *NewIntrinsic);
  return true;
}

using InstLane = std::pair<Use *, int>;

static InstLane lookThroughShuffles(Use *U, int Lane) {
  while (auto *SV = dyn_cast<ShuffleVectorInst>(U->get())) {
    unsigned NumElts =
        cast<FixedVectorType>(SV->getOperand(0)->getType())->getNumElements();
    int M = SV->getMaskValue(Lane);
    if (M < 0)
      return {nullptr, PoisonMaskElem};
    if (static_cast<unsigned>(M) < NumElts) {
      U = &SV->getOperandUse(0);
      Lane = M;
    } else {
      U = &SV->getOperandUse(1);
      Lane = M - NumElts;
    }
  }
  return InstLane{U, Lane};
}

static SmallVector<InstLane>
generateInstLaneVectorFromOperand(ArrayRef<InstLane> Item, int Op) {
  SmallVector<InstLane> NItem;
  for (InstLane IL : Item) {
    auto [U, Lane] = IL;
    InstLane OpLane =
        U ? lookThroughShuffles(&cast<Instruction>(U->get())->getOperandUse(Op),
                                Lane)
          : InstLane{nullptr, PoisonMaskElem};
    NItem.emplace_back(OpLane);
  }
  return NItem;
}

/// Detect concat of multiple values into a vector
static bool isFreeConcat(ArrayRef<InstLane> Item, TTI::TargetCostKind CostKind,
                         const TargetTransformInfo &TTI) {
  auto *Ty = cast<FixedVectorType>(Item.front().first->get()->getType());
  unsigned NumElts = Ty->getNumElements();
  if (Item.size() == NumElts || NumElts == 1 || Item.size() % NumElts != 0)
    return false;

  // Check that the concat is free, usually meaning that the type will be split
  // during legalization.
  SmallVector<int, 16> ConcatMask(NumElts * 2);
  std::iota(ConcatMask.begin(), ConcatMask.end(), 0);
  if (TTI.getShuffleCost(TTI::SK_PermuteTwoSrc,
                         FixedVectorType::get(Ty->getScalarType(), NumElts * 2),
                         Ty, ConcatMask, CostKind) != 0)
    return false;

  unsigned NumSlices = Item.size() / NumElts;
  // Currently we generate a tree of shuffles for the concats, which limits us
  // to a power2.
  if (!isPowerOf2_32(NumSlices))
    return false;
  for (unsigned Slice = 0; Slice < NumSlices; ++Slice) {
    Use *SliceV = Item[Slice * NumElts].first;
    if (!SliceV || SliceV->get()->getType() != Ty)
      return false;
    for (unsigned Elt = 0; Elt < NumElts; ++Elt) {
      auto [V, Lane] = Item[Slice * NumElts + Elt];
      if (Lane != static_cast<int>(Elt) || SliceV->get() != V->get())
        return false;
    }
  }
  return true;
}

static Value *generateNewInstTree(ArrayRef<InstLane> Item, FixedVectorType *Ty,
                                  const SmallPtrSet<Use *, 4> &IdentityLeafs,
                                  const SmallPtrSet<Use *, 4> &SplatLeafs,
                                  const SmallPtrSet<Use *, 4> &ConcatLeafs,
                                  IRBuilderBase &Builder,
                                  const TargetTransformInfo *TTI) {
  auto [FrontU, FrontLane] = Item.front();

  if (IdentityLeafs.contains(FrontU)) {
    return FrontU->get();
  }
  if (SplatLeafs.contains(FrontU)) {
    SmallVector<int, 16> Mask(Ty->getNumElements(), FrontLane);
    return Builder.CreateShuffleVector(FrontU->get(), Mask);
  }
  if (ConcatLeafs.contains(FrontU)) {
    unsigned NumElts =
        cast<FixedVectorType>(FrontU->get()->getType())->getNumElements();
    SmallVector<Value *> Values(Item.size() / NumElts, nullptr);
    for (unsigned S = 0; S < Values.size(); ++S)
      Values[S] = Item[S * NumElts].first->get();

    while (Values.size() > 1) {
      NumElts *= 2;
      SmallVector<int, 16> Mask(NumElts, 0);
      std::iota(Mask.begin(), Mask.end(), 0);
      SmallVector<Value *> NewValues(Values.size() / 2, nullptr);
      for (unsigned S = 0; S < NewValues.size(); ++S)
        NewValues[S] =
            Builder.CreateShuffleVector(Values[S * 2], Values[S * 2 + 1], Mask);
      Values = NewValues;
    }
    return Values[0];
  }

  auto *I = cast<Instruction>(FrontU->get());
  auto *II = dyn_cast<IntrinsicInst>(I);
  unsigned NumOps = I->getNumOperands() - (II ? 1 : 0);
  SmallVector<Value *> Ops(NumOps);
  for (unsigned Idx = 0; Idx < NumOps; Idx++) {
    if (II &&
        isVectorIntrinsicWithScalarOpAtArg(II->getIntrinsicID(), Idx, TTI)) {
      Ops[Idx] = II->getOperand(Idx);
      continue;
    }
    Ops[Idx] = generateNewInstTree(generateInstLaneVectorFromOperand(Item, Idx),
                                   Ty, IdentityLeafs, SplatLeafs, ConcatLeafs,
                                   Builder, TTI);
  }

  SmallVector<Value *, 8> ValueList;
  for (const auto &Lane : Item)
    if (Lane.first)
      ValueList.push_back(Lane.first->get());

  Type *DstTy =
      FixedVectorType::get(I->getType()->getScalarType(), Ty->getNumElements());
  if (auto *BI = dyn_cast<BinaryOperator>(I)) {
    auto *Value = Builder.CreateBinOp((Instruction::BinaryOps)BI->getOpcode(),
                                      Ops[0], Ops[1]);
    propagateIRFlags(Value, ValueList);
    return Value;
  }
  if (auto *CI = dyn_cast<CmpInst>(I)) {
    auto *Value = Builder.CreateCmp(CI->getPredicate(), Ops[0], Ops[1]);
    propagateIRFlags(Value, ValueList);
    return Value;
  }
  if (auto *SI = dyn_cast<SelectInst>(I)) {
    auto *Value = Builder.CreateSelect(Ops[0], Ops[1], Ops[2], "", SI);
    propagateIRFlags(Value, ValueList);
    return Value;
  }
  if (auto *CI = dyn_cast<CastInst>(I)) {
    auto *Value = Builder.CreateCast(CI->getOpcode(), Ops[0], DstTy);
    propagateIRFlags(Value, ValueList);
    return Value;
  }
  if (II) {
    auto *Value = Builder.CreateIntrinsic(DstTy, II->getIntrinsicID(), Ops);
    propagateIRFlags(Value, ValueList);
    return Value;
  }
  assert(isa<UnaryInstruction>(I) && "Unexpected instruction type in Generate");
  auto *Value =
      Builder.CreateUnOp((Instruction::UnaryOps)I->getOpcode(), Ops[0]);
  propagateIRFlags(Value, ValueList);
  return Value;
}

// Starting from a shuffle, look up through operands tracking the shuffled index
// of each lane. If we can simplify away the shuffles to identities then
// do so.
bool VectorCombine::foldShuffleToIdentity(Instruction &I) {
  auto *Ty = dyn_cast<FixedVectorType>(I.getType());
  if (!Ty || I.use_empty())
    return false;

  SmallVector<InstLane> Start(Ty->getNumElements());
  for (unsigned M = 0, E = Ty->getNumElements(); M < E; ++M)
    Start[M] = lookThroughShuffles(&*I.use_begin(), M);

  SmallVector<SmallVector<InstLane>> Worklist;
  Worklist.push_back(Start);
  SmallPtrSet<Use *, 4> IdentityLeafs, SplatLeafs, ConcatLeafs;
  unsigned NumVisited = 0;

  while (!Worklist.empty()) {
    if (++NumVisited > MaxInstrsToScan)
      return false;

    SmallVector<InstLane> Item = Worklist.pop_back_val();
    auto [FrontU, FrontLane] = Item.front();

    // If we found an undef first lane then bail out to keep things simple.
    if (!FrontU)
      return false;

    // Helper to peek through bitcasts to the same value.
    auto IsEquiv = [&](Value *X, Value *Y) {
      return X->getType() == Y->getType() &&
             peekThroughBitcasts(X) == peekThroughBitcasts(Y);
    };

    // Look for an identity value.
    if (FrontLane == 0 &&
        cast<FixedVectorType>(FrontU->get()->getType())->getNumElements() ==
            Ty->getNumElements() &&
        all_of(drop_begin(enumerate(Item)), [IsEquiv, Item](const auto &E) {
          Value *FrontV = Item.front().first->get();
          return !E.value().first || (IsEquiv(E.value().first->get(), FrontV) &&
                                      E.value().second == (int)E.index());
        })) {
      IdentityLeafs.insert(FrontU);
      continue;
    }
    // Look for constants, for the moment only supporting constant splats.
    if (auto *C = dyn_cast<Constant>(FrontU);
        C && C->getSplatValue() &&
        all_of(drop_begin(Item), [Item](InstLane &IL) {
          Value *FrontV = Item.front().first->get();
          Use *U = IL.first;
          return !U || (isa<Constant>(U->get()) &&
                        cast<Constant>(U->get())->getSplatValue() ==
                            cast<Constant>(FrontV)->getSplatValue());
        })) {
      SplatLeafs.insert(FrontU);
      continue;
    }
    // Look for a splat value.
    if (all_of(drop_begin(Item), [Item](InstLane &IL) {
          auto [FrontU, FrontLane] = Item.front();
          auto [U, Lane] = IL;
          return !U || (U->get() == FrontU->get() && Lane == FrontLane);
        })) {
      SplatLeafs.insert(FrontU);
      continue;
    }

    // We need each element to be the same type of value, and check that each
    // element has a single use.
    auto CheckLaneIsEquivalentToFirst = [Item](InstLane IL) {
      Value *FrontV = Item.front().first->get();
      if (!IL.first)
        return true;
      Value *V = IL.first->get();
      if (auto *I = dyn_cast<Instruction>(V); I && !I->hasOneUser())
        return false;
      if (V->getValueID() != FrontV->getValueID())
        return false;
      if (auto *CI = dyn_cast<CmpInst>(V))
        if (CI->getPredicate() != cast<CmpInst>(FrontV)->getPredicate())
          return false;
      if (auto *CI = dyn_cast<CastInst>(V))
        if (CI->getSrcTy()->getScalarType() !=
            cast<CastInst>(FrontV)->getSrcTy()->getScalarType())
          return false;
      if (auto *SI = dyn_cast<SelectInst>(V))
        if (!isa<VectorType>(SI->getOperand(0)->getType()) ||
            SI->getOperand(0)->getType() !=
                cast<SelectInst>(FrontV)->getOperand(0)->getType())
          return false;
      if (isa<CallInst>(V) && !isa<IntrinsicInst>(V))
        return false;
      auto *II = dyn_cast<IntrinsicInst>(V);
      return !II || (isa<IntrinsicInst>(FrontV) &&
                     II->getIntrinsicID() ==
                         cast<IntrinsicInst>(FrontV)->getIntrinsicID() &&
                     !II->hasOperandBundles());
    };
    if (all_of(drop_begin(Item), CheckLaneIsEquivalentToFirst)) {
      // Check the operator is one that we support.
      if (isa<BinaryOperator, CmpInst>(FrontU)) {
        //  We exclude div/rem in case they hit UB from poison lanes.
        if (auto *BO = dyn_cast<BinaryOperator>(FrontU);
            BO && BO->isIntDivRem())
          return false;
        Worklist.push_back(generateInstLaneVectorFromOperand(Item, 0));
        Worklist.push_back(generateInstLaneVectorFromOperand(Item, 1));
        continue;
      } else if (isa<UnaryOperator, TruncInst, ZExtInst, SExtInst, FPToSIInst,
                     FPToUIInst, SIToFPInst, UIToFPInst>(FrontU)) {
        Worklist.push_back(generateInstLaneVectorFromOperand(Item, 0));
        continue;
      } else if (auto *BitCast = dyn_cast<BitCastInst>(FrontU)) {
        // TODO: Handle vector widening/narrowing bitcasts.
        auto *DstTy = dyn_cast<FixedVectorType>(BitCast->getDestTy());
        auto *SrcTy = dyn_cast<FixedVectorType>(BitCast->getSrcTy());
        if (DstTy && SrcTy &&
            SrcTy->getNumElements() == DstTy->getNumElements()) {
          Worklist.push_back(generateInstLaneVectorFromOperand(Item, 0));
          continue;
        }
      } else if (isa<SelectInst>(FrontU)) {
        Worklist.push_back(generateInstLaneVectorFromOperand(Item, 0));
        Worklist.push_back(generateInstLaneVectorFromOperand(Item, 1));
        Worklist.push_back(generateInstLaneVectorFromOperand(Item, 2));
        continue;
      } else if (auto *II = dyn_cast<IntrinsicInst>(FrontU);
                 II && isTriviallyVectorizable(II->getIntrinsicID()) &&
                 !II->hasOperandBundles()) {
        for (unsigned Op = 0, E = II->getNumOperands() - 1; Op < E; Op++) {
          if (isVectorIntrinsicWithScalarOpAtArg(II->getIntrinsicID(), Op,
                                                 &TTI)) {
            if (!all_of(drop_begin(Item), [Item, Op](InstLane &IL) {
                  Value *FrontV = Item.front().first->get();
                  Use *U = IL.first;
                  return !U || (cast<Instruction>(U->get())->getOperand(Op) ==
                                cast<Instruction>(FrontV)->getOperand(Op));
                }))
              return false;
            continue;
          }
          Worklist.push_back(generateInstLaneVectorFromOperand(Item, Op));
        }
        continue;
      }
    }

    if (isFreeConcat(Item, CostKind, TTI)) {
      ConcatLeafs.insert(FrontU);
      continue;
    }

    return false;
  }

  if (NumVisited <= 1)
    return false;

  LLVM_DEBUG(dbgs() << "Found a superfluous identity shuffle: " << I << "\n");

  // If we got this far, we know the shuffles are superfluous and can be
  // removed. Scan through again and generate the new tree of instructions.
  Builder.SetInsertPoint(&I);
  Value *V = generateNewInstTree(Start, Ty, IdentityLeafs, SplatLeafs,
                                 ConcatLeafs, Builder, &TTI);
  replaceValue(I, *V);
  return true;
}

/// Given a commutative reduction, the order of the input lanes does not alter
/// the results. We can use this to remove certain shuffles feeding the
/// reduction, removing the need to shuffle at all.
bool VectorCombine::foldShuffleFromReductions(Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_mul:
  case Intrinsic::vector_reduce_and:
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_xor:
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_smax:
  case Intrinsic::vector_reduce_umin:
  case Intrinsic::vector_reduce_umax:
    break;
  default:
    return false;
  }

  // Find all the inputs when looking through operations that do not alter the
  // lane order (binops, for example). Currently we look for a single shuffle,
  // and can ignore splat values.
  std::queue<Value *> Worklist;
  SmallPtrSet<Value *, 4> Visited;
  ShuffleVectorInst *Shuffle = nullptr;
  if (auto *Op = dyn_cast<Instruction>(I.getOperand(0)))
    Worklist.push(Op);

  while (!Worklist.empty()) {
    Value *CV = Worklist.front();
    Worklist.pop();
    if (Visited.contains(CV))
      continue;

    // Splats don't change the order, so can be safely ignored.
    if (isSplatValue(CV))
      continue;

    Visited.insert(CV);

    if (auto *CI = dyn_cast<Instruction>(CV)) {
      if (CI->isBinaryOp()) {
        for (auto *Op : CI->operand_values())
          Worklist.push(Op);
        continue;
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(CI)) {
        if (Shuffle && Shuffle != SV)
          return false;
        Shuffle = SV;
        continue;
      }
    }

    // Anything else is currently an unknown node.
    return false;
  }

  if (!Shuffle)
    return false;

  // Check all uses of the binary ops and shuffles are also included in the
  // lane-invariant operations (Visited should be the list of lanewise
  // instructions, including the shuffle that we found).
  for (auto *V : Visited)
    for (auto *U : V->users())
      if (!Visited.contains(U) && U != &I)
        return false;

  FixedVectorType *VecType =
      dyn_cast<FixedVectorType>(II->getOperand(0)->getType());
  if (!VecType)
    return false;
  FixedVectorType *ShuffleInputType =
      dyn_cast<FixedVectorType>(Shuffle->getOperand(0)->getType());
  if (!ShuffleInputType)
    return false;
  unsigned NumInputElts = ShuffleInputType->getNumElements();

  // Find the mask from sorting the lanes into order. This is most likely to
  // become a identity or concat mask. Undef elements are pushed to the end.
  SmallVector<int> ConcatMask;
  Shuffle->getShuffleMask(ConcatMask);
  sort(ConcatMask, [](int X, int Y) { return (unsigned)X < (unsigned)Y; });
  bool UsesSecondVec =
      any_of(ConcatMask, [&](int M) { return M >= (int)NumInputElts; });

  InstructionCost OldCost = TTI.getShuffleCost(
      UsesSecondVec ? TTI::SK_PermuteTwoSrc : TTI::SK_PermuteSingleSrc, VecType,
      ShuffleInputType, Shuffle->getShuffleMask(), CostKind);
  InstructionCost NewCost = TTI.getShuffleCost(
      UsesSecondVec ? TTI::SK_PermuteTwoSrc : TTI::SK_PermuteSingleSrc, VecType,
      ShuffleInputType, ConcatMask, CostKind);

  LLVM_DEBUG(dbgs() << "Found a reduction feeding from a shuffle: " << *Shuffle
                    << "\n");
  LLVM_DEBUG(dbgs() << "  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");
  bool MadeChanges = false;
  if (NewCost < OldCost) {
    Builder.SetInsertPoint(Shuffle);
    Value *NewShuffle = Builder.CreateShuffleVector(
        Shuffle->getOperand(0), Shuffle->getOperand(1), ConcatMask);
    LLVM_DEBUG(dbgs() << "Created new shuffle: " << *NewShuffle << "\n");
    replaceValue(*Shuffle, *NewShuffle);
    return true;
  }

  // See if we can re-use foldSelectShuffle, getting it to reduce the size of
  // the shuffle into a nicer order, as it can ignore the order of the shuffles.
  MadeChanges |= foldSelectShuffle(*Shuffle, true);
  return MadeChanges;
}

/// For a given chain of patterns of the following form:
///
/// ```
///   %1 = shufflevector <n x ty1> %0, <n x ty1> poison <n x ty2> mask
///
///   %2 = tail call <n x ty1> llvm.<umin/umax/smin/smax>(<n x ty1> %0, <n x
///   ty1> %1)
///     OR
///   %2 = add/mul/or/and/xor <n x ty1> %0, %1
///
///   %3 = shufflevector <n x ty1> %2, <n x ty1> poison <n x ty2> mask
///   ...
///   ...
///   %(i - 1) = tail call <n x ty1> llvm.<umin/umax/smin/smax>(<n x ty1> %(i -
///   3), <n x ty1> %(i - 2)
///     OR
///   %(i - 1) = add/mul/or/and/xor <n x ty1> %(i - 3), %(i - 2)
///
///   %(i) = extractelement <n x ty1> %(i - 1), 0
/// ```
///
/// Where:
///    `mask` follows a partition pattern:
///
/// Ex:
///    [n = 8, p = poison]
///
///    4 5 6 7 | p p p p
///    2 3 | p p p p p p
///    1 | p p p p p p p
///
///    For powers of 2, there's a consistent pattern, but for other cases
///    the parity of the current half value at each step decides the
///    next partition half (see `ExpectedParityMask` for more logical details
///    in generalising this).
///
/// Ex:
///    [n = 6]
///
///    3 4 5 | p p p
///    1 2 | p p p p
///    1 | p p p p p
bool VectorCombine::foldShuffleChainsToReduce(Instruction &I) {
  // Going bottom-up for the pattern.
  std::queue<Value *> InstWorklist;
  InstructionCost OrigCost = 0;

  // Common instruction operation after each shuffle op.
  std::optional<unsigned int> CommonCallOp = std::nullopt;
  std::optional<Instruction::BinaryOps> CommonBinOp = std::nullopt;

  bool IsFirstCallOrBinInst = true;
  bool ShouldBeCallOrBinInst = true;

  // This stores the last used instructions for shuffle/common op.
  //
  // PrevVecV[0] / PrevVecV[1] store the last two simultaneous
  // instructions from either shuffle/common op.
  SmallVector<Value *, 2> PrevVecV(2, nullptr);

  Value *VecOpEE;
  if (!match(&I, m_ExtractElt(m_Value(VecOpEE), m_Zero())))
    return false;

  auto *FVT = dyn_cast<FixedVectorType>(VecOpEE->getType());
  if (!FVT)
    return false;

  int64_t VecSize = FVT->getNumElements();
  if (VecSize < 2)
    return false;

  // Number of levels would be ~log2(n), considering we always partition
  // by half for this fold pattern.
  unsigned int NumLevels = Log2_64_Ceil(VecSize), VisitedCnt = 0;
  int64_t ShuffleMaskHalf = 1, ExpectedParityMask = 0;

  // This is how we generalise for all element sizes.
  // At each step, if vector size is odd, we need non-poison
  // values to cover the dominant half so we don't miss out on any element.
  //
  // This mask will help us retrieve this as we go from bottom to top:
  //
  // Mask Set -> N = N * 2 - 1
  // Mask Unset -> N = N * 2
  for (int Cur = VecSize, Mask = NumLevels - 1; Cur > 1;
       Cur = (Cur + 1) / 2, --Mask) {
    if (Cur & 1)
      ExpectedParityMask |= (1ll << Mask);
  }

  InstWorklist.push(VecOpEE);

  while (!InstWorklist.empty()) {
    Value *CI = InstWorklist.front();
    InstWorklist.pop();

    if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
      if (!ShouldBeCallOrBinInst)
        return false;

      if (!IsFirstCallOrBinInst &&
          any_of(PrevVecV, [](Value *VecV) { return VecV == nullptr; }))
        return false;

      // For the first found call/bin op, the vector has to come from the
      // extract element op.
      if (II != (IsFirstCallOrBinInst ? VecOpEE : PrevVecV[0]))
        return false;
      IsFirstCallOrBinInst = false;

      if (!CommonCallOp)
        CommonCallOp = II->getIntrinsicID();
      if (II->getIntrinsicID() != *CommonCallOp)
        return false;

      switch (II->getIntrinsicID()) {
      case Intrinsic::umin:
      case Intrinsic::umax:
      case Intrinsic::smin:
      case Intrinsic::smax: {
        auto *Op0 = II->getOperand(0);
        auto *Op1 = II->getOperand(1);
        PrevVecV[0] = Op0;
        PrevVecV[1] = Op1;
        break;
      }
      default:
        return false;
      }
      ShouldBeCallOrBinInst ^= 1;

      IntrinsicCostAttributes ICA(
          *CommonCallOp, II->getType(),
          {PrevVecV[0]->getType(), PrevVecV[1]->getType()});
      OrigCost += TTI.getIntrinsicInstrCost(ICA, CostKind);

      // We may need a swap here since it can be (a, b) or (b, a)
      // and accordingly change as we go up.
      if (!isa<ShuffleVectorInst>(PrevVecV[1]))
        std::swap(PrevVecV[0], PrevVecV[1]);
      InstWorklist.push(PrevVecV[1]);
      InstWorklist.push(PrevVecV[0]);
    } else if (auto *BinOp = dyn_cast<BinaryOperator>(CI)) {
      // Similar logic for bin ops.

      if (!ShouldBeCallOrBinInst)
        return false;

      if (!IsFirstCallOrBinInst &&
          any_of(PrevVecV, [](Value *VecV) { return VecV == nullptr; }))
        return false;

      if (BinOp != (IsFirstCallOrBinInst ? VecOpEE : PrevVecV[0]))
        return false;
      IsFirstCallOrBinInst = false;

      if (!CommonBinOp)
        CommonBinOp = BinOp->getOpcode();

      if (BinOp->getOpcode() != *CommonBinOp)
        return false;

      switch (*CommonBinOp) {
      case BinaryOperator::Add:
      case BinaryOperator::Mul:
      case BinaryOperator::Or:
      case BinaryOperator::And:
      case BinaryOperator::Xor: {
        auto *Op0 = BinOp->getOperand(0);
        auto *Op1 = BinOp->getOperand(1);
        PrevVecV[0] = Op0;
        PrevVecV[1] = Op1;
        break;
      }
      default:
        return false;
      }
      ShouldBeCallOrBinInst ^= 1;

      OrigCost +=
          TTI.getArithmeticInstrCost(*CommonBinOp, BinOp->getType(), CostKind);

      if (!isa<ShuffleVectorInst>(PrevVecV[1]))
        std::swap(PrevVecV[0], PrevVecV[1]);
      InstWorklist.push(PrevVecV[1]);
      InstWorklist.push(PrevVecV[0]);
    } else if (auto *SVInst = dyn_cast<ShuffleVectorInst>(CI)) {
      // We shouldn't have any null values in the previous vectors,
      // is so, there was a mismatch in pattern.
      if (ShouldBeCallOrBinInst ||
          any_of(PrevVecV, [](Value *VecV) { return VecV == nullptr; }))
        return false;

      if (SVInst != PrevVecV[1])
        return false;

      ArrayRef<int> CurMask;
      if (!match(SVInst, m_Shuffle(m_Specific(PrevVecV[0]), m_Poison(),
                                   m_Mask(CurMask))))
        return false;

      // Subtract the parity mask when checking the condition.
      for (int Mask = 0, MaskSize = CurMask.size(); Mask != MaskSize; ++Mask) {
        if (Mask < ShuffleMaskHalf &&
            CurMask[Mask] != ShuffleMaskHalf + Mask - (ExpectedParityMask & 1))
          return false;
        if (Mask >= ShuffleMaskHalf && CurMask[Mask] != -1)
          return false;
      }

      // Update mask values.
      ShuffleMaskHalf *= 2;
      ShuffleMaskHalf -= (ExpectedParityMask & 1);
      ExpectedParityMask >>= 1;

      OrigCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                     SVInst->getType(), SVInst->getType(),
                                     CurMask, CostKind);

      VisitedCnt += 1;
      if (!ExpectedParityMask && VisitedCnt == NumLevels)
        break;

      ShouldBeCallOrBinInst ^= 1;
    } else {
      return false;
    }
  }

  // Pattern should end with a shuffle op.
  if (ShouldBeCallOrBinInst)
    return false;

  assert(VecSize != -1 && "Expected Match for Vector Size");

  Value *FinalVecV = PrevVecV[0];
  if (!FinalVecV)
    return false;

  auto *FinalVecVTy = cast<FixedVectorType>(FinalVecV->getType());

  Intrinsic::ID ReducedOp =
      (CommonCallOp ? getMinMaxReductionIntrinsicID(*CommonCallOp)
                    : getReductionForBinop(*CommonBinOp));
  if (!ReducedOp)
    return false;

  IntrinsicCostAttributes ICA(ReducedOp, FinalVecVTy, {FinalVecV});
  InstructionCost NewCost = TTI.getIntrinsicInstrCost(ICA, CostKind);

  if (NewCost >= OrigCost)
    return false;

  auto *ReducedResult =
      Builder.CreateIntrinsic(ReducedOp, {FinalVecV->getType()}, {FinalVecV});
  replaceValue(I, *ReducedResult);

  return true;
}

/// Determine if its more efficient to fold:
///   reduce(trunc(x)) -> trunc(reduce(x)).
///   reduce(sext(x))  -> sext(reduce(x)).
///   reduce(zext(x))  -> zext(reduce(x)).
bool VectorCombine::foldCastFromReductions(Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;

  bool TruncOnly = false;
  Intrinsic::ID IID = II->getIntrinsicID();
  switch (IID) {
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_mul:
    TruncOnly = true;
    break;
  case Intrinsic::vector_reduce_and:
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_xor:
    break;
  default:
    return false;
  }

  unsigned ReductionOpc = getArithmeticReductionInstruction(IID);
  Value *ReductionSrc = I.getOperand(0);

  Value *Src;
  if (!match(ReductionSrc, m_OneUse(m_Trunc(m_Value(Src)))) &&
      (TruncOnly || !match(ReductionSrc, m_OneUse(m_ZExtOrSExt(m_Value(Src))))))
    return false;

  auto CastOpc =
      (Instruction::CastOps)cast<Instruction>(ReductionSrc)->getOpcode();

  auto *SrcTy = cast<VectorType>(Src->getType());
  auto *ReductionSrcTy = cast<VectorType>(ReductionSrc->getType());
  Type *ResultTy = I.getType();

  InstructionCost OldCost = TTI.getArithmeticReductionCost(
      ReductionOpc, ReductionSrcTy, std::nullopt, CostKind);
  OldCost += TTI.getCastInstrCost(CastOpc, ReductionSrcTy, SrcTy,
                                  TTI::CastContextHint::None, CostKind,
                                  cast<CastInst>(ReductionSrc));
  InstructionCost NewCost =
      TTI.getArithmeticReductionCost(ReductionOpc, SrcTy, std::nullopt,
                                     CostKind) +
      TTI.getCastInstrCost(CastOpc, ResultTy, ReductionSrcTy->getScalarType(),
                           TTI::CastContextHint::None, CostKind);

  if (OldCost <= NewCost || !NewCost.isValid())
    return false;

  Value *NewReduction = Builder.CreateIntrinsic(SrcTy->getScalarType(),
                                                II->getIntrinsicID(), {Src});
  Value *NewCast = Builder.CreateCast(CastOpc, NewReduction, ResultTy);
  replaceValue(I, *NewCast);
  return true;
}

/// Returns true if this ShuffleVectorInst eventually feeds into a
/// vector reduction intrinsic (e.g., vector_reduce_add) by only following
/// chains of shuffles and binary operators (in any combination/order).
/// The search does not go deeper than the given Depth.
static bool feedsIntoVectorReduction(ShuffleVectorInst *SVI) {
  constexpr unsigned MaxVisited = 32;
  SmallPtrSet<Instruction *, 8> Visited;
  SmallVector<Instruction *, 4> WorkList;
  bool FoundReduction = false;

  WorkList.push_back(SVI);
  while (!WorkList.empty()) {
    Instruction *I = WorkList.pop_back_val();
    for (User *U : I->users()) {
      auto *UI = cast<Instruction>(U);
      if (!UI || !Visited.insert(UI).second)
        continue;
      if (Visited.size() > MaxVisited)
        return false;
      if (auto *II = dyn_cast<IntrinsicInst>(UI)) {
        // More than one reduction reached
        if (FoundReduction)
          return false;
        switch (II->getIntrinsicID()) {
        case Intrinsic::vector_reduce_add:
        case Intrinsic::vector_reduce_mul:
        case Intrinsic::vector_reduce_and:
        case Intrinsic::vector_reduce_or:
        case Intrinsic::vector_reduce_xor:
        case Intrinsic::vector_reduce_smin:
        case Intrinsic::vector_reduce_smax:
        case Intrinsic::vector_reduce_umin:
        case Intrinsic::vector_reduce_umax:
          FoundReduction = true;
          continue;
        default:
          return false;
        }
      }

      if (!isa<BinaryOperator>(UI) && !isa<ShuffleVectorInst>(UI))
        return false;

      WorkList.emplace_back(UI);
    }
  }
  return FoundReduction;
}

/// This method looks for groups of shuffles acting on binops, of the form:
///  %x = shuffle ...
///  %y = shuffle ...
///  %a = binop %x, %y
///  %b = binop %x, %y
///  shuffle %a, %b, selectmask
/// We may, especially if the shuffle is wider than legal, be able to convert
/// the shuffle to a form where only parts of a and b need to be computed. On
/// architectures with no obvious "select" shuffle, this can reduce the total
/// number of operations if the target reports them as cheaper.
bool VectorCombine::foldSelectShuffle(Instruction &I, bool FromReduction) {
  auto *SVI = cast<ShuffleVectorInst>(&I);
  auto *VT = cast<FixedVectorType>(I.getType());
  auto *Op0 = dyn_cast<Instruction>(SVI->getOperand(0));
  auto *Op1 = dyn_cast<Instruction>(SVI->getOperand(1));
  if (!Op0 || !Op1 || Op0 == Op1 || !Op0->isBinaryOp() || !Op1->isBinaryOp() ||
      VT != Op0->getType())
    return false;

  auto *SVI0A = dyn_cast<Instruction>(Op0->getOperand(0));
  auto *SVI0B = dyn_cast<Instruction>(Op0->getOperand(1));
  auto *SVI1A = dyn_cast<Instruction>(Op1->getOperand(0));
  auto *SVI1B = dyn_cast<Instruction>(Op1->getOperand(1));
  SmallPtrSet<Instruction *, 4> InputShuffles({SVI0A, SVI0B, SVI1A, SVI1B});
  auto checkSVNonOpUses = [&](Instruction *I) {
    if (!I || I->getOperand(0)->getType() != VT)
      return true;
    return any_of(I->users(), [&](User *U) {
      return U != Op0 && U != Op1 &&
             !(isa<ShuffleVectorInst>(U) &&
               (InputShuffles.contains(cast<Instruction>(U)) ||
                isInstructionTriviallyDead(cast<Instruction>(U))));
    });
  };
  if (checkSVNonOpUses(SVI0A) || checkSVNonOpUses(SVI0B) ||
      checkSVNonOpUses(SVI1A) || checkSVNonOpUses(SVI1B))
    return false;

  // Collect all the uses that are shuffles that we can transform together. We
  // may not have a single shuffle, but a group that can all be transformed
  // together profitably.
  SmallVector<ShuffleVectorInst *> Shuffles;
  auto collectShuffles = [&](Instruction *I) {
    for (auto *U : I->users()) {
      auto *SV = dyn_cast<ShuffleVectorInst>(U);
      if (!SV || SV->getType() != VT)
        return false;
      if ((SV->getOperand(0) != Op0 && SV->getOperand(0) != Op1) ||
          (SV->getOperand(1) != Op0 && SV->getOperand(1) != Op1))
        return false;
      if (!llvm::is_contained(Shuffles, SV))
        Shuffles.push_back(SV);
    }
    return true;
  };
  if (!collectShuffles(Op0) || !collectShuffles(Op1))
    return false;
  // From a reduction, we need to be processing a single shuffle, otherwise the
  // other uses will not be lane-invariant.
  if (FromReduction && Shuffles.size() > 1)
    return false;

  // Add any shuffle uses for the shuffles we have found, to include them in our
  // cost calculations.
  if (!FromReduction) {
    for (ShuffleVectorInst *SV : Shuffles) {
      for (auto *U : SV->users()) {
        ShuffleVectorInst *SSV = dyn_cast<ShuffleVectorInst>(U);
        if (SSV && isa<UndefValue>(SSV->getOperand(1)) && SSV->getType() == VT)
          Shuffles.push_back(SSV);
      }
    }
  }

  // For each of the output shuffles, we try to sort all the first vector
  // elements to the beginning, followed by the second array elements at the
  // end. If the binops are legalized to smaller vectors, this may reduce total
  // number of binops. We compute the ReconstructMask mask needed to convert
  // back to the original lane order.
  SmallVector<std::pair<int, int>> V1, V2;
  SmallVector<SmallVector<int>> OrigReconstructMasks;
  int MaxV1Elt = 0, MaxV2Elt = 0;
  unsigned NumElts = VT->getNumElements();
  for (ShuffleVectorInst *SVN : Shuffles) {
    SmallVector<int> Mask;
    SVN->getShuffleMask(Mask);

    // Check the operands are the same as the original, or reversed (in which
    // case we need to commute the mask).
    Value *SVOp0 = SVN->getOperand(0);
    Value *SVOp1 = SVN->getOperand(1);
    if (isa<UndefValue>(SVOp1)) {
      auto *SSV = cast<ShuffleVectorInst>(SVOp0);
      SVOp0 = SSV->getOperand(0);
      SVOp1 = SSV->getOperand(1);
      for (int &Elem : Mask) {
        if (Elem >= static_cast<int>(SSV->getShuffleMask().size()))
          return false;
        Elem = Elem < 0 ? Elem : SSV->getMaskValue(Elem);
      }
    }
    if (SVOp0 == Op1 && SVOp1 == Op0) {
      std::swap(SVOp0, SVOp1);
      ShuffleVectorInst::commuteShuffleMask(Mask, NumElts);
    }
    if (SVOp0 != Op0 || SVOp1 != Op1)
      return false;

    // Calculate the reconstruction mask for this shuffle, as the mask needed to
    // take the packed values from Op0/Op1 and reconstructing to the original
    // order.
    SmallVector<int> ReconstructMask;
    for (unsigned I = 0; I < Mask.size(); I++) {
      if (Mask[I] < 0) {
        ReconstructMask.push_back(-1);
      } else if (Mask[I] < static_cast<int>(NumElts)) {
        MaxV1Elt = std::max(MaxV1Elt, Mask[I]);
        auto It = find_if(V1, [&](const std::pair<int, int> &A) {
          return Mask[I] == A.first;
        });
        if (It != V1.end())
          ReconstructMask.push_back(It - V1.begin());
        else {
          ReconstructMask.push_back(V1.size());
          V1.emplace_back(Mask[I], V1.size());
        }
      } else {
        MaxV2Elt = std::max<int>(MaxV2Elt, Mask[I] - NumElts);
        auto It = find_if(V2, [&](const std::pair<int, int> &A) {
          return Mask[I] - static_cast<int>(NumElts) == A.first;
        });
        if (It != V2.end())
          ReconstructMask.push_back(NumElts + It - V2.begin());
        else {
          ReconstructMask.push_back(NumElts + V2.size());
          V2.emplace_back(Mask[I] - NumElts, NumElts + V2.size());
        }
      }
    }

    // For reductions, we know that the lane ordering out doesn't alter the
    // result. In-order can help simplify the shuffle away.
    if (FromReduction)
      sort(ReconstructMask);
    OrigReconstructMasks.push_back(std::move(ReconstructMask));
  }

  // If the Maximum element used from V1 and V2 are not larger than the new
  // vectors, the vectors are already packes and performing the optimization
  // again will likely not help any further. This also prevents us from getting
  // stuck in a cycle in case the costs do not also rule it out.
  if (V1.empty() || V2.empty() ||
      (MaxV1Elt == static_cast<int>(V1.size()) - 1 &&
       MaxV2Elt == static_cast<int>(V2.size()) - 1))
    return false;

  // GetBaseMaskValue takes one of the inputs, which may either be a shuffle, a
  // shuffle of another shuffle, or not a shuffle (that is treated like a
  // identity shuffle).
  auto GetBaseMaskValue = [&](Instruction *I, int M) {
    auto *SV = dyn_cast<ShuffleVectorInst>(I);
    if (!SV)
      return M;
    if (isa<UndefValue>(SV->getOperand(1)))
      if (auto *SSV = dyn_cast<ShuffleVectorInst>(SV->getOperand(0)))
        if (InputShuffles.contains(SSV))
          return SSV->getMaskValue(SV->getMaskValue(M));
    return SV->getMaskValue(M);
  };

  // Attempt to sort the inputs my ascending mask values to make simpler input
  // shuffles and push complex shuffles down to the uses. We sort on the first
  // of the two input shuffle orders, to try and get at least one input into a
  // nice order.
  auto SortBase = [&](Instruction *A, std::pair<int, int> X,
                      std::pair<int, int> Y) {
    int MXA = GetBaseMaskValue(A, X.first);
    int MYA = GetBaseMaskValue(A, Y.first);
    return MXA < MYA;
  };
  stable_sort(V1, [&](std::pair<int, int> A, std::pair<int, int> B) {
    return SortBase(SVI0A, A, B);
  });
  stable_sort(V2, [&](std::pair<int, int> A, std::pair<int, int> B) {
    return SortBase(SVI1A, A, B);
  });
  // Calculate our ReconstructMasks from the OrigReconstructMasks and the
  // modified order of the input shuffles.
  SmallVector<SmallVector<int>> ReconstructMasks;
  for (const auto &Mask : OrigReconstructMasks) {
    SmallVector<int> ReconstructMask;
    for (int M : Mask) {
      auto FindIndex = [](const SmallVector<std::pair<int, int>> &V, int M) {
        auto It = find_if(V, [M](auto A) { return A.second == M; });
        assert(It != V.end() && "Expected all entries in Mask");
        return std::distance(V.begin(), It);
      };
      if (M < 0)
        ReconstructMask.push_back(-1);
      else if (M < static_cast<int>(NumElts)) {
        ReconstructMask.push_back(FindIndex(V1, M));
      } else {
        ReconstructMask.push_back(NumElts + FindIndex(V2, M));
      }
    }
    ReconstructMasks.push_back(std::move(ReconstructMask));
  }

  // Calculate the masks needed for the new input shuffles, which get padded
  // with undef
  SmallVector<int> V1A, V1B, V2A, V2B;
  for (unsigned I = 0; I < V1.size(); I++) {
    V1A.push_back(GetBaseMaskValue(SVI0A, V1[I].first));
    V1B.push_back(GetBaseMaskValue(SVI0B, V1[I].first));
  }
  for (unsigned I = 0; I < V2.size(); I++) {
    V2A.push_back(GetBaseMaskValue(SVI1A, V2[I].first));
    V2B.push_back(GetBaseMaskValue(SVI1B, V2[I].first));
  }
  while (V1A.size() < NumElts) {
    V1A.push_back(PoisonMaskElem);
    V1B.push_back(PoisonMaskElem);
  }
  while (V2A.size() < NumElts) {
    V2A.push_back(PoisonMaskElem);
    V2B.push_back(PoisonMaskElem);
  }

  auto AddShuffleCost = [&](InstructionCost C, Instruction *I) {
    auto *SV = dyn_cast<ShuffleVectorInst>(I);
    if (!SV)
      return C;
    return C + TTI.getShuffleCost(isa<UndefValue>(SV->getOperand(1))
                                      ? TTI::SK_PermuteSingleSrc
                                      : TTI::SK_PermuteTwoSrc,
                                  VT, VT, SV->getShuffleMask(), CostKind);
  };
  auto AddShuffleMaskCost = [&](InstructionCost C, ArrayRef<int> Mask) {
    return C +
           TTI.getShuffleCost(TTI::SK_PermuteTwoSrc, VT, VT, Mask, CostKind);
  };

  unsigned ElementSize = VT->getElementType()->getPrimitiveSizeInBits();
  unsigned MaxVectorSize =
      TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector);
  unsigned MaxElementsInVector = MaxVectorSize / ElementSize;
  // When there are multiple shufflevector operations on the same input,
  // especially when the vector length is larger than the register size,
  // identical shuffle patterns may occur across different groups of elements.
  // To avoid overestimating the cost by counting these repeated shuffles more
  // than once, we only account for unique shuffle patterns. This adjustment
  // prevents inflated costs in the cost model for wide vectors split into
  // several register-sized groups.
  std::set<SmallVector<int, 4>> UniqueShuffles;
  auto AddShuffleMaskAdjustedCost = [&](InstructionCost C, ArrayRef<int> Mask) {
    // Compute the cost for performing the shuffle over the full vector.
    auto ShuffleCost =
        TTI.getShuffleCost(TTI::SK_PermuteTwoSrc, VT, VT, Mask, CostKind);
    unsigned NumFullVectors = Mask.size() / MaxElementsInVector;
    if (NumFullVectors < 2)
      return C + ShuffleCost;
    SmallVector<int, 4> SubShuffle(MaxElementsInVector);
    unsigned NumUniqueGroups = 0;
    unsigned NumGroups = Mask.size() / MaxElementsInVector;
    // For each group of MaxElementsInVector contiguous elements,
    // collect their shuffle pattern and insert into the set of unique patterns.
    for (unsigned I = 0; I < NumFullVectors; ++I) {
      for (unsigned J = 0; J < MaxElementsInVector; ++J)
        SubShuffle[J] = Mask[MaxElementsInVector * I + J];
      if (UniqueShuffles.insert(SubShuffle).second)
        NumUniqueGroups += 1;
    }
    return C + ShuffleCost * NumUniqueGroups / NumGroups;
  };
  auto AddShuffleAdjustedCost = [&](InstructionCost C, Instruction *I) {
    auto *SV = dyn_cast<ShuffleVectorInst>(I);
    if (!SV)
      return C;
    SmallVector<int, 16> Mask;
    SV->getShuffleMask(Mask);
    return AddShuffleMaskAdjustedCost(C, Mask);
  };
  // Check that input consists of ShuffleVectors applied to the same input
  auto AllShufflesHaveSameOperands =
      [](SmallPtrSetImpl<Instruction *> &InputShuffles) {
        if (InputShuffles.size() < 2)
          return false;
        ShuffleVectorInst *FirstSV =
            dyn_cast<ShuffleVectorInst>(*InputShuffles.begin());
        if (!FirstSV)
          return false;

        Value *In0 = FirstSV->getOperand(0), *In1 = FirstSV->getOperand(1);
        return std::all_of(
            std::next(InputShuffles.begin()), InputShuffles.end(),
            [&](Instruction *I) {
              ShuffleVectorInst *SV = dyn_cast<ShuffleVectorInst>(I);
              return SV && SV->getOperand(0) == In0 && SV->getOperand(1) == In1;
            });
      };

  // Get the costs of the shuffles + binops before and after with the new
  // shuffle masks.
  InstructionCost CostBefore =
      TTI.getArithmeticInstrCost(Op0->getOpcode(), VT, CostKind) +
      TTI.getArithmeticInstrCost(Op1->getOpcode(), VT, CostKind);
  CostBefore += std::accumulate(Shuffles.begin(), Shuffles.end(),
                                InstructionCost(0), AddShuffleCost);
  if (AllShufflesHaveSameOperands(InputShuffles)) {
    UniqueShuffles.clear();
    CostBefore += std::accumulate(InputShuffles.begin(), InputShuffles.end(),
                                  InstructionCost(0), AddShuffleAdjustedCost);
  } else {
    CostBefore += std::accumulate(InputShuffles.begin(), InputShuffles.end(),
                                  InstructionCost(0), AddShuffleCost);
  }

  // The new binops will be unused for lanes past the used shuffle lengths.
  // These types attempt to get the correct cost for that from the target.
  FixedVectorType *Op0SmallVT =
      FixedVectorType::get(VT->getScalarType(), V1.size());
  FixedVectorType *Op1SmallVT =
      FixedVectorType::get(VT->getScalarType(), V2.size());
  InstructionCost CostAfter =
      TTI.getArithmeticInstrCost(Op0->getOpcode(), Op0SmallVT, CostKind) +
      TTI.getArithmeticInstrCost(Op1->getOpcode(), Op1SmallVT, CostKind);
  UniqueShuffles.clear();
  CostAfter += std::accumulate(ReconstructMasks.begin(), ReconstructMasks.end(),
                               InstructionCost(0), AddShuffleMaskAdjustedCost);
  std::set<SmallVector<int>> OutputShuffleMasks({V1A, V1B, V2A, V2B});
  CostAfter +=
      std::accumulate(OutputShuffleMasks.begin(), OutputShuffleMasks.end(),
                      InstructionCost(0), AddShuffleMaskCost);

  LLVM_DEBUG(dbgs() << "Found a binop select shuffle pattern: " << I << "\n");
  LLVM_DEBUG(dbgs() << "  CostBefore: " << CostBefore
                    << " vs CostAfter: " << CostAfter << "\n");
  if (CostBefore < CostAfter ||
      (CostBefore == CostAfter && !feedsIntoVectorReduction(SVI)))
    return false;

  // The cost model has passed, create the new instructions.
  auto GetShuffleOperand = [&](Instruction *I, unsigned Op) -> Value * {
    auto *SV = dyn_cast<ShuffleVectorInst>(I);
    if (!SV)
      return I;
    if (isa<UndefValue>(SV->getOperand(1)))
      if (auto *SSV = dyn_cast<ShuffleVectorInst>(SV->getOperand(0)))
        if (InputShuffles.contains(SSV))
          return SSV->getOperand(Op);
    return SV->getOperand(Op);
  };
  Builder.SetInsertPoint(*SVI0A->getInsertionPointAfterDef());
  Value *NSV0A = Builder.CreateShuffleVector(GetShuffleOperand(SVI0A, 0),
                                             GetShuffleOperand(SVI0A, 1), V1A);
  Builder.SetInsertPoint(*SVI0B->getInsertionPointAfterDef());
  Value *NSV0B = Builder.CreateShuffleVector(GetShuffleOperand(SVI0B, 0),
                                             GetShuffleOperand(SVI0B, 1), V1B);
  Builder.SetInsertPoint(*SVI1A->getInsertionPointAfterDef());
  Value *NSV1A = Builder.CreateShuffleVector(GetShuffleOperand(SVI1A, 0),
                                             GetShuffleOperand(SVI1A, 1), V2A);
  Builder.SetInsertPoint(*SVI1B->getInsertionPointAfterDef());
  Value *NSV1B = Builder.CreateShuffleVector(GetShuffleOperand(SVI1B, 0),
                                             GetShuffleOperand(SVI1B, 1), V2B);
  Builder.SetInsertPoint(Op0);
  Value *NOp0 = Builder.CreateBinOp((Instruction::BinaryOps)Op0->getOpcode(),
                                    NSV0A, NSV0B);
  if (auto *I = dyn_cast<Instruction>(NOp0))
    I->copyIRFlags(Op0, true);
  Builder.SetInsertPoint(Op1);
  Value *NOp1 = Builder.CreateBinOp((Instruction::BinaryOps)Op1->getOpcode(),
                                    NSV1A, NSV1B);
  if (auto *I = dyn_cast<Instruction>(NOp1))
    I->copyIRFlags(Op1, true);

  for (int S = 0, E = ReconstructMasks.size(); S != E; S++) {
    Builder.SetInsertPoint(Shuffles[S]);
    Value *NSV = Builder.CreateShuffleVector(NOp0, NOp1, ReconstructMasks[S]);
    replaceValue(*Shuffles[S], *NSV, false);
  }

  Worklist.pushValue(NSV0A);
  Worklist.pushValue(NSV0B);
  Worklist.pushValue(NSV1A);
  Worklist.pushValue(NSV1B);
  return true;
}

/// Check if instruction depends on ZExt and this ZExt can be moved after the
/// instruction. Move ZExt if it is profitable. For example:
///     logic(zext(x),y) -> zext(logic(x,trunc(y)))
///     lshr((zext(x),y) -> zext(lshr(x,trunc(y)))
/// Cost model calculations takes into account if zext(x) has other users and
/// whether it can be propagated through them too.
bool VectorCombine::shrinkType(Instruction &I) {
  Value *ZExted, *OtherOperand;
  if (!match(&I, m_c_BitwiseLogic(m_ZExt(m_Value(ZExted)),
                                  m_Value(OtherOperand))) &&
      !match(&I, m_LShr(m_ZExt(m_Value(ZExted)), m_Value(OtherOperand))))
    return false;

  Value *ZExtOperand = I.getOperand(I.getOperand(0) == OtherOperand ? 1 : 0);

  auto *BigTy = cast<FixedVectorType>(I.getType());
  auto *SmallTy = cast<FixedVectorType>(ZExted->getType());
  unsigned BW = SmallTy->getElementType()->getPrimitiveSizeInBits();

  if (I.getOpcode() == Instruction::LShr) {
    // Check that the shift amount is less than the number of bits in the
    // smaller type. Otherwise, the smaller lshr will return a poison value.
    KnownBits ShAmtKB = computeKnownBits(I.getOperand(1), *DL);
    if (ShAmtKB.getMaxValue().uge(BW))
      return false;
  } else {
    // Check that the expression overall uses at most the same number of bits as
    // ZExted
    KnownBits KB = computeKnownBits(&I, *DL);
    if (KB.countMaxActiveBits() > BW)
      return false;
  }

  // Calculate costs of leaving current IR as it is and moving ZExt operation
  // later, along with adding truncates if needed
  InstructionCost ZExtCost = TTI.getCastInstrCost(
      Instruction::ZExt, BigTy, SmallTy,
      TargetTransformInfo::CastContextHint::None, CostKind);
  InstructionCost CurrentCost = ZExtCost;
  InstructionCost ShrinkCost = 0;

  // Calculate total cost and check that we can propagate through all ZExt users
  for (User *U : ZExtOperand->users()) {
    auto *UI = cast<Instruction>(U);
    if (UI == &I) {
      CurrentCost +=
          TTI.getArithmeticInstrCost(UI->getOpcode(), BigTy, CostKind);
      ShrinkCost +=
          TTI.getArithmeticInstrCost(UI->getOpcode(), SmallTy, CostKind);
      ShrinkCost += ZExtCost;
      continue;
    }

    if (!Instruction::isBinaryOp(UI->getOpcode()))
      return false;

    // Check if we can propagate ZExt through its other users
    KnownBits KB = computeKnownBits(UI, *DL);
    if (KB.countMaxActiveBits() > BW)
      return false;

    CurrentCost += TTI.getArithmeticInstrCost(UI->getOpcode(), BigTy, CostKind);
    ShrinkCost +=
        TTI.getArithmeticInstrCost(UI->getOpcode(), SmallTy, CostKind);
    ShrinkCost += ZExtCost;
  }

  // If the other instruction operand is not a constant, we'll need to
  // generate a truncate instruction. So we have to adjust cost
  if (!isa<Constant>(OtherOperand))
    ShrinkCost += TTI.getCastInstrCost(
        Instruction::Trunc, SmallTy, BigTy,
        TargetTransformInfo::CastContextHint::None, CostKind);

  // If the cost of shrinking types and leaving the IR is the same, we'll lean
  // towards modifying the IR because shrinking opens opportunities for other
  // shrinking optimisations.
  if (ShrinkCost > CurrentCost)
    return false;

  Builder.SetInsertPoint(&I);
  Value *Op0 = ZExted;
  Value *Op1 = Builder.CreateTrunc(OtherOperand, SmallTy);
  // Keep the order of operands the same
  if (I.getOperand(0) == OtherOperand)
    std::swap(Op0, Op1);
  Value *NewBinOp =
      Builder.CreateBinOp((Instruction::BinaryOps)I.getOpcode(), Op0, Op1);
  cast<Instruction>(NewBinOp)->copyIRFlags(&I);
  cast<Instruction>(NewBinOp)->copyMetadata(I);
  Value *NewZExtr = Builder.CreateZExt(NewBinOp, BigTy);
  replaceValue(I, *NewZExtr);
  return true;
}

/// insert (DstVec, (extract SrcVec, ExtIdx), InsIdx) -->
/// shuffle (DstVec, SrcVec, Mask)
bool VectorCombine::foldInsExtVectorToShuffle(Instruction &I) {
  Value *DstVec, *SrcVec;
  uint64_t ExtIdx, InsIdx;
  if (!match(&I,
             m_InsertElt(m_Value(DstVec),
                         m_ExtractElt(m_Value(SrcVec), m_ConstantInt(ExtIdx)),
                         m_ConstantInt(InsIdx))))
    return false;

  auto *DstVecTy = dyn_cast<FixedVectorType>(I.getType());
  auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcVec->getType());
  // We can try combining vectors with different element sizes.
  if (!DstVecTy || !SrcVecTy ||
      SrcVecTy->getElementType() != DstVecTy->getElementType())
    return false;

  unsigned NumDstElts = DstVecTy->getNumElements();
  unsigned NumSrcElts = SrcVecTy->getNumElements();
  if (InsIdx >= NumDstElts || ExtIdx >= NumSrcElts || NumDstElts == 1)
    return false;

  // Insertion into poison is a cheaper single operand shuffle.
  TargetTransformInfo::ShuffleKind SK;
  SmallVector<int> Mask(NumDstElts, PoisonMaskElem);

  bool NeedExpOrNarrow = NumSrcElts != NumDstElts;
  bool IsExtIdxInBounds = ExtIdx < NumDstElts;
  bool NeedDstSrcSwap = isa<PoisonValue>(DstVec) && !isa<UndefValue>(SrcVec);
  if (NeedDstSrcSwap) {
    SK = TargetTransformInfo::SK_PermuteSingleSrc;
    if (!IsExtIdxInBounds && NeedExpOrNarrow)
      Mask[InsIdx] = 0;
    else
      Mask[InsIdx] = ExtIdx;
    std::swap(DstVec, SrcVec);
  } else {
    SK = TargetTransformInfo::SK_PermuteTwoSrc;
    std::iota(Mask.begin(), Mask.end(), 0);
    if (!IsExtIdxInBounds && NeedExpOrNarrow)
      Mask[InsIdx] = NumDstElts;
    else
      Mask[InsIdx] = ExtIdx + NumDstElts;
  }

  // Cost
  auto *Ins = cast<InsertElementInst>(&I);
  auto *Ext = cast<ExtractElementInst>(I.getOperand(1));
  InstructionCost InsCost =
      TTI.getVectorInstrCost(*Ins, DstVecTy, CostKind, InsIdx);
  InstructionCost ExtCost =
      TTI.getVectorInstrCost(*Ext, DstVecTy, CostKind, ExtIdx);
  InstructionCost OldCost = ExtCost + InsCost;

  InstructionCost NewCost = 0;
  SmallVector<int> ExtToVecMask;
  if (!NeedExpOrNarrow) {
    // Ignore 'free' identity insertion shuffle.
    // TODO: getShuffleCost should return TCC_Free for Identity shuffles.
    if (!ShuffleVectorInst::isIdentityMask(Mask, NumSrcElts))
      NewCost += TTI.getShuffleCost(SK, DstVecTy, DstVecTy, Mask, CostKind, 0,
                                    nullptr, {DstVec, SrcVec});
  } else {
    // When creating length-changing-vector, always create with a Mask whose
    // first element has an ExtIdx, so that the first element of the vector
    // being created is always the target to be extracted.
    ExtToVecMask.assign(NumDstElts, PoisonMaskElem);
    if (IsExtIdxInBounds)
      ExtToVecMask[ExtIdx] = ExtIdx;
    else
      ExtToVecMask[0] = ExtIdx;
    // Add cost for expanding or narrowing
    NewCost = TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc,
                                 DstVecTy, SrcVecTy, ExtToVecMask, CostKind);
    NewCost += TTI.getShuffleCost(SK, DstVecTy, DstVecTy, Mask, CostKind);
  }

  if (!Ext->hasOneUse())
    NewCost += ExtCost;

  LLVM_DEBUG(dbgs() << "Found a insert/extract shuffle-like pair: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");

  if (OldCost < NewCost)
    return false;

  if (NeedExpOrNarrow) {
    if (!NeedDstSrcSwap)
      SrcVec = Builder.CreateShuffleVector(SrcVec, ExtToVecMask);
    else
      DstVec = Builder.CreateShuffleVector(DstVec, ExtToVecMask);
  }

  // Canonicalize undef param to RHS to help further folds.
  if (isa<UndefValue>(DstVec) && !isa<UndefValue>(SrcVec)) {
    ShuffleVectorInst::commuteShuffleMask(Mask, NumDstElts);
    std::swap(DstVec, SrcVec);
  }

  Value *Shuf = Builder.CreateShuffleVector(DstVec, SrcVec, Mask);
  replaceValue(I, *Shuf);

  return true;
}

/// If we're interleaving 2 constant splats, for instance `<vscale x 8 x i32>
/// <splat of 666>` and `<vscale x 8 x i32> <splat of 777>`, we can create a
/// larger splat `<vscale x 8 x i64> <splat of ((777 << 32) | 666)>` first
/// before casting it back into `<vscale x 16 x i32>`.
bool VectorCombine::foldInterleaveIntrinsics(Instruction &I) {
  const APInt *SplatVal0, *SplatVal1;
  if (!match(&I, m_Intrinsic<Intrinsic::vector_interleave2>(
                     m_APInt(SplatVal0), m_APInt(SplatVal1))))
    return false;

  LLVM_DEBUG(dbgs() << "VC: Folding interleave2 with two splats: " << I
                    << "\n");

  auto *VTy =
      cast<VectorType>(cast<IntrinsicInst>(I).getArgOperand(0)->getType());
  auto *ExtVTy = VectorType::getExtendedElementVectorType(VTy);
  unsigned Width = VTy->getElementType()->getIntegerBitWidth();

  // Just in case the cost of interleave2 intrinsic and bitcast are both
  // invalid, in which case we want to bail out, we use <= rather
  // than < here. Even they both have valid and equal costs, it's probably
  // not a good idea to emit a high-cost constant splat.
  if (TTI.getInstructionCost(&I, CostKind) <=
      TTI.getCastInstrCost(Instruction::BitCast, I.getType(), ExtVTy,
                           TTI::CastContextHint::None, CostKind)) {
    LLVM_DEBUG(dbgs() << "VC: The cost to cast from " << *ExtVTy << " to "
                      << *I.getType() << " is too high.\n");
    return false;
  }

  APInt NewSplatVal = SplatVal1->zext(Width * 2);
  NewSplatVal <<= Width;
  NewSplatVal |= SplatVal0->zext(Width * 2);
  auto *NewSplat = ConstantVector::getSplat(
      ExtVTy->getElementCount(), ConstantInt::get(F.getContext(), NewSplatVal));

  IRBuilder<> Builder(&I);
  replaceValue(I, *Builder.CreateBitCast(NewSplat, I.getType()));
  return true;
}

// Attempt to shrink loads that are only used by shufflevector instructions.
bool VectorCombine::shrinkLoadForShuffles(Instruction &I) {
  auto *OldLoad = dyn_cast<LoadInst>(&I);
  if (!OldLoad || !OldLoad->isSimple())
    return false;

  auto *OldLoadTy = dyn_cast<FixedVectorType>(OldLoad->getType());
  if (!OldLoadTy)
    return false;

  unsigned const OldNumElements = OldLoadTy->getNumElements();

  // Search all uses of load. If all uses are shufflevector instructions, and
  // the second operands are all poison values, find the minimum and maximum
  // indices of the vector elements referenced by all shuffle masks.
  // Otherwise return `std::nullopt`.
  using IndexRange = std::pair<int, int>;
  auto GetIndexRangeInShuffles = [&]() -> std::optional<IndexRange> {
    IndexRange OutputRange = IndexRange(OldNumElements, -1);
    for (llvm::Use &Use : I.uses()) {
      // Ensure all uses match the required pattern.
      User *Shuffle = Use.getUser();
      ArrayRef<int> Mask;

      if (!match(Shuffle,
                 m_Shuffle(m_Specific(OldLoad), m_Undef(), m_Mask(Mask))))
        return std::nullopt;

      // Ignore shufflevector instructions that have no uses.
      if (Shuffle->use_empty())
        continue;

      // Find the min and max indices used by the shufflevector instruction.
      for (int Index : Mask) {
        if (Index >= 0 && Index < static_cast<int>(OldNumElements)) {
          OutputRange.first = std::min(Index, OutputRange.first);
          OutputRange.second = std::max(Index, OutputRange.second);
        }
      }
    }

    if (OutputRange.second < OutputRange.first)
      return std::nullopt;

    return OutputRange;
  };

  // Get the range of vector elements used by shufflevector instructions.
  if (std::optional<IndexRange> Indices = GetIndexRangeInShuffles()) {
    unsigned const NewNumElements = Indices->second + 1u;

    // If the range of vector elements is smaller than the full load, attempt
    // to create a smaller load.
    if (NewNumElements < OldNumElements) {
      IRBuilder Builder(&I);
      Builder.SetCurrentDebugLocation(I.getDebugLoc());

      // Calculate costs of old and new ops.
      Type *ElemTy = OldLoadTy->getElementType();
      FixedVectorType *NewLoadTy = FixedVectorType::get(ElemTy, NewNumElements);
      Value *PtrOp = OldLoad->getPointerOperand();

      InstructionCost OldCost = TTI.getMemoryOpCost(
          Instruction::Load, OldLoad->getType(), OldLoad->getAlign(),
          OldLoad->getPointerAddressSpace(), CostKind);
      InstructionCost NewCost =
          TTI.getMemoryOpCost(Instruction::Load, NewLoadTy, OldLoad->getAlign(),
                              OldLoad->getPointerAddressSpace(), CostKind);

      using UseEntry = std::pair<ShuffleVectorInst *, std::vector<int>>;
      SmallVector<UseEntry, 4u> NewUses;
      unsigned const MaxIndex = NewNumElements * 2u;

      for (llvm::Use &Use : I.uses()) {
        auto *Shuffle = cast<ShuffleVectorInst>(Use.getUser());
        ArrayRef<int> OldMask = Shuffle->getShuffleMask();

        // Create entry for new use.
        NewUses.push_back({Shuffle, OldMask});

        // Validate mask indices.
        for (int Index : OldMask) {
          if (Index >= static_cast<int>(MaxIndex))
            return false;
        }

        // Update costs.
        OldCost +=
            TTI.getShuffleCost(TTI::SK_PermuteSingleSrc, Shuffle->getType(),
                               OldLoadTy, OldMask, CostKind);
        NewCost +=
            TTI.getShuffleCost(TTI::SK_PermuteSingleSrc, Shuffle->getType(),
                               NewLoadTy, OldMask, CostKind);
      }

      LLVM_DEBUG(
          dbgs() << "Found a load used only by shufflevector instructions: "
                 << I << "\n  OldCost: " << OldCost
                 << " vs NewCost: " << NewCost << "\n");

      if (OldCost < NewCost || !NewCost.isValid())
        return false;

      // Create new load of smaller vector.
      auto *NewLoad = cast<LoadInst>(
          Builder.CreateAlignedLoad(NewLoadTy, PtrOp, OldLoad->getAlign()));
      NewLoad->copyMetadata(I);

      // Replace all uses.
      for (UseEntry &Use : NewUses) {
        ShuffleVectorInst *Shuffle = Use.first;
        std::vector<int> &NewMask = Use.second;

        Builder.SetInsertPoint(Shuffle);
        Builder.SetCurrentDebugLocation(Shuffle->getDebugLoc());
        Value *NewShuffle = Builder.CreateShuffleVector(
            NewLoad, PoisonValue::get(NewLoadTy), NewMask);

        replaceValue(*Shuffle, *NewShuffle, false);
      }

      return true;
    }
  }
  return false;
}

// Attempt to narrow a phi of shufflevector instructions where the two incoming
// values have the same operands but different masks. If the two shuffle masks
// are offsets of one another we can use one branch to rotate the incoming
// vector and perform one larger shuffle after the phi.
bool VectorCombine::shrinkPhiOfShuffles(Instruction &I) {
  auto *Phi = dyn_cast<PHINode>(&I);
  if (!Phi || Phi->getNumIncomingValues() != 2u)
    return false;

  Value *Op = nullptr;
  ArrayRef<int> Mask0;
  ArrayRef<int> Mask1;

  if (!match(Phi->getOperand(0u),
             m_OneUse(m_Shuffle(m_Value(Op), m_Poison(), m_Mask(Mask0)))) ||
      !match(Phi->getOperand(1u),
             m_OneUse(m_Shuffle(m_Specific(Op), m_Poison(), m_Mask(Mask1)))))
    return false;

  auto *Shuf = cast<ShuffleVectorInst>(Phi->getOperand(0u));

  // Ensure result vectors are wider than the argument vector.
  auto *InputVT = cast<FixedVectorType>(Op->getType());
  auto *ResultVT = cast<FixedVectorType>(Shuf->getType());
  auto const InputNumElements = InputVT->getNumElements();

  if (InputNumElements >= ResultVT->getNumElements())
    return false;

  // Take the difference of the two shuffle masks at each index. Ignore poison
  // values at the same index in both masks.
  SmallVector<int, 16> NewMask;
  NewMask.reserve(Mask0.size());

  for (auto [M0, M1] : zip(Mask0, Mask1)) {
    if (M0 >= 0 && M1 >= 0)
      NewMask.push_back(M0 - M1);
    else if (M0 == -1 && M1 == -1)
      continue;
    else
      return false;
  }

  // Ensure all elements of the new mask are equal. If the difference between
  // the incoming mask elements is the same, the two must be constant offsets
  // of one another.
  if (NewMask.empty() || !all_equal(NewMask))
    return false;

  // Create new mask using difference of the two incoming masks.
  int MaskOffset = NewMask[0u];
  unsigned Index = (InputNumElements - MaskOffset) % InputNumElements;
  NewMask.clear();

  for (unsigned I = 0u; I < InputNumElements; ++I) {
    NewMask.push_back(Index);
    Index = (Index + 1u) % InputNumElements;
  }

  // Calculate costs for worst cases and compare.
  auto const Kind = TTI::SK_PermuteSingleSrc;
  auto OldCost =
      std::max(TTI.getShuffleCost(Kind, ResultVT, InputVT, Mask0, CostKind),
               TTI.getShuffleCost(Kind, ResultVT, InputVT, Mask1, CostKind));
  auto NewCost = TTI.getShuffleCost(Kind, InputVT, InputVT, NewMask, CostKind) +
                 TTI.getShuffleCost(Kind, ResultVT, InputVT, Mask1, CostKind);

  LLVM_DEBUG(dbgs() << "Found a phi of mergeable shuffles: " << I
                    << "\n  OldCost: " << OldCost << " vs NewCost: " << NewCost
                    << "\n");

  if (NewCost > OldCost)
    return false;

  // Create new shuffles and narrowed phi.
  auto Builder = IRBuilder(Shuf);
  Builder.SetCurrentDebugLocation(Shuf->getDebugLoc());
  auto *PoisonVal = PoisonValue::get(InputVT);
  auto *NewShuf0 = Builder.CreateShuffleVector(Op, PoisonVal, NewMask);
  Worklist.push(cast<Instruction>(NewShuf0));

  Builder.SetInsertPoint(Phi);
  Builder.SetCurrentDebugLocation(Phi->getDebugLoc());
  auto *NewPhi = Builder.CreatePHI(NewShuf0->getType(), 2u);
  NewPhi->addIncoming(NewShuf0, Phi->getIncomingBlock(0u));
  NewPhi->addIncoming(Op, Phi->getIncomingBlock(1u));

  Builder.SetInsertPoint(*NewPhi->getInsertionPointAfterDef());
  PoisonVal = PoisonValue::get(NewPhi->getType());
  auto *NewShuf1 = Builder.CreateShuffleVector(NewPhi, PoisonVal, Mask1);

  replaceValue(*Phi, *NewShuf1);
  return true;
}

/// This is the entry point for all transforms. Pass manager differences are
/// handled in the callers of this function.
bool VectorCombine::run() {
  if (DisableVectorCombine)
    return false;

  // Don't attempt vectorization if the target does not support vectors.
  if (!TTI.getNumberOfRegisters(TTI.getRegisterClassForType(/*Vector*/ true)))
    return false;

  LLVM_DEBUG(dbgs() << "\n\nVECTORCOMBINE on " << F.getName() << "\n");

  auto FoldInst = [this](Instruction &I) {
    Builder.SetInsertPoint(&I);
    bool IsVectorType = isa<VectorType>(I.getType());
    bool IsFixedVectorType = isa<FixedVectorType>(I.getType());
    auto Opcode = I.getOpcode();

    LLVM_DEBUG(dbgs() << "VC: Visiting: " << I << '\n');

    // These folds should be beneficial regardless of when this pass is run
    // in the optimization pipeline.
    // The type checking is for run-time efficiency. We can avoid wasting time
    // dispatching to folding functions if there's no chance of matching.
    if (IsFixedVectorType) {
      switch (Opcode) {
      case Instruction::InsertElement:
        if (vectorizeLoadInsert(I))
          return true;
        break;
      case Instruction::ShuffleVector:
        if (widenSubvectorLoad(I))
          return true;
        break;
      default:
        break;
      }
    }

    // This transform works with scalable and fixed vectors
    // TODO: Identify and allow other scalable transforms
    if (IsVectorType) {
      if (scalarizeOpOrCmp(I))
        return true;
      if (scalarizeLoadExtract(I))
        return true;
      if (scalarizeExtExtract(I))
        return true;
      if (scalarizeVPIntrinsic(I))
        return true;
      if (foldInterleaveIntrinsics(I))
        return true;
    }

    if (Opcode == Instruction::Store)
      if (foldSingleElementStore(I))
        return true;

    // If this is an early pipeline invocation of this pass, we are done.
    if (TryEarlyFoldsOnly)
      return false;

    // Otherwise, try folds that improve codegen but may interfere with
    // early IR canonicalizations.
    // The type checking is for run-time efficiency. We can avoid wasting time
    // dispatching to folding functions if there's no chance of matching.
    if (IsFixedVectorType) {
      switch (Opcode) {
      case Instruction::InsertElement:
        if (foldInsExtFNeg(I))
          return true;
        if (foldInsExtBinop(I))
          return true;
        if (foldInsExtVectorToShuffle(I))
          return true;
        break;
      case Instruction::ShuffleVector:
        if (foldPermuteOfBinops(I))
          return true;
        if (foldShuffleOfBinops(I))
          return true;
        if (foldShuffleOfSelects(I))
          return true;
        if (foldShuffleOfCastops(I))
          return true;
        if (foldShuffleOfShuffles(I))
          return true;
        if (foldShuffleOfIntrinsics(I))
          return true;
        if (foldSelectShuffle(I))
          return true;
        if (foldShuffleToIdentity(I))
          return true;
        break;
      case Instruction::Load:
        if (shrinkLoadForShuffles(I))
          return true;
        break;
      case Instruction::BitCast:
        if (foldBitcastShuffle(I))
          return true;
        break;
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
        if (foldBitOpOfCastops(I))
          return true;
        if (foldBitOpOfCastConstant(I))
          return true;
        break;
      case Instruction::PHI:
        if (shrinkPhiOfShuffles(I))
          return true;
        break;
      default:
        if (shrinkType(I))
          return true;
        break;
      }
    } else {
      switch (Opcode) {
      case Instruction::Call:
        if (foldShuffleFromReductions(I))
          return true;
        if (foldCastFromReductions(I))
          return true;
        break;
      case Instruction::ExtractElement:
        if (foldShuffleChainsToReduce(I))
          return true;
        break;
      case Instruction::ICmp:
      case Instruction::FCmp:
        if (foldExtractExtract(I))
          return true;
        break;
      case Instruction::Or:
        if (foldConcatOfBoolMasks(I))
          return true;
        [[fallthrough]];
      default:
        if (Instruction::isBinaryOp(Opcode)) {
          if (foldExtractExtract(I))
            return true;
          if (foldExtractedCmps(I))
            return true;
          if (foldBinopOfReductions(I))
            return true;
        }
        break;
      }
    }
    return false;
  };

  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    // Ignore unreachable basic blocks.
    if (!DT.isReachableFromEntry(&BB))
      continue;
    // Use early increment range so that we can erase instructions in loop.
    // make_early_inc_range is not applicable here, as the next iterator may
    // be invalidated by RecursivelyDeleteTriviallyDeadInstructions.
    // We manually maintain the next instruction and update it when it is about
    // to be deleted.
    Instruction *I = &BB.front();
    while (I) {
      NextInst = I->getNextNode();
      if (!I->isDebugOrPseudoInst())
        MadeChange |= FoldInst(*I);
      I = NextInst;
    }
  }

  NextInst = nullptr;

  while (!Worklist.isEmpty()) {
    Instruction *I = Worklist.removeOne();
    if (!I)
      continue;

    if (isInstructionTriviallyDead(I)) {
      eraseInstruction(*I);
      continue;
    }

    MadeChange |= FoldInst(*I);
  }

  return MadeChange;
}

PreservedAnalyses VectorCombinePass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  AAResults &AA = FAM.getResult<AAManager>(F);
  const DataLayout *DL = &F.getDataLayout();
  VectorCombine Combiner(F, TTI, DT, AA, AC, DL, TTI::TCK_RecipThroughput,
                         TryEarlyFoldsOnly);
  if (!Combiner.run())
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
