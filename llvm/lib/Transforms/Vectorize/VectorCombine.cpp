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
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "vector-combine"
STATISTIC(NumVecLoad, "Number of vector loads formed");
STATISTIC(NumVecCmp, "Number of vector compares formed");
STATISTIC(NumVecBO, "Number of vector binops formed");
STATISTIC(NumVecCmpBO, "Number of vector compare + binop formed");
STATISTIC(NumShufOfBitcast, "Number of shuffles moved after bitcast");
STATISTIC(NumScalarBO, "Number of scalar binops formed");
STATISTIC(NumScalarCmp, "Number of scalar compares formed");

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
                const DominatorTree &DT, AAResults &AA, AssumptionCache &AC)
      : F(F), Builder(F.getContext()), TTI(TTI), DT(DT), AA(AA), AC(AC) {}

  bool run();

private:
  Function &F;
  IRBuilder<> Builder;
  const TargetTransformInfo &TTI;
  const DominatorTree &DT;
  AAResults &AA;
  AssumptionCache &AC;

  bool vectorizeLoadInsert(Instruction &I);
  ExtractElementInst *getShuffleExtract(ExtractElementInst *Ext0,
                                        ExtractElementInst *Ext1,
                                        unsigned PreferredExtractIndex) const;
  bool isExtractExtractCheap(ExtractElementInst *Ext0, ExtractElementInst *Ext1,
                             unsigned Opcode,
                             ExtractElementInst *&ConvertToShuffle,
                             unsigned PreferredExtractIndex);
  void foldExtExtCmp(ExtractElementInst *Ext0, ExtractElementInst *Ext1,
                     Instruction &I);
  void foldExtExtBinop(ExtractElementInst *Ext0, ExtractElementInst *Ext1,
                       Instruction &I);
  bool foldExtractExtract(Instruction &I);
  bool foldBitcastShuf(Instruction &I);
  bool scalarizeBinopOrCmp(Instruction &I);
  bool foldExtractedCmps(Instruction &I);
  bool foldSingleElementStore(Instruction &I);
  bool scalarizeLoadExtract(Instruction &I);
};
} // namespace

static void replaceValue(Value &Old, Value &New) {
  Old.replaceAllUsesWith(&New);
  New.takeName(&Old);
}

bool VectorCombine::vectorizeLoadInsert(Instruction &I) {
  // Match insert into fixed vector of scalar value.
  // TODO: Handle non-zero insert index.
  auto *Ty = dyn_cast<FixedVectorType>(I.getType());
  Value *Scalar;
  if (!Ty || !match(&I, m_InsertElt(m_Undef(), m_Value(Scalar), m_ZeroInt())) ||
      !Scalar->hasOneUse())
    return false;

  // Optionally match an extract from another vector.
  Value *X;
  bool HasExtract = match(Scalar, m_ExtractElt(m_Value(X), m_ZeroInt()));
  if (!HasExtract)
    X = Scalar;

  // Match source value as load of scalar or vector.
  // Do not vectorize scalar load (widening) if atomic/volatile or under
  // asan/hwasan/memtag/tsan. The widened load may load data from dirty regions
  // or create data races non-existent in the source.
  auto *Load = dyn_cast<LoadInst>(X);
  if (!Load || !Load->isSimple() || !Load->hasOneUse() ||
      Load->getFunction()->hasFnAttribute(Attribute::SanitizeMemTag) ||
      mustSuppressSpeculation(*Load))
    return false;

  const DataLayout &DL = I.getModule()->getDataLayout();
  Value *SrcPtr = Load->getPointerOperand()->stripPointerCasts();
  assert(isa<PointerType>(SrcPtr->getType()) && "Expected a pointer type");

  // If original AS != Load's AS, we can't bitcast the original pointer and have
  // to use Load's operand instead. Ideally we would want to strip pointer casts
  // without changing AS, but there's no API to do that ATM.
  unsigned AS = Load->getPointerAddressSpace();
  if (AS != SrcPtr->getType()->getPointerAddressSpace())
    SrcPtr = Load->getPointerOperand();

  // We are potentially transforming byte-sized (8-bit) memory accesses, so make
  // sure we have all of our type-based constraints in place for this target.
  Type *ScalarTy = Scalar->getType();
  uint64_t ScalarSize = ScalarTy->getPrimitiveSizeInBits();
  unsigned MinVectorSize = TTI.getMinVectorRegisterBitWidth();
  if (!ScalarSize || !MinVectorSize || MinVectorSize % ScalarSize != 0 ||
      ScalarSize % 8 != 0)
    return false;

  // Check safety of replacing the scalar load with a larger vector load.
  // We use minimal alignment (maximum flexibility) because we only care about
  // the dereferenceable region. When calculating cost and creating a new op,
  // we may use a larger value based on alignment attributes.
  unsigned MinVecNumElts = MinVectorSize / ScalarSize;
  auto *MinVecTy = VectorType::get(ScalarTy, MinVecNumElts, false);
  unsigned OffsetEltIndex = 0;
  Align Alignment = Load->getAlign();
  if (!isSafeToLoadUnconditionally(SrcPtr, MinVecTy, Align(1), DL, Load, &DT)) {
    // It is not safe to load directly from the pointer, but we can still peek
    // through gep offsets and check if it safe to load from a base address with
    // updated alignment. If it is, we can shuffle the element(s) into place
    // after loading.
    unsigned OffsetBitWidth = DL.getIndexTypeSizeInBits(SrcPtr->getType());
    APInt Offset(OffsetBitWidth, 0);
    SrcPtr = SrcPtr->stripAndAccumulateInBoundsConstantOffsets(DL, Offset);

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

    if (!isSafeToLoadUnconditionally(SrcPtr, MinVecTy, Align(1), DL, Load, &DT))
      return false;

    // Update alignment with offset value. Note that the offset could be negated
    // to more accurately represent "(new) SrcPtr - Offset = (old) SrcPtr", but
    // negation does not change the result of the alignment calculation.
    Alignment = commonAlignment(Alignment, Offset.getZExtValue());
  }

  // Original pattern: insertelt undef, load [free casts of] PtrOp, 0
  // Use the greater of the alignment on the load or its source pointer.
  Alignment = std::max(SrcPtr->getPointerAlignment(DL), Alignment);
  Type *LoadTy = Load->getType();
  InstructionCost OldCost =
      TTI.getMemoryOpCost(Instruction::Load, LoadTy, Alignment, AS);
  APInt DemandedElts = APInt::getOneBitSet(MinVecNumElts, 0);
  OldCost += TTI.getScalarizationOverhead(MinVecTy, DemandedElts,
                                          /* Insert */ true, HasExtract);

  // New pattern: load VecPtr
  InstructionCost NewCost =
      TTI.getMemoryOpCost(Instruction::Load, MinVecTy, Alignment, AS);
  // Optionally, we are shuffling the loaded vector element(s) into place.
  // For the mask set everything but element 0 to undef to prevent poison from
  // propagating from the extra loaded memory. This will also optionally
  // shrink/grow the vector from the loaded size to the output size.
  // We assume this operation has no cost in codegen if there was no offset.
  // Note that we could use freeze to avoid poison problems, but then we might
  // still need a shuffle to change the vector size.
  unsigned OutputNumElts = Ty->getNumElements();
  SmallVector<int, 16> Mask(OutputNumElts, UndefMaskElem);
  assert(OffsetEltIndex < MinVecNumElts && "Address offset too big");
  Mask[0] = OffsetEltIndex;
  if (OffsetEltIndex)
    NewCost += TTI.getShuffleCost(TTI::SK_PermuteSingleSrc, MinVecTy, Mask);

  // We can aggressively convert to the vector form because the backend can
  // invert this transform if it does not result in a performance win.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // It is safe and potentially profitable to load a vector directly:
  // inselt undef, load Scalar, 0 --> load VecPtr
  IRBuilder<> Builder(Load);
  Value *CastedPtr = Builder.CreateBitCast(SrcPtr, MinVecTy->getPointerTo(AS));
  Value *VecLd = Builder.CreateAlignedLoad(MinVecTy, CastedPtr, Alignment);
  VecLd = Builder.CreateShuffleVector(VecLd, Mask);

  replaceValue(I, *VecLd);
  ++NumVecLoad;
  return true;
}

/// Determine which, if any, of the inputs should be replaced by a shuffle
/// followed by extract from a different index.
ExtractElementInst *VectorCombine::getShuffleExtract(
    ExtractElementInst *Ext0, ExtractElementInst *Ext1,
    unsigned PreferredExtractIndex = InvalidIndex) const {
  assert(isa<ConstantInt>(Ext0->getIndexOperand()) &&
         isa<ConstantInt>(Ext1->getIndexOperand()) &&
         "Expected constant extract indexes");

  unsigned Index0 = cast<ConstantInt>(Ext0->getIndexOperand())->getZExtValue();
  unsigned Index1 = cast<ConstantInt>(Ext1->getIndexOperand())->getZExtValue();

  // If the extract indexes are identical, no shuffle is needed.
  if (Index0 == Index1)
    return nullptr;

  Type *VecTy = Ext0->getVectorOperand()->getType();
  assert(VecTy == Ext1->getVectorOperand()->getType() && "Need matching types");
  InstructionCost Cost0 =
      TTI.getVectorInstrCost(Ext0->getOpcode(), VecTy, Index0);
  InstructionCost Cost1 =
      TTI.getVectorInstrCost(Ext1->getOpcode(), VecTy, Index1);

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
                                          unsigned Opcode,
                                          ExtractElementInst *&ConvertToShuffle,
                                          unsigned PreferredExtractIndex) {
  assert(isa<ConstantInt>(Ext0->getOperand(1)) &&
         isa<ConstantInt>(Ext1->getOperand(1)) &&
         "Expected constant extract indexes");
  Type *ScalarTy = Ext0->getType();
  auto *VecTy = cast<VectorType>(Ext0->getOperand(0)->getType());
  InstructionCost ScalarOpCost, VectorOpCost;

  // Get cost estimates for scalar and vector versions of the operation.
  bool IsBinOp = Instruction::isBinaryOp(Opcode);
  if (IsBinOp) {
    ScalarOpCost = TTI.getArithmeticInstrCost(Opcode, ScalarTy);
    VectorOpCost = TTI.getArithmeticInstrCost(Opcode, VecTy);
  } else {
    assert((Opcode == Instruction::ICmp || Opcode == Instruction::FCmp) &&
           "Expected a compare");
    ScalarOpCost = TTI.getCmpSelInstrCost(Opcode, ScalarTy,
                                          CmpInst::makeCmpResultType(ScalarTy));
    VectorOpCost = TTI.getCmpSelInstrCost(Opcode, VecTy,
                                          CmpInst::makeCmpResultType(VecTy));
  }

  // Get cost estimates for the extract elements. These costs will factor into
  // both sequences.
  unsigned Ext0Index = cast<ConstantInt>(Ext0->getOperand(1))->getZExtValue();
  unsigned Ext1Index = cast<ConstantInt>(Ext1->getOperand(1))->getZExtValue();

  InstructionCost Extract0Cost =
      TTI.getVectorInstrCost(Instruction::ExtractElement, VecTy, Ext0Index);
  InstructionCost Extract1Cost =
      TTI.getVectorInstrCost(Instruction::ExtractElement, VecTy, Ext1Index);

  // A more expensive extract will always be replaced by a splat shuffle.
  // For example, if Ext0 is more expensive:
  // opcode (extelt V0, Ext0), (ext V1, Ext1) -->
  // extelt (opcode (splat V0, Ext0), V1), Ext1
  // TODO: Evaluate whether that always results in lowest cost. Alternatively,
  //       check the cost of creating a broadcast shuffle and shuffling both
  //       operands to element 0.
  InstructionCost CheapExtractCost = std::min(Extract0Cost, Extract1Cost);

  // Extra uses of the extracts mean that we include those costs in the
  // vector total because those instructions will not be eliminated.
  InstructionCost OldCost, NewCost;
  if (Ext0->getOperand(0) == Ext1->getOperand(0) && Ext0Index == Ext1Index) {
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
    // undefined except for 1 lane that is being translated to the remaining
    // extraction lane. Therefore, it is a splat shuffle. Ex:
    // ShufMask = { undef, undef, 0, undef }
    // TODO: The cost model has an option for a "broadcast" shuffle
    //       (splat-from-element-0), but no option for a more general splat.
    NewCost +=
        TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, VecTy);
  }

  // Aggressively form a vector op if the cost is equal because the transform
  // may enable further optimization.
  // Codegen can reverse this transform (scalarize) if it was not profitable.
  return OldCost < NewCost;
}

/// Create a shuffle that translates (shifts) 1 element from the input vector
/// to a new element location.
static Value *createShiftShuffle(Value *Vec, unsigned OldIndex,
                                 unsigned NewIndex, IRBuilder<> &Builder) {
  // The shuffle mask is undefined except for 1 lane that is being translated
  // to the new element index. Example for OldIndex == 2 and NewIndex == 0:
  // ShufMask = { 2, undef, undef, undef }
  auto *VecTy = cast<FixedVectorType>(Vec->getType());
  SmallVector<int, 32> ShufMask(VecTy->getNumElements(), UndefMaskElem);
  ShufMask[NewIndex] = OldIndex;
  return Builder.CreateShuffleVector(Vec, ShufMask, "shift");
}

/// Given an extract element instruction with constant index operand, shuffle
/// the source vector (shift the scalar element) to a NewIndex for extraction.
/// Return null if the input can be constant folded, so that we are not creating
/// unnecessary instructions.
static ExtractElementInst *translateExtract(ExtractElementInst *ExtElt,
                                            unsigned NewIndex,
                                            IRBuilder<> &Builder) {
  // If the extract can be constant-folded, this code is unsimplified. Defer
  // to other passes to handle that.
  Value *X = ExtElt->getVectorOperand();
  Value *C = ExtElt->getIndexOperand();
  assert(isa<ConstantInt>(C) && "Expected a constant index operand");
  if (isa<Constant>(X))
    return nullptr;

  Value *Shuf = createShiftShuffle(X, cast<ConstantInt>(C)->getZExtValue(),
                                   NewIndex, Builder);
  return cast<ExtractElementInst>(Builder.CreateExtractElement(Shuf, NewIndex));
}

/// Try to reduce extract element costs by converting scalar compares to vector
/// compares followed by extract.
/// cmp (ext0 V0, C), (ext1 V1, C)
void VectorCombine::foldExtExtCmp(ExtractElementInst *Ext0,
                                  ExtractElementInst *Ext1, Instruction &I) {
  assert(isa<CmpInst>(&I) && "Expected a compare");
  assert(cast<ConstantInt>(Ext0->getIndexOperand())->getZExtValue() ==
             cast<ConstantInt>(Ext1->getIndexOperand())->getZExtValue() &&
         "Expected matching constant extract indexes");

  // cmp Pred (extelt V0, C), (extelt V1, C) --> extelt (cmp Pred V0, V1), C
  ++NumVecCmp;
  CmpInst::Predicate Pred = cast<CmpInst>(&I)->getPredicate();
  Value *V0 = Ext0->getVectorOperand(), *V1 = Ext1->getVectorOperand();
  Value *VecCmp = Builder.CreateCmp(Pred, V0, V1);
  Value *NewExt = Builder.CreateExtractElement(VecCmp, Ext0->getIndexOperand());
  replaceValue(I, *NewExt);
}

/// Try to reduce extract element costs by converting scalar binops to vector
/// binops followed by extract.
/// bo (ext0 V0, C), (ext1 V1, C)
void VectorCombine::foldExtExtBinop(ExtractElementInst *Ext0,
                                    ExtractElementInst *Ext1, Instruction &I) {
  assert(isa<BinaryOperator>(&I) && "Expected a binary operator");
  assert(cast<ConstantInt>(Ext0->getIndexOperand())->getZExtValue() ==
             cast<ConstantInt>(Ext1->getIndexOperand())->getZExtValue() &&
         "Expected matching constant extract indexes");

  // bo (extelt V0, C), (extelt V1, C) --> extelt (bo V0, V1), C
  ++NumVecBO;
  Value *V0 = Ext0->getVectorOperand(), *V1 = Ext1->getVectorOperand();
  Value *VecBO =
      Builder.CreateBinOp(cast<BinaryOperator>(&I)->getOpcode(), V0, V1);

  // All IR flags are safe to back-propagate because any potential poison
  // created in unused vector elements is discarded by the extract.
  if (auto *VecBOInst = dyn_cast<Instruction>(VecBO))
    VecBOInst->copyIRFlags(&I);

  Value *NewExt = Builder.CreateExtractElement(VecBO, Ext0->getIndexOperand());
  replaceValue(I, *NewExt);
}

/// Match an instruction with extracted vector operands.
bool VectorCombine::foldExtractExtract(Instruction &I) {
  // It is not safe to transform things like div, urem, etc. because we may
  // create undefined behavior when executing those on unknown vector elements.
  if (!isSafeToSpeculativelyExecute(&I))
    return false;

  Instruction *I0, *I1;
  CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
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
  if (isExtractExtractCheap(Ext0, Ext1, I.getOpcode(), ExtractToChange,
                            InsertIndex))
    return false;

  if (ExtractToChange) {
    unsigned CheapExtractIdx = ExtractToChange == Ext0 ? C1 : C0;
    ExtractElementInst *NewExtract =
        translateExtract(ExtractToChange, CheapExtractIdx, Builder);
    if (!NewExtract)
      return false;
    if (ExtractToChange == Ext0)
      Ext0 = NewExtract;
    else
      Ext1 = NewExtract;
  }

  if (Pred != CmpInst::BAD_ICMP_PREDICATE)
    foldExtExtCmp(Ext0, Ext1, I);
  else
    foldExtExtBinop(Ext0, Ext1, I);

  return true;
}

/// If this is a bitcast of a shuffle, try to bitcast the source vector to the
/// destination type followed by shuffle. This can enable further transforms by
/// moving bitcasts or shuffles together.
bool VectorCombine::foldBitcastShuf(Instruction &I) {
  Value *V;
  ArrayRef<int> Mask;
  if (!match(&I, m_BitCast(
                     m_OneUse(m_Shuffle(m_Value(V), m_Undef(), m_Mask(Mask))))))
    return false;

  // 1) Do not fold bitcast shuffle for scalable type. First, shuffle cost for
  // scalable type is unknown; Second, we cannot reason if the narrowed shuffle
  // mask for scalable type is a splat or not.
  // 2) Disallow non-vector casts and length-changing shuffles.
  // TODO: We could allow any shuffle.
  auto *DestTy = dyn_cast<FixedVectorType>(I.getType());
  auto *SrcTy = dyn_cast<FixedVectorType>(V->getType());
  if (!SrcTy || !DestTy || I.getOperand(0)->getType() != SrcTy)
    return false;

  unsigned DestNumElts = DestTy->getNumElements();
  unsigned SrcNumElts = SrcTy->getNumElements();
  SmallVector<int, 16> NewMask;
  if (SrcNumElts <= DestNumElts) {
    // The bitcast is from wide to narrow/equal elements. The shuffle mask can
    // always be expanded to the equivalent form choosing narrower elements.
    assert(DestNumElts % SrcNumElts == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = DestNumElts / SrcNumElts;
    narrowShuffleMaskElts(ScaleFactor, Mask, NewMask);
  } else {
    // The bitcast is from narrow elements to wide elements. The shuffle mask
    // must choose consecutive elements to allow casting first.
    assert(SrcNumElts % DestNumElts == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = SrcNumElts / DestNumElts;
    if (!widenShuffleMaskElts(ScaleFactor, Mask, NewMask))
      return false;
  }

  // The new shuffle must not cost more than the old shuffle. The bitcast is
  // moved ahead of the shuffle, so assume that it has the same cost as before.
  InstructionCost DestCost = TTI.getShuffleCost(
      TargetTransformInfo::SK_PermuteSingleSrc, DestTy, NewMask);
  InstructionCost SrcCost =
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, SrcTy, Mask);
  if (DestCost > SrcCost || !DestCost.isValid())
    return false;

  // bitcast (shuf V, MaskC) --> shuf (bitcast V), MaskC'
  ++NumShufOfBitcast;
  Value *CastV = Builder.CreateBitCast(V, DestTy);
  Value *Shuf = Builder.CreateShuffleVector(CastV, NewMask);
  replaceValue(I, *Shuf);
  return true;
}

/// Match a vector binop or compare instruction with at least one inserted
/// scalar operand and convert to scalar binop/cmp followed by insertelement.
bool VectorCombine::scalarizeBinopOrCmp(Instruction &I) {
  CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
  Value *Ins0, *Ins1;
  if (!match(&I, m_BinOp(m_Value(Ins0), m_Value(Ins1))) &&
      !match(&I, m_Cmp(Pred, m_Value(Ins0), m_Value(Ins1))))
    return false;

  // Do not convert the vector condition of a vector select into a scalar
  // condition. That may cause problems for codegen because of differences in
  // boolean formats and register-file transfers.
  // TODO: Can we account for that in the cost model?
  bool IsCmp = Pred != CmpInst::Predicate::BAD_ICMP_PREDICATE;
  if (IsCmp)
    for (User *U : I.users())
      if (match(U, m_Select(m_Specific(&I), m_Value(), m_Value())))
        return false;

  // Match against one or both scalar values being inserted into constant
  // vectors:
  // vec_op VecC0, (inselt VecC1, V1, Index)
  // vec_op (inselt VecC0, V0, Index), VecC1
  // vec_op (inselt VecC0, V0, Index), (inselt VecC1, V1, Index)
  // TODO: Deal with mismatched index constants and variable indexes?
  Constant *VecC0 = nullptr, *VecC1 = nullptr;
  Value *V0 = nullptr, *V1 = nullptr;
  uint64_t Index0 = 0, Index1 = 0;
  if (!match(Ins0, m_InsertElt(m_Constant(VecC0), m_Value(V0),
                               m_ConstantInt(Index0))) &&
      !match(Ins0, m_Constant(VecC0)))
    return false;
  if (!match(Ins1, m_InsertElt(m_Constant(VecC1), m_Value(V1),
                               m_ConstantInt(Index1))) &&
      !match(Ins1, m_Constant(VecC1)))
    return false;

  bool IsConst0 = !V0;
  bool IsConst1 = !V1;
  if (IsConst0 && IsConst1)
    return false;
  if (!IsConst0 && !IsConst1 && Index0 != Index1)
    return false;

  // Bail for single insertion if it is a load.
  // TODO: Handle this once getVectorInstrCost can cost for load/stores.
  auto *I0 = dyn_cast_or_null<Instruction>(V0);
  auto *I1 = dyn_cast_or_null<Instruction>(V1);
  if ((IsConst0 && I1 && I1->mayReadFromMemory()) ||
      (IsConst1 && I0 && I0->mayReadFromMemory()))
    return false;

  uint64_t Index = IsConst0 ? Index1 : Index0;
  Type *ScalarTy = IsConst0 ? V1->getType() : V0->getType();
  Type *VecTy = I.getType();
  assert(VecTy->isVectorTy() &&
         (IsConst0 || IsConst1 || V0->getType() == V1->getType()) &&
         (ScalarTy->isIntegerTy() || ScalarTy->isFloatingPointTy() ||
          ScalarTy->isPointerTy()) &&
         "Unexpected types for insert element into binop or cmp");

  unsigned Opcode = I.getOpcode();
  InstructionCost ScalarOpCost, VectorOpCost;
  if (IsCmp) {
    ScalarOpCost = TTI.getCmpSelInstrCost(Opcode, ScalarTy);
    VectorOpCost = TTI.getCmpSelInstrCost(Opcode, VecTy);
  } else {
    ScalarOpCost = TTI.getArithmeticInstrCost(Opcode, ScalarTy);
    VectorOpCost = TTI.getArithmeticInstrCost(Opcode, VecTy);
  }

  // Get cost estimate for the insert element. This cost will factor into
  // both sequences.
  InstructionCost InsertCost =
      TTI.getVectorInstrCost(Instruction::InsertElement, VecTy, Index);
  InstructionCost OldCost =
      (IsConst0 ? 0 : InsertCost) + (IsConst1 ? 0 : InsertCost) + VectorOpCost;
  InstructionCost NewCost = ScalarOpCost + InsertCost +
                            (IsConst0 ? 0 : !Ins0->hasOneUse() * InsertCost) +
                            (IsConst1 ? 0 : !Ins1->hasOneUse() * InsertCost);

  // We want to scalarize unless the vector variant actually has lower cost.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // vec_op (inselt VecC0, V0, Index), (inselt VecC1, V1, Index) -->
  // inselt NewVecC, (scalar_op V0, V1), Index
  if (IsCmp)
    ++NumScalarCmp;
  else
    ++NumScalarBO;

  // For constant cases, extract the scalar element, this should constant fold.
  if (IsConst0)
    V0 = ConstantExpr::getExtractElement(VecC0, Builder.getInt64(Index));
  if (IsConst1)
    V1 = ConstantExpr::getExtractElement(VecC1, Builder.getInt64(Index));

  Value *Scalar =
      IsCmp ? Builder.CreateCmp(Pred, V0, V1)
            : Builder.CreateBinOp((Instruction::BinaryOps)Opcode, V0, V1);

  Scalar->setName(I.getName() + ".scalar");

  // All IR flags are safe to back-propagate. There is no potential for extra
  // poison to be created by the scalar instruction.
  if (auto *ScalarInst = dyn_cast<Instruction>(Scalar))
    ScalarInst->copyIRFlags(&I);

  // Fold the vector constants in the original vectors into a new base vector.
  Constant *NewVecC = IsCmp ? ConstantExpr::getCompare(Pred, VecC0, VecC1)
                            : ConstantExpr::get(Opcode, VecC0, VecC1);
  Value *Insert = Builder.CreateInsertElement(NewVecC, Scalar, Index);
  replaceValue(I, *Insert);
  return true;
}

/// Try to combine a scalar binop + 2 scalar compares of extracted elements of
/// a vector into vector operations followed by extract. Note: The SLP pass
/// may miss this pattern because of implementation problems.
bool VectorCombine::foldExtractedCmps(Instruction &I) {
  // We are looking for a scalar binop of booleans.
  // binop i1 (cmp Pred I0, C0), (cmp Pred I1, C1)
  if (!I.isBinaryOp() || !I.getType()->isIntegerTy(1))
    return false;

  // The compare predicates should match, and each compare should have a
  // constant operand.
  // TODO: Relax the one-use constraints.
  Value *B0 = I.getOperand(0), *B1 = I.getOperand(1);
  Instruction *I0, *I1;
  Constant *C0, *C1;
  CmpInst::Predicate P0, P1;
  if (!match(B0, m_OneUse(m_Cmp(P0, m_Instruction(I0), m_Constant(C0)))) ||
      !match(B1, m_OneUse(m_Cmp(P1, m_Instruction(I1), m_Constant(C1)))) ||
      P0 != P1)
    return false;

  // The compare operands must be extracts of the same vector with constant
  // extract indexes.
  // TODO: Relax the one-use constraints.
  Value *X;
  uint64_t Index0, Index1;
  if (!match(I0, m_OneUse(m_ExtractElt(m_Value(X), m_ConstantInt(Index0)))) ||
      !match(I1, m_OneUse(m_ExtractElt(m_Specific(X), m_ConstantInt(Index1)))))
    return false;

  auto *Ext0 = cast<ExtractElementInst>(I0);
  auto *Ext1 = cast<ExtractElementInst>(I1);
  ExtractElementInst *ConvertToShuf = getShuffleExtract(Ext0, Ext1);
  if (!ConvertToShuf)
    return false;

  // The original scalar pattern is:
  // binop i1 (cmp Pred (ext X, Index0), C0), (cmp Pred (ext X, Index1), C1)
  CmpInst::Predicate Pred = P0;
  unsigned CmpOpcode = CmpInst::isFPPredicate(Pred) ? Instruction::FCmp
                                                    : Instruction::ICmp;
  auto *VecTy = dyn_cast<FixedVectorType>(X->getType());
  if (!VecTy)
    return false;

  InstructionCost OldCost =
      TTI.getVectorInstrCost(Ext0->getOpcode(), VecTy, Index0);
  OldCost += TTI.getVectorInstrCost(Ext1->getOpcode(), VecTy, Index1);
  OldCost += TTI.getCmpSelInstrCost(CmpOpcode, I0->getType()) * 2;
  OldCost += TTI.getArithmeticInstrCost(I.getOpcode(), I.getType());

  // The proposed vector pattern is:
  // vcmp = cmp Pred X, VecC
  // ext (binop vNi1 vcmp, (shuffle vcmp, Index1)), Index0
  int CheapIndex = ConvertToShuf == Ext0 ? Index1 : Index0;
  int ExpensiveIndex = ConvertToShuf == Ext0 ? Index0 : Index1;
  auto *CmpTy = cast<FixedVectorType>(CmpInst::makeCmpResultType(X->getType()));
  InstructionCost NewCost = TTI.getCmpSelInstrCost(CmpOpcode, X->getType());
  SmallVector<int, 32> ShufMask(VecTy->getNumElements(), UndefMaskElem);
  ShufMask[CheapIndex] = ExpensiveIndex;
  NewCost += TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, CmpTy,
                                ShufMask);
  NewCost += TTI.getArithmeticInstrCost(I.getOpcode(), CmpTy);
  NewCost += TTI.getVectorInstrCost(Ext0->getOpcode(), CmpTy, CheapIndex);

  // Aggressively form vector ops if the cost is equal because the transform
  // may enable further optimization.
  // Codegen can reverse this transform (scalarize) if it was not profitable.
  if (OldCost < NewCost || !NewCost.isValid())
    return false;

  // Create a vector constant from the 2 scalar constants.
  SmallVector<Constant *, 32> CmpC(VecTy->getNumElements(),
                                   UndefValue::get(VecTy->getElementType()));
  CmpC[Index0] = C0;
  CmpC[Index1] = C1;
  Value *VCmp = Builder.CreateCmp(Pred, X, ConstantVector::get(CmpC));

  Value *Shuf = createShiftShuffle(VCmp, ExpensiveIndex, CheapIndex, Builder);
  Value *VecLogic = Builder.CreateBinOp(cast<BinaryOperator>(I).getOpcode(),
                                        VCmp, Shuf);
  Value *NewExt = Builder.CreateExtractElement(VecLogic, CheapIndex);
  replaceValue(I, *NewExt);
  ++NumVecCmpBO;
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

/// Check if it is legal to scalarize a memory access to \p VecTy at index \p
/// Idx. \p Idx must access a valid vector element.
static bool canScalarizeAccess(FixedVectorType *VecTy, Value *Idx,
                               Instruction *CtxI, AssumptionCache &AC) {
  if (auto *C = dyn_cast<ConstantInt>(Idx))
    return C->getValue().ult(VecTy->getNumElements());

  APInt Zero(Idx->getType()->getScalarSizeInBits(), 0);
  APInt MaxElts(Idx->getType()->getScalarSizeInBits(), VecTy->getNumElements());
  ConstantRange ValidIndices(Zero, MaxElts);
  ConstantRange IdxRange = computeConstantRange(Idx, true, &AC, CtxI, 0);
  return ValidIndices.contains(IdxRange);
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
  StoreInst *SI = dyn_cast<StoreInst>(&I);
  if (!SI || !SI->isSimple() ||
      !isa<FixedVectorType>(SI->getValueOperand()->getType()))
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
    auto VecTy = cast<FixedVectorType>(SI->getValueOperand()->getType());
    const DataLayout &DL = I.getModule()->getDataLayout();
    Value *SrcAddr = Load->getPointerOperand()->stripPointerCasts();
    // Don't optimize for atomic/volatile load or store. Ensure memory is not
    // modified between, vector type matches store size, and index is inbounds.
    if (!Load->isSimple() || Load->getParent() != SI->getParent() ||
        !DL.typeSizeEqualsStoreSize(Load->getType()) ||
        !canScalarizeAccess(VecTy, Idx, Load, AC) ||
        SrcAddr != SI->getPointerOperand()->stripPointerCasts() ||
        isMemModifiedBetween(Load->getIterator(), SI->getIterator(),
                             MemoryLocation::get(SI), AA))
      return false;

    Value *GEP = GetElementPtrInst::CreateInBounds(
        SI->getPointerOperand(), {ConstantInt::get(Idx->getType(), 0), Idx});
    Builder.Insert(GEP);
    StoreInst *NSI = Builder.CreateStore(NewElement, GEP);
    NSI->copyMetadata(*SI);
    if (SI->getAlign() < NSI->getAlign())
      NSI->setAlignment(SI->getAlign());
    replaceValue(I, *NSI);
    // Need erasing the store manually.
    I.eraseFromParent();
    return true;
  }

  return false;
}

/// Try to scalarize vector loads feeding extractelement instructions.
bool VectorCombine::scalarizeLoadExtract(Instruction &I) {
  Value *Ptr;
  Value *Idx;
  if (!match(&I, m_ExtractElt(m_Load(m_Value(Ptr)), m_Value(Idx))))
    return false;

  auto *LI = cast<LoadInst>(I.getOperand(0));
  const DataLayout &DL = I.getModule()->getDataLayout();
  if (LI->isVolatile() || !DL.typeSizeEqualsStoreSize(LI->getType()))
    return false;

  auto *FixedVT = dyn_cast<FixedVectorType>(LI->getType());
  if (!FixedVT)
    return false;

  InstructionCost OriginalCost = TTI.getMemoryOpCost(
      Instruction::Load, LI->getType(), Align(LI->getAlignment()),
      LI->getPointerAddressSpace());
  InstructionCost ScalarizedCost = 0;

  Instruction *LastCheckedInst = LI;
  unsigned NumInstChecked = 0;
  // Check if all users of the load are extracts with no memory modifications
  // between the load and the extract. Compute the cost of both the original
  // code and the scalarized version.
  for (User *U : LI->users()) {
    auto *UI = dyn_cast<ExtractElementInst>(U);
    if (!UI || UI->getParent() != LI->getParent())
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
    }

    if (!LastCheckedInst)
      LastCheckedInst = UI;
    else if (LastCheckedInst->comesBefore(UI))
      LastCheckedInst = UI;

    if (!canScalarizeAccess(FixedVT, UI->getOperand(1), &I, AC))
      return false;

    auto *Index = dyn_cast<ConstantInt>(UI->getOperand(1));
    OriginalCost +=
        TTI.getVectorInstrCost(Instruction::ExtractElement, LI->getType(),
                               Index ? Index->getZExtValue() : -1);
    ScalarizedCost +=
        TTI.getMemoryOpCost(Instruction::Load, FixedVT->getElementType(),
                            Align(1), LI->getPointerAddressSpace());
    ScalarizedCost += TTI.getAddressComputationCost(FixedVT->getElementType());
  }

  if (ScalarizedCost >= OriginalCost)
    return false;

  // Replace extracts with narrow scalar loads.
  for (User *U : LI->users()) {
    auto *EI = cast<ExtractElementInst>(U);
    Builder.SetInsertPoint(EI);

    Value *Idx = EI->getOperand(1);
    if (!isGuaranteedNotToBePoison(Idx, &AC, LI, &DT))
      Idx = Builder.CreateFreeze(Idx);

    Value *GEP =
        Builder.CreateInBoundsGEP(FixedVT, Ptr, {Builder.getInt32(0), Idx});
    auto *NewLoad = cast<LoadInst>(Builder.CreateLoad(
        FixedVT->getElementType(), GEP, EI->getName() + ".scalar"));

    // Set the alignment for the new load. For index 0, we can use the original
    // alignment. Otherwise choose the common alignment of the load's align and
    // the alignment for the scalar type.
    auto *ConstIdx = dyn_cast<ConstantInt>(EI->getOperand(1));
    if (ConstIdx && ConstIdx->isNullValue())
      NewLoad->setAlignment(LI->getAlign());
    else
      NewLoad->setAlignment(commonAlignment(
          DL.getABITypeAlign(NewLoad->getType()), LI->getAlign()));
    replaceValue(*EI, *NewLoad);
  }

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

  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    // Ignore unreachable basic blocks.
    if (!DT.isReachableFromEntry(&BB))
      continue;
    // Use early increment range so that we can erase instructions in loop.
    for (Instruction &I : make_early_inc_range(BB)) {
      if (isa<DbgInfoIntrinsic>(I))
        continue;
      Builder.SetInsertPoint(&I);
      MadeChange |= vectorizeLoadInsert(I);
      MadeChange |= foldExtractExtract(I);
      MadeChange |= foldBitcastShuf(I);
      MadeChange |= scalarizeBinopOrCmp(I);
      MadeChange |= foldExtractedCmps(I);
      MadeChange |= scalarizeLoadExtract(I);
      MadeChange |= foldSingleElementStore(I);
    }
  }

  // We're done with transforms, so remove dead instructions.
  if (MadeChange)
    for (BasicBlock &BB : F)
      SimplifyInstructionsInBlock(&BB);

  return MadeChange;
}

// Pass manager boilerplate below here.

namespace {
class VectorCombineLegacyPass : public FunctionPass {
public:
  static char ID;
  VectorCombineLegacyPass() : FunctionPass(ID) {
    initializeVectorCombineLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.setPreservesCFG();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    VectorCombine Combiner(F, TTI, DT, AA, AC);
    return Combiner.run();
  }
};
} // namespace

char VectorCombineLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(VectorCombineLegacyPass, "vector-combine",
                      "Optimize scalar/vector ops", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(VectorCombineLegacyPass, "vector-combine",
                    "Optimize scalar/vector ops", false, false)
Pass *llvm::createVectorCombinePass() {
  return new VectorCombineLegacyPass();
}

PreservedAnalyses VectorCombinePass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  AAResults &AA = FAM.getResult<AAManager>(F);
  VectorCombine Combiner(F, TTI, DT, AA, AC);
  if (!Combiner.run())
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
