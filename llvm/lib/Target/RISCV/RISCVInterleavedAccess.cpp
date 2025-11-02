//===-- RISCVInterleavedAccess.cpp - RISC-V Interleaved Access Transform --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions and callbacks related to the InterleavedAccessPass.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVISelLowering.h"
#include "RISCVSubtarget.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;

bool RISCVTargetLowering::isLegalInterleavedAccessType(
    VectorType *VTy, unsigned Factor, Align Alignment, unsigned AddrSpace,
    const DataLayout &DL) const {
  EVT VT = getValueType(DL, VTy);
  // Don't lower vlseg/vsseg for vector types that can't be split.
  if (!isTypeLegal(VT))
    return false;

  if (!isLegalElementTypeForRVV(VT.getScalarType()) ||
      !allowsMemoryAccessForAlignment(VTy->getContext(), DL, VT, AddrSpace,
                                      Alignment))
    return false;

  MVT ContainerVT = VT.getSimpleVT();

  if (auto *FVTy = dyn_cast<FixedVectorType>(VTy)) {
    if (!Subtarget.useRVVForFixedLengthVectors())
      return false;
    // Sometimes the interleaved access pass picks up splats as interleaves of
    // one element. Don't lower these.
    if (FVTy->getNumElements() < 2)
      return false;

    ContainerVT = getContainerForFixedLengthVector(VT.getSimpleVT());
  }

  // Need to make sure that EMUL * NFIELDS ≤ 8
  auto [LMUL, Fractional] = RISCVVType::decodeVLMUL(getLMUL(ContainerVT));
  if (Fractional)
    return true;
  return Factor * LMUL <= 8;
}

static const Intrinsic::ID FixedVlsegIntrIds[] = {
    Intrinsic::riscv_seg2_load_mask, Intrinsic::riscv_seg3_load_mask,
    Intrinsic::riscv_seg4_load_mask, Intrinsic::riscv_seg5_load_mask,
    Intrinsic::riscv_seg6_load_mask, Intrinsic::riscv_seg7_load_mask,
    Intrinsic::riscv_seg8_load_mask};

static const Intrinsic::ID FixedVlssegIntrIds[] = {
    Intrinsic::riscv_sseg2_load_mask, Intrinsic::riscv_sseg3_load_mask,
    Intrinsic::riscv_sseg4_load_mask, Intrinsic::riscv_sseg5_load_mask,
    Intrinsic::riscv_sseg6_load_mask, Intrinsic::riscv_sseg7_load_mask,
    Intrinsic::riscv_sseg8_load_mask};

static const Intrinsic::ID ScalableVlsegIntrIds[] = {
    Intrinsic::riscv_vlseg2_mask, Intrinsic::riscv_vlseg3_mask,
    Intrinsic::riscv_vlseg4_mask, Intrinsic::riscv_vlseg5_mask,
    Intrinsic::riscv_vlseg6_mask, Intrinsic::riscv_vlseg7_mask,
    Intrinsic::riscv_vlseg8_mask};

static const Intrinsic::ID FixedVssegIntrIds[] = {
    Intrinsic::riscv_seg2_store_mask, Intrinsic::riscv_seg3_store_mask,
    Intrinsic::riscv_seg4_store_mask, Intrinsic::riscv_seg5_store_mask,
    Intrinsic::riscv_seg6_store_mask, Intrinsic::riscv_seg7_store_mask,
    Intrinsic::riscv_seg8_store_mask};

static const Intrinsic::ID FixedVsssegIntrIds[] = {
    Intrinsic::riscv_sseg2_store_mask, Intrinsic::riscv_sseg3_store_mask,
    Intrinsic::riscv_sseg4_store_mask, Intrinsic::riscv_sseg5_store_mask,
    Intrinsic::riscv_sseg6_store_mask, Intrinsic::riscv_sseg7_store_mask,
    Intrinsic::riscv_sseg8_store_mask};

static const Intrinsic::ID ScalableVssegIntrIds[] = {
    Intrinsic::riscv_vsseg2_mask, Intrinsic::riscv_vsseg3_mask,
    Intrinsic::riscv_vsseg4_mask, Intrinsic::riscv_vsseg5_mask,
    Intrinsic::riscv_vsseg6_mask, Intrinsic::riscv_vsseg7_mask,
    Intrinsic::riscv_vsseg8_mask};

static bool isMultipleOfN(const Value *V, const DataLayout &DL, unsigned N) {
  assert(N);
  if (N == 1)
    return true;

  using namespace PatternMatch;
  // Right now we're only recognizing the simplest pattern.
  uint64_t C;
  if (match(V, m_CombineOr(m_ConstantInt(C),
                           m_NUWMul(m_Value(), m_ConstantInt(C)))) &&
      C && C % N == 0)
    return true;

  if (isPowerOf2_32(N)) {
    KnownBits KB = llvm::computeKnownBits(V, DL);
    return KB.countMinTrailingZeros() >= Log2_32(N);
  }

  return false;
}

/// Do the common operand retrieval and validition required by the
/// routines below.
static bool getMemOperands(unsigned Factor, VectorType *VTy, Type *XLenTy,
                           Instruction *I, Value *&Ptr, Value *&Mask,
                           Value *&VL, Align &Alignment) {

  IRBuilder<> Builder(I);
  const DataLayout &DL = I->getDataLayout();
  ElementCount EC = VTy->getElementCount();
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    assert(LI->isSimple());
    Ptr = LI->getPointerOperand();
    Alignment = LI->getAlign();
    assert(!Mask && "Unexpected mask on a load");
    Mask = Builder.getAllOnesMask(EC);
    VL = isa<FixedVectorType>(VTy) ? Builder.CreateElementCount(XLenTy, EC)
                                   : Constant::getAllOnesValue(XLenTy);
    return true;
  }
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    assert(SI->isSimple());
    Ptr = SI->getPointerOperand();
    Alignment = SI->getAlign();
    assert(!Mask && "Unexpected mask on a store");
    Mask = Builder.getAllOnesMask(EC);
    VL = isa<FixedVectorType>(VTy) ? Builder.CreateElementCount(XLenTy, EC)
                                   : Constant::getAllOnesValue(XLenTy);
    return true;
  }

  auto *II = cast<IntrinsicInst>(I);
  switch (II->getIntrinsicID()) {
  default:
    llvm_unreachable("Unsupported intrinsic type");
  case Intrinsic::vp_load:
  case Intrinsic::vp_store: {
    auto *VPLdSt = cast<VPIntrinsic>(I);
    Ptr = VPLdSt->getMemoryPointerParam();
    Alignment = VPLdSt->getPointerAlignment().value_or(
        DL.getABITypeAlign(VTy->getElementType()));

    assert(Mask && "vp.load and vp.store needs a mask!");

    Value *WideEVL = VPLdSt->getVectorLengthParam();
    // Conservatively check if EVL is a multiple of factor, otherwise some
    // (trailing) elements might be lost after the transformation.
    if (!isMultipleOfN(WideEVL, I->getDataLayout(), Factor))
      return false;

    auto *FactorC = ConstantInt::get(WideEVL->getType(), Factor);
    VL = Builder.CreateZExt(Builder.CreateExactUDiv(WideEVL, FactorC), XLenTy);
    return true;
  }
  case Intrinsic::masked_load: {
    Ptr = II->getOperand(0);
    Alignment = II->getParamAlign(0).valueOrOne();

    if (!isa<UndefValue>(II->getOperand(2)))
      return false;

    assert(Mask && "masked.load needs a mask!");

    VL = isa<FixedVectorType>(VTy)
             ? Builder.CreateElementCount(XLenTy, VTy->getElementCount())
             : Constant::getAllOnesValue(XLenTy);
    return true;
  }
  case Intrinsic::masked_store: {
    Ptr = II->getOperand(1);
    Alignment = II->getParamAlign(1).valueOrOne();

    assert(Mask && "masked.store needs a mask!");

    VL = isa<FixedVectorType>(VTy)
             ? Builder.CreateElementCount(XLenTy, VTy->getElementCount())
             : Constant::getAllOnesValue(XLenTy);
    return true;
  }
  }
}

/// Lower an interleaved load into a vlsegN intrinsic.
///
/// E.g. Lower an interleaved load (Factor = 2):
/// %wide.vec = load <8 x i32>, <8 x i32>* %ptr
/// %v0 = shuffle %wide.vec, undef, <0, 2, 4, 6>  ; Extract even elements
/// %v1 = shuffle %wide.vec, undef, <1, 3, 5, 7>  ; Extract odd elements
///
/// Into:
/// %ld2 = { <4 x i32>, <4 x i32> } call llvm.riscv.seg2.load.v4i32.p0.i64(
///                                        %ptr, i64 4)
/// %vec0 = extractelement { <4 x i32>, <4 x i32> } %ld2, i32 0
/// %vec1 = extractelement { <4 x i32>, <4 x i32> } %ld2, i32 1
bool RISCVTargetLowering::lowerInterleavedLoad(
    Instruction *Load, Value *Mask, ArrayRef<ShuffleVectorInst *> Shuffles,
    ArrayRef<unsigned> Indices, unsigned Factor, const APInt &GapMask) const {
  assert(Indices.size() == Shuffles.size());
  assert(GapMask.getBitWidth() == Factor);

  // We only support cases where the skipped fields are the trailing ones.
  // TODO: Lower to strided load if there is only a single active field.
  unsigned MaskFactor = GapMask.popcount();
  if (MaskFactor < 2 || !GapMask.isMask())
    return false;
  IRBuilder<> Builder(Load);

  const DataLayout &DL = Load->getDataLayout();
  auto *VTy = cast<FixedVectorType>(Shuffles[0]->getType());
  auto *XLenTy = Builder.getIntNTy(Subtarget.getXLen());

  Value *Ptr, *VL;
  Align Alignment;
  if (!getMemOperands(MaskFactor, VTy, XLenTy, Load, Ptr, Mask, VL, Alignment))
    return false;

  Type *PtrTy = Ptr->getType();
  unsigned AS = PtrTy->getPointerAddressSpace();
  if (!isLegalInterleavedAccessType(VTy, MaskFactor, Alignment, AS, DL))
    return false;

  CallInst *SegLoad = nullptr;
  if (MaskFactor < Factor) {
    // Lower to strided segmented load.
    unsigned ScalarSizeInBytes = DL.getTypeStoreSize(VTy->getElementType());
    Value *Stride = ConstantInt::get(XLenTy, Factor * ScalarSizeInBytes);
    SegLoad = Builder.CreateIntrinsic(FixedVlssegIntrIds[MaskFactor - 2],
                                      {VTy, PtrTy, XLenTy, XLenTy},
                                      {Ptr, Stride, Mask, VL});
  } else {
    // Lower to normal segmented load.
    SegLoad = Builder.CreateIntrinsic(FixedVlsegIntrIds[Factor - 2],
                                      {VTy, PtrTy, XLenTy}, {Ptr, Mask, VL});
  }

  for (unsigned i = 0; i < Shuffles.size(); i++) {
    unsigned FactorIdx = Indices[i];
    if (FactorIdx >= MaskFactor) {
      // Replace masked-off factors (that are still extracted) with poison.
      Shuffles[i]->replaceAllUsesWith(PoisonValue::get(VTy));
    } else {
      Value *SubVec = Builder.CreateExtractValue(SegLoad, FactorIdx);
      Shuffles[i]->replaceAllUsesWith(SubVec);
    }
  }

  return true;
}

/// Lower an interleaved store into a vssegN intrinsic.
///
/// E.g. Lower an interleaved store (Factor = 3):
/// %i.vec = shuffle <8 x i32> %v0, <8 x i32> %v1,
///                  <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>
/// store <12 x i32> %i.vec, <12 x i32>* %ptr
///
/// Into:
/// %sub.v0 = shuffle <8 x i32> %v0, <8 x i32> v1, <0, 1, 2, 3>
/// %sub.v1 = shuffle <8 x i32> %v0, <8 x i32> v1, <4, 5, 6, 7>
/// %sub.v2 = shuffle <8 x i32> %v0, <8 x i32> v1, <8, 9, 10, 11>
/// call void llvm.riscv.seg3.store.v4i32.p0.i64(%sub.v0, %sub.v1, %sub.v2,
///                                              %ptr, i32 4)
///
/// Note that the new shufflevectors will be removed and we'll only generate one
/// vsseg3 instruction in CodeGen.
bool RISCVTargetLowering::lowerInterleavedStore(Instruction *Store,
                                                Value *LaneMask,
                                                ShuffleVectorInst *SVI,
                                                unsigned Factor,
                                                const APInt &GapMask) const {
  assert(GapMask.getBitWidth() == Factor);

  // We only support cases where the skipped fields are the trailing ones.
  // TODO: Lower to strided store if there is only a single active field.
  unsigned MaskFactor = GapMask.popcount();
  if (MaskFactor < 2 || !GapMask.isMask())
    return false;

  IRBuilder<> Builder(Store);
  const DataLayout &DL = Store->getDataLayout();
  auto Mask = SVI->getShuffleMask();
  auto *ShuffleVTy = cast<FixedVectorType>(SVI->getType());
  // Given SVI : <n*factor x ty>, then VTy : <n x ty>
  auto *VTy = FixedVectorType::get(ShuffleVTy->getElementType(),
                                   ShuffleVTy->getNumElements() / Factor);
  auto *XLenTy = Builder.getIntNTy(Subtarget.getXLen());

  Value *Ptr, *VL;
  Align Alignment;
  if (!getMemOperands(MaskFactor, VTy, XLenTy, Store, Ptr, LaneMask, VL,
                      Alignment))
    return false;

  Type *PtrTy = Ptr->getType();
  unsigned AS = PtrTy->getPointerAddressSpace();
  if (!isLegalInterleavedAccessType(VTy, MaskFactor, Alignment, AS, DL))
    return false;

  Function *SegStoreFunc;
  if (MaskFactor < Factor)
    // Strided segmented store.
    SegStoreFunc = Intrinsic::getOrInsertDeclaration(
        Store->getModule(), FixedVsssegIntrIds[MaskFactor - 2],
        {VTy, PtrTy, XLenTy, XLenTy});
  else
    // Normal segmented store.
    SegStoreFunc = Intrinsic::getOrInsertDeclaration(
        Store->getModule(), FixedVssegIntrIds[Factor - 2],
        {VTy, PtrTy, XLenTy});

  SmallVector<Value *, 10> Ops;
  SmallVector<int, 16> NewShuffleMask;

  for (unsigned i = 0; i < MaskFactor; i++) {
    // Collect shuffle mask for this lane.
    for (unsigned j = 0; j < VTy->getNumElements(); j++)
      NewShuffleMask.push_back(Mask[i + Factor * j]);

    Value *Shuffle = Builder.CreateShuffleVector(
        SVI->getOperand(0), SVI->getOperand(1), NewShuffleMask);
    Ops.push_back(Shuffle);

    NewShuffleMask.clear();
  }
  Ops.push_back(Ptr);
  if (MaskFactor < Factor) {
    // Insert the stride argument.
    unsigned ScalarSizeInBytes = DL.getTypeStoreSize(VTy->getElementType());
    Ops.push_back(ConstantInt::get(XLenTy, Factor * ScalarSizeInBytes));
  }
  Ops.append({LaneMask, VL});
  Builder.CreateCall(SegStoreFunc, Ops);

  return true;
}

bool RISCVTargetLowering::lowerDeinterleaveIntrinsicToLoad(
    Instruction *Load, Value *Mask, IntrinsicInst *DI) const {
  const unsigned Factor = getDeinterleaveIntrinsicFactor(DI->getIntrinsicID());
  if (Factor > 8)
    return false;

  IRBuilder<> Builder(Load);

  VectorType *ResVTy = getDeinterleavedVectorType(DI);

  const DataLayout &DL = Load->getDataLayout();
  auto *XLenTy = Builder.getIntNTy(Subtarget.getXLen());

  Value *Ptr, *VL;
  Align Alignment;
  if (!getMemOperands(Factor, ResVTy, XLenTy, Load, Ptr, Mask, VL, Alignment))
    return false;

  Type *PtrTy = Ptr->getType();
  unsigned AS = PtrTy->getPointerAddressSpace();
  if (!isLegalInterleavedAccessType(ResVTy, Factor, Alignment, AS, DL))
    return false;

  Value *Return;
  if (isa<FixedVectorType>(ResVTy)) {
    Return = Builder.CreateIntrinsic(FixedVlsegIntrIds[Factor - 2],
                                     {ResVTy, PtrTy, XLenTy}, {Ptr, Mask, VL});
  } else {
    unsigned SEW = DL.getTypeSizeInBits(ResVTy->getElementType());
    unsigned NumElts = ResVTy->getElementCount().getKnownMinValue();
    Type *VecTupTy = TargetExtType::get(
        Load->getContext(), "riscv.vector.tuple",
        ScalableVectorType::get(Builder.getInt8Ty(), NumElts * SEW / 8),
        Factor);
    Function *VlsegNFunc = Intrinsic::getOrInsertDeclaration(
        Load->getModule(), ScalableVlsegIntrIds[Factor - 2],
        {VecTupTy, PtrTy, Mask->getType(), VL->getType()});

    Value *Operands[] = {
        PoisonValue::get(VecTupTy),
        Ptr,
        Mask,
        VL,
        ConstantInt::get(XLenTy,
                         RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC),
        ConstantInt::get(XLenTy, Log2_64(SEW))};

    CallInst *Vlseg = Builder.CreateCall(VlsegNFunc, Operands);

    SmallVector<Type *, 2> AggrTypes{Factor, ResVTy};
    Return = PoisonValue::get(StructType::get(Load->getContext(), AggrTypes));
    for (unsigned i = 0; i < Factor; ++i) {
      Value *VecExtract = Builder.CreateIntrinsic(
          Intrinsic::riscv_tuple_extract, {ResVTy, VecTupTy},
          {Vlseg, Builder.getInt32(i)});
      Return = Builder.CreateInsertValue(Return, VecExtract, i);
    }
  }

  DI->replaceAllUsesWith(Return);
  return true;
}

bool RISCVTargetLowering::lowerInterleaveIntrinsicToStore(
    Instruction *Store, Value *Mask, ArrayRef<Value *> InterleaveValues) const {
  unsigned Factor = InterleaveValues.size();
  if (Factor > 8)
    return false;

  IRBuilder<> Builder(Store);

  auto *InVTy = cast<VectorType>(InterleaveValues[0]->getType());
  const DataLayout &DL = Store->getDataLayout();
  Type *XLenTy = Builder.getIntNTy(Subtarget.getXLen());

  Value *Ptr, *VL;
  Align Alignment;
  if (!getMemOperands(Factor, InVTy, XLenTy, Store, Ptr, Mask, VL, Alignment))
    return false;
  Type *PtrTy = Ptr->getType();
  unsigned AS = Ptr->getType()->getPointerAddressSpace();
  if (!isLegalInterleavedAccessType(InVTy, Factor, Alignment, AS, DL))
    return false;

  if (isa<FixedVectorType>(InVTy)) {
    Function *VssegNFunc = Intrinsic::getOrInsertDeclaration(
        Store->getModule(), FixedVssegIntrIds[Factor - 2],
        {InVTy, PtrTy, XLenTy});
    SmallVector<Value *, 10> Ops(InterleaveValues);
    Ops.append({Ptr, Mask, VL});
    Builder.CreateCall(VssegNFunc, Ops);
    return true;
  }
  unsigned SEW = DL.getTypeSizeInBits(InVTy->getElementType());
  unsigned NumElts = InVTy->getElementCount().getKnownMinValue();
  Type *VecTupTy = TargetExtType::get(
      Store->getContext(), "riscv.vector.tuple",
      ScalableVectorType::get(Builder.getInt8Ty(), NumElts * SEW / 8), Factor);

  Value *StoredVal = PoisonValue::get(VecTupTy);
  for (unsigned i = 0; i < Factor; ++i)
    StoredVal = Builder.CreateIntrinsic(
        Intrinsic::riscv_tuple_insert, {VecTupTy, InVTy},
        {StoredVal, InterleaveValues[i], Builder.getInt32(i)});

  Function *VssegNFunc = Intrinsic::getOrInsertDeclaration(
      Store->getModule(), ScalableVssegIntrIds[Factor - 2],
      {VecTupTy, PtrTy, Mask->getType(), VL->getType()});

  Value *Operands[] = {StoredVal, Ptr, Mask, VL,
                       ConstantInt::get(XLenTy, Log2_64(SEW))};
  Builder.CreateCall(VssegNFunc, Operands);
  return true;
}
