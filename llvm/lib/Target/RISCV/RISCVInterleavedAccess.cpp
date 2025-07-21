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
    Alignment = cast<ConstantInt>(II->getArgOperand(1))->getAlignValue();

    if (!isa<UndefValue>(II->getOperand(3)))
      return false;

    assert(Mask && "masked.load needs a mask!");

    VL = isa<FixedVectorType>(VTy)
             ? Builder.CreateElementCount(XLenTy, VTy->getElementCount())
             : Constant::getAllOnesValue(XLenTy);
    return true;
  }
  case Intrinsic::masked_store: {
    Ptr = II->getOperand(1);
    Alignment = cast<ConstantInt>(II->getArgOperand(2))->getAlignValue();

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
    ArrayRef<unsigned> Indices, unsigned Factor) const {
  assert(Indices.size() == Shuffles.size());

  IRBuilder<> Builder(Load);

  const DataLayout &DL = Load->getDataLayout();
  auto *VTy = cast<FixedVectorType>(Shuffles[0]->getType());
  auto *XLenTy = Type::getIntNTy(Load->getContext(), Subtarget.getXLen());

  Value *Ptr, *VL;
  Align Alignment;
  if (!getMemOperands(Factor, VTy, XLenTy, Load, Ptr, Mask, VL, Alignment))
    return false;

  Type *PtrTy = Ptr->getType();
  unsigned AS = PtrTy->getPointerAddressSpace();
  if (!isLegalInterleavedAccessType(VTy, Factor, Alignment, AS, DL))
    return false;

  // If the segment load is going to be performed segment at a time anyways
  // and there's only one element used, use a strided load instead.  This
  // will be equally fast, and create less vector register pressure.
  if (Indices.size() == 1 && !Subtarget.hasOptimizedSegmentLoadStore(Factor)) {
    unsigned ScalarSizeInBytes = DL.getTypeStoreSize(VTy->getElementType());
    Value *Stride = ConstantInt::get(XLenTy, Factor * ScalarSizeInBytes);
    Value *Offset = ConstantInt::get(XLenTy, Indices[0] * ScalarSizeInBytes);
    Value *BasePtr = Builder.CreatePtrAdd(Ptr, Offset);
    // Note: Same VL as above, but i32 not xlen due to signature of
    // vp.strided.load
    VL = Builder.CreateElementCount(Builder.getInt32Ty(),
                                    VTy->getElementCount());
    CallInst *CI =
        Builder.CreateIntrinsic(Intrinsic::experimental_vp_strided_load,
                                {VTy, BasePtr->getType(), Stride->getType()},
                                {BasePtr, Stride, Mask, VL});
    CI->addParamAttr(0,
                     Attribute::getWithAlignment(CI->getContext(), Alignment));
    Shuffles[0]->replaceAllUsesWith(CI);
    return true;
  };

  CallInst *VlsegN = Builder.CreateIntrinsic(
      FixedVlsegIntrIds[Factor - 2], {VTy, PtrTy, XLenTy}, {Ptr, Mask, VL});

  for (unsigned i = 0; i < Shuffles.size(); i++) {
    Value *SubVec = Builder.CreateExtractValue(VlsegN, Indices[i]);
    Shuffles[i]->replaceAllUsesWith(SubVec);
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
bool RISCVTargetLowering::lowerInterleavedStore(StoreInst *SI,
                                                ShuffleVectorInst *SVI,
                                                unsigned Factor) const {
  IRBuilder<> Builder(SI);
  const DataLayout &DL = SI->getDataLayout();
  auto Mask = SVI->getShuffleMask();
  auto *ShuffleVTy = cast<FixedVectorType>(SVI->getType());
  // Given SVI : <n*factor x ty>, then VTy : <n x ty>
  auto *VTy = FixedVectorType::get(ShuffleVTy->getElementType(),
                                   ShuffleVTy->getNumElements() / Factor);
  if (!isLegalInterleavedAccessType(VTy, Factor, SI->getAlign(),
                                    SI->getPointerAddressSpace(), DL))
    return false;

  auto *PtrTy = SI->getPointerOperandType();
  auto *XLenTy = Type::getIntNTy(SI->getContext(), Subtarget.getXLen());

  unsigned Index;
  // If the segment store only has one active lane (i.e. the interleave is
  // just a spread shuffle), we can use a strided store instead.  This will
  // be equally fast, and create less vector register pressure.
  if (!Subtarget.hasOptimizedSegmentLoadStore(Factor) &&
      isSpreadMask(Mask, Factor, Index)) {
    unsigned ScalarSizeInBytes =
        DL.getTypeStoreSize(ShuffleVTy->getElementType());
    Value *Data = SVI->getOperand(0);
    auto *DataVTy = cast<FixedVectorType>(Data->getType());
    Value *Stride = ConstantInt::get(XLenTy, Factor * ScalarSizeInBytes);
    Value *Offset = ConstantInt::get(XLenTy, Index * ScalarSizeInBytes);
    Value *BasePtr = Builder.CreatePtrAdd(SI->getPointerOperand(), Offset);
    Value *Mask = Builder.getAllOnesMask(DataVTy->getElementCount());
    Value *VL = Builder.CreateElementCount(Builder.getInt32Ty(),
                                           VTy->getElementCount());

    CallInst *CI = Builder.CreateIntrinsic(
        Intrinsic::experimental_vp_strided_store,
        {Data->getType(), BasePtr->getType(), Stride->getType()},
        {Data, BasePtr, Stride, Mask, VL});
    CI->addParamAttr(
        1, Attribute::getWithAlignment(CI->getContext(), SI->getAlign()));

    return true;
  }

  Function *VssegNFunc = Intrinsic::getOrInsertDeclaration(
      SI->getModule(), FixedVssegIntrIds[Factor - 2], {VTy, PtrTy, XLenTy});

  SmallVector<Value *, 10> Ops;
  SmallVector<int, 16> NewShuffleMask;

  for (unsigned i = 0; i < Factor; i++) {
    // Collect shuffle mask for this lane.
    for (unsigned j = 0; j < VTy->getNumElements(); j++)
      NewShuffleMask.push_back(Mask[i + Factor * j]);

    Value *Shuffle = Builder.CreateShuffleVector(
        SVI->getOperand(0), SVI->getOperand(1), NewShuffleMask);
    Ops.push_back(Shuffle);

    NewShuffleMask.clear();
  }
  // This VL should be OK (should be executable in one vsseg instruction,
  // potentially under larger LMULs) because we checked that the fixed vector
  // type fits in isLegalInterleavedAccessType
  Value *VL = Builder.CreateElementCount(XLenTy, VTy->getElementCount());
  Value *StoreMask = Builder.getAllOnesMask(VTy->getElementCount());
  Ops.append({SI->getPointerOperand(), StoreMask, VL});

  Builder.CreateCall(VssegNFunc, Ops);

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
  auto *XLenTy = Type::getIntNTy(Load->getContext(), Subtarget.getXLen());

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
        ScalableVectorType::get(Type::getInt8Ty(Load->getContext()),
                                NumElts * SEW / 8),
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
  Type *XLenTy = Type::getIntNTy(Store->getContext(), Subtarget.getXLen());

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
      ScalableVectorType::get(Type::getInt8Ty(Store->getContext()),
                              NumElts * SEW / 8),
      Factor);

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

/// Lower an interleaved vp.store into a vssegN intrinsic.
///
/// E.g. Lower an interleaved vp.store (Factor = 2):
///
///   %is = tail call <vscale x 64 x i8>
///             @llvm.vector.interleave2.nxv64i8(
///                               <vscale x 32 x i8> %load0,
///                               <vscale x 32 x i8> %load1
///   %wide.rvl = shl nuw nsw i32 %rvl, 1
///   tail call void @llvm.vp.store.nxv64i8.p0(
///                               <vscale x 64 x i8> %is, ptr %ptr,
///                               %mask,
///                               i32 %wide.rvl)
///
/// Into:
///   call void @llvm.riscv.vsseg2.mask.nxv32i8.i64(
///                               <vscale x 32 x i8> %load1,
///                               <vscale x 32 x i8> %load2, ptr %ptr,
///                               %mask,
///                               i64 %rvl)
bool RISCVTargetLowering::lowerInterleavedVPStore(
    VPIntrinsic *Store, Value *Mask,
    ArrayRef<Value *> InterleaveOperands) const {
  assert(Mask && "Expect a valid mask");
  assert(Store->getIntrinsicID() == Intrinsic::vp_store &&
         "Unexpected intrinsic");

  const unsigned Factor = InterleaveOperands.size();

  auto *VTy = dyn_cast<VectorType>(InterleaveOperands[0]->getType());
  if (!VTy)
    return false;

  const DataLayout &DL = Store->getDataLayout();
  Align Alignment = Store->getParamAlign(1).value_or(
      DL.getABITypeAlign(VTy->getElementType()));
  if (!isLegalInterleavedAccessType(
          VTy, Factor, Alignment,
          Store->getArgOperand(1)->getType()->getPointerAddressSpace(), DL))
    return false;

  IRBuilder<> Builder(Store);
  Value *WideEVL = Store->getArgOperand(3);
  // Conservatively check if EVL is a multiple of factor, otherwise some
  // (trailing) elements might be lost after the transformation.
  if (!isMultipleOfN(WideEVL, Store->getDataLayout(), Factor))
    return false;

  auto *PtrTy = Store->getArgOperand(1)->getType();
  auto *XLenTy = Type::getIntNTy(Store->getContext(), Subtarget.getXLen());
  auto *FactorC = ConstantInt::get(WideEVL->getType(), Factor);
  Value *EVL =
      Builder.CreateZExt(Builder.CreateExactUDiv(WideEVL, FactorC), XLenTy);

  if (isa<FixedVectorType>(VTy)) {
    SmallVector<Value *, 8> Operands(InterleaveOperands);
    Operands.append({Store->getArgOperand(1), Mask, EVL});
    Builder.CreateIntrinsic(FixedVssegIntrIds[Factor - 2],
                            {VTy, PtrTy, XLenTy}, Operands);
    return true;
  }

  unsigned SEW = DL.getTypeSizeInBits(VTy->getElementType());
  unsigned NumElts = VTy->getElementCount().getKnownMinValue();
  Type *VecTupTy = TargetExtType::get(
      Store->getContext(), "riscv.vector.tuple",
      ScalableVectorType::get(Type::getInt8Ty(Store->getContext()),
                              NumElts * SEW / 8),
      Factor);

  Function *VecInsertFunc = Intrinsic::getOrInsertDeclaration(
      Store->getModule(), Intrinsic::riscv_tuple_insert, {VecTupTy, VTy});
  Value *StoredVal = PoisonValue::get(VecTupTy);
  for (unsigned i = 0; i < Factor; ++i)
    StoredVal = Builder.CreateCall(
        VecInsertFunc, {StoredVal, InterleaveOperands[i], Builder.getInt32(i)});

  Function *VssegNFunc = Intrinsic::getOrInsertDeclaration(
      Store->getModule(), ScalableVssegIntrIds[Factor - 2],
      {VecTupTy, PtrTy, Mask->getType(), EVL->getType()});

  Value *Operands[] = {StoredVal, Store->getArgOperand(1), Mask, EVL,
                       ConstantInt::get(XLenTy, Log2_64(SEW))};

  Builder.CreateCall(VssegNFunc, Operands);
  return true;
}
