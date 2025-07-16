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
    LoadInst *LI, ArrayRef<ShuffleVectorInst *> Shuffles,
    ArrayRef<unsigned> Indices, unsigned Factor) const {
  assert(Indices.size() == Shuffles.size());

  IRBuilder<> Builder(LI);

  const DataLayout &DL = LI->getDataLayout();

  auto *VTy = cast<FixedVectorType>(Shuffles[0]->getType());
  if (!isLegalInterleavedAccessType(VTy, Factor, LI->getAlign(),
                                    LI->getPointerAddressSpace(), DL))
    return false;

  auto *PtrTy = LI->getPointerOperandType();
  auto *XLenTy = Type::getIntNTy(LI->getContext(), Subtarget.getXLen());

  // If the segment load is going to be performed segment at a time anyways
  // and there's only one element used, use a strided load instead.  This
  // will be equally fast, and create less vector register pressure.
  if (Indices.size() == 1 && !Subtarget.hasOptimizedSegmentLoadStore(Factor)) {
    unsigned ScalarSizeInBytes = DL.getTypeStoreSize(VTy->getElementType());
    Value *Stride = ConstantInt::get(XLenTy, Factor * ScalarSizeInBytes);
    Value *Offset = ConstantInt::get(XLenTy, Indices[0] * ScalarSizeInBytes);
    Value *BasePtr = Builder.CreatePtrAdd(LI->getPointerOperand(), Offset);
    Value *Mask = Builder.getAllOnesMask(VTy->getElementCount());
    Value *VL = Builder.CreateElementCount(Builder.getInt32Ty(),
                                           VTy->getElementCount());

    CallInst *CI =
        Builder.CreateIntrinsic(Intrinsic::experimental_vp_strided_load,
                                {VTy, BasePtr->getType(), Stride->getType()},
                                {BasePtr, Stride, Mask, VL});
    CI->addParamAttr(
        0, Attribute::getWithAlignment(CI->getContext(), LI->getAlign()));
    Shuffles[0]->replaceAllUsesWith(CI);
    return true;
  };

  Value *VL = Builder.CreateElementCount(XLenTy, VTy->getElementCount());
  Value *Mask = Builder.getAllOnesMask(VTy->getElementCount());
  CallInst *VlsegN = Builder.CreateIntrinsic(
      FixedVlsegIntrIds[Factor - 2], {VTy, PtrTy, XLenTy},
      {LI->getPointerOperand(), Mask, VL});

  for (unsigned i = 0; i < Shuffles.size(); i++) {
    Value *SubVec = Builder.CreateExtractValue(VlsegN, Indices[i]);
    Shuffles[i]->replaceAllUsesWith(SubVec);
  }

  return true;
}

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

static bool isMultipleOfN(const Value *V, const DataLayout &DL, unsigned N) {
  assert(N);
  if (N == 1)
    return true;

  using namespace PatternMatch;
  // Right now we're only recognizing the simplest pattern.
  uint64_t C;
  if (match(V, m_CombineOr(m_ConstantInt(C),
                           m_c_Mul(m_Value(), m_ConstantInt(C)))) &&
      C && C % N == 0)
    return true;

  if (isPowerOf2_32(N)) {
    KnownBits KB = llvm::computeKnownBits(V, DL);
    return KB.countMinTrailingZeros() >= Log2_32(N);
  }

  return false;
}

bool RISCVTargetLowering::lowerDeinterleaveIntrinsicToLoad(
    Instruction *Load, Value *Mask,
    ArrayRef<Value *> DeinterleaveValues) const {
  const unsigned Factor = DeinterleaveValues.size();
  if (Factor > 8)
    return false;

  IRBuilder<> Builder(Load);

  Value *FirstActive =
      *llvm::find_if(DeinterleaveValues, [](Value *V) { return V != nullptr; });
  VectorType *ResVTy = cast<VectorType>(FirstActive->getType());

  const DataLayout &DL = Load->getDataLayout();
  auto *XLenTy = Type::getIntNTy(Load->getContext(), Subtarget.getXLen());

  Value *Ptr, *VL;
  Align Alignment;
  if (auto *LI = dyn_cast<LoadInst>(Load)) {
    assert(LI->isSimple());
    Ptr = LI->getPointerOperand();
    Alignment = LI->getAlign();
    assert(!Mask && "Unexpected mask on a load\n");
    Mask = Builder.getAllOnesMask(ResVTy->getElementCount());
    VL = isa<FixedVectorType>(ResVTy)
             ? Builder.CreateElementCount(XLenTy, ResVTy->getElementCount())
             : Constant::getAllOnesValue(XLenTy);
  } else {
    auto *VPLoad = cast<VPIntrinsic>(Load);
    assert(VPLoad->getIntrinsicID() == Intrinsic::vp_load &&
           "Unexpected intrinsic");
    Ptr = VPLoad->getMemoryPointerParam();
    Alignment = VPLoad->getPointerAlignment().value_or(
        DL.getABITypeAlign(ResVTy->getElementType()));

    assert(Mask && "vp.load needs a mask!");

    Value *WideEVL = VPLoad->getVectorLengthParam();
    // Conservatively check if EVL is a multiple of factor, otherwise some
    // (trailing) elements might be lost after the transformation.
    if (!isMultipleOfN(WideEVL, Load->getDataLayout(), Factor))
      return false;

    VL = Builder.CreateZExt(
        Builder.CreateUDiv(WideEVL,
                           ConstantInt::get(WideEVL->getType(), Factor)),
        XLenTy);
  }

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

  for (auto [Idx, DIV] : enumerate(DeinterleaveValues)) {
    if (!DIV)
      continue;
    // We have to create a brand new ExtractValue to replace each
    // of these old ExtractValue instructions.
    Value *NewEV =
        Builder.CreateExtractValue(Return, {static_cast<unsigned>(Idx)});
    DIV->replaceAllUsesWith(NewEV);
  }

  return true;
}

bool RISCVTargetLowering::lowerInterleaveIntrinsicToStore(
    StoreInst *SI, ArrayRef<Value *> InterleaveValues) const {
  unsigned Factor = InterleaveValues.size();
  if (Factor > 8)
    return false;

  assert(SI->isSimple());
  IRBuilder<> Builder(SI);

  auto *InVTy = cast<VectorType>(InterleaveValues[0]->getType());
  auto *PtrTy = SI->getPointerOperandType();
  const DataLayout &DL = SI->getDataLayout();

  if (!isLegalInterleavedAccessType(InVTy, Factor, SI->getAlign(),
                                    SI->getPointerAddressSpace(), DL))
    return false;

  Type *XLenTy = Type::getIntNTy(SI->getContext(), Subtarget.getXLen());

  if (isa<FixedVectorType>(InVTy)) {
    Function *VssegNFunc = Intrinsic::getOrInsertDeclaration(
        SI->getModule(), FixedVssegIntrIds[Factor - 2], {InVTy, PtrTy, XLenTy});

    SmallVector<Value *, 10> Ops(InterleaveValues);
    Value *VL = Builder.CreateElementCount(XLenTy, InVTy->getElementCount());
    Value *Mask = Builder.getAllOnesMask(InVTy->getElementCount());
    Ops.append({SI->getPointerOperand(), Mask, VL});

    Builder.CreateCall(VssegNFunc, Ops);
    return true;
  }
  unsigned SEW = DL.getTypeSizeInBits(InVTy->getElementType());
  unsigned NumElts = InVTy->getElementCount().getKnownMinValue();
  Type *VecTupTy = TargetExtType::get(
      SI->getContext(), "riscv.vector.tuple",
      ScalableVectorType::get(Type::getInt8Ty(SI->getContext()),
                              NumElts * SEW / 8),
      Factor);

  Value *VL = Constant::getAllOnesValue(XLenTy);
  Value *Mask = Builder.getAllOnesMask(InVTy->getElementCount());

  Value *StoredVal = PoisonValue::get(VecTupTy);
  for (unsigned i = 0; i < Factor; ++i)
    StoredVal = Builder.CreateIntrinsic(
        Intrinsic::riscv_tuple_insert, {VecTupTy, InVTy},
        {StoredVal, InterleaveValues[i], Builder.getInt32(i)});

  Function *VssegNFunc = Intrinsic::getOrInsertDeclaration(
      SI->getModule(), ScalableVssegIntrIds[Factor - 2],
      {VecTupTy, PtrTy, Mask->getType(), VL->getType()});

  Value *Operands[] = {StoredVal, SI->getPointerOperand(), Mask, VL,
                       ConstantInt::get(XLenTy, Log2_64(SEW))};
  Builder.CreateCall(VssegNFunc, Operands);
  return true;
}

/// Lower an interleaved vp.load into a vlsegN intrinsic.
///
/// E.g. Lower an interleaved vp.load (Factor = 2):
///   %l = call <vscale x 64 x i8> @llvm.vp.load.nxv64i8.p0(ptr %ptr,
///                                                         %mask,
///                                                         i32 %wide.rvl)
///   %dl = tail call { <vscale x 32 x i8>, <vscale x 32 x i8> }
///             @llvm.vector.deinterleave2.nxv64i8(
///               <vscale x 64 x i8> %l)
///   %r0 = extractvalue { <vscale x 32 x i8>, <vscale x 32 x i8> } %dl, 0
///   %r1 = extractvalue { <vscale x 32 x i8>, <vscale x 32 x i8> } %dl, 1
///
/// Into:
///   %rvl = udiv %wide.rvl, 2
///   %sl = call { <vscale x 32 x i8>, <vscale x 32 x i8> }
///             @llvm.riscv.vlseg2.mask.nxv32i8.i64(<vscale x 32 x i8> undef,
///                                                 <vscale x 32 x i8> undef,
///                                                 ptr %ptr,
///                                                 %mask,
///                                                 i64 %rvl,
///                                                 i64 1)
///   %r0 = extractvalue { <vscale x 32 x i8>, <vscale x 32 x i8> } %sl, 0
///   %r1 = extractvalue { <vscale x 32 x i8>, <vscale x 32 x i8> } %sl, 1
///
/// NOTE: the deinterleave2 intrinsic won't be touched and is expected to be
/// removed by the caller
/// TODO: We probably can loosen the dependency on matching extractvalue when
/// dealing with factor of 2 (extractvalue is still required for most of other
/// factors though).
bool RISCVTargetLowering::lowerInterleavedVPLoad(
    VPIntrinsic *Load, Value *Mask,
    ArrayRef<Value *> DeinterleaveResults) const {
  const unsigned Factor = DeinterleaveResults.size();
  assert(Mask && "Expect a valid mask");
  assert(Load->getIntrinsicID() == Intrinsic::vp_load &&
         "Unexpected intrinsic");

  Value *FirstActive = *llvm::find_if(DeinterleaveResults,
                                      [](Value *V) { return V != nullptr; });
  VectorType *VTy = cast<VectorType>(FirstActive->getType());

  auto &DL = Load->getModule()->getDataLayout();
  Align Alignment = Load->getParamAlign(0).value_or(
      DL.getABITypeAlign(VTy->getElementType()));
  if (!isLegalInterleavedAccessType(
          VTy, Factor, Alignment,
          Load->getArgOperand(0)->getType()->getPointerAddressSpace(), DL))
    return false;

  IRBuilder<> Builder(Load);

  Value *WideEVL = Load->getVectorLengthParam();
  // Conservatively check if EVL is a multiple of factor, otherwise some
  // (trailing) elements might be lost after the transformation.
  if (!isMultipleOfN(WideEVL, Load->getDataLayout(), Factor))
    return false;

  auto *PtrTy = Load->getArgOperand(0)->getType();
  auto *XLenTy = Type::getIntNTy(Load->getContext(), Subtarget.getXLen());
  Value *EVL = Builder.CreateZExt(
      Builder.CreateUDiv(WideEVL, ConstantInt::get(WideEVL->getType(), Factor)),
      XLenTy);

  Value *Return = nullptr;
  if (isa<FixedVectorType>(VTy)) {
    Return = Builder.CreateIntrinsic(FixedVlsegIntrIds[Factor - 2],
                                     {VTy, PtrTy, XLenTy},
                                     {Load->getArgOperand(0), Mask, EVL});
  } else {
    unsigned SEW = DL.getTypeSizeInBits(VTy->getElementType());
    unsigned NumElts = VTy->getElementCount().getKnownMinValue();
    Type *VecTupTy = TargetExtType::get(
        Load->getContext(), "riscv.vector.tuple",
        ScalableVectorType::get(Type::getInt8Ty(Load->getContext()),
                                NumElts * SEW / 8),
        Factor);

    Function *VlsegNFunc = Intrinsic::getOrInsertDeclaration(
        Load->getModule(), ScalableVlsegIntrIds[Factor - 2],
        {VecTupTy, PtrTy, Mask->getType(), EVL->getType()});

    Value *Operands[] = {
        PoisonValue::get(VecTupTy),
        Load->getArgOperand(0),
        Mask,
        EVL,
        ConstantInt::get(XLenTy,
                         RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC),
        ConstantInt::get(XLenTy, Log2_64(SEW))};

    CallInst *VlsegN = Builder.CreateCall(VlsegNFunc, Operands);

    SmallVector<Type *, 8> AggrTypes{Factor, VTy};
    Return = PoisonValue::get(StructType::get(Load->getContext(), AggrTypes));
    Function *VecExtractFunc = Intrinsic::getOrInsertDeclaration(
        Load->getModule(), Intrinsic::riscv_tuple_extract, {VTy, VecTupTy});
    for (unsigned i = 0; i < Factor; ++i) {
      Value *VecExtract =
          Builder.CreateCall(VecExtractFunc, {VlsegN, Builder.getInt32(i)});
      Return = Builder.CreateInsertValue(Return, VecExtract, i);
    }
  }

  for (auto [Idx, DIO] : enumerate(DeinterleaveResults)) {
    if (!DIO)
      continue;
    // We have to create a brand new ExtractValue to replace each
    // of these old ExtractValue instructions.
    Value *NewEV =
        Builder.CreateExtractValue(Return, {static_cast<unsigned>(Idx)});
    DIO->replaceAllUsesWith(NewEV);
  }

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
  Value *EVL = Builder.CreateZExt(
      Builder.CreateUDiv(WideEVL, ConstantInt::get(WideEVL->getType(), Factor)),
      XLenTy);

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
