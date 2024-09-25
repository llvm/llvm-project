//===AMDGPUAsanInstrumentation.cpp - ASAN related helper functions===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------===//

#include "AMDGPUAsanInstrumentation.h"

#define DEBUG_TYPE "amdgpu-asan-instrumentation"

using namespace llvm;

namespace llvm {
namespace AMDGPU {

static uint64_t getRedzoneSizeForScale(int AsanScale) {
  // Redzone used for stack and globals is at least 32 bytes.
  // For scales 6 and 7, the redzone has to be 64 and 128 bytes respectively.
  return std::max(32U, 1U << AsanScale);
}

static uint64_t getMinRedzoneSizeForGlobal(int AsanScale) {
  return getRedzoneSizeForScale(AsanScale);
}

uint64_t getRedzoneSizeForGlobal(int AsanScale, uint64_t SizeInBytes) {
  constexpr uint64_t kMaxRZ = 1 << 18;
  const uint64_t MinRZ = getMinRedzoneSizeForGlobal(AsanScale);

  uint64_t RZ = 0;
  if (SizeInBytes <= MinRZ / 2) {
    // Reduce redzone size for small size objects, e.g. int, char[1]. MinRZ is
    // at least 32 bytes, optimize when SizeInBytes is less than or equal to
    // half of MinRZ.
    RZ = MinRZ - SizeInBytes;
  } else {
    // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 * SizeInBytes.
    RZ = std::clamp((SizeInBytes / MinRZ / 4) * MinRZ, MinRZ, kMaxRZ);

    // Round up to multiple of MinRZ.
    if (SizeInBytes % MinRZ)
      RZ += MinRZ - (SizeInBytes % MinRZ);
  }

  assert((RZ + SizeInBytes) % MinRZ == 0);

  return RZ;
}

static size_t TypeStoreSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = llvm::countr_zero(TypeSize / 8);
  return Res;
}

static Instruction *genAMDGPUReportBlock(Module &M, IRBuilder<> &IRB,
                                         Value *Cond, bool Recover) {
  Value *ReportCond = Cond;
  if (!Recover) {
    auto *Ballot =
        IRB.CreateIntrinsic(Intrinsic::amdgcn_ballot, IRB.getInt64Ty(), {Cond});
    ReportCond = IRB.CreateIsNotNull(Ballot);
  }

  auto *Trm = SplitBlockAndInsertIfThen(
      ReportCond, &*IRB.GetInsertPoint(), false,
      MDBuilder(M.getContext()).createUnlikelyBranchWeights());
  Trm->getParent()->setName("asan.report");

  if (Recover)
    return Trm;

  Trm = SplitBlockAndInsertIfThen(Cond, Trm, false);
  IRB.SetInsertPoint(Trm);
  return IRB.CreateIntrinsic(Intrinsic::amdgcn_unreachable, {}, {});
}

static Value *createSlowPathCmp(Module &M, IRBuilder<> &IRB, Type *IntptrTy,
                                Value *AddrLong, Value *ShadowValue,
                                uint32_t TypeStoreSize, int AsanScale) {
  uint64_t Granularity = static_cast<uint64_t>(1) << AsanScale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte =
      IRB.CreateAnd(AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeStoreSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeStoreSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte =
      IRB.CreateIntCast(LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

static Instruction *generateCrashCode(Module &M, IRBuilder<> &IRB,
                                      Type *IntptrTy, Instruction *InsertBefore,
                                      Value *Addr, bool IsWrite,
                                      size_t AccessSizeIndex,
                                      Value *SizeArgument, bool Recover) {
  IRB.SetInsertPoint(InsertBefore);
  CallInst *Call = nullptr;
  SmallString<128> kAsanReportErrorTemplate{"__asan_report_"};
  SmallString<64> TypeStr{IsWrite ? "store" : "load"};
  SmallString<64> EndingStr{Recover ? "_noabort" : ""};

  SmallString<128> AsanErrorCallbackSizedString;
  raw_svector_ostream AsanErrorCallbackSizedOS(AsanErrorCallbackSizedString);
  AsanErrorCallbackSizedOS << kAsanReportErrorTemplate << TypeStr << "_n"
                           << EndingStr;

  SmallVector<Type *, 3> Args2 = {IntptrTy, IntptrTy};
  AttributeList AL2;
  FunctionCallee AsanErrorCallbackSized = M.getOrInsertFunction(
      AsanErrorCallbackSizedOS.str(),
      FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);
  SmallVector<Type *, 2> Args1{1, IntptrTy};
  AttributeList AL1;

  SmallString<128> AsanErrorCallbackString;
  raw_svector_ostream AsanErrorCallbackOS(AsanErrorCallbackString);
  AsanErrorCallbackOS << kAsanReportErrorTemplate << TypeStr
                      << (1ULL << AccessSizeIndex) << EndingStr;

  FunctionCallee AsanErrorCallback = M.getOrInsertFunction(
      AsanErrorCallbackOS.str(),
      FunctionType::get(IRB.getVoidTy(), Args1, false), AL1);
  if (SizeArgument) {
    Call = IRB.CreateCall(AsanErrorCallbackSized, {Addr, SizeArgument});
  } else {
    Call = IRB.CreateCall(AsanErrorCallback, Addr);
  }

  Call->setCannotMerge();
  return Call;
}

static Value *memToShadow(Module &M, IRBuilder<> &IRB, Type *IntptrTy,
                          Value *Shadow, int AsanScale, uint32_t AsanOffset) {
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, AsanScale);
  if (AsanOffset == 0)
    return Shadow;
  // (Shadow >> scale) | offset
  Value *ShadowBase = ConstantInt::get(IntptrTy, AsanOffset);
  return IRB.CreateAdd(Shadow, ShadowBase);
}

static void instrumentAddressImpl(Module &M, IRBuilder<> &IRB,
                                  Instruction *OrigIns,
                                  Instruction *InsertBefore, Value *Addr,
                                  Align Alignment, uint32_t TypeStoreSize,
                                  bool IsWrite, Value *SizeArgument,
                                  bool UseCalls, bool Recover, int AsanScale,
                                  int AsanOffset) {
  Type *AddrTy = Addr->getType();
  Type *IntptrTy = M.getDataLayout().getIntPtrType(
      M.getContext(), AddrTy->getPointerAddressSpace());
  IRB.SetInsertPoint(InsertBefore);
  size_t AccessSizeIndex = TypeStoreSizeToSizeIndex(TypeStoreSize);
  Type *ShadowTy = IntegerType::get(M.getContext(),
                                    std::max(8U, TypeStoreSize >> AsanScale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *AddrLong = IRB.CreatePtrToInt(Addr, IntptrTy);
  Value *ShadowPtr =
      memToShadow(M, IRB, IntptrTy, AddrLong, AsanScale, AsanOffset);
  const uint64_t ShadowAlign =
      std::max<uint64_t>(Alignment.value() >> AsanScale, 1);
  Value *ShadowValue = IRB.CreateAlignedLoad(
      ShadowTy, IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy), Align(ShadowAlign));
  Value *Cmp = IRB.CreateIsNotNull(ShadowValue);
  auto *Cmp2 = createSlowPathCmp(M, IRB, IntptrTy, AddrLong, ShadowValue,
                                 TypeStoreSize, AsanScale);
  Cmp = IRB.CreateAnd(Cmp, Cmp2);
  Instruction *CrashTerm = genAMDGPUReportBlock(M, IRB, Cmp, Recover);
  Instruction *Crash =
      generateCrashCode(M, IRB, IntptrTy, CrashTerm, AddrLong, IsWrite,
                        AccessSizeIndex, SizeArgument, Recover);
  Crash->setDebugLoc(OrigIns->getDebugLoc());
  return;
}

void instrumentAddress(Module &M, IRBuilder<> &IRB, Instruction *OrigIns,
                       Instruction *InsertBefore, Value *Addr, Align Alignment,
                       TypeSize TypeStoreSize, bool IsWrite,
                       Value *SizeArgument, bool UseCalls, bool Recover,
                       int AsanScale, int AsanOffset) {
  if (!TypeStoreSize.isScalable()) {
    unsigned Granularity = 1 << AsanScale;
    const auto FixedSize = TypeStoreSize.getFixedValue();
    switch (FixedSize) {
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      if (Alignment.value() >= Granularity ||
          Alignment.value() >= FixedSize / 8)
        return instrumentAddressImpl(
            M, IRB, OrigIns, InsertBefore, Addr, Alignment, FixedSize, IsWrite,
            SizeArgument, UseCalls, Recover, AsanScale, AsanOffset);
    }
  }
  // Instrument unusual size or unusual alignment.
  IRB.SetInsertPoint(InsertBefore);
  Type *AddrTy = Addr->getType();
  Type *IntptrTy = M.getDataLayout().getIntPtrType(AddrTy);
  Value *NumBits = IRB.CreateTypeSize(IntptrTy, TypeStoreSize);
  Value *Size = IRB.CreateLShr(NumBits, ConstantInt::get(IntptrTy, 3));
  Value *AddrLong = IRB.CreatePtrToInt(Addr, IntptrTy);
  Value *SizeMinusOne = IRB.CreateAdd(Size, ConstantInt::get(IntptrTy, -1));
  Value *LastByte =
      IRB.CreateIntToPtr(IRB.CreateAdd(AddrLong, SizeMinusOne), AddrTy);
  instrumentAddressImpl(M, IRB, OrigIns, InsertBefore, Addr, {}, 8, IsWrite,
                        SizeArgument, UseCalls, Recover, AsanScale, AsanOffset);
  instrumentAddressImpl(M, IRB, OrigIns, InsertBefore, LastByte, {}, 8, IsWrite,
                        SizeArgument, UseCalls, Recover, AsanScale, AsanOffset);
}

void getInterestingMemoryOperands(
    Module &M, Instruction *I,
    SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  const DataLayout &DL = M.getDataLayout();
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                             LI->getType(), LI->getAlign());
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                             SI->getValueOperand()->getType(), SI->getAlign());
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    Interesting.emplace_back(I, RMW->getPointerOperandIndex(), true,
                             RMW->getValOperand()->getType(), std::nullopt);
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    Interesting.emplace_back(I, XCHG->getPointerOperandIndex(), true,
                             XCHG->getCompareOperand()->getType(),
                             std::nullopt);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    switch (CI->getIntrinsicID()) {
    case Intrinsic::masked_load:
    case Intrinsic::masked_store:
    case Intrinsic::masked_gather:
    case Intrinsic::masked_scatter: {
      bool IsWrite = CI->getType()->isVoidTy();
      // Masked store has an initial operand for the value.
      unsigned OpOffset = IsWrite ? 1 : 0;
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = Align(1);
      // Otherwise no alignment guarantees. We probably got Undef.
      if (auto *Op = dyn_cast<ConstantInt>(CI->getOperand(1 + OpOffset)))
        Alignment = Op->getMaybeAlignValue();
      Value *Mask = CI->getOperand(2 + OpOffset);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, Mask);
      break;
    }
    case Intrinsic::masked_expandload:
    case Intrinsic::masked_compressstore: {
      bool IsWrite = CI->getIntrinsicID() == Intrinsic::masked_compressstore;
      unsigned OpOffset = IsWrite ? 1 : 0;
      auto BasePtr = CI->getOperand(OpOffset);
      MaybeAlign Alignment = BasePtr->getPointerAlignment(DL);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      IRBuilder<> IB(I);
      Value *Mask = CI->getOperand(1 + OpOffset);
      Type *IntptrTy = M.getDataLayout().getIntPtrType(
          M.getContext(), BasePtr->getType()->getPointerAddressSpace());
      // Use the popcount of Mask as the effective vector length.
      Type *ExtTy = VectorType::get(IntptrTy, cast<VectorType>(Ty));
      Value *ExtMask = IB.CreateZExt(Mask, ExtTy);
      Value *EVL = IB.CreateAddReduce(ExtMask);
      Value *TrueMask = ConstantInt::get(Mask->getType(), 1);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, TrueMask,
                               EVL);
      break;
    }
    case Intrinsic::vp_load:
    case Intrinsic::vp_store:
    case Intrinsic::experimental_vp_strided_load:
    case Intrinsic::experimental_vp_strided_store: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = CI->getType()->isVoidTy();
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getOperand(PtrOpNo)->getPointerAlignment(DL);
      Value *Stride = nullptr;
      if (IID == Intrinsic::experimental_vp_strided_store ||
          IID == Intrinsic::experimental_vp_strided_load) {
        Stride = VPI->getOperand(PtrOpNo + 1);
        // Use the pointer alignment as the element alignment if the stride is a
        // mutiple of the pointer alignment. Otherwise, the element alignment
        // should be Align(1).
        unsigned PointerAlign = Alignment.valueOrOne().value();
        if (!isa<ConstantInt>(Stride) ||
            cast<ConstantInt>(Stride)->getZExtValue() % PointerAlign != 0)
          Alignment = Align(1);
      }
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(), VPI->getVectorLengthParam(),
                               Stride);
      break;
    }
    case Intrinsic::vp_gather:
    case Intrinsic::vp_scatter: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = IID == Intrinsic::vp_scatter;
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getPointerAlignment();
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(),
                               VPI->getVectorLengthParam());
      break;
    }
    case Intrinsic::amdgcn_raw_buffer_load:
    case Intrinsic::amdgcn_raw_ptr_buffer_load:
    case Intrinsic::amdgcn_raw_buffer_load_format:
    case Intrinsic::amdgcn_raw_ptr_buffer_load_format:
    case Intrinsic::amdgcn_raw_tbuffer_load:
    case Intrinsic::amdgcn_raw_ptr_tbuffer_load:
    case Intrinsic::amdgcn_struct_buffer_load:
    case Intrinsic::amdgcn_struct_ptr_buffer_load:
    case Intrinsic::amdgcn_struct_buffer_load_format:
    case Intrinsic::amdgcn_struct_ptr_buffer_load_format:
    case Intrinsic::amdgcn_struct_tbuffer_load:
    case Intrinsic::amdgcn_struct_ptr_tbuffer_load:
    case Intrinsic::amdgcn_s_buffer_load:
    case Intrinsic::amdgcn_global_load_tr_b64:
    case Intrinsic::amdgcn_global_load_tr_b128: {
      unsigned PtrOpNo = 0;
      bool IsWrite = false;
      Type *Ty = CI->getType();
      Value *Ptr = CI->getArgOperand(PtrOpNo);
      MaybeAlign Alignment = Ptr->getPointerAlignment(DL);
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment);
      break;
    }
    case Intrinsic::amdgcn_raw_tbuffer_store:
    case Intrinsic::amdgcn_raw_ptr_tbuffer_store:
    case Intrinsic::amdgcn_raw_buffer_store:
    case Intrinsic::amdgcn_raw_ptr_buffer_store:
    case Intrinsic::amdgcn_raw_buffer_store_format:
    case Intrinsic::amdgcn_raw_ptr_buffer_store_format:
    case Intrinsic::amdgcn_struct_buffer_store:
    case Intrinsic::amdgcn_struct_ptr_buffer_store:
    case Intrinsic::amdgcn_struct_buffer_store_format:
    case Intrinsic::amdgcn_struct_ptr_buffer_store_format:
    case Intrinsic::amdgcn_struct_tbuffer_store:
    case Intrinsic::amdgcn_struct_ptr_tbuffer_store: {
      unsigned PtrOpNo = 1;
      bool IsWrite = true;
      Value *Ptr = CI->getArgOperand(PtrOpNo);
      Type *Ty = Ptr->getType();
      MaybeAlign Alignment = Ptr->getPointerAlignment(DL);
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment);
      break;
    }
    default:
      for (unsigned ArgNo = 0; ArgNo < CI->arg_size(); ArgNo++) {
        if (Type *Ty = CI->getParamByRefType(ArgNo)) {
          Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
        } else if (Type *Ty = CI->getParamByValType(ArgNo)) {
          Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
        }
      }
    }
  }
}
} // end namespace AMDGPU
} // end namespace llvm
