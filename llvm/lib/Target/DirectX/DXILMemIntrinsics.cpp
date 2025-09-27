//===- DXILMemIntrinsics.cpp - Eliminate Memory Intrinsics ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILMemIntrinsics.h"
#include "DirectX.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "dxil-mem-intrinsics"

using namespace llvm;

void expandMemSet(MemSetInst *MemSet) {
  IRBuilder<> Builder(MemSet);
  Value *Dst = MemSet->getDest();
  Value *Val = MemSet->getValue();
  ConstantInt *LengthCI = dyn_cast<ConstantInt>(MemSet->getLength());
  assert(LengthCI && "Expected length to be a ConstantInt");

  [[maybe_unused]] const DataLayout &DL =
      Builder.GetInsertBlock()->getModule()->getDataLayout();
  [[maybe_unused]] uint64_t OrigLength = LengthCI->getZExtValue();

  AllocaInst *Alloca = dyn_cast<AllocaInst>(Dst);

  assert(Alloca && "Expected memset on an Alloca");
  assert(OrigLength == Alloca->getAllocationSize(DL)->getFixedValue() &&
         "Expected for memset size to match DataLayout size");

  Type *AllocatedTy = Alloca->getAllocatedType();
  ArrayType *ArrTy = dyn_cast<ArrayType>(AllocatedTy);
  assert(ArrTy && "Expected Alloca for an Array Type");

  Type *ElemTy = ArrTy->getElementType();
  uint64_t Size = ArrTy->getArrayNumElements();

  [[maybe_unused]] uint64_t ElemSize = DL.getTypeStoreSize(ElemTy);

  assert(ElemSize > 0 && "Size must be set");
  assert(OrigLength == ElemSize * Size && "Size in bytes must match");

  Value *TypedVal = Val;

  if (Val->getType() != ElemTy)
    TypedVal = Builder.CreateIntCast(Val, ElemTy, false);

  for (uint64_t I = 0; I < Size; ++I) {
    Value *Zero = Builder.getInt32(0);
    Value *Offset = Builder.getInt32(I);
    Value *Ptr = Builder.CreateGEP(ArrTy, Dst, {Zero, Offset}, "gep");
    Builder.CreateStore(TypedVal, Ptr);
  }

  MemSet->eraseFromParent();
}

void expandMemCpy(MemCpyInst *MemCpy) {
  IRBuilder<> Builder(MemCpy);
  Value *Dst = MemCpy->getDest();
  Value *Src = MemCpy->getSource();
  ConstantInt *LengthCI = dyn_cast<ConstantInt>(MemCpy->getLength());
  assert(LengthCI && "Expected Length to be a ConstantInt");
  assert(!MemCpy->isVolatile() && "Handling for volatile not implemented");

  uint64_t ByteLength = LengthCI->getZExtValue();
  // If length to copy is zero, no memcpy is needed.
  if (ByteLength == 0)
    return;

  const DataLayout &DL = Builder.GetInsertBlock()->getModule()->getDataLayout();

  auto GetArrTyFromVal = [](Value *Val) -> ArrayType * {
    assert(isa<AllocaInst>(Val) ||
           isa<GlobalVariable>(Val) &&
               "Expected Val to be an Alloca or Global Variable");
    if (auto *Alloca = dyn_cast<AllocaInst>(Val))
      return dyn_cast<ArrayType>(Alloca->getAllocatedType());
    if (auto *GlobalVar = dyn_cast<GlobalVariable>(Val))
      return dyn_cast<ArrayType>(GlobalVar->getValueType());
    return nullptr;
  };

  ArrayType *DstArrTy = GetArrTyFromVal(Dst);
  assert(DstArrTy && "Expected Dst of memcpy to be a Pointer to an Array Type");
  if (auto *DstGlobalVar = dyn_cast<GlobalVariable>(Dst))
    assert(!DstGlobalVar->isConstant() &&
           "The Dst of memcpy must not be a constant Global Variable");
  [[maybe_unused]] ArrayType *SrcArrTy = GetArrTyFromVal(Src);
  assert(SrcArrTy && "Expected Src of memcpy to be a Pointer to an Array Type");

  Type *DstElemTy = DstArrTy->getElementType();
  uint64_t DstElemByteSize = DL.getTypeStoreSize(DstElemTy);
  assert(DstElemByteSize > 0 && "Dst element type store size must be set");
  Type *SrcElemTy = SrcArrTy->getElementType();
  [[maybe_unused]] uint64_t SrcElemByteSize = DL.getTypeStoreSize(SrcElemTy);
  assert(SrcElemByteSize > 0 && "Src element type store size must be set");

  // This assumption simplifies implementation and covers currently-known
  // use-cases for DXIL. It may be relaxed in the future if required.
  assert(DstElemTy == SrcElemTy &&
         "The element types of Src and Dst arrays must match");

  [[maybe_unused]] uint64_t DstArrNumElems = DstArrTy->getArrayNumElements();
  assert(DstElemByteSize * DstArrNumElems >= ByteLength &&
         "Dst array size must be at least as large as the memcpy length");
  [[maybe_unused]] uint64_t SrcArrNumElems = SrcArrTy->getArrayNumElements();
  assert(SrcElemByteSize * SrcArrNumElems >= ByteLength &&
         "Src array size must be at least as large as the memcpy length");

  uint64_t NumElemsToCopy = ByteLength / DstElemByteSize;
  assert(ByteLength % DstElemByteSize == 0 &&
         "memcpy length must be divisible by array element type");
  for (uint64_t I = 0; I < NumElemsToCopy; ++I) {
    SmallVector<Value *, 2> Indices = {Builder.getInt32(0),
                                       Builder.getInt32(I)};
    Value *SrcPtr = Builder.CreateInBoundsGEP(SrcArrTy, Src, Indices, "gep");
    Value *SrcVal = Builder.CreateLoad(SrcElemTy, SrcPtr);
    Value *DstPtr = Builder.CreateInBoundsGEP(DstArrTy, Dst, Indices, "gep");
    Builder.CreateStore(SrcVal, DstPtr);
  }

  MemCpy->eraseFromParent();
}

void expandMemMove(MemMoveInst *MemMove) {
  report_fatal_error("memmove expansion is not implemented yet.");
}

static bool eliminateMemIntrinsics(Module &M) {
  bool HadMemIntrinsicUses = false;
  for (auto &F : make_early_inc_range(M.functions())) {
    Intrinsic::ID IID = F.getIntrinsicID();
    switch (IID) {
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memmove:
    case Intrinsic::memset:
    case Intrinsic::memset_inline:
      break;
    default:
      continue;
    }
    for (User *U : make_early_inc_range(F.users())) {
      HadMemIntrinsicUses = true;
      if (auto *MemSet = dyn_cast<MemSetInst>(U))
        expandMemSet(MemSet);
      else if (auto *MemCpy = dyn_cast<MemCpyInst>(U))
        expandMemCpy(MemCpy);
      else if (auto *MemMove = dyn_cast<MemMoveInst>(U))
        expandMemMove(MemMove);
      else
        llvm_unreachable("Unhandled memory intrinsic");
    }
    assert(F.user_empty() && "Mem intrinsic not eliminated?");
    F.eraseFromParent();
  }
  return HadMemIntrinsicUses;
}

PreservedAnalyses DXILMemIntrinsics::run(Module & M, ModuleAnalysisManager &) {
  if (eliminateMemIntrinsics(M))
      return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

class DXILMemIntrinsicsLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override { return eliminateMemIntrinsics(M); }
  DXILMemIntrinsicsLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};
char DXILMemIntrinsicsLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILMemIntrinsicsLegacy, DEBUG_TYPE,
                      "DXIL Memory Intrinsic Elimination", false, false)
INITIALIZE_PASS_END(DXILMemIntrinsicsLegacy, DEBUG_TYPE,
                    "DXIL Memory Intrinsic Elimination", false, false)

ModulePass *llvm::createDXILMemIntrinsicsLegacyPass() {
  return new DXILMemIntrinsicsLegacy();
}
