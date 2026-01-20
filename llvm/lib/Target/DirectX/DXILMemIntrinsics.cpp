//===- DXILMemIntrinsics.cpp - Eliminate Memory Intrinsics ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILMemIntrinsics.h"
#include "DirectX.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsDirectX.h"
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

static Type *getPointeeType(Value *Ptr, const DataLayout &DL) {
  if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
    return GV->getValueType();
  if (auto *AI = dyn_cast<AllocaInst>(Ptr))
    return AI->getAllocatedType();

  if (auto *II = dyn_cast<IntrinsicInst>(Ptr)) {
    if (II->getIntrinsicID() == Intrinsic::dx_resource_getpointer) {
      Type *Ty = cast<dxil::AnyResourceExtType>(II->getArgOperand(0)->getType())
                     ->getResourceType();
      assert(Ty && "getpointer used on untyped resource");
      return Ty;
    }
  }

  if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
    Type *Ty = GEP->getResultElementType();
    if (!Ty->isIntegerTy(8))
      return Ty;

    // We have ptradd, so we have to hope there's enough information to work out
    // what we're indexing.
    Type *IndexedType = getPointeeType(GEP->getPointerOperand(), DL);
    if (auto *AT = dyn_cast<ArrayType>(IndexedType))
      return AT->getElementType();

    if (auto *ST = dyn_cast<StructType>(IndexedType)) {
      // Indexing a struct should always be constant
      APInt ConstantOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      [[maybe_unused]] bool IsConst =
          GEP->accumulateConstantOffset(DL, ConstantOffset);
      assert(IsConst && "Non-constant GEP into struct?");

      // Now, work out what we'll find at that offset.
      const StructLayout *Layout = DL.getStructLayout(ST);
      unsigned Idx =
          Layout->getElementContainingOffset(ConstantOffset.getZExtValue());

      return ST->getTypeAtIndex(Idx);
    }

    llvm_unreachable("Could not infer type from GEP");
  }

  llvm_unreachable("Could not calculate pointee type");
}

static size_t flattenTypes(Type *ContainerTy, const DataLayout &DL,
                           SmallVectorImpl<std::pair<Type *, size_t>> &FlatTys,
                           size_t NextOffset = 0) {
  if (auto *AT = dyn_cast<ArrayType>(ContainerTy)) {
    for (uint64_t I = 0, E = AT->getNumElements(); I != E; ++I)
      NextOffset = flattenTypes(AT->getElementType(), DL, FlatTys, NextOffset);
    return NextOffset;
  }
  if (auto *ST = dyn_cast<StructType>(ContainerTy)) {
    for (Type *Ty : ST->elements())
      NextOffset = flattenTypes(Ty, DL, FlatTys, NextOffset);
    return NextOffset;
  }

  FlatTys.emplace_back(ContainerTy, NextOffset);
  return NextOffset + DL.getTypeStoreSize(ContainerTy);
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

  SmallVector<std::pair<Type *, size_t>> FlattenedTypes;
  [[maybe_unused]] size_t MaxLength =
      flattenTypes(getPointeeType(Dst, DL), DL, FlattenedTypes);
  assert(MaxLength >= ByteLength && "Dst not large enough for memcpy");

  LLVM_DEBUG({
    // Check if Src is layout compatible with Dst. This should always be true
    // unless the frontend did something wrong.
    SmallVector<std::pair<Type *, size_t>> SrcTypes;
    size_t SrcLength = flattenTypes(getPointeeType(Src, DL), DL, SrcTypes);
    assert(SrcLength >= ByteLength && "Src not large enough for memcpy");
    for (const auto &[LHS, RHS] : zip(FlattenedTypes, SrcTypes)) {
      auto &[DstTy, DstOffset] = LHS;
      auto &[SrcTy, SrcOffset] = RHS;
      assert(DstTy == SrcTy && "Mismatched types for memcpy");
      assert(DstOffset == SrcOffset && "Incompatible layouts for memcpy");
      if (DstOffset >= ByteLength)
        break;
    }
  });

  for (const auto &[Ty, Offset] : FlattenedTypes) {
    if (Offset >= ByteLength)
      break;
    // TODO: Should we skip padding types here?
    Type *Int8Ty = Builder.getInt8Ty();
    Value *ByteOffset = Builder.getInt32(Offset);
    Value *SrcPtr = Builder.CreateInBoundsGEP(Int8Ty, Src, ByteOffset);
    Value *SrcVal = Builder.CreateLoad(Ty, SrcPtr);
    Value *DstPtr = Builder.CreateInBoundsGEP(Int8Ty, Dst, ByteOffset);
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

PreservedAnalyses DXILMemIntrinsics::run(Module &M, ModuleAnalysisManager &) {
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
