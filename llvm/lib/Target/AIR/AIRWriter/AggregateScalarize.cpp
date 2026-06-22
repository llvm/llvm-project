//===- AggregateScalarize.cpp - Scalarize aggregate/bool-vec ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIR v1 bitcode / the AGX JIT reject aggregate-typed loads/stores and
// bool-vector bitcasts. These transforms scalarize each into per-element /
// per-bit operations the writer and JIT accept.
//
//===----------------------------------------------------------------------===//

#include "AggregateScalarize.h"
#include "PointerRepairUtil.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

static bool isI1VecScalarBitcast(BitCastInst *BC) {
  auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
  auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
  return (SV && SV->getElementType()->isIntegerTy(1) &&
          BC->getDestTy()->isIntegerTy(SV->getNumElements())) ||
         (DV && DV->getElementType()->isIntegerTy(1) &&
          BC->getSrcTy()->isIntegerTy(DV->getNumElements()));
}

static bool isI1VecToSubByteVecBitcast(BitCastInst *BC) {
  auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
  auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
  if (!SV || !DV || !SV->getElementType()->isIntegerTy(1) ||
      !DV->getElementType()->isIntegerTy())
    return false;
  unsigned K = DV->getElementType()->getIntegerBitWidth();
  return SV->getNumElements() == DV->getNumElements() * K;
}

void scalarizeBoolVectorCasts(Module &M) {
  auto Casts = collectInsts<BitCastInst>(M, [](BitCastInst *BC) {
    return isI1VecScalarBitcast(BC) || isI1VecToSubByteVecBitcast(BC);
  });
  for (auto *BC : Casts) {
    IRBuilder<> B(BC);
    Value *R;
    if (isI1VecToSubByteVecBitcast(BC)) {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      Type *ElemTy = DV->getElementType();
      unsigned K = ElemTy->getIntegerBitWidth();
      R = UndefValue::get(DV);
      for (unsigned E = 0; E < DV->getNumElements(); ++E) {
        Value *Packed = ConstantInt::get(ElemTy, 0);
        for (unsigned K0 = 0; K0 < K; ++K0) {
          Value *Bit = B.CreateZExt(
              B.CreateExtractElement(BC->getOperand(0), B.getInt64(E * K + K0)),
              ElemTy);
          Packed = B.CreateOr(Packed, B.CreateShl(Bit, K0));
        }
        R = B.CreateInsertElement(R, Packed, B.getInt64(E));
      }
    } else if (auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy())) {
      R = ConstantInt::get(BC->getDestTy(), 0);
      for (unsigned L = 0; L < SV->getNumElements(); ++L) {
        Value *Bit = B.CreateZExt(
            B.CreateExtractElement(BC->getOperand(0), B.getInt64(L)),
            BC->getDestTy());
        R = B.CreateOr(R, B.CreateShl(Bit, L));
      }
    } else {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      R = UndefValue::get(DV);
      for (unsigned L = 0; L < DV->getNumElements(); ++L) {
        Value *Bit =
            B.CreateTrunc(B.CreateLShr(BC->getOperand(0), L), B.getInt1Ty());
        R = B.CreateInsertElement(R, Bit, B.getInt64(L));
      }
    }
    BC->replaceAllUsesWith(R);
    BC->eraseFromParent();
  }
}

void scalarizeAggregateLoads(Module &M, PointeeTypeMap &PTM) {
  auto Aggs = collectInsts<LoadInst>(
      M, [](LoadInst *LI) { return isa<ArrayType>(LI->getType()); });
  for (LoadInst *LI : Aggs) {
    auto *AT = cast<ArrayType>(LI->getType());
    Type *ElemTy = AT->getElementType();
    Value *Ptr = retypePointerVia(LI->getPointerOperand(), ElemTy, LI, PTM);
    Value *Agg = UndefValue::get(AT);
    IRBuilder<> B(LI);
    for (uint64_t E = 0; E < AT->getNumElements(); ++E) {
      Value *EP = B.CreateGEP(ElemTy, Ptr, B.getInt64(E));
      Value *EV = B.CreateLoad(ElemTy, EP);
      Agg = B.CreateInsertValue(Agg, EV, {unsigned(E)});
    }
    LI->replaceAllUsesWith(Agg);
    LI->eraseFromParent();
  }
}

void scalarizeAggregateStores(Module &M, PointeeTypeMap &PTM) {

  auto Aggs = collectInsts<StoreInst>(M, [](StoreInst *SI) {
    return isa<ArrayType>(SI->getValueOperand()->getType());
  });
  for (StoreInst *SI : Aggs) {
    Value *Val = SI->getValueOperand();
    auto *AT = cast<ArrayType>(Val->getType());
    Type *ElemTy = AT->getElementType();
    Value *Ptr = retypePointerVia(SI->getPointerOperand(), ElemTy, SI, PTM);
    IRBuilder<> B(SI);
    for (uint64_t E = 0; E < AT->getNumElements(); ++E) {
      Value *EP = B.CreateGEP(ElemTy, Ptr, B.getInt64(E));
      Value *EV = B.CreateExtractValue(Val, {unsigned(E)});
      B.CreateStore(EV, EP);
    }
    SI->eraseFromParent();
  }

  auto VecAggs = collectInsts<StoreInst>(M, [&](StoreInst *SI) {
    auto *VT = dyn_cast<FixedVectorType>(SI->getValueOperand()->getType());
    if (!VT)
      return false;
    auto *AI =
        dyn_cast<AllocaInst>(SI->getPointerOperand()->stripPointerCasts());
    if (!AI)
      return false;
    Type *AT = AI->getAllocatedType();
    if (!AT->isAggregateType())
      return false;
    Type *ElemTy =
        AT->isArrayTy() ? cast<ArrayType>(AT)->getElementType() : nullptr;
    return ElemTy != VT;
  });
  for (StoreInst *SI : VecAggs) {
    auto *VT = cast<FixedVectorType>(SI->getValueOperand()->getType());
    Type *ElemTy = VT->getElementType();
    auto *AI = cast<AllocaInst>(SI->getPointerOperand()->stripPointerCasts());
    Value *Val = SI->getValueOperand();
    IRBuilder<> B(SI);
    Value *Base = retypePointerVia(AI, ElemTy, SI, PTM);
    for (uint64_t E = 0; E < VT->getNumElements(); ++E) {
      Value *EP = B.CreateGEP(ElemTy, Base, B.getInt64(E));
      Value *EV = B.CreateExtractElement(Val, B.getInt64(E));
      B.CreateStore(EV, EP);
    }
    SI->eraseFromParent();
  }
}

}
}
