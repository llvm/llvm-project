//===- PointeeRules.cpp - Single-authority pointee-typing rules -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointeeRules.h"
#include "AIRConstraints.h"
#include "PointerRepairUtil.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"

using namespace llvm;

namespace llvm {
namespace metal {

Type *atomicPointeeFromUsers(Value *Ptr) {
  for (auto *U : Ptr->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI || !CI->getCalledFunction())
      continue;
    StringRef Name = CI->getCalledFunction()->getName();
    if (!Name.starts_with("air.atomic."))
      continue;
    if (CI->arg_size() == 0 || CI->getArgOperand(0) != Ptr)
      continue;
    if (Name.ends_with(".i32"))
      return Type::getInt32Ty(Ptr->getContext());
    if (Name.ends_with(".f32"))
      return Type::getFloatTy(Ptr->getContext());
  }
  return nullptr;
}

Type *mmaElemFromName(StringRef Name, LLVMContext &Ctx) {
  if (!Name.starts_with("air.simdgroup_matrix_8x8_"))
    return nullptr;
  if (Name.contains("p1i8") || Name.contains("p3i8"))
    return Type::getInt8Ty(Ctx);
  if (Name.contains("bf16"))
    return Type::getBFloatTy(Ctx);
  if (Name.contains("f16"))
    return Type::getHalfTy(Ctx);
  if (Name.contains("f32"))
    return Type::getFloatTy(Ctx);
  return nullptr;
}

Type *requiredSelectPointee(SelectInst *Sel, const PointeeTypeMap &PTM) {
  Value *TV = Sel->getTrueValue();
  Value *FV = Sel->getFalseValue();
  Type *TT = PTM.get(TV);
  Type *FT = PTM.get(FV);
  if (TT && !isa<IntToPtrInst>(TV))
    return TT;
  if (FT && !isa<IntToPtrInst>(FV))
    return FT;
  return TT ? TT : FT;
}

Type *requiredPhiPointee(PHINode *PN, const PointeeTypeMap &PTM) {
  if (PN->getType()->getPointerAddressSpace() == AS::Device) {
    if (Type *AtomicTy = atomicPointeeFromUsers(PN))
      return AtomicTy;
    return Type::getFloatTy(PN->getContext());
  }
  if (Type *Existing = PTM.get(PN))
    return Existing;
  for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
    Value *In = PN->getIncomingValue(J);
    if (isa<Constant>(In))
      continue;
    if (auto *GV = dyn_cast<GlobalVariable>(In))
      return GV->getValueType();
    if (auto *G = dyn_cast<GetElementPtrInst>(In))
      return G->getResultElementType();
    if (Type *T = PTM.get(In))
      return T;
  }
  return nullptr;
}

bool reconcileGEPBaseType(GetElementPtrInst *GEP, PointeeTypeMap &PTM) {
  Value *Base = GEP->getPointerOperand();
  if (isa<GlobalVariable>(Base) || isa<BitCastInst>(Base))
    return false;
  Type *SrcTy = GEP->getSourceElementType();

  if (auto *AI = dyn_cast<AllocaInst>(Base)) {
    if (AI->getAllocatedType() == SrcTy)
      return false;
  } else {
    Type *Pointee = effectivePointee(Base, PTM);
    if (!Pointee || Pointee == SrcTy)
      return false;
  }
  GEP->setOperand(0, retypePointerVia(Base, SrcTy, GEP, PTM));
  return true;
}

}
}
