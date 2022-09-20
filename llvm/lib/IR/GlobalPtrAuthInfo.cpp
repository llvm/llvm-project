//===- GlobalPtrAuthInfo.cpp - Analysis tools for ptrauth globals ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/GlobalPtrAuthInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

Expected<GlobalPtrAuthInfo> GlobalPtrAuthInfo::tryAnalyze(const Value *V) {
  auto Invalid = [](const Twine &Reason) {
    return make_error<StringError>(Reason, inconvertibleErrorCode());
  };

  auto &Ctx = V->getContext();

  V = V->stripPointerCasts();

  auto *GV = dyn_cast<GlobalVariable>(V);
  if (!GV)
    return Invalid("value isn't a global");

  if (GV->getSection() != "llvm.ptrauth")
    return Invalid("global isn't in section \"llvm.ptrauth\"");

  if (!GV->hasInitializer())
    return Invalid("global doesn't have an initializer");

  auto *Init = GV->getInitializer();

  auto *Ty = dyn_cast<StructType>(GV->getInitializer()->getType());
  if (!Ty)
    return Invalid("global isn't a struct");

  auto *I64Ty = Type::getInt64Ty(Ctx);
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *P0I8Ty = Type::getInt8PtrTy(Ctx);
  // Check that the struct matches its expected shape:
  //   { i8*, i32, i64, i64 }
  if (!Ty->isLayoutIdentical(
        StructType::get(Ctx, {P0I8Ty, I32Ty, I64Ty, I64Ty})))
    return Invalid("global doesn't have type '{ i8*, i32, i64, i64 }'");

  auto *Key = dyn_cast<ConstantInt>(Init->getOperand(1));
  if (!Key)
    return Invalid("key isn't a constant integer");

  auto *AddrDiscriminator = Init->getOperand(2);
  if (!isa<ConstantInt>(AddrDiscriminator) &&
      !isa<ConstantExpr>(AddrDiscriminator))
    return Invalid("address discriminator isn't a constant integer or expr");

  auto *Discriminator = dyn_cast<ConstantInt>(Init->getOperand(3));
  if (!Discriminator)
    return Invalid("discriminator isn't a constant integer");

  return GlobalPtrAuthInfo(GV);
}

Optional<GlobalPtrAuthInfo> GlobalPtrAuthInfo::analyze(const Value *V) {
  if (auto PAIOrErr = tryAnalyze(V)) {
    return *PAIOrErr;
  } else {
    consumeError(PAIOrErr.takeError());
    return None;
  }
}

static bool areEquivalentAddrDiscriminators(const Value *V1, const Value *V2,
                                            const DataLayout &DL) {
  APInt V1Off(DL.getPointerSizeInBits(), 0);
  APInt V2Off(DL.getPointerSizeInBits(), 0);

  if (auto *V1Cast = dyn_cast<PtrToIntOperator>(V1))
    V1 = V1Cast->getPointerOperand();
  if (auto *V2Cast = dyn_cast<PtrToIntOperator>(V2))
    V2 = V2Cast->getPointerOperand();
  auto *V1Base = V1->stripAndAccumulateInBoundsConstantOffsets(DL, V1Off);
  auto *V2Base = V2->stripAndAccumulateInBoundsConstantOffsets(DL, V2Off);
  return V1Base == V2Base && V1Off == V2Off;
}

bool GlobalPtrAuthInfo::isCompatibleWith(const Value *Key,
                                         const Value *Discriminator,
                                         const DataLayout &DL) const {
  // If the keys are different, there's no chance for this to be compatible.
  if (Key != getKey())
    return false;

  // If the discriminators are the same, this is compatible iff there is no
  // address discriminator.
  if (Discriminator == getDiscriminator())
    return getAddrDiscriminator()->isNullValue();

  // If we dynamically blend the discriminator with the address discriminator,
  // this is compatible.
  if (auto *DiscBlend = dyn_cast<IntrinsicInst>(Discriminator)) {
    if (DiscBlend->getIntrinsicID() == Intrinsic::ptrauth_blend &&
        DiscBlend->getOperand(1) == getDiscriminator() &&
        areEquivalentAddrDiscriminators(DiscBlend->getOperand(0),
                                        getAddrDiscriminator(), DL))
      return true;
  }

  // If we don't have a non-address discriminator, we don't need a blend in
  // the first place:  accept the address discriminator as the discriminator.
  if (getDiscriminator()->isNullValue() &&
      areEquivalentAddrDiscriminators(getAddrDiscriminator(), Discriminator,
                                      DL))
    return true;

  // Otherwise, we don't know.
  return false;
}

Constant *GlobalPtrAuthInfo::createWithSameSchema(Module &M,
                                                  Constant *Pointer) const {
  return create(M, Pointer, const_cast<ConstantInt*>(getKey()),
                const_cast<Constant*>(getAddrDiscriminator()),
                const_cast<ConstantInt*>(getDiscriminator()));
}

Constant *GlobalPtrAuthInfo::create(Module &M, Constant *Pointer,
                                    ConstantInt *Key,
                                    Constant *AddrDiscriminator,
                                    ConstantInt *Discriminator) {
  auto CastPointer =
    ConstantExpr::getBitCast(Pointer, Type::getInt8PtrTy(M.getContext()));

  auto Init = ConstantStruct::getAnon({CastPointer, Key, AddrDiscriminator,
                                       Discriminator}, /*packed*/ false);

  // TODO: look for an existing global with the right setup?
  auto GV = new GlobalVariable(M, Init->getType(), /*constant*/ true,
                               GlobalVariable::PrivateLinkage, Init);
  GV->setSection("llvm.ptrauth");

  auto Result = ConstantExpr::getBitCast(GV, Pointer->getType());

  assert(analyze(Result).has_value() && "invalid ptrauth constant");

  return Result;
}
