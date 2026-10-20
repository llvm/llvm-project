//===-- AGXLowerGenericAddresses.cpp - Generic pointers to threadgroup memory ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX has no generic address space: device memory and threadgroup memory are
// reached by different instructions, and a 64-bit device address cannot name
// threadgroup memory. PTX's generic addressing (cvta.shared, then ld / st on
// the generic pointer) is emulated, in the functions that convert threadgroup
// pointers to generic ones, with a tagged representation:
//
//   generic pointer to threadgroup offset o:  (1 << 63) | o
//   generic pointer to device address a:      a          (device addresses are
//                                                          below 2^48)
//
// The casts become integer operations, and every load, store and atomic
// through a generic pointer that InferAddressSpaces could not resolve tests
// the tag and branches to a threadgroup or a device access. Functions without
// such casts are unchanged: their generic pointers are device pointers.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-generic-addresses"

namespace {

class AGXLowerGenericAddresses : public FunctionPass {
public:
  static char ID;
  AGXLowerGenericAddresses() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Generic Addresses";
  }
  bool runOnFunction(Function &F) override;
};

} // namespace

char AGXLowerGenericAddresses::ID = 0;

INITIALIZE_PASS(AGXLowerGenericAddresses, DEBUG_TYPE,
                "AGX Lower Generic Addresses", false, false)

FunctionPass *llvm::createAGXLowerGenericAddressesPass() {
  return new AGXLowerGenericAddresses();
}

static constexpr unsigned Generic = AGX::GENERIC_ADDRESS;
static constexpr uint64_t SharedTag = uint64_t(1) << 63;

static bool isSharedToGeneric(const Value *V) {
  auto *C = dyn_cast<AddrSpaceCastOperator>(V);
  return C && C->getSrcAddressSpace() == AGX::THREADGROUP_ADDRESS &&
         C->getDestAddressSpace() == Generic;
}

/// The pointer operand of a generic memory access, or null.
static Value *genericPointer(Instruction &I) {
  Value *P = nullptr;
  if (auto *L = dyn_cast<LoadInst>(&I))
    P = L->getPointerOperand();
  else if (auto *S = dyn_cast<StoreInst>(&I))
    P = S->getPointerOperand();
  else if (auto *A = dyn_cast<AtomicRMWInst>(&I))
    P = A->getPointerOperand();
  else if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
    P = X->getPointerOperand();
  return P && P->getType()->getPointerAddressSpace() == Generic ? P : nullptr;
}

static void setPointer(Instruction &I, Value *P) {
  if (auto *L = dyn_cast<LoadInst>(&I))
    L->setOperand(LoadInst::getPointerOperandIndex(), P);
  else if (auto *S = dyn_cast<StoreInst>(&I))
    S->setOperand(StoreInst::getPointerOperandIndex(), P);
  else if (auto *A = dyn_cast<AtomicRMWInst>(&I))
    A->setOperand(AtomicRMWInst::getPointerOperandIndex(), P);
  else
    cast<AtomicCmpXchgInst>(I).setOperand(
        AtomicCmpXchgInst::getPointerOperandIndex(), P);
}

bool AGXLowerGenericAddresses::runOnFunction(Function &F) {
  // Constant casts of threadgroup globals become instructions first.
  SmallVector<Constant *, 4> Consts;
  for (GlobalVariable &G : F.getParent()->globals())
    if (G.getAddressSpace() == AGX::THREADGROUP_ADDRESS)
      Consts.push_back(&G);
  convertUsersOfConstantsToInstructions(Consts, &F);

  SmallVector<AddrSpaceCastInst *, 8> ToGeneric, FromGeneric;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<AddrSpaceCastInst>(&I)) {
      if (isSharedToGeneric(C))
        ToGeneric.push_back(C);
      else if (C->getSrcAddressSpace() == Generic &&
               C->getDestAddressSpace() == AGX::THREADGROUP_ADDRESS)
        FromGeneric.push_back(C);
    }
  if (ToGeneric.empty() && FromGeneric.empty())
    return false;

  LLVMContext &Ctx = F.getContext();
  Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  PointerType *GenericPtr = PointerType::get(Ctx, Generic);
  PointerType *SharedPtr = PointerType::get(Ctx, AGX::THREADGROUP_ADDRESS);
  PointerType *DevicePtr = PointerType::get(Ctx, AGX::DEVICE_ADDRESS);

  for (AddrSpaceCastInst *C : ToGeneric) {
    IRBuilder<> B(C);
    Value *Off = B.CreateZExt(B.CreatePtrToInt(C->getOperand(0), I32), I64);
    Value *G = B.CreateIntToPtr(B.CreateOr(Off, SharedTag), GenericPtr);
    C->replaceAllUsesWith(G);
    C->eraseFromParent();
  }
  for (AddrSpaceCastInst *C : FromGeneric) {
    IRBuilder<> B(C);
    Value *Off = B.CreateTrunc(B.CreatePtrToInt(C->getOperand(0), I64), I32);
    C->replaceAllUsesWith(B.CreateIntToPtr(Off, SharedPtr));
    C->eraseFromParent();
  }

  SmallVector<Instruction *, 16> Accesses;
  for (Instruction &I : instructions(F))
    if (genericPointer(I))
      Accesses.push_back(&I);
  for (Instruction *I : Accesses) {
    Value *P = genericPointer(*I);
    IRBuilder<> B(I);
    Value *Bits = B.CreatePtrToInt(P, I64);
    // The tag is the sign of the high word, which is cheaper to test than the
    // sign of the 64-bit value.
    Value *Hi = B.CreateTrunc(B.CreateLShr(Bits, 32), I32);
    Value *IsShared = B.CreateICmpSLT(Hi, ConstantInt::get(I32, 0));
    Instruction *ThenT, *ElseT;
    SplitBlockAndInsertIfThenElse(IsShared, I, &ThenT, &ElseT);
    // Threadgroup access.
    Instruction *S = I->clone();
    S->insertBefore(ThenT->getIterator());
    IRBuilder<> BS(S);
    setPointer(*S, BS.CreateIntToPtr(BS.CreateTrunc(Bits, I32), SharedPtr));
    // Device access.
    Instruction *D = I->clone();
    D->insertBefore(ElseT->getIterator());
    IRBuilder<> BD(D);
    setPointer(*D, BD.CreateIntToPtr(Bits, DevicePtr));
    if (!I->getType()->isVoidTy()) {
      PHINode *Phi = PHINode::Create(I->getType(), 2, "", I->getIterator());
      Phi->addIncoming(S, S->getParent());
      Phi->addIncoming(D, D->getParent());
      Phi->takeName(I);
      I->replaceAllUsesWith(Phi);
    }
    I->eraseFromParent();
  }
  return true;
}
