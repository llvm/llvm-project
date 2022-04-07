//===- LowerAtomic.cpp - Lower atomic intrinsics --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers atomic intrinsics to non-atomic form for use in a known
// non-preemptible environment.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerAtomic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "loweratomic"

bool llvm::lowerAtomicCmpXchgInst(AtomicCmpXchgInst *CXI) {
  IRBuilder<> Builder(CXI);
  Value *Ptr = CXI->getPointerOperand();
  Value *Cmp = CXI->getCompareOperand();
  Value *Val = CXI->getNewValOperand();

  LoadInst *Orig = Builder.CreateLoad(Val->getType(), Ptr);
  Value *Equal = Builder.CreateICmpEQ(Orig, Cmp);
  Value *Res = Builder.CreateSelect(Equal, Val, Orig);
  Builder.CreateStore(Res, Ptr);

  Res = Builder.CreateInsertValue(UndefValue::get(CXI->getType()), Orig, 0);
  Res = Builder.CreateInsertValue(Res, Equal, 1);

  CXI->replaceAllUsesWith(Res);
  CXI->eraseFromParent();
  return true;
}

bool llvm::lowerAtomicRMWInst(AtomicRMWInst *RMWI) {
  IRBuilder<> Builder(RMWI);
  Value *Ptr = RMWI->getPointerOperand();
  Value *Val = RMWI->getValOperand();

  LoadInst *Orig = Builder.CreateLoad(Val->getType(), Ptr);
  Value *Res = nullptr;

  switch (RMWI->getOperation()) {
  default: llvm_unreachable("Unexpected RMW operation");
  case AtomicRMWInst::Xchg:
    Res = Val;
    break;
  case AtomicRMWInst::Add:
    Res = Builder.CreateAdd(Orig, Val);
    break;
  case AtomicRMWInst::Sub:
    Res = Builder.CreateSub(Orig, Val);
    break;
  case AtomicRMWInst::And:
    Res = Builder.CreateAnd(Orig, Val);
    break;
  case AtomicRMWInst::Nand:
    Res = Builder.CreateNot(Builder.CreateAnd(Orig, Val));
    break;
  case AtomicRMWInst::Or:
    Res = Builder.CreateOr(Orig, Val);
    break;
  case AtomicRMWInst::Xor:
    Res = Builder.CreateXor(Orig, Val);
    break;
  case AtomicRMWInst::Max:
    Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Val),
                               Val, Orig);
    break;
  case AtomicRMWInst::Min:
    Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Val),
                               Orig, Val);
    break;
  case AtomicRMWInst::UMax:
    Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Val),
                               Val, Orig);
    break;
  case AtomicRMWInst::UMin:
    Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Val),
                               Orig, Val);
    break;
  case AtomicRMWInst::FAdd:
    Res = Builder.CreateFAdd(Orig, Val);
    break;
  case AtomicRMWInst::FSub:
    Res = Builder.CreateFSub(Orig, Val);
    break;
  }
  Builder.CreateStore(Res, Ptr);
  RMWI->replaceAllUsesWith(Orig);
  RMWI->eraseFromParent();
  return true;
}
