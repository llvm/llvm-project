//===- NextSiliconHostAtomicFixup.cpp - Add atomic fixup ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NextSiliconHostAtomicFixup is an LLVM pass that bumps the pointers of atomic
// operations in order detect cases where accessing memory that is
// migrated on device.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/NextSiliconHostAtomicFixup.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "next-silicon-host-atomic-fixup"

static bool processAtomics(Module &M) {
  bool Changed = false;
  GlobalVariable *AtomicFixupPtrBias = nullptr;
  GlobalVariable *AtomicFixupPtrStart = nullptr;
  GlobalVariable *AtomicFixupPtrEnd = nullptr;
  Type *Int64Ty = Type::getInt64Ty(M.getContext());

  auto createFixupSymbol = [&](StringRef Name) {
    GlobalVariable *GV =
        new GlobalVariable(M, Int64Ty, false, GlobalVariable::ExternalLinkage,
                           Constant::getNullValue(Int64Ty), Name);
    GV->setDSOLocal(true);
    GV->setVisibility(GlobalValue::DefaultVisibility);
    return GV;
  };

  auto createFixupLoad = [&](GlobalVariable *GV, Function *F) {
    IRBuilder<> LIBuilder(&F->getEntryBlock().front());
    return LIBuilder.CreateLoad(Int64Ty, GV);
  };

  for (Function &F : M) {
    LoadInst *LIBias = nullptr;
    LoadInst *LIStart = nullptr;
    LoadInst *LIEnd = nullptr;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (!I.isAtomic() || I.getOpcode() == Instruction::Fence)
          continue;

        if (!AtomicFixupPtrBias)
          AtomicFixupPtrBias = createFixupSymbol("__atomic_fixup_ptr_bias");
        if (!AtomicFixupPtrStart)
          AtomicFixupPtrStart = createFixupSymbol("__atomic_fixup_ptr_start");
        if (!AtomicFixupPtrEnd)
          AtomicFixupPtrEnd = createFixupSymbol("__atomic_fixup_ptr_end");

        if (!LIBias)
          LIBias = createFixupLoad(AtomicFixupPtrBias, &F);
        if (!LIStart)
          LIStart = createFixupLoad(AtomicFixupPtrStart, &F);
        if (!LIEnd)
          LIEnd = createFixupLoad(AtomicFixupPtrEnd, &F);

        IRBuilder<> Builder(&I);
        unsigned PtrOperandIdx = 0;
        if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
          PtrOperandIdx = LI->getPointerOperandIndex();
        } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
          PtrOperandIdx = SI->getPointerOperandIndex();
        } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(&I)) {
          PtrOperandIdx = RMW->getPointerOperandIndex();
        } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(&I)) {
          PtrOperandIdx = XCHG->getPointerOperandIndex();
        }
        auto AtomicPtrToInt =
            Builder.CreatePtrToInt(I.getOperand(PtrOperandIdx), Int64Ty);
        Value *Cmp1 =
            Builder.CreateCmp(ICmpInst::ICMP_UGE, AtomicPtrToInt, LIStart);
        Value *Cmp2 =
            Builder.CreateCmp(ICmpInst::ICMP_ULT, AtomicPtrToInt, LIEnd);
        Value *CmpBoth = Builder.CreateAnd(Cmp1, Cmp2);
        auto *Select =
            Builder.CreateSelect(CmpBoth, LIBias, ConstantInt::get(Int64Ty, 0));
        auto *Add = Builder.CreateAdd(AtomicPtrToInt, Select);
        I.setOperand(PtrOperandIdx,
                     Builder.CreateIntToPtr(
                         Add, I.getOperand(PtrOperandIdx)->getType()));
        Changed = true;
      }
    }
  }
  return Changed;
}

PreservedAnalyses
NextSiliconHostAtomicFixupPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!processAtomics(M))
    return PreservedAnalyses::all();

  // Be conservative for now, optimize later if necessary
  return PreservedAnalyses::none();
}
