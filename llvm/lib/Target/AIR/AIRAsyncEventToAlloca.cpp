//===- AIRAsyncEventToAlloca.cpp - async-event pointer-arg bitcasts ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRAsyncEventToAlloca.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-async-event-to-alloca"

static constexpr StringLiteral kAsyncCopyPrefix("air.simdgroup_async_copy_2d.");
static constexpr StringLiteral
    kWaitSimdgroupEvents("air.wait_simdgroup_events");

static bool asyncEventToAlloca(Module &M) {
  bool Changed = false;

  Type *I64 = Type::getInt64Ty(M.getContext());
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        bool IsAsyncCopy = Name.starts_with(kAsyncCopyPrefix);
        bool IsWaitEvents = (Name == kWaitSimdgroupEvents);
        if (!IsAsyncCopy && !IsWaitEvents)
          continue;

        for (unsigned K = 0; K < CI->arg_size(); ++K) {
          Value *Arg = CI->getArgOperand(K);
          if (!Arg->getType()->isPointerTy())
            continue;
          if (isa<BitCastInst>(Arg))
            continue;
          if (IsAsyncCopy) {
            auto *PI = new PtrToIntInst(Arg, I64, "", CI->getIterator());
            auto *IP =
                new IntToPtrInst(PI, Arg->getType(), "", CI->getIterator());
            CI->setArgOperand(K, IP);
          } else {
            auto *BC = CastInst::Create(Instruction::BitCast, Arg,
                                        Arg->getType(), "", CI->getIterator());
            CI->setArgOperand(K, BC);
          }
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

PreservedAnalyses AIRAsyncEventToAllocaPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  return asyncEventToAlloca(M) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}

bool AIRAsyncEventToAllocaLegacy::runOnModule(Module &M) {
  return asyncEventToAlloca(M);
}

char AIRAsyncEventToAllocaLegacy::ID = 0;

INITIALIZE_PASS(AIRAsyncEventToAllocaLegacy, DEBUG_TYPE,
                "AIR Async Event to Alloca", false, false)

ModulePass *llvm::createAIRAsyncEventToAllocaLegacyPass() {
  return new AIRAsyncEventToAllocaLegacy();
}
