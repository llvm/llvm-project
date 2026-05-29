//===- MetalAsyncEventToAlloca.cpp - async-event pointer-arg bitcasts ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalAsyncEventToAlloca.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-async-event-to-alloca"

static constexpr StringLiteral kAsyncCopyPrefix("air.simdgroup_async_copy_2d.");
static constexpr StringLiteral
    kWaitSimdgroupEvents("air.wait_simdgroup_events");

static bool asyncEventToAlloca(Module &M) {
  bool Changed = false;

  // Insert no-op bitcasts before async-copy / wait-event pointer args so the
  // writer's PointeeTypeMap sees a fresh typed-pointer slot per call site.
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
          auto *BC = CastInst::Create(Instruction::BitCast, Arg, Arg->getType(),
                                      "", CI->getIterator());
          CI->setArgOperand(K, BC);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

PreservedAnalyses MetalAsyncEventToAllocaPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  return asyncEventToAlloca(M) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}

bool MetalAsyncEventToAllocaLegacy::runOnModule(Module &M) {
  return asyncEventToAlloca(M);
}

char MetalAsyncEventToAllocaLegacy::ID = 0;

INITIALIZE_PASS(MetalAsyncEventToAllocaLegacy, DEBUG_TYPE,
                "Metal Async Event to Alloca", false, false)

ModulePass *llvm::createMetalAsyncEventToAllocaLegacyPass() {
  return new MetalAsyncEventToAllocaLegacy();
}
