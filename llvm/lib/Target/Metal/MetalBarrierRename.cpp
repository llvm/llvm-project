//===- MetalBarrierRename.cpp - Rename threadgroup barrier ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalBarrierRename.h"
#include "Metal.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-barrier-rename"

static constexpr StringLiteral kBarrier("air.wg.barrier");
static constexpr StringLiteral kBarrierOld("air.threadgroup.barrier");

// The Metal 4 JIT rejects calls to air.threadgroup.barrier as an "unlowered
// function call"; only the function name needs updating to air.wg.barrier,
// which the JIT accepts with either the (2,1) or the legacy (1,4) args.
static bool barrierRename(Module &M) {
  Function *OldBarrier = M.getFunction(kBarrierOld);
  if (!OldBarrier)
    return false;

  Function *NewBarrier = M.getFunction(kBarrier);
  if (!NewBarrier)
    NewBarrier = Function::Create(OldBarrier->getFunctionType(),
                                  OldBarrier->getLinkage(), kBarrier, &M);

  bool Changed = false;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() == OldBarrier) {
            CI->setCalledFunction(NewBarrier);
            Changed = true;
          }

  if (OldBarrier->use_empty())
    OldBarrier->eraseFromParent();

  return Changed;
}

PreservedAnalyses MetalBarrierRenamePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return barrierRename(M) ? PreservedAnalyses::none()
                          : PreservedAnalyses::all();
}

bool MetalBarrierRenameLegacy::runOnModule(Module &M) {
  return barrierRename(M);
}

char MetalBarrierRenameLegacy::ID = 0;

INITIALIZE_PASS(MetalBarrierRenameLegacy, DEBUG_TYPE, "Metal Barrier Rename",
                false, false)

ModulePass *llvm::createMetalBarrierRenameLegacyPass() {
  return new MetalBarrierRenameLegacy();
}
