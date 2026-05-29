//===- MetalInlineNonKernel.cpp - Inline all non-kernel functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalInlineNonKernel.h"
#include "Metal.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "metal-inline-non-kernel"

// AIR has no call stack, so any call to a defined (non-declaration) function
// must be inlined. Repeats until fixpoint to handle nested chains (A->B->C).
static bool inlineNonKernel(Module &M) {
  bool Changed = false;
  SmallPtrSet<Function *, 4> WasCallee;

  // Strip noinline so InlineFunction succeeds even on frontend-marked
  // functions.
  for (Function &F : M)
    if (!F.isDeclaration() && F.hasFnAttribute(Attribute::NoInline))
      F.removeFnAttr(Attribute::NoInline);

  bool Inlined = true;
  while (Inlined) {
    Inlined = false;
    SmallVector<CallInst *, 16> ToInline;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (Function *Callee = CI->getCalledFunction())
              if (!Callee->isDeclaration())
                ToInline.push_back(CI);
    }

    for (CallInst *CI : ToInline) {
      if (Function *Callee = CI->getCalledFunction())
        WasCallee.insert(Callee);
      InlineFunctionInfo IFI;
      if (InlineFunction(*CI, IFI).isSuccess()) {
        Changed = true;
        Inlined = true;
      }
    }
  }

  // Erase functions that were fully inlined and are now unused. Entry points
  // (never called by another function) are preserved.
  SmallVector<Function *, 4> Dead;
  for (Function &F : M)
    if (!F.isDeclaration() && F.use_empty() && WasCallee.contains(&F))
      Dead.push_back(&F);
  for (Function *F : Dead)
    F->eraseFromParent();

  return Changed;
}

PreservedAnalyses MetalInlineNonKernelPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return inlineNonKernel(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool MetalInlineNonKernelLegacy::runOnModule(Module &M) {
  return inlineNonKernel(M);
}

char MetalInlineNonKernelLegacy::ID = 0;

INITIALIZE_PASS(MetalInlineNonKernelLegacy, DEBUG_TYPE,
                "Metal Inline Non-Kernel Functions", false, false)

ModulePass *llvm::createMetalInlineNonKernelLegacyPass() {
  return new MetalInlineNonKernelLegacy();
}
