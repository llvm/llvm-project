//===- AIRInlineNonKernel.cpp - Inline all non-kernel functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRInlineNonKernel.h"
#include "AIR.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "air-inline-non-kernel"

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

  // Erase functions that were fully inlined and are now unused. When the
  // frontend marked the launchable entry ("air-kernel" fn attribute), also
  // erase any other unused defined function: the mid-end optimizer may
  // inline a helper's only call site itself, leaving it uncalled here, and
  // a stray second function corrupts the metallib container.
  bool HaveKernelAttr = llvm::any_of(M, [](const Function &F) {
    return !F.isDeclaration() && F.hasFnAttribute("air-kernel");
  });
  SmallVector<Function *, 4> Dead;
  for (Function &F : M)
    if (!F.isDeclaration() && F.use_empty() &&
        (WasCallee.contains(&F) ||
         (HaveKernelAttr && !F.hasFnAttribute("air-kernel"))))
      Dead.push_back(&F);
  for (Function *F : Dead)
    F->eraseFromParent();
  Changed |= !Dead.empty();

  return Changed;
}

PreservedAnalyses AIRInlineNonKernelPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return inlineNonKernel(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool AIRInlineNonKernelLegacy::runOnModule(Module &M) {
  return inlineNonKernel(M);
}

char AIRInlineNonKernelLegacy::ID = 0;

INITIALIZE_PASS(AIRInlineNonKernelLegacy, DEBUG_TYPE,
                "AIR Inline Non-Kernel Functions", false, false)

ModulePass *llvm::createAIRInlineNonKernelLegacyPass() {
  return new AIRInlineNonKernelLegacy();
}
