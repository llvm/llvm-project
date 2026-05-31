//===- AIRSplitI64Shuffle.cpp - Rewrite i64 simd shuffle ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRSplitI64Shuffle.h"
#include "AIR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-split-i64-shuffle"

static constexpr StringLiteral kShufflePrefix("air.simd_shuffle");
static constexpr StringLiteral kI64Suffix(".i64");
static constexpr StringLiteral kV2I32Suffix(".v2i32");

static bool isI64ShuffleDecl(const Function &F) {
  return F.isDeclaration() && F.getName().starts_with(kShufflePrefix) &&
         F.getName().ends_with(kI64Suffix);
}

static bool splitI64Shuffle(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I16 = Type::getInt16Ty(Ctx);
  Type *V2I32 = FixedVectorType::get(Type::getInt32Ty(Ctx), 2);

  SmallVector<CallInst *, 16> Calls;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction())
            if (isI64ShuffleDecl(*Callee))
              Calls.push_back(CI);

  for (CallInst *CI : Calls) {
    // Swap the trailing ".i64" for ".v2i32"; the JIT accepts the vector form.
    std::string NewName = CI->getCalledFunction()->getName().str();
    NewName.replace(NewName.size() - kI64Suffix.size(), kI64Suffix.size(),
                    kV2I32Suffix);

    FunctionType *NewFTy = FunctionType::get(V2I32, {V2I32, I16}, false);
    FunctionCallee NewFC = M.getOrInsertFunction(NewName, NewFTy);

    IRBuilder<> B(CI);
    Value *Vec =
        B.CreateBitCast(CI->getArgOperand(0), V2I32, CI->getName() + "_vv");
    Value *Shuf =
        B.CreateCall(NewFC, {Vec, CI->getArgOperand(1)}, CI->getName() + "_sv");
    Value *Result = B.CreateBitCast(Shuf, I64, CI->getName());

    CI->replaceAllUsesWith(Result);
    CI->eraseFromParent();
  }

  SmallVector<Function *, 4> DeadDecls;
  for (Function &F : M)
    if (isI64ShuffleDecl(F) && F.use_empty())
      DeadDecls.push_back(&F);
  for (Function *F : DeadDecls)
    F->eraseFromParent();

  return !Calls.empty();
}

PreservedAnalyses AIRSplitI64ShufflePass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return splitI64Shuffle(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool AIRSplitI64ShuffleLegacy::runOnModule(Module &M) {
  return splitI64Shuffle(M);
}

char AIRSplitI64ShuffleLegacy::ID = 0;

INITIALIZE_PASS(AIRSplitI64ShuffleLegacy, DEBUG_TYPE,
                "AIR rewrite i64 simd shuffle to v2i32", false, false)

ModulePass *llvm::createAIRSplitI64ShuffleLegacyPass() {
  return new AIRSplitI64ShuffleLegacy();
}
