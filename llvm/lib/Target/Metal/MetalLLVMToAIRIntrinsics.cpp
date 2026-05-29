//===- MetalLLVMToAIRIntrinsics.cpp - Rename LLVM intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalLLVMToAIRIntrinsics.h"
#include "Metal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-llvm-to-air-intrinsics"

namespace {
// Layout must match the struct emitted by -gen-searchable-tables for the
// IntrinsicRename class in MetalAIRIntrinsicMappings.td.
struct IntrinsicRename {
  const char *LLVMName;
  const char *AIRName;
};
} // namespace

#define GET_AIRIntrinsicRenames_DECL
#include "MetalGenAIRIntrinsicMappings.inc"

namespace {
#define GET_AIRIntrinsicRenames_IMPL
#include "MetalGenAIRIntrinsicMappings.inc"
} // namespace

// Lower __mulhi(a, b) -> (i32)((zext64(a) * zext64(b)) >> 32).
static bool lowerMulHi(Module &M) {
  Function *F = M.getFunction("__mulhi");
  if (!F)
    return false;

  Type *I32 = Type::getInt32Ty(M.getContext());
  Type *I64 = Type::getInt64Ty(M.getContext());

  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);

  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Value *A = B.CreateZExt(CI->getArgOperand(0), I64);
    Value *Bv = B.CreateZExt(CI->getArgOperand(1), I64);
    Value *Mul = B.CreateMul(A, Bv);
    Value *Hi = B.CreateLShr(Mul, 32);
    Value *Result = B.CreateTrunc(Hi, I32);
    CI->replaceAllUsesWith(Result);
    CI->eraseFromParent();
  }

  if (F->use_empty())
    F->eraseFromParent();
  return true;
}

static bool llvmToAIRIntrinsics(Module &M) {
  bool Changed = false;
  for (const IntrinsicRename &Mapping : AIRIntrinsicRenames) {
    if (Function *F = M.getFunction(Mapping.LLVMName)) {
      F->setName(Mapping.AIRName);
      Changed = true;
    }
  }
  Changed |= lowerMulHi(M);
  return Changed;
}

PreservedAnalyses MetalLLVMToAIRIntrinsicsPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  return llvmToAIRIntrinsics(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

bool MetalLLVMToAIRIntrinsicsLegacy::runOnModule(Module &M) {
  return llvmToAIRIntrinsics(M);
}

char MetalLLVMToAIRIntrinsicsLegacy::ID = 0;

INITIALIZE_PASS(MetalLLVMToAIRIntrinsicsLegacy, DEBUG_TYPE,
                "Metal Lower LLVM Intrinsics to AIR", false, false)

ModulePass *llvm::createMetalLLVMToAIRIntrinsicsLegacyPass() {
  return new MetalLLVMToAIRIntrinsicsLegacy();
}
