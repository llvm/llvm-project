//===- LLVMToAIRIntrinsics.cpp - Rename LLVM intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMToAIRIntrinsics.h"
#include "AIR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-llvm-to-air-intrinsics"

namespace {
// Layout must match the struct emitted by -gen-searchable-tables for the
// AIRFunction class in AIRIntrinsicMappings.td.
struct AIRFunction {
  StringTable::Offset LLVMName;
  StringTable::Offset AIRName;
  uint8_t Signedness;
  bool HasFast;
  uint8_t NumArgs;
  bool Invertible;
};
} // namespace

namespace {
#define GET_AIRFunctions_IMPL
#include "AIRGenAIRIntrinsicMappings.inc"
} // namespace

static const AIRFunction *lookupAIRFunction(StringRef LLVMBaseName) {
  static const StringMap<const AIRFunction *> Map = [] {
    StringMap<const AIRFunction *> M;
    for (const AIRFunction &F : AIRFunctions)
      M[getAIRFunctionStr(F.LLVMName)] = &F;
    return M;
  }();
  auto It = Map.find(LLVMBaseName);
  return It == Map.end() ? nullptr : It->second;
}

/// The AIR function a call to intrinsic F maps to, or "" if none: see
/// AIRIntrinsicMappings.td for the naming scheme.
static std::string getAIRName(const Function &F, const AIRFunction &Fn,
                              bool Fast) {
  StringRef Base = Intrinsic::getBaseName(F.getIntrinsicID());
  StringRef Suffix = F.getName().drop_front(Base.size()); // ".v4f32"
  std::string Name = "air.";
  if (Fast)
    Name += "fast_";
  Name += getAIRFunctionStr(Fn.AIRName);
  Name += '.';
  if (Fn.Signedness == 1)
    Name += "s.";
  else if (Fn.Signedness == 2)
    Name += "u.";
  Name += Suffix.drop_front();
  return Name;
}

static bool lowerToAIRFunctions(Module &M) {
  bool Changed = false;
  for (Function &F : make_early_inc_range(M)) {
    if (!F.isIntrinsic())
      continue;
    const AIRFunction *Fn =
        lookupAIRFunction(Intrinsic::getBaseName(F.getIntrinsicID()));
    if (!Fn)
      continue;
    unsigned NumArgs = Fn->NumArgs ? Fn->NumArgs : F.arg_size();
    FunctionType *FTy = FunctionType::get(
        F.getReturnType(), F.getFunctionType()->params().take_front(NumArgs),
        false);
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &F)
        continue;
      // Relaxed AIR functions exist for float only.
      bool Fast = Fn->HasFast && isa<FPMathOperator>(CI) &&
                  CI->hasApproxFunc() &&
                  CI->getType()->getScalarType()->isFloatTy();
      FunctionCallee AIRF =
          M.getOrInsertFunction(getAIRName(F, *Fn, Fast), FTy);
      if (auto *AF = dyn_cast<Function>(AIRF.getCallee())) {
        AF->setDoesNotThrow();
        AF->setWillReturn();
        AF->setDoesNotAccessMemory();
      }
      IRBuilder<> B(CI);
      SmallVector<Value *, 3> Args(CI->args().begin(),
                                   CI->args().begin() + NumArgs);
      CallInst *New = B.CreateCall(AIRF, Args);
      New->takeName(CI);
      New->copyMetadata(*CI);
      if (isa<FPMathOperator>(New))
        New->setFastMathFlags(CI->getFastMathFlags());
      CI->replaceAllUsesWith(New);
      CI->eraseFromParent();
      Changed = true;
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
  return Changed;
}

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
  bool Changed = lowerToAIRFunctions(M);
  Changed |= lowerMulHi(M);
  return Changed;
}

PreservedAnalyses LLVMToAIRIntrinsicsPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  return llvmToAIRIntrinsics(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

bool LLVMToAIRIntrinsicsLegacy::runOnModule(Module &M) {
  return llvmToAIRIntrinsics(M);
}

char LLVMToAIRIntrinsicsLegacy::ID = 0;

INITIALIZE_PASS(LLVMToAIRIntrinsicsLegacy, DEBUG_TYPE,
                "AIR Lower LLVM Intrinsics to AIR", false, false)

ModulePass *llvm::createLLVMToAIRIntrinsicsLegacyPass() {
  return new LLVMToAIRIntrinsicsLegacy();
}
