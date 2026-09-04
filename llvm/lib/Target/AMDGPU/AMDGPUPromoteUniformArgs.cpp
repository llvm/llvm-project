//===-- AMDGPUPromoteUniformArgs.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promote scalar and pointer arguments of internal callees to \c inreg when
// every visible call-site operand is trivially uniform: a constant, an
// argument passed in an SGPR, or an always-uniform intrinsic in the same
// block as the call. Vectors are not promoted.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-promote-uniform-args"

STATISTIC(NumPromotedInRegArgs,
          "Number of uniform arguments promoted to inreg");
STATISTIC(NumPromotedInRegFuncs,
          "Number of functions with a promoted uniform argument");

static cl::opt<bool> EnablePromoteUniformArgs(
    "amdgpu-enable-promote-uniform-args", cl::Hidden, cl::init(true),
    cl::desc("Promote provably uniform internal scalar and pointer arguments "
             "to inreg"));

namespace {

static bool canPromoteArgToInReg(const Argument &A) {
  Type *Ty = A.getType();
  if (!(Ty->isIntOrPtrTy() || Ty->isFloatingPointTy()) || A.hasInRegAttr())
    return false;
  // inreg is mutually exclusive with byval, inalloca, preallocated, byref,
  // sret, and nest. The first five are covered by hasPointeeInMemoryValueAttr.
  if (A.hasPointeeInMemoryValueAttr() || A.hasNestAttr())
    return false;
  return !A.hasAttribute("amdgpu-hidden-argument");
}

static bool isEligibleInRegUniformCallee(const Function &F) {
  if (F.isDeclaration() || F.isVarArg() || !F.canChangeSignature())
    return false;
  if (!F.hasLocalLinkage())
    return false;
  switch (F.getCallingConv()) {
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  default:
    return false;
  }

  // A musttail call requires the enclosing function's parameter ABI attributes
  // to match the callee's positionally, so adding inreg to any parameter of F
  // breaks the contract, not just to one forwarded to the tail call.
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (const auto *CB = dyn_cast<CallBase>(&I))
        if (CB->isMustTailCall())
          return false;

  // Every use must be a direct call to F. This subsumes hasAddressTaken(),
  // which by default ignores some uses we care about (e.g. assume-like calls),
  // and it is what lets the transform treat each user as a call site to update.
  for (const User *U : F.users()) {
    const auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledFunction() != &F)
      return false;
    if (CB->isMustTailCall() || isa<InvokeInst>(CB))
      return false;
  }
  return !F.user_empty();
}

static bool isTriviallyUniform(const Use &U) {
  Value *V = U.get();
  if (isa<Constant>(V))
    return true;
  if (const auto *A = dyn_cast<Argument>(V))
    return AMDGPU::isArgPassedInSGPR(A);
  if (const auto *II = dyn_cast<IntrinsicInst>(V)) {
    if (!AMDGPU::isIntrinsicAlwaysUniform(II->getIntrinsicID()))
      return false;
    // If II and U are in different blocks then there is a possibility of
    // temporal divergence.
    return II->getParent() == cast<Instruction>(U.getUser())->getParent();
  }
  return false;
}

static bool promoteUniformArgsToInReg(Module &M) {
  bool Changed = false;

  for (Function &F : M) {
    if (!isEligibleInRegUniformCallee(F))
      continue;

    SmallVector<CallBase *, 8> Calls;
    for (User *U : F.users()) {
      auto *CB = cast<CallBase>(U);
      Calls.push_back(CB);
    }

    bool FuncChanged = false;
    for (Argument &A : F.args()) {
      if (!canPromoteArgToInReg(A))
        continue;

      bool AllUniform = true;
      for (CallBase *CB : Calls) {
        if (!isTriviallyUniform(CB->getArgOperandUse(A.getArgNo()))) {
          AllUniform = false;
          break;
        }
      }
      if (!AllUniform)
        continue;

      A.addAttr(Attribute::InReg);
      for (CallBase *CB : Calls)
        CB->addParamAttr(A.getArgNo(), Attribute::InReg);
      ++NumPromotedInRegArgs;
      FuncChanged = Changed = true;
    }

    if (FuncChanged)
      ++NumPromotedInRegFuncs;
  }

  return Changed;
}

} // namespace

PreservedAnalyses AMDGPUPromoteUniformArgsPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (!EnablePromoteUniformArgs || !Triple(M.getTargetTriple()).isAMDGCN())
    return PreservedAnalyses::all();
  if (!promoteUniformArgsToInReg(M))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
