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

using namespace llvm;

#define DEBUG_TYPE "amdgpu-promote-uniform-args"

STATISTIC(NumPromotedInRegArgs,
          "Number of uniform arguments promoted to inreg");
STATISTIC(NumPromotedInRegFuncs,
          "Number of functions with a promoted uniform argument");

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

static bool collectDirectCallSites(Function &F,
                                   SmallVectorImpl<CallBase *> &Calls) {
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

  // Every use must be the callee of a direct call to F. isCallee distinguishes
  // that from F appearing as a call operand. Reject musttail/invoke *to* F.
  for (Use &U : F.uses()) {
    auto *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB || !CB->isCallee(&U) || CB->isMustTailCall() || isa<InvokeInst>(CB))
      return false;
    Calls.push_back(CB);
  }
  return !Calls.empty();
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
    SmallVector<CallBase *, 8> Calls;
    if (!collectDirectCallSites(F, Calls))
      continue;

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
  if (!M.getTargetTriple().isAMDGCN())
    return PreservedAnalyses::all();
  if (!promoteUniformArgsToInReg(M))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
