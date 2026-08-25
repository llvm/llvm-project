//===-- AMDGPUPromoteUniformArgs.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promote pointer arguments of internal callees to \c inreg when every visible
// call-site operand is trivially uniform per \c GCNTTIImpl::isAlwaysUniform
// (queried through TTI). Arg-chain propagation, private guards, and full
// \c UniformityInfo are planned follow-ups; see
// \c AMDGPUPromoteUniformArgs.cpp.advanced for a more complete prototype.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-promote-uniform-args"

STATISTIC(NumPromotedInRegArgs,
          "Number of uniform pointer arguments promoted to inreg");
STATISTIC(NumPromotedInRegFuncs,
          "Number of functions with a promoted uniform pointer argument");

static cl::opt<bool> EnablePromoteUniformArgs(
    "amdgpu-enable-promote-uniform-args", cl::Hidden, cl::init(true),
    cl::desc("Promote provably uniform internal pointer arguments to inreg"));

namespace {

static bool canPromoteArgToInReg(const Argument &A) {
  if (!A.getType()->isPointerTy() || A.hasInRegAttr())
    return false;
  if (A.hasPointeeInMemoryValueAttr() || A.hasNestAttr() ||
      A.hasReturnedAttr() || A.hasSwiftSelfAttr() || A.hasSwiftErrorAttr() ||
      A.hasAttribute(Attribute::SwiftAsync))
    return false;
  return !A.hasAttribute("amdgpu-hidden-argument");
}

static bool isEligibleInRegUniformCallee(const Function &F) {
  if (F.isDeclaration() || F.isVarArg() || F.hasOptNone())
    return false;
  if (!F.hasLocalLinkage() || F.hasAddressTaken())
    return false;
  switch (F.getCallingConv()) {
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  default:
    return false;
  }
  for (const User *U : F.users()) {
    const auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledFunction() != &F)
      return false;
    if (CB->isMustTailCall() || isa<InvokeInst>(CB))
      return false;
    if (CB->getFunction()->hasOptNone())
      return false;
  }
  return !F.user_empty();
}

static bool isAlwaysUniformValue(const TargetTransformInfo &TTI,
                                 const Value *V) {
  if (isa<Constant>(V))
    return true;
  return TTI.getValueUniformity(V) == ValueUniformity::AlwaysUniform;
}

static bool promoteUniformPointerArgsToInReg(Module &M,
                                             ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
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
        const TargetTransformInfo &TTI =
            FAM.getResult<TargetIRAnalysis>(*CB->getFunction());
        if (!isAlwaysUniformValue(TTI, CB->getArgOperand(A.getArgNo()))) {
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

    if (FuncChanged) {
      ++NumPromotedInRegFuncs;
      PreservedAnalyses FuncPA;
      FuncPA.preserveSet<CFGAnalyses>();
      FAM.invalidate(F, FuncPA);
    }
  }

  return Changed;
}

} // namespace

PreservedAnalyses AMDGPUPromoteUniformArgsPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (!EnablePromoteUniformArgs || !Triple(M.getTargetTriple()).isAMDGCN())
    return PreservedAnalyses::all();
  if (!promoteUniformPointerArgsToInReg(M, AM))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
