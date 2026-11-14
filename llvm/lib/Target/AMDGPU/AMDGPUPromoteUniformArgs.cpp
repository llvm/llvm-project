//===-- AMDGPUPromoteUniformArgs.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promote arguments of internal callees to \c inreg when every visible
// call-site operand is trivially uniform: a constant, an argument passed in an
// SGPR, or an always-uniform value in the same block as the call.
//
// FIXME: Promotion is monotone -- marking an argument \c inreg can make a
// call-site operand that forwards it uniform in turn -- but the module is
// visited once, so how much is promoted along a call chain depends on the
// order functions appear in the module. Iterating to a fixpoint would remove
// that dependence.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Uniformity.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
using namespace llvm;

#define DEBUG_TYPE "amdgpu-promote-uniform-args"

STATISTIC(NumPromotedInRegArgs,
          "Number of uniform arguments promoted to inreg");
STATISTIC(NumPromotedInRegFuncs,
          "Number of functions with a promoted uniform argument");

static bool canPromoteArgToInReg(const Argument &A) {
  if (A.hasInRegAttr())
    return false;
  // inreg is mutually exclusive with byval, inalloca, preallocated, byref,
  // sret, and nest. The first five are covered by hasPointeeInMemoryValueAttr.
  if (A.hasPointeeInMemoryValueAttr() || A.hasNestAttr())
    return false;
  return !A.hasAttribute("amdgpu-hidden-argument");
}

// A call site's parameter attributes need not agree with the callee's, so the
// exclusions in canPromoteArgToInReg are rechecked here. paramHasAttr would
// fall back to the declaration, which has already been found clean.
static bool hasIncompatibleParamAttr(const CallBase &CB, unsigned ArgNo) {
  AttributeSet PA = CB.getParamAttributes(ArgNo);
  return PA.hasAttribute(Attribute::ByVal) ||
         PA.hasAttribute(Attribute::StructRet) ||
         PA.hasAttribute(Attribute::InAlloca) ||
         PA.hasAttribute(Attribute::Preallocated) ||
         PA.hasAttribute(Attribute::ByRef) || PA.hasAttribute(Attribute::Nest);
}

static bool collectDirectCallSites(Function &F,
                                   SmallVectorImpl<CallBase *> &Calls) {
  if (F.isDeclaration() || !F.canChangeSignature() || !F.hasLocalLinkage())
    return false;

  if (!any_of(F.args(), canPromoteArgToInReg))
    return false;

  // A use is a call site only when it is the callee of a direct call with this
  // function type. Assume-like intrinsics can mention F without calling it, so
  // they are skipped.
  for (Use &U : F.uses()) {
    if (const auto *II = dyn_cast<IntrinsicInst>(U.getUser()))
      if (II->isAssumeLikeIntrinsic())
        continue;

    CallBase *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB || !CB->isCallee(&U) ||
        CB->getFunctionType() != F.getFunctionType())
      return false;
    if (CB->isMustTailCall())
      return false;
    Calls.push_back(CB);
  }
  if (Calls.empty())
    return false;

  // Checked last: the only gate here that walks F's body. A musttail call must
  // immediately precede a ret, and it requires F's ABI-impacting parameter
  // attributes to agree positionally with its callee's, so F cannot be promoted
  // on its own. Promoting both in lockstep is left for later.
  for (const BasicBlock &BB : F) {
    const Instruction *Term = BB.getTerminator();
    if (!isa<ReturnInst>(Term))
      continue;
    const auto *CI = dyn_cast_or_null<CallInst>(Term->getPrevNode());
    if (CI && CI->isMustTailCall())
      return false;
  }
  return true;
}

static bool isTriviallyUniform(
    CallBase &CB, unsigned ArgNo,
    function_ref<const TargetTransformInfo &(Function &)> GetTTI) {
  Value *V = CB.getArgOperand(ArgNo);
  if (isa<Constant>(V))
    return true;
  if (const auto *A = dyn_cast<Argument>(V))
    return AMDGPU::isArgPassedInSGPR(A);
  if (const auto *I = dyn_cast<Instruction>(V)) {
    // Only an instruction operand needs TTI, so it is fetched lazily.
    if (GetTTI(*CB.getFunction()).getValueUniformity(I) !=
        ValueUniformity::AlwaysUniform)
      return false;
    // If I and the call are in different blocks then there is a possibility of
    // temporal divergence.
    return I->getParent() == CB.getParent();
  }
  return false;
}

static bool promoteUniformArgsToInReg(
    Module &M, function_ref<const TargetTransformInfo &(Function &)> GetTTI) {
  bool Changed = false;

  for (Function &F : M) {
    SmallVector<CallBase *, 8> Calls;
    if (!collectDirectCallSites(F, Calls))
      continue;

    bool FuncChanged = false;
    for (Argument &A : F.args()) {
      if (!canPromoteArgToInReg(A))
        continue;

      unsigned ArgNo = A.getArgNo();
      bool AllUniform = true;
      for (CallBase *CB : Calls) {
        if (hasIncompatibleParamAttr(*CB, ArgNo) ||
            !isTriviallyUniform(*CB, ArgNo, GetTTI)) {
          AllUniform = false;
          break;
        }
      }
      if (!AllUniform)
        continue;

      A.addAttr(Attribute::InReg);
      for (CallBase *CB : Calls)
        CB->addParamAttr(ArgNo, Attribute::InReg);
      ++NumPromotedInRegArgs;
      FuncChanged = Changed = true;
    }

    if (FuncChanged)
      ++NumPromotedInRegFuncs;
  }

  return Changed;
}

PreservedAnalyses AMDGPUPromoteUniformArgsPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTTI = [&FAM](Function &F) -> const TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  if (!promoteUniformArgsToInReg(M, GetTTI))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
