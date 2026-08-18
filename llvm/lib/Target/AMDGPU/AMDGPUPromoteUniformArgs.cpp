//===-- AMDGPUPromoteUniformArgs.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promote provably-uniform pointer arguments of internal callees to \c inreg
// (SGPR passing) on the definition and at every direct call site. Pointers are
// the primary target: non-entry pointer args default to VGPR unless marked
// \c inreg, while many scalars already use the SGPR path. Scalar promotion is
// a planned follow-up. Uniformity is established conservatively via
// \c GCNTTIImpl::isAlwaysUniform (queried through TTI) plus recursive
// propagation through visible caller chains. A full \c UniformityInfo check is
// a planned follow-up.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
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
  if (F.isDeclaration() || F.isVarArg())
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
  }
  if (F.user_empty())
    return false;
  return true;
}

// musttail requires the enclosing function's parameter ABI (including inreg) to
// match positionally, so skip promotion for any function containing one.
static bool hasMustTailCall(const Function &F) {
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (const auto *CB = dyn_cast<CallBase>(&I))
        if (CB->isMustTailCall())
          return true;
  return false;
}

static bool mayBePrivateDerivedPointer(const Value *V) {
  assert(V->getType()->isPointerTy());
  if (V->getType()->getPointerAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS)
    return true;

  SmallVector<const Value *, 8> Objects;
  getUnderlyingObjects(V, Objects);
  for (const Value *Obj : Objects) {
    if (isa<AllocaInst>(Obj))
      return true;
    if (Obj->getType()->isPointerTy() &&
        Obj->getType()->getPointerAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS)
      return true;
  }
  return false;
}

static bool calleeCastsArgToPrivate(const Argument &A) {
  SmallVector<const Value *, 16> Worklist;
  SmallPtrSet<const Value *, 16> Visited;
  Worklist.push_back(&A);
  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;
    for (const User *U : V->users()) {
      if (const auto *ASC = dyn_cast<AddrSpaceCastInst>(U)) {
        if (ASC->getDestAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS)
          return true;
        Worklist.push_back(ASC);
        continue;
      }
      if (const auto *II = dyn_cast<IntrinsicInst>(U)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::amdgcn_addrspacecast_nonnull:
          if (II->getType()->getPointerAddressSpace() ==
              AMDGPUAS::PRIVATE_ADDRESS)
            return true;
          Worklist.push_back(II);
          break;
        case Intrinsic::ptrmask:
        case Intrinsic::launder_invariant_group:
        case Intrinsic::strip_invariant_group:
          Worklist.push_back(II);
          break;
        default:
          break;
        }
        continue;
      }
      if (isa<GetElementPtrInst, BitCastInst, PHINode, SelectInst, FreezeInst>(
              U))
        Worklist.push_back(U);
    }
  }
  return false;
}

static bool isAlwaysUniformValue(const TargetTransformInfo &TTI,
                                 const Value *V) {
  if (isa<Constant>(V))
    return true;
  return TTI.getValueUniformity(V) == ValueUniformity::AlwaysUniform;
}

static bool isTriviallyUniformArg(
    const Argument *Arg,
    function_ref<TargetTransformInfo &(const Function &)> GetTTI,
    SmallPtrSetImpl<const Argument *> &Visited);

static bool isTriviallyUniformOperand(
    const Value *V,
    function_ref<TargetTransformInfo &(const Function &)> GetTTI,
    SmallPtrSetImpl<const Argument *> &Visited) {
  if (const auto *Arg = dyn_cast<Argument>(V))
    return isTriviallyUniformArg(Arg, GetTTI, Visited);

  if (const auto *I = dyn_cast<Instruction>(V))
    return isAlwaysUniformValue(GetTTI(*I->getFunction()), V);

  return false;
}

static bool isTriviallyUniformArg(
    const Argument *Arg,
    function_ref<TargetTransformInfo &(const Function &)> GetTTI,
    SmallPtrSetImpl<const Argument *> &Visited) {
  if (Arg->hasInRegAttr())
    return true;

  const Function *F = Arg->getParent();
  if (AMDGPU::isEntryFunctionCC(F->getCallingConv()))
    return AMDGPU::isArgPassedInSGPR(Arg);

  // Only direct-call-only internal callees have fully visible callers; an
  // indirect or address-taken caller could pass a divergent value we never see.
  if (!isEligibleInRegUniformCallee(*F))
    return false;

  // Path-based cycle guard: a (mutually) recursive chain that forwards the
  // argument back to itself cannot be proven here and must terminate. The
  // argument is removed on the way out so a diamond-shaped call graph can still
  // reach the same argument through an independent path.
  if (!Visited.insert(Arg).second)
    return false;

  bool HasCallSite = false;
  bool AllUniform = true;
  for (const User *U : F->users()) {
    const auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledFunction() != F)
      continue;
    HasCallSite = true;
    if (!isTriviallyUniformOperand(CB->getArgOperand(Arg->getArgNo()), GetTTI,
                                   Visited)) {
      AllUniform = false;
      break;
    }
  }
  Visited.erase(Arg);
  return HasCallSite && AllUniform;
}

static bool promoteUniformPointerArgsToInReg(Module &M,
                                             ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTTI = [&FAM](const Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(const_cast<Function &>(F));
  };
  bool Changed = false;
  bool RoundChanged = true;
  while (RoundChanged) {
    RoundChanged = false;
    for (Function &F : M) {
      if (F.hasOptNone())
        continue;
      if (!isEligibleInRegUniformCallee(F))
        continue;

      // Promoting any argument changes F's ABI attributes, which a musttail
      // call in F's body would then no longer match positionally. Skip
      // conservatively.
      if (hasMustTailCall(F))
        continue;

      // isEligibleInRegUniformCallee guarantees every user is a direct,
      // non-musttail, non-invoke call to F, so we can just gather them.
      SmallVector<CallBase *, 8> Calls;
      bool HasOptNoneCaller = false;
      for (User *U : F.users()) {
        auto *CB = dyn_cast<CallBase>(U);
        if (!CB || CB->getCalledFunction() != &F)
          continue;
        if (CB->getFunction()->hasOptNone())
          HasOptNoneCaller = true;
        Calls.push_back(CB);
      }
      if (HasOptNoneCaller || Calls.empty())
        continue;

      bool FuncChanged = false;
      for (Argument &A : F.args()) {
        if (!canPromoteArgToInReg(A))
          continue;
        if (calleeCastsArgToPrivate(A))
          continue;

        bool AllUniform = true;
        for (CallBase *CB : Calls) {
          Value *ArgOp = CB->getArgOperand(A.getArgNo());
          if (mayBePrivateDerivedPointer(ArgOp)) {
            AllUniform = false;
            break;
          }

          SmallPtrSet<const Argument *, 4> VisitedArgs;
          if (!isTriviallyUniformOperand(ArgOp, GetTTI, VisitedArgs)) {
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
        FuncChanged = Changed = RoundChanged = true;
      }

      if (FuncChanged) {
        ++NumPromotedInRegFuncs;
        PreservedAnalyses FuncPA;
        // Attribute-only change; the CFG is unchanged.
        FuncPA.preserveSet<CFGAnalyses>();
        FAM.invalidate(F, FuncPA);
        SmallPtrSet<Function *, 8> InvalidatedCallers;
        for (CallBase *CB : Calls) {
          Function *Caller = CB->getFunction();
          if (Caller != &F && InvalidatedCallers.insert(Caller).second)
            FAM.invalidate(*Caller, FuncPA);
        }
      }
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
