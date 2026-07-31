//===-- AMDGPUPromoteUniformArgs.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For non-entry AMDGPU functions, pointer arguments are passed in VGPRs unless
// marked \c inreg. When such an argument is actually uniform across the
// wavefront at every visible call site, passing it in VGPRs forces every lane
// to carry the same value and can inflate register pressure. This pass promotes
// provably-uniform pointer arguments of internal callees to \c inreg (SGPR
// passing) on the definition and at each direct call site.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Analysis.h"
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

static cl::opt<bool> EnablePromoteUniformPointerArgs(
    "amdgpu-promote-uniform-pointer-args", cl::Hidden, cl::init(true),
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

static bool argForwardedByMustTail(const Argument &A) {
  SmallVector<const Value *, 8> Worklist;
  SmallPtrSet<const Value *, 8> Visited;
  Worklist.push_back(&A);
  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;
    for (const User *U : V->users()) {
      if (const auto *CB = dyn_cast<CallBase>(U)) {
        if (CB->isMustTailCall())
          return true;
        continue;
      }
      if (isa<BitCastInst, GetElementPtrInst, AddrSpaceCastInst, PHINode,
              SelectInst>(U))
        Worklist.push_back(cast<Instruction>(U));
    }
  }
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
        if (II->getIntrinsicID() == Intrinsic::amdgcn_addrspacecast_nonnull) {
          if (II->getType()->getPointerAddressSpace() ==
              AMDGPUAS::PRIVATE_ADDRESS)
            return true;
          Worklist.push_back(II);
        }
        continue;
      }
      if (isa<GetElementPtrInst, BitCastInst, PHINode, SelectInst>(U))
        Worklist.push_back(U);
    }
  }
  return false;
}

static bool collectCallSites(Function &F, SmallVectorImpl<CallBase *> &Calls) {
  for (User *U : F.users()) {
    auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledFunction() != &F)
      return false;
    if (CB->isMustTailCall())
      return false;
    if (isa<InvokeInst>(CB))
      return false;
    Calls.push_back(CB);
  }
  return !Calls.empty();
}

static bool promoteUniformPointerArgsToInReg(Module &M,
                                             ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool Changed = false;
  bool RoundChanged = true;
  while (RoundChanged) {
    RoundChanged = false;
    for (Function &F : M) {
      if (!isEligibleInRegUniformCallee(F))
        continue;

      SmallVector<CallBase *, 8> Calls;
      if (!collectCallSites(F, Calls))
        continue;

      bool FuncChanged = false;
      for (Argument &A : F.args()) {
        if (!canPromoteArgToInReg(A))
          continue;
        if (argForwardedByMustTail(A))
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

          Function *Caller = CB->getFunction();
          UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(*Caller);
          if (UI.isDivergentAtUse(CB->getArgOperandUse(A.getArgNo()))) {
            AllUniform = false;
            break;
          }

          const TargetTransformInfo &TTI =
              FAM.getResult<TargetIRAnalysis>(*Caller);
          if (TTI.getValueUniformity(ArgOp) == ValueUniformity::NeverUniform) {
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
  if (!EnablePromoteUniformPointerArgs ||
      !Triple(M.getTargetTriple()).isAMDGCN())
    return PreservedAnalyses::all();
  if (!promoteUniformPointerArgsToInReg(M, AM))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
