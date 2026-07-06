//===- NVPTXUtilities.cpp - Utility Functions -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions
//
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "NVVMProperties.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>

using namespace llvm;

static cl::opt<bool> ForceMinByValParamAlign(
    "nvptx-force-min-byval-param-align", cl::Hidden,
    cl::desc("NVPTX Specific: force 4-byte minimal alignment for byval"
             " params of device functions."),
    cl::init(false));

Function *llvm::getMaybeBitcastedCallee(const CallBase *CB) {
  return dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
}

Align llvm::getPTXPromotedParamTypeAlign(const Function *F, Type *ArgTy,
                                         const DataLayout &DL) {
  // Capping the alignment to 128 bytes as that is the maximum alignment
  // supported by PTX.
  const Align ABITypeAlign = std::min(Align(128), DL.getABITypeAlign(ArgTy));

  // If a function has linkage different from internal or private, we
  // must use default ABI alignment as external users rely on it. Same
  // for a function that may be called from a function pointer.
  const bool MayOptimizeAlign =
      F && F->hasLocalLinkage() &&
      !F->hasAddressTaken(/*Users=*/nullptr,
                          /*IgnoreCallbackUses=*/false,
                          /*IgnoreAssumeLikeCalls=*/true,
                          /*IgnoreLLVMUsed=*/true);
  assert(!(MayOptimizeAlign && isKernelFunction(*F)) &&
         "Expect kernels to have non-local linkage");
  const Align OptimizedAlign = MayOptimizeAlign ? Align(16) : Align(1);
  return std::max(OptimizedAlign, ABITypeAlign);
}

static Align getPTXPromotedParamTypeAlign(const CallBase *CB, Type *Ty,
                                          const DataLayout &DL) {
  const Function *F = CB ? getMaybeBitcastedCallee(CB) : nullptr;
  return getPTXPromotedParamTypeAlign(F, Ty, DL);
}

static bool isKernelFunction(const CallBase &CB) {
  // A call target is never a kernel.
  return false;
}

template <typename AttrSourceT>
static Type *getParamByValType(const AttrSourceT *AttrSource,
                               unsigned AttrIdx) {
  if (AttrSource && AttrIdx >= AttributeList::FirstArgIndex) {
    const unsigned ArgNo = AttrIdx - AttributeList::FirstArgIndex;
    return AttrSource->getParamByValType(ArgNo);
  }
  return nullptr;
}

template <typename AttrSourceT>
static Align getPTXArgAlign(const AttrSourceT *AttrSource, unsigned AttrIdx,
                            Type *Ty, const DataLayout &DL) {

  const Align OptimizedAlign = getPTXPromotedParamTypeAlign(AttrSource, Ty, DL);

  const MaybeAlign StackAlign =
      AttrSource ? getStackAlign(*AttrSource, AttrIdx) : std::nullopt;

  const bool IsByVal = !!getParamByValType(AttrSource, AttrIdx);
  const MaybeAlign ByValAlign =
      IsByVal
          ? AttrSource->getParamAlign(AttrIdx - AttributeList::FirstArgIndex)
          : std::nullopt;

  if (IsByVal && !isKernelFunction(*AttrSource)) {
    const Align InitialAlign = StackAlign.value_or(ByValAlign.valueOrOne());

  // Old ptx versions have a bug. When PTX code takes address of
  // byval parameter with alignment < 4, ptxas generates code to
  // spill argument into memory. Alas on sm_50+ ptxas generates
  // SASS code that fails with misaligned access. To work around
  // the problem, make sure that we align byval parameters by at
  // least 4. This bug seems to be fixed at least starting from
  // ptxas > 9.0.
  // TODO: remove this after verifying the bug is not reproduced
  // on non-deprecated ptxas versions.
    const Align AlignFloor = ForceMinByValParamAlign ? Align(4) : Align(1);

    return std::max({InitialAlign, OptimizedAlign, AlignFloor});
  }

  return StackAlign.value_or(
      std::max({OptimizedAlign, ByValAlign.valueOrOne()}));
}

Align llvm::getPTXArgAlign(const Function *F, const Argument &Arg,
                           const DataLayout &DL) {
  Type *Ty = Arg.hasByValAttr() ? Arg.getParamByValType() : Arg.getType();
  return ::getPTXArgAlign(F, Arg.getArgNo() + AttributeList::FirstArgIndex, Ty,
                          DL);
}

Align llvm::getPTXArgAlign(const CallBase *CB,
                           unsigned ArgNo, Type *Ty, const DataLayout &DL) {
  return ::getPTXArgAlign(CB, ArgNo + AttributeList::FirstArgIndex, Ty, DL);
}

Align llvm::getPTXReturnAlign(const Function *F, Type *Ty,
                              const DataLayout &DL) {
  return ::getPTXArgAlign(F, AttributeList::ReturnIndex, Ty, DL);
}
Align llvm::getPTXReturnAlign(const CallBase *CB, Type *Ty,
                              const DataLayout &DL) {
  return ::getPTXArgAlign(CB, AttributeList::ReturnIndex, Ty, DL);
}

bool llvm::shouldEmitPTXNoReturn(const Value *V, const TargetMachine &TM) {
  const auto &ST =
      *static_cast<const NVPTXTargetMachine &>(TM).getSubtargetImpl();
  if (!ST.hasNoReturn())
    return false;

  assert((isa<Function>(V) || isa<CallInst>(V)) &&
         "Expect either a call instruction or a function");

  if (const CallInst *CallI = dyn_cast<CallInst>(V))
    return CallI->doesNotReturn() &&
           CallI->getFunctionType()->getReturnType()->isVoidTy();

  const Function *F = cast<Function>(V);
  return F->doesNotReturn() &&
         F->getFunctionType()->getReturnType()->isVoidTy() &&
         !isKernelFunction(*F);
}
