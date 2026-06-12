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

namespace llvm {

static cl::opt<bool> ForceMinByValParamAlign(
    "nvptx-force-min-byval-param-align", cl::Hidden,
    cl::desc("NVPTX Specific: force 4-byte minimal alignment for byval"
             " params of device functions."),
    cl::init(false));

Function *getMaybeBitcastedCallee(const CallBase *CB) {
  return dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
}

Align getPTXPromotedParamTypeAlign(const Function *F, Type *ArgTy,
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

Align getDeviceByValParamAlign(const Function *F, Type *ArgTy,
                               Align InitialAlign, const DataLayout &DL) {
  const Align OptimizedAlign = getPTXPromotedParamTypeAlign(F, ArgTy, DL);

  // Old ptx versions have a bug. When PTX code takes address of
  // byval parameter with alignment < 4, ptxas generates code to
  // spill argument into memory. Alas on sm_50+ ptxas generates
  // SASS code that fails with misaligned access. To work around
  // the problem, make sure that we align byval parameters by at
  // least 4. This bug seems to be fixed at least starting from
  // ptxas > 9.0.
  // TODO: remove this after verifying the bug is not reproduced
  // on non-deprecated ptxas versions.
  const bool ShouldForceMinAlign =
      ForceMinByValParamAlign && (!F || !isKernelFunction(*F));
  const Align AlignFloor = ShouldForceMinAlign ? Align(4) : Align(1);

  return std::max({InitialAlign, OptimizedAlign, AlignFloor});
}

Align getPTXParamAlign(const Function *F, Type *Ty, unsigned AttrIdx,
                       const DataLayout &DL) {
  if (F)
    if (MaybeAlign StackAlign = getStackAlign(*F, AttrIdx))
      return StackAlign.value();

  Align TypeAlign = getPTXPromotedParamTypeAlign(F, Ty, DL);
  if (F && AttrIdx >= AttributeList::FirstArgIndex) {
    unsigned ArgNo = AttrIdx - AttributeList::FirstArgIndex;
    if (F->getAttributes().hasParamAttr(ArgNo, Attribute::ByVal))
      return std::max(TypeAlign, F->getParamAlign(ArgNo).valueOrOne());
  }
  return TypeAlign;
}

Align getPTXParamAlign(const CallBase *CB, Type *Ty, unsigned Idx,
                       const DataLayout &DL) {
  const Function *DirectCallee = CB ? CB->getCalledFunction() : nullptr;

  if (!DirectCallee && CB) {
    if (MaybeAlign StackAlign = getStackAlign(*CB, Idx))
      return StackAlign.value();

    DirectCallee = getMaybeBitcastedCallee(CB);
  }

  return getPTXParamAlign(DirectCallee, Ty, Idx, DL);
}

bool shouldEmitPTXNoReturn(const Value *V, const TargetMachine &TM) {
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

} // namespace llvm
