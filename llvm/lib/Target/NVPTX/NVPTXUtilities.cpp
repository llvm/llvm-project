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

Align llvm::getPTXParamTypeAlign(Type *ArgTy, const DataLayout &DL) {
  // Capping the alignment to 128 bytes as that is the maximum alignment
  // supported by PTX.
  return std::min(Align(128), DL.getABITypeAlign(ArgTy));
}

static Align getByValParamAlignFloor(const Function *F) {
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
  return ShouldForceMinAlign ? Align(4) : Align(1);
}

Align llvm::getDeviceByValParamAlign(const Function *F, Type *ArgTy,
                                     unsigned AttrIdx, const DataLayout &DL) {
  return std::max(getPTXParamAlign(F, ArgTy, AttrIdx, DL),
                  getByValParamAlignFloor(F));
}

Align llvm::getDeviceByValParamAlign(const CallBase *CB, Type *ArgTy,
                                     unsigned AttrIdx, const DataLayout &DL) {
  Align ParamAlign = getPTXParamAlign(CB, ArgTy, AttrIdx, DL);

  // For an indirect call getPTXParamAlign can't see the call's own byval
  // alignment, so fold it in.
  if (CB && AttrIdx >= AttributeList::FirstArgIndex)
    ParamAlign = std::max(
        ParamAlign,
        CB->getParamAlign(AttrIdx - AttributeList::FirstArgIndex).valueOrOne());

  return std::max(ParamAlign, getByValParamAlignFloor(
                                  CB ? CB->getCalledFunction() : nullptr));
}

Align llvm::getPTXParamAlign(const Function *F, Type *Ty, unsigned AttrIdx,
                             const DataLayout &DL) {
  if (F)
    if (MaybeAlign StackAlign = getStackAlign(*F, AttrIdx))
      return StackAlign.value();

  Align TypeAlign = getPTXParamTypeAlign(Ty, DL);
  if (F && AttrIdx >= AttributeList::FirstArgIndex) {
    unsigned ArgNo = AttrIdx - AttributeList::FirstArgIndex;
    if (F->getAttributes().hasParamAttr(ArgNo, Attribute::ByVal))
      return std::max(TypeAlign, F->getParamAlign(ArgNo).valueOrOne());
  }
  return TypeAlign;
}

Align llvm::getPTXParamAlign(const CallBase *CB, Type *Ty, unsigned Idx,
                             const DataLayout &DL) {
  if (CB)
    if (MaybeAlign StackAlign = getStackAlign(*CB, Idx))
      return StackAlign.value();

  // Otherwise resolve the direct callee and use its parameter alignment.
  const Function *DirectCallee = CB ? CB->getCalledFunction() : nullptr;
  if (!DirectCallee && CB)
    DirectCallee = getMaybeBitcastedCallee(CB);

  return getPTXParamAlign(DirectCallee, Ty, Idx, DL);
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
