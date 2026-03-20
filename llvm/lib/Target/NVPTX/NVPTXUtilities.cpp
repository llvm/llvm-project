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

Align getFunctionParamOptimizedAlign(const Function *F, Type *ArgTy,
                                     const DataLayout &DL) {
  // Capping the alignment to 128 bytes as that is the maximum alignment
  // supported by PTX.
  const Align ABITypeAlign = std::min(Align(128), DL.getABITypeAlign(ArgTy));

  // If a function has linkage different from internal or private, we
  // must use default ABI alignment as external users rely on it. Same
  // for a function that may be called from a function pointer.
  if (!F || !F->hasLocalLinkage() ||
      F->hasAddressTaken(/*Users=*/nullptr,
                         /*IgnoreCallbackUses=*/false,
                         /*IgnoreAssumeLikeCalls=*/true,
                         /*IgnoreLLVMUsed=*/true))
    return ABITypeAlign;

  assert(!isKernelFunction(*F) && "Expect kernels to have non-local linkage");
  return std::max(Align(16), ABITypeAlign);
}

Align getFunctionArgumentAlignment(const Function *F, Type *Ty, unsigned Idx,
                                   const DataLayout &DL) {
  return getAlign(*F, Idx).value_or(getFunctionParamOptimizedAlign(F, Ty, DL));
}

Align getFunctionByValParamAlign(const Function *F, Type *ArgTy,
                                 Align InitialAlign, const DataLayout &DL) {
  Align ArgAlign = InitialAlign;
  if (F)
    ArgAlign = std::max(ArgAlign, getFunctionParamOptimizedAlign(F, ArgTy, DL));

  // Old ptx versions have a bug. When PTX code takes address of
  // byval parameter with alignment < 4, ptxas generates code to
  // spill argument into memory. Alas on sm_50+ ptxas generates
  // SASS code that fails with misaligned access. To work around
  // the problem, make sure that we align byval parameters by at
  // least 4. This bug seems to be fixed at least starting from
  // ptxas > 9.0.
  // TODO: remove this after verifying the bug is not reproduced
  // on non-deprecated ptxas versions.
  if (ForceMinByValParamAlign)
    ArgAlign = std::max(ArgAlign, Align(4));

  return ArgAlign;
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
