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

Align llvm::getPTXParamTypeAlign(Type *ArgTy, const DataLayout &DL) {
  // Capping the alignment to 128 bytes as that is the maximum alignment
  // supported by PTX.
  return std::min(Align(128), DL.getABITypeAlign(ArgTy));
}

static Align getByValParamAlignFloor(const bool IsKernelFunction) {
  // Old ptx versions have a bug. When PTX code takes address of
  // byval parameter with alignment < 4, ptxas generates code to
  // spill argument into memory. Alas on sm_50+ ptxas generates
  // SASS code that fails with misaligned access. To work around
  // the problem, make sure that we align byval parameters by at
  // least 4. This bug seems to be fixed at least starting from
  // ptxas > 9.0.
  // TODO: remove this after verifying the bug is not reproduced
  // on non-deprecated ptxas versions.
  const bool ShouldForceMinAlign = ForceMinByValParamAlign && !IsKernelFunction;
  return ShouldForceMinAlign ? Align(4) : Align(1);
}

Align llvm::getPTXParamAlign(const Function *F, Type *Ty, unsigned AttrIdx,
                             const DataLayout &DL) {

  const Align TypeAlign = getPTXParamTypeAlign(Ty, DL);
  if (F) {
    if (MaybeAlign StackAlign = getStackAlign(*F, AttrIdx))
      return StackAlign.value();

    if (AttrIdx >= AttributeList::FirstArgIndex) {
      unsigned ArgNo = AttrIdx - AttributeList::FirstArgIndex;
      if (F->getAttributes().hasParamAttr(ArgNo, Attribute::ByVal))
        return std::max({TypeAlign, F->getParamAlign(ArgNo).valueOrOne(),
                         getByValParamAlignFloor(isKernelFunction(*F))});
    }
  }
  return TypeAlign;
}

Align llvm::getPTXParamAlign(const CallBase *CB, Type *Ty, unsigned AttrIdx,
                             const DataLayout &DL) {
  const Align TypeAlign = getPTXParamTypeAlign(Ty, DL);
  if (CB) {
    if (MaybeAlign StackAlign = getStackAlign(*CB, AttrIdx))
      return StackAlign.value();

    if (AttrIdx >= AttributeList::FirstArgIndex) {
      unsigned ArgNo = AttrIdx - AttributeList::FirstArgIndex;
      if (CB->getAttributes().hasParamAttr(ArgNo, Attribute::ByVal))
        return std::max({TypeAlign, CB->getParamAlign(ArgNo).valueOrOne(),
                         getByValParamAlignFloor(/*IsKernelFunction=*/false)});
    }
  }
  return TypeAlign;
}
