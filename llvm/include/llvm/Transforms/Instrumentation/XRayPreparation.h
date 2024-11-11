//===- XRayPreparation.h - Preparation for XRay instrumentation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Pass does some IR-level preparations (e.g. inserting global variable
// that carries default options, if there is any) for XRay instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_XRAYPREPARATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_XRAYPREPARATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct XRayPreparationPass : public PassInfoMixin<XRayPreparationPass> {
  PreservedAnalyses run(Module &, ModuleAnalysisManager &);
};
} // namespace llvm
#endif
