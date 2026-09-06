//===- InjectOpenMPVFABIMappings.h - OpenMP _ZGV to VFABI conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts raw `_ZGV...` function attributes (emitted by OMPIRBuilder for
// `declare simd` functions) into the structured `vector-function-abi-variant`
// attribute that LoopVectorize / VFDatabase consumes, and emits external
// declarations for the vector variants.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_INJECTOPENMPVFABIMAPPINGS_H
#define LLVM_TRANSFORMS_UTILS_INJECTOPENMPVFABIMAPPINGS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

class InjectOpenMPVFABIMappings
    : public PassInfoMixin<InjectOpenMPVFABIMappings> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_INJECTOPENMPVFABIMAPPINGS_H
