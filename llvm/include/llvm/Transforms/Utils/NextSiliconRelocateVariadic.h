//===- NextSiliconRelocateVariadic.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// NextSiliconRelocateVariadic is an LLVM pass that relocates variadic
// call-sites from functions into thunks, thus making the variadic call an
// indirect call. The original call is replaces with a call to the thunk which
// has a non-variadic function signature.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NEXTSILICONRELOCATEVARIADIC_H
#define LLVM_TRANSFORMS_UTILS_NEXTSILICONRELOCATEVARIADIC_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;

class NextSiliconRelocateVariadicPass
    : public PassInfoMixin<NextSiliconRelocateVariadicPass> {
public:
  NextSiliconRelocateVariadicPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_NEXTSILICONRELOCATEVARIADIC_H
