//===-- CrossDSOCFI.cpp - Externalize this module's CFI checks --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass exports all llvm.bitset's found in the module in the form of a
// __cfi_check function, which can be used to verify cross-DSO call targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_CROSSDSOCFI_H
#define LLVM_TRANSFORMS_IPO_CROSSDSOCFI_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class CrossDSOCFIPass : public PassInfoMixin<CrossDSOCFIPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
}
#endif // LLVM_TRANSFORMS_IPO_CROSSDSOCFI_H

