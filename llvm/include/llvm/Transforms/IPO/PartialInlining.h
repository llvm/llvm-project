//===- PartialInlining.h - Inline parts of functions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs partial inlining, typically by inlining an if statement
// that surrounds the body of the function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PARTIALINLINING_H
#define LLVM_TRANSFORMS_IPO_PARTIALINLINING_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Module;

/// Pass to remove unused function declarations.
class PartialInlinerPass : public PassInfoMixin<PartialInlinerPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_PARTIALINLINING_H
