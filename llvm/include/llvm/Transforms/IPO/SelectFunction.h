//===-- SelectFunction.h - Compile only a selected function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass keeps only the named function and its transitive dependencies,
// removing everything else from the module. It works by chaining:
//   1. InternalizePass  — marks everything except the target as internal
//   2. GlobalDCEPass    — removes unreachable internal globals
//   3. StripDeadPrototypesPass — cleans up leftover dead declarations
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SELECTFUNCTION_H
#define LLVM_TRANSFORMS_IPO_SELECTFUNCTION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {

class Module;

struct SelectFunctionPass : PassInfoMixin<SelectFunctionPass> {
  SmallVector<std::string, 2> FunctionNames;

  SelectFunctionPass(SmallVector<std::string, 0> Names)
      : FunctionNames(std::move(Names)) {}

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName) {
    OS << MapClassName2PassName("SelectFunctionPass");
    OS << "<";
    for (size_t I = 0; I < FunctionNames.size(); ++I) {
      if (I)
        OS << ";";
      OS << "fn=" << FunctionNames[I];
    }
    OS << ">";
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_SELECTFUNCTION_H
