//===------ DumpFunctionPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write a function to a file.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_DUMPFUNCTIONPASS_H
#define POLLY_SUPPORT_DUMPFUNCTIONPASS_H

#include "llvm/IR/PassManager.h"
#include <string>

namespace llvm {
class ModulePass;
} // namespace llvm

namespace polly {
llvm::FunctionPass *createDumpFunctionWrapperPass(std::string Suffix);

/// A pass that isolates a function into a new Module and writes it into a file.
struct DumpFunctionPass : llvm::PassInfoMixin<DumpFunctionPass> {
  std::string Suffix;

  DumpFunctionPass(std::string Suffix) : Suffix(std::move(Suffix)) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
};

} // namespace polly

namespace llvm {
class PassRegistry;
void initializeDumpFunctionWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_SUPPORT_DUMPFUNCTIONPASS_H */
