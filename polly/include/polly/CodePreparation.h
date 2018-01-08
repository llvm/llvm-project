//===- polly/ScopPreparation.h - Code preparation pass ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Prepare the Function for polyhedral codegeneration.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEPREPARATION_H
#define POLLY_CODEPREPARATION_H

#include "llvm/IR/PassManager.h"

namespace polly {
struct CodePreparationPass : public llvm::PassInfoMixin<CodePreparationPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
} // namespace polly

#endif /* POLLY_CODEPREPARATION_H */
