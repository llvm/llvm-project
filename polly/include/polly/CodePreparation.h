//===- polly/ScopPreparation.h - Code preparation pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
struct CodePreparationPass final : llvm::PassInfoMixin<CodePreparationPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
} // namespace polly

#endif /* POLLY_CODEPREPARATION_H */
