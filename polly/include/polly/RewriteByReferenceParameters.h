//===- RewriteByReferenceParameters.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_REWRITEBYREFERENCEPARAMETERS_H
#define POLLY_REWRITEBYREFERENCEPARAMETERS_H

#include "polly/ScopPass.h"

namespace llvm {
class PassRegistry;
class Pass;
class raw_ostream;
} // namespace llvm

namespace polly {
llvm::Pass *createRewriteByrefParamsWrapperPass();

struct RewriteByrefParamsPass : llvm::PassInfoMixin<RewriteByrefParamsPass> {
  RewriteByrefParamsPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace polly

namespace llvm {
void initializeRewriteByrefParamsWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_REWRITEBYREFERENCEPARAMETERS_H */
