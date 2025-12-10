//===------ PollyModulePass.h - Polly module pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PASS_POLLYMODULEPASS_H_
#define POLLY_PASS_POLLYMODULEPASS_H_

#include "polly/Pass/PhaseManager.h"
#include "llvm/IR/PassManager.h"

namespace polly {

class PollyModulePass : public llvm::PassInfoMixin<PollyModulePass> {
public:
  PollyModulePass() {}
  PollyModulePass(PollyPassOptions Opts) : Opts(std::move(Opts)) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);

private:
  PollyPassOptions Opts;
};

} // namespace polly

#endif /* POLLY_PASS_POLLYMODULEPASS_H_ */
