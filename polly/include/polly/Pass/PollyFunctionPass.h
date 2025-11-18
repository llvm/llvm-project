//===------ PollyFunctionPass.h - Polly function pass ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PASS_POLLYFUNCTIONPASS_H_
#define POLLY_PASS_POLLYFUNCTIONPASS_H_

#include "polly/Pass/PhaseManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include <utility>

namespace polly {

class PollyFunctionPass : public llvm::PassInfoMixin<PollyFunctionPass> {
public:
  PollyFunctionPass() {}
  PollyFunctionPass(PollyPassOptions Opts) : Opts(std::move(Opts)) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &);

private:
  PollyPassOptions Opts;
};
} // namespace polly

#endif /* POLLY_PASS_POLLYFUNCTIONPASS_H_ */
