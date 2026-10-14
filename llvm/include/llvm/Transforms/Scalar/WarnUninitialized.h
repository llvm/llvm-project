//===- WarnUninitialized.h - Warn about uninitialized loads -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_WARNUNINITIALIZED_H
#define LLVM_TRANSFORMS_SCALAR_WARNUNINITIALIZED_H

#include "llvm/IR/PassManager.h"
#include <memory>

namespace llvm {

class WarnUninitializedDiagnosticState;

LLVM_ABI std::shared_ptr<WarnUninitializedDiagnosticState>
createWarnUninitializedDiagnosticState();

class WarnUninitializedEarlyPass
    : public RequiredPassInfoMixin<WarnUninitializedEarlyPass> {
public:
  explicit WarnUninitializedEarlyPass(
      std::shared_ptr<WarnUninitializedDiagnosticState> State = nullptr)
      : State(State) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  std::shared_ptr<WarnUninitializedDiagnosticState> State;
};

class WarnUninitializedLatePass
    : public RequiredPassInfoMixin<WarnUninitializedLatePass> {
public:
  explicit WarnUninitializedLatePass(
      std::shared_ptr<WarnUninitializedDiagnosticState> State = nullptr)
      : State(State) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  std::shared_ptr<WarnUninitializedDiagnosticState> State;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_WARNUNINITIALIZED_H
