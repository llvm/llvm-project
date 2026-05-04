//===-- EJitStructFieldPass.h - JIT Constant Substitution Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSTRUCTFIELDPASS_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSTRUCTFIELDPASS_H

#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
namespace ejit {

/// PASS6: JIT-time specialization pass. Scans the module for load instructions
/// with !ejit.may_const metadata, reads the actual runtime values from process
/// memory via the PeriodArrayRegistry, and replaces the loads with LLVM
/// constants.
class EJitStructFieldPass : public PassInfoMixin<EJitStructFieldPass> {
public:
  EJitStructFieldPass(PeriodArrayRegistry &reg) : registry_(reg) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  PeriodArrayRegistry &registry_;
};

} // namespace ejit
} // namespace llvm

#endif
