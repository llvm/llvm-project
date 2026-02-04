//===-- CFGuard.h - CFGuard Transformations ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// Windows Control Flow Guard passes (/guard:cf).
//===---------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CFGUARD_H
#define LLVM_TRANSFORMS_CFGUARD_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class CallBase;
class FunctionPass;
class GlobalValue;

class CFGuardPass : public PassInfoMixin<CFGuardPass> {
public:
  enum class Mechanism { Check, Dispatch };

  CFGuardPass(Mechanism M = Mechanism::Check) : GuardMechanism(M) {}
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  Mechanism GuardMechanism;
};

/// Insert Control FLow Guard checks on indirect function calls.
LLVM_ABI FunctionPass *createCFGuardCheckPass();

/// Insert Control FLow Guard dispatches on indirect function calls.
LLVM_ABI FunctionPass *createCFGuardDispatchPass();

LLVM_ABI bool isCFGuardCall(const CallBase *CB);
LLVM_ABI bool isCFGuardFunction(const GlobalValue *GV);

} // namespace llvm

#endif
