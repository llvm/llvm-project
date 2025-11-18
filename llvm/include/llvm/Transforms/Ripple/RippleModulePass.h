//===--------- RippleModulePass.h - Expand RIpple intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands Ripple intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_RIPPLEMODULEPASS_H
#define LLVM_TRANSFORMS_VECTORIZE_RIPPLEMODULEPASS_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class TargetMachine;

class RippleModulePass : public PassInfoMixin<RippleModulePass> {
  TargetMachine *TM;

public:
  RippleModulePass(TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_RIPPLEMODULEPASS_H
