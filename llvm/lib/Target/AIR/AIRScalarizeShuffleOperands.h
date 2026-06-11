//===- AIRScalarizeShuffleOperands.h - Scalarize shuffle inputs -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scalarize the vector data-flow entangled with `air.simd_shuffle*` operands.
///
/// The Apple AGX GPU JIT miscompiles a cross-lane shuffle whose scalar operand
/// is `extractelement`-ed from a vector SSA value (vector register): the
/// permute reads the wrong physical lane for some SIMD threads, silently
/// corrupting cross-lane reductions. The SLP vectorizer (LLVM O1+) produces
/// exactly this shape in reduce/scan/thread-locality kernels.
///
/// This pass runs the standard LLVM Scalarizer, but only on functions where a
/// shuffle operand is fed from a vector extract — so GEMM kernels (whose only
/// vectors are memory load/store quads, never feeding a shuffle) are left
/// byte-identical.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRSCALARIZESHUFFLEOPERANDS_H
#define LLVM_LIB_TARGET_AIR_AIRSCALARIZESHUFFLEOPERANDS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRScalarizeShuffleOperandsPass
    : public PassInfoMixin<AIRScalarizeShuffleOperandsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRScalarizeShuffleOperandsLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRScalarizeShuffleOperandsLegacy() : ModulePass(ID) {}
  static char ID;
};

ModulePass *createAIRScalarizeShuffleOperandsLegacyPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRSCALARIZESHUFFLEOPERANDS_H
