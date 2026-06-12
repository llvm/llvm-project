//===- AIRArgumentBuffer.h - Pack kernel buffers into an arg buffer -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Packs a kernel's device/constant buffer pointer arguments into a single
/// Metal argument buffer (an addrspace(1) struct of pointers passed at
/// [[buffer(0)]], with each member at [[id(N)]]). This matches the indirect
/// argument-buffer ABI expected by argument-buffer-based host runtimes (e.g.
/// IREE's Metal HAL). Kernels that already carry pre-baked !air.kernel metadata
/// (Triton's direct-binding path) are left untouched.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRARGUMENTBUFFER_H
#define LLVM_LIB_TARGET_AIR_AIRARGUMENTBUFFER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRArgumentBufferPass : public PassInfoMixin<AIRArgumentBufferPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRArgumentBufferLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRArgumentBufferLegacy() : ModulePass(ID) {}
  static char ID;
};

ModulePass *createAIRArgumentBufferLegacyPass();
void initializeAIRArgumentBufferLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRARGUMENTBUFFER_H
