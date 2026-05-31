//===- AIREmbedderPass.h - Embed AIR --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_AIREMBEDDERPASS_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_AIREMBEDDERPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
class PassRegistry;

void initializeAIREmbedderLegacyPassPass(PassRegistry &);
ModulePass *createAIREmbedderPass();

/// New pass-manager wrapper for the AIR embedder. Serializes the
/// module into a .metallib byte blob and embeds it as a private global
/// in the ".metallib" section.
class AIREmbedderPass : public PassInfoMixin<AIREmbedderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRWRITER_AIREMBEDDERPASS_H
