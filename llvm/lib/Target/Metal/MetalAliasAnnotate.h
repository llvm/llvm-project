//===- MetalAliasAnnotate.h - AIR-style aliasing metadata -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Emit Apple-style alias-scope metadata and the "air-buffer-no-alias"
/// parameter attribute on every kernel buffer argument. Models what Apple's
/// frontend ships in modern AIR so the Metal JIT can perform the same
/// load/store optimisations without the volatility / widening workarounds.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALALIASANNOTATE_H
#define LLVM_LIB_TARGET_METAL_METALALIASANNOTATE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalAliasAnnotatePass
    : public PassInfoMixin<MetalAliasAnnotatePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalAliasAnnotateLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalAliasAnnotateLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALALIASANNOTATE_H
