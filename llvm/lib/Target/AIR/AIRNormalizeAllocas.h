//===- AIRNormalizeAllocas.h - Pre-serialization IR cleanup ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Pre-serialization IR cleanup for AIR v1 bitcode: hoist allocas to the
/// entry block, normalize alloca i64 sizes to i32, strip `disjoint` flags,
/// and insert ptr->ptr bitcasts where typed-pointer encoding needs them.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRNORMALIZEALLOCAS_H
#define LLVM_LIB_TARGET_AIR_AIRNORMALIZEALLOCAS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRNormalizeAllocasPass
    : public PassInfoMixin<AIRNormalizeAllocasPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRNormalizeAllocasLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRNormalizeAllocasLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRNORMALIZEALLOCAS_H
