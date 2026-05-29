//===- MetalBarrierRename.h - Rename threadgroup barrier --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Rename `air.threadgroup.barrier` calls to `air.wg.barrier` and fix the
/// legacy `(1, 4)` argument pair to `(2, 1)`. Existing `air.wg.barrier`
/// calls are left untouched.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALBARRIERRENAME_H
#define LLVM_LIB_TARGET_METAL_METALBARRIERRENAME_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalBarrierRenamePass : public PassInfoMixin<MetalBarrierRenamePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalBarrierRenameLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalBarrierRenameLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALBARRIERRENAME_H
