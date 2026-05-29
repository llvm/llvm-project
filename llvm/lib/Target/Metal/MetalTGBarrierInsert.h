//===- MetalTGBarrierInsert.h - Insert TG memory barriers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Insert `air.wg.barrier` calls around threadgroup memory accesses to
/// guarantee inter-thread coherence required by Metal.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALTGBARRIERINSERT_H
#define LLVM_LIB_TARGET_METAL_METALTGBARRIERINSERT_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalTGBarrierInsertPass
    : public PassInfoMixin<MetalTGBarrierInsertPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalTGBarrierInsertLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalTGBarrierInsertLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALTGBARRIERINSERT_H
