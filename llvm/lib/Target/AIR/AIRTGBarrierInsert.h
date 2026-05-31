//===- AIRTGBarrierInsert.h - Insert TG memory barriers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Insert `air.wg.barrier` calls around threadgroup memory accesses to
/// guarantee inter-thread coherence required by AIR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRTGBARRIERINSERT_H
#define LLVM_LIB_TARGET_AIR_AIRTGBARRIERINSERT_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRTGBarrierInsertPass
    : public PassInfoMixin<AIRTGBarrierInsertPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRTGBarrierInsertLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRTGBarrierInsertLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRTGBARRIERINSERT_H
