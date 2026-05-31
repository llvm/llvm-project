//===- AIRTGGlobalCoalesce.h - Merge cvt/dot TG globals ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Merge `__tg_cvt_*` threadgroup globals into `__tg_dot_ab_*` globals when
/// MMA intrinsics are present. The cvt and dot buffers don't overlap in
/// lifetime, so they can share storage; this works around a AIR GPU JIT
/// crash that triggers when a kernel contains multiple TG globals together
/// with MMA intrinsics.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRTGGLOBALCOALESCE_H
#define LLVM_LIB_TARGET_AIR_AIRTGGLOBALCOALESCE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRTGGlobalCoalescePass
    : public PassInfoMixin<AIRTGGlobalCoalescePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRTGGlobalCoalesceLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRTGGlobalCoalesceLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRTGGLOBALCOALESCE_H
