//===- MetalAsyncEventToAlloca.h - TG event global to alloca ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Convert the `@__tg_async_events` threadgroup global to a stack alloca and
/// insert no-op bitcasts before async-copy / wait-event calls so subsequent
/// typed-pointer inference assigns the right element type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALASYNCEVENTTOALLOCA_H
#define LLVM_LIB_TARGET_METAL_METALASYNCEVENTTOALLOCA_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalAsyncEventToAllocaPass
    : public PassInfoMixin<MetalAsyncEventToAllocaPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalAsyncEventToAllocaLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalAsyncEventToAllocaLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALASYNCEVENTTOALLOCA_H
