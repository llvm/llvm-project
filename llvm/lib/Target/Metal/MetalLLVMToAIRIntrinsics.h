//===- MetalLLVMToAIRIntrinsics.h - Rename LLVM intrinsics ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Rename a fixed set of LLVM intrinsic declarations to their AIR
/// equivalents (e.g. `llvm.sin.f32` -> `air.fast_sin.f32`), and inline the
/// software implementation of `__mulhi`.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALLLVMTOAIRINTRINSICS_H
#define LLVM_LIB_TARGET_METAL_METALLLVMTOAIRINTRINSICS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalLLVMToAIRIntrinsicsPass
    : public PassInfoMixin<MetalLLVMToAIRIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalLLVMToAIRIntrinsicsLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalLLVMToAIRIntrinsicsLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALLLVMTOAIRINTRINSICS_H
