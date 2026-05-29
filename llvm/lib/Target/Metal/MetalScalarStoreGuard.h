//===- MetalScalarStoreGuard.h - Guard scalar device stores ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Guard scalar device stores with a `tid.x == 0` check. A function is
/// considered a scalar kernel when it has at least one device-space write
/// (store or global atomic) but no per-thread index intrinsic calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALSCALARSTOREGUARD_H
#define LLVM_LIB_TARGET_METAL_METALSCALARSTOREGUARD_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalScalarStoreGuardPass
    : public PassInfoMixin<MetalScalarStoreGuardPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalScalarStoreGuardLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalScalarStoreGuardLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALSCALARSTOREGUARD_H
