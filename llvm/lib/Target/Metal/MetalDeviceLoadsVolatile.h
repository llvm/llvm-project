//===- MetalDeviceLoadsVolatile.h - Mark loop device loads volatile *- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Mark device (`addrspace(1)`) loads inside loops as volatile when the loop
/// also stores to the same pointer (prevents Metal's GPU JIT from hoisting
/// the load). Functions using CAS atomic spin-locks mark all device
/// loads/stores volatile to suppress reordering across the lock.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALDEVICELOADSVOLATILE_H
#define LLVM_LIB_TARGET_METAL_METALDEVICELOADSVOLATILE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalDeviceLoadsVolatilePass
    : public PassInfoMixin<MetalDeviceLoadsVolatilePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalDeviceLoadsVolatileLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalDeviceLoadsVolatileLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALDEVICELOADSVOLATILE_H
