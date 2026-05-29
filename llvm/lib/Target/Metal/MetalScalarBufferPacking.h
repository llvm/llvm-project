//===- MetalScalarBufferPacking.h - Pack scalar params into one buffer
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Pack all scalar kernel parameters (and `addrspace(2)` constant-buffer
/// pointers) into one `addrspace(1)` device-buffer parameter appended at the
/// end of the signature, matching the Triton Python driver's argument packing
/// for the Metal backend. Must run BEFORE MetalAIRSystemValues so the latter
/// sees the packed signature when emitting `!air.kernel` metadata.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALSCALARBUFFERPACKING_H
#define LLVM_LIB_TARGET_METAL_METALSCALARBUFFERPACKING_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalScalarBufferPackingPass
    : public PassInfoMixin<MetalScalarBufferPackingPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalScalarBufferPackingLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalScalarBufferPackingLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALSCALARBUFFERPACKING_H
