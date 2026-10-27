//===- AIRScalarArgsToConstantBuffers.h - Scalars as constant buffers ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AIR kernels take no arguments by value: a scalar is a reference to constant
/// memory (`constant T &x [[buffer(n)]]` in MSL, a `ptr addrspace(2)` buffer
/// argument in AIR). Give each by-value integer, floating-point or vector
/// kernel argument that form: a constant buffer argument holding the value,
/// loaded once at the kernel's entry. `i1` is held in an `i8`. The buffer is
/// read-only, not aliased and dereferenceable for the value's size, which lets
/// a consumer pass its contents in registers (the AGX target does).
///
/// This is the portable counterpart of `air-scalar-buffer-packing`, which
/// packs the same arguments into one device buffer for the Triton host.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRSCALARARGSTOCONSTANTBUFFERS_H
#define LLVM_LIB_TARGET_AIR_AIRSCALARARGSTOCONSTANTBUFFERS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRScalarArgsToConstantBuffersPass
    : public RequiredPassInfoMixin<AIRScalarArgsToConstantBuffersPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AIRScalarArgsToConstantBuffersLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRScalarArgsToConstantBuffersLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRSCALARARGSTOCONSTANTBUFFERS_H
