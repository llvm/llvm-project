//===- Passes.h - LLVM Target Pass Construction and Registration ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_TRANSFORMS_PASSES_H
#define MLIR_TARGET_LLVMIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace LLVM {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir/Target/LLVMIR/Transforms/Passes.h.inc"

void registerTargetLLVMPasses();

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_TRANSFORMS_PASSES_H
