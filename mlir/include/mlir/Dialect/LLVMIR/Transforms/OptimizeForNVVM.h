//===- OptimizeForNVVM.h - Optimize LLVM IR for NVVM -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_OPTIMIZENVVM_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_OPTIMIZENVVM_H

#include <memory>

namespace mlir {
class Pass;

namespace NVVM {

/// Creates a pass that optimizes LLVM IR for the NVVM target.
std::unique_ptr<Pass> createOptimizeForTargetPass();

} // namespace NVVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_OPTIMIZENVVM_H
