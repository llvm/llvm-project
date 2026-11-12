//===- OptimizeForNVVM.h - Optimize LLVM IR for NVVM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVVM_TRANSFORMS_OPTIMIZEFORNVVM_H
#define MLIR_DIALECT_NVVM_TRANSFORMS_OPTIMIZEFORNVVM_H

#include <memory>

namespace mlir {
class Pass;
namespace NVVM {
#define GEN_PASS_DECL_NVVMOPTIMIZEFORTARGETPASS
#include "mlir/Dialect/NVVM/Transforms/Passes.h.inc"
} // namespace NVVM
} // namespace mlir

#endif // MLIR_DIALECT_NVVM_TRANSFORMS_OPTIMIZEFORNVVM_H
