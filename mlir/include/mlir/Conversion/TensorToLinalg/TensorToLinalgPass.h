//===- TensorToLinalgPass.h - Tensor to Linalg Passes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALGPASS_H
#define MLIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALGPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Creates a pass to convert Tensor ops to Linalg ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTensorToLinalgPass();

} // namespace mlir

#endif // MLIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALGPASS_H
