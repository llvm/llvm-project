//===- Serialization.h - MLIR SPIR-V (De)serialization ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point for serializing SPIR-V binary modules.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_SERIALIZATION_H
#define MLIR_TARGET_SPIRV_SERIALIZATION_H

#include "mlir/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class MLIRContext;

namespace spirv {
class ModuleOp;

/// Serializes the given SPIR-V `module` and writes to `binary`. On failure,
/// reports errors to the error handler registered with the MLIR context for
/// `module`.
LogicalResult serialize(ModuleOp module, SmallVectorImpl<uint32_t> &binary,
                        bool emitDebugInfo = false);

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_TARGET_SPIRV_SERIALIZATION_H
