//===- Serialization.h - MLIR SPIR-V (De)serialization ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry points for deserializing SPIR-V binary modules.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_DESERIALIZATION_H
#define MLIR_TARGET_SPIRV_DESERIALIZATION_H

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>

namespace mlir {
class MLIRContext;

namespace spirv {
class ModuleOp;

/// Deserializes the given SPIR-V `binary` module and creates a MLIR ModuleOp
/// in the given `context`. Returns the ModuleOp on success; otherwise, reports
/// errors to the error handler registered with `context` and returns a null
/// module.
OwningOpRef<spirv::ModuleOp> deserialize(ArrayRef<uint32_t> binary,
                                         MLIRContext *context);

} // namespace spirv
} // namespace mlir

#endif // MLIR_TARGET_SPIRV_DESERIALIZATION_H
