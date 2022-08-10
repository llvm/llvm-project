//===- MemRefToSPIRVPass.h - MemRef to SPIR-V Passes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert MemRef dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRVPASS_H
#define MLIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRVPASS_H

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Creates a pass to map numeric MemRef memory spaces to symbolic SPIR-V
/// storage classes. The mapping is read from the command-line option.
std::unique_ptr<OperationPass<>> createMapMemRefStorageClassPass();

/// Creates a pass to convert MemRef ops to SPIR-V ops.
std::unique_ptr<OperationPass<>> createConvertMemRefToSPIRVPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRVPASS_H
