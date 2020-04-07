//===- LinalgToSPIRVPass.h -  Linalg to SPIR-V conversion pass --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass for Linalg to SPIR-V dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSPIRV_LINALGTOSPIRVPASS_H
#define MLIR_CONVERSION_STANDARDTOSPIRV_LINALGTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates and returns a pass to convert Linalg ops to SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createLinalgToSPIRVPass();

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_LINALGTOSPIRVPASS_H
