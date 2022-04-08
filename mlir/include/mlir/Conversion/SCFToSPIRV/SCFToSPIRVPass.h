//===- SCFToSPIRVPass.h - SCF to SPIR-V Passes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert SCF dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOSPIRV_SCFTOSPIRVPASS_H
#define MLIR_CONVERSION_SCFTOSPIRV_SCFTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Creates a pass to convert SCF ops into SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSCFToSPIRVPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOSPIRV_SCFTOSPIRVPASS_H
