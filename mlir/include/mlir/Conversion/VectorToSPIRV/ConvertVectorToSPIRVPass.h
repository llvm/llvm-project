//=- ConvertVectorToSPIRVPass.h - Pass converting Vector to SPIRV -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a pass to convert Vector ops to SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOSPIRV_CONVERTGPUTOSPIRVPASS_H
#define MLIR_CONVERSION_VECTORTOSPIRV_CONVERTGPUTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Pass to convert Vector Ops to SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertVectorToSPIRVPass();

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOSPIRV_CONVERTGPUTOSPIRVPASS_H
