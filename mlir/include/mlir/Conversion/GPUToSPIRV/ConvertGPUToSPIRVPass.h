//===- ConvertGPUToSPIRVPass.h - GPU to SPIR-V conversion pass --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a pass to convert GPU ops to SPIRV ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRVPASS_H
#define MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRVPASS_H

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

/// Pass to convert GPU Ops to SPIR-V ops. For a gpu.func to be converted, it
/// should have a spv.entry_point_abi attribute.
std::unique_ptr<OperationPass<ModuleOp>> createConvertGPUToSPIRVPass();

} // namespace mlir
#endif // MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRVPASS_H
