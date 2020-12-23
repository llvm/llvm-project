//===- GPUToSPIRVPass.h - GPU to SPIR-V Passes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert GPU dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOSPIRV_GPUTOSPIRVPASS_H
#define MLIR_CONVERSION_GPUTOSPIRV_GPUTOSPIRVPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T>
class OperationPass;

/// Creates a pass to convert GPU Ops to SPIR-V ops. For a gpu.func to be
/// converted, it should have a spv.entry_point_abi attribute.
std::unique_ptr<OperationPass<ModuleOp>> createConvertGPUToSPIRVPass();

} // namespace mlir
#endif // MLIR_CONVERSION_GPUTOSPIRV_GPUTOSPIRVPASS_H
