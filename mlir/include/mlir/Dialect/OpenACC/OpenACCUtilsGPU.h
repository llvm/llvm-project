//===- OpenACCUtilsGPU.h - OpenACC GPU Utilities -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for OpenACC that depend on the GPU
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSGPU_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSGPU_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace mlir {
namespace acc {

/// Default GPU module name used by OpenACC.
constexpr llvm::StringLiteral kDefaultGPUModuleName = "acc_gpu_module";

/// Get or create a GPU module in the given module.
///
/// If a GPU module with the specified name already exists, it is returned.
/// If `create` is true and no GPU module exists, one is created.
/// If `create` is false and no GPU module exists, std::nullopt is returned.
///
/// \param mod The module to search or create the GPU module in.
/// \param create If true (default), create the GPU module if it doesn't exist.
/// \param name The name for the GPU module. If empty, uses
/// kDefaultGPUModuleName.
/// \return The GPU module if found or created, std::nullopt otherwise.
std::optional<gpu::GPUModuleOp>
getOrCreateGPUModule(ModuleOp mod, bool create = true,
                     llvm::StringRef name = kDefaultGPUModuleName);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSGPU_H_
