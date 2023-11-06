//===- Target.h - MLIR SPIR-V target registration ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the SPIR-V target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_TARGET_H
#define MLIR_TARGET_SPIRV_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace spirv {
/// Registers the `TargetAttrInterface` for the `#spirv.target_env` attribute in
/// the given registry.
void registerSPIRVTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#spirv.target_env` attribute in
/// the registry associated with the given context.
void registerSPIRVTargetInterfaceExternalModels(MLIRContext &context);
} // namespace spirv
} // namespace mlir

#endif // MLIR_TARGET_SPIRV_TARGET_H
