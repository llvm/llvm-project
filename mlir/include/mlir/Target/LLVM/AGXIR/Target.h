//===- Target.h - MLIR AGXIR target registration ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the AGXIR target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_AGXIR_TARGET_H
#define MLIR_TARGET_LLVM_AGXIR_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace AGXIR {
/// Registers the `TargetAttrInterface` for the `#agxir.target` attribute in
/// the given registry.
void registerAGXIRTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#agxir.target` attribute in
/// the registry associated with the given context.
void registerAGXIRTargetInterfaceExternalModels(MLIRContext &context);
} // namespace AGXIR
} // namespace mlir

#endif // MLIR_TARGET_LLVM_AGXIR_TARGET_H
