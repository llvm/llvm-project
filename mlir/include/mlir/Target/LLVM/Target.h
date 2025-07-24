//===- Target.h - MLIR LLVM target registration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the LLVM target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_TARGET_H
#define MLIR_TARGET_LLVM_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace LLVM {
/// Registers the `TargetAttrInterface` for the `#llvm.target` attribute in the
/// given registry.
void registerLLVMTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#llvm.target` attribute in the
/// registry associated with the given context.
void registerLLVMTargetInterfaceExternalModels(MLIRContext &context);
} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVM_TARGET_H
