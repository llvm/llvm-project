//===-- Target.h - MLIR XeVM target registration ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the XeVM target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_XEVM_TARGET_H
#define MLIR_TARGET_LLVM_XEVM_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace xevm {
/// Registers the `TargetAttrInterface` for the `#xevm.target` attribute in
/// the given registry.
void registerXeVMTargetInterfaceExternalModels(mlir::DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#xevm.target` attribute in
/// the registry associated with the given context.
void registerXeVMTargetInterfaceExternalModels(mlir::MLIRContext &context);
} // namespace xevm
} // namespace mlir

#endif // MLIR_TARGET_LLVM_XEVM_TARGET_H
