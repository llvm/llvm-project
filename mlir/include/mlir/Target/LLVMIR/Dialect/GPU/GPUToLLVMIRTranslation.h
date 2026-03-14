//===- GPUToLLVMIRTranslation.h - GPU Dialect to LLVM IR --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for GPU dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_GPU_GPUTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_GPU_GPUTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the GPU dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerGPUDialectTranslation(DialectRegistry &registry);

/// Register the GPU dialect and the translation from it in the registry
/// associated with the given context.
void registerGPUDialectTranslation(MLIRContext &context);

namespace gpu {
/// Registers the offloading LLVM translation interfaces for
/// `gpu.select_object`.
void registerOffloadingLLVMTranslationInterfaceExternalModels(
    mlir::DialectRegistry &registry);
} // namespace gpu

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_GPU_GPUTOLLVMIRTRANSLATION_H
