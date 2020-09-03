//===- ConvertSPIRVToLLVMPass.h - SPIR-V dialect to LLVM pass ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a pass to lower from SPIR-V dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SPIRVTOLLVM_CONVERTSPIRVTOLLVMPASS_H
#define MLIR_CONVERSION_SPIRVTOLLVM_CONVERTSPIRVTOLLVMPASS_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

/// Creates a pass to convert SPIR-V operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSPIRVToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVTOLLVM_CONVERTSPIRVTOLLVMPASS_H_
