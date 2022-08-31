//===- SPIRVToLLVMPass.h - SPIR-V to LLVM Passes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert SPIR-V dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVMPASS_H
#define MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVMPASS_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

#define GEN_PASS_DECL_LOWERHOSTCODETOLLVM
#define GEN_PASS_DECL_CONVERTSPIRVTOLLVM
#include "mlir/Conversion/Passes.h.inc"

/// Creates a pass to emulate `gpu.launch_func` call in LLVM dialect and lower
/// the host module code to LLVM.
///
/// This transformation creates a sequence of global variables that are later
/// linked to the variables in the kernel module, and a series of copies to/from
/// them to emulate the memory transfer from the host or to the device sides. It
/// also converts the remaining Arithmetic, Func, and MemRef dialects into LLVM
/// dialect, emitting C wrappers.
std::unique_ptr<OperationPass<ModuleOp>> createLowerHostCodeToLLVMPass();

/// Creates a pass to convert SPIR-V operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSPIRVToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVMPASS_H
