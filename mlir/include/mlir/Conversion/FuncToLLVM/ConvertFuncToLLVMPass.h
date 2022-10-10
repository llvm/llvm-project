//===- ConvertFuncToLLVMPass.h - Pass entrypoint ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_
#define MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_

#include <memory>
#include <string>

namespace mlir {
class LowerToLLVMOptions;
class ModuleOp;
template <typename T>
class OperationPass;
class Pass;

#define GEN_PASS_DECL_CONVERTFUNCTOLLVM
#include "mlir/Conversion/Passes.h.inc"

/// Creates a pass to convert the Func dialect into the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertFuncToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncToLLVMPass(const LowerToLLVMOptions &options);

} // namespace mlir

#endif // MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_
