//===- ConvertStandardToLLVMPass.h - Pass entrypoint ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_

#include <memory>

namespace mlir {
class LowerToLLVMOptions;
class ModuleOp;
template <typename T>
class OperationPass;

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// stdlib malloc/free is used by default for allocating memrefs allocated with
/// memref.alloc, while LLVM's alloca is used for those allocated with
/// memref.alloca.
std::unique_ptr<OperationPass<ModuleOp>> createLowerToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLowerToLLVMPass(const LowerToLLVMOptions &options);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
