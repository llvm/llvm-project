//===- AsyncToLLVM.h - Convert Async to LLVM dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ASYNCTOLLVM_ASYNCTOLLVM_H
#define MLIR_CONVERSION_ASYNCTOLLVM_ASYNCTOLLVM_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T>
class OperationPass;

/// Create a pass to convert Async operations to the LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertAsyncToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_ASYNCTOLLVM_ASYNCTOLLVM_H
