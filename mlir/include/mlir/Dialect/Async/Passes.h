//===- Passes.h - Async pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_PASSES_H_
#define MLIR_DIALECT_ASYNC_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createAsyncParallelForPass();

std::unique_ptr<OperationPass<FuncOp>>
createAsyncParallelForPass(int numWorkerThreads);

std::unique_ptr<OperationPass<FuncOp>> createAsyncRefCountingPass();

std::unique_ptr<OperationPass<FuncOp>> createAsyncRefCountingOptimizationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Async/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_ASYNC_PASSES_H_
