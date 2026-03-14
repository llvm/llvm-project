//===- OptUtils.h - MLIR Execution Engine opt pass utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_OPTUTILS_H
#define MLIR_EXECUTIONENGINE_OPTUTILS_H

#include <functional>
#include <string>

namespace llvm {
class Module;
class Error;
class TargetMachine;
} // namespace llvm

namespace mlir {

/// Create a module transformer function for MLIR ExecutionEngine that runs
/// LLVM IR passes corresponding to the given speed and size optimization
/// levels (e.g. -O2 or -Os). If not null, `targetMachine` is used to
/// initialize passes that provide target-specific information to the LLVM
/// optimizer. `targetMachine` must outlive the returned std::function.
std::function<llvm::Error(llvm::Module *)>
makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                          llvm::TargetMachine *targetMachine);

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_OPTUTILS_H
