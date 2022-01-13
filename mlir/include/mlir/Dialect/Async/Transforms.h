//===- Transforms.h - Async dialect transformation utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines transformations on Async operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_TRANSFORMS_H_
#define MLIR_DIALECT_ASYNC_TRANSFORMS_H_

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace mlir {
namespace async {

/// Emit the IR to compute the minimum number of iterations of scf.parallel body
/// that would be viable for a single parallel task. Allows the user to avoid
/// incurring the overheads of spawning costly parallel tasks in absence of
/// sufficient amount of parallelizable work.
///
/// Must return an index type.
using AsyncMinTaskSizeComputationFunction =
    std::function<Value(ImplicitLocOpBuilder, scf::ParallelOp)>;

/// Add a pattern to the given pattern list to lower scf.parallel to async
/// operations.
void populateAsyncParallelForPatterns(
    RewritePatternSet &patterns, bool asyncDispatch, int32_t numWorkerThreads,
    const AsyncMinTaskSizeComputationFunction &computeMinTaskSize);

} // namespace async
} // namespace mlir

#endif // MLIR_DIALECT_ASYNC_TRANSFORMS_H_
