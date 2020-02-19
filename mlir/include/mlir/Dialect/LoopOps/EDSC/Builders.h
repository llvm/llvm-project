//===- Builders.h - MLIR Declarative Builder Classes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LOOPOPS_EDSC_BUILDERS_H_
#define MLIR_DIALECT_LOOPOPS_EDSC_BUILDERS_H_

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace edsc {

/// Constructs a new loop::ParallelOp and captures the associated induction
/// variables. An array of ValueHandle pointers is passed as the first
/// argument and is the *only* way to capture loop induction variables.
LoopBuilder makeParallelLoopBuilder(ArrayRef<ValueHandle *> ivs,
                                    ArrayRef<ValueHandle> lbHandles,
                                    ArrayRef<ValueHandle> ubHandles,
                                    ArrayRef<ValueHandle> steps);
/// Constructs a new loop::ForOp and captures the associated induction
/// variable. A ValueHandle pointer is passed as the first argument and is the
/// *only* way to capture the loop induction variable.
LoopBuilder makeLoopBuilder(ValueHandle *iv, ValueHandle lbHandle,
                            ValueHandle ubHandle, ValueHandle stepHandle);

/// Helper class to sugar building loop.parallel loop nests from lower/upper
/// bounds and step sizes.
class ParallelLoopNestBuilder {
public:
  ParallelLoopNestBuilder(ArrayRef<ValueHandle *> ivs,
                          ArrayRef<ValueHandle> lbs, ArrayRef<ValueHandle> ubs,
                          ArrayRef<ValueHandle> steps);

  void operator()(function_ref<void(void)> fun = nullptr);

private:
  SmallVector<LoopBuilder, 4> loops;
};

/// Helper class to sugar building loop.for loop nests from ranges.
/// This is similar to edsc::AffineLoopNestBuilder except it operates on
/// loop.for.
class LoopNestBuilder {
public:
  LoopNestBuilder(ArrayRef<edsc::ValueHandle *> ivs, ArrayRef<ValueHandle> lbs,
                  ArrayRef<ValueHandle> ubs, ArrayRef<ValueHandle> steps);
  void operator()(std::function<void(void)> fun = nullptr);

private:
  SmallVector<LoopBuilder, 4> loops;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LOOPOPS_EDSC_BUILDERS_H_
