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

#ifndef MLIR_DIALECT_SCF_EDSC_BUILDERS_H_
#define MLIR_DIALECT_SCF_EDSC_BUILDERS_H_

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace edsc {

/// Constructs a new scf::ParallelOp and captures the associated induction
/// variables. An array of Value pointers is passed as the first
/// argument and is the *only* way to capture loop induction variables.
LoopBuilder makeParallelLoopBuilder(MutableArrayRef<Value> ivs,
                                    ArrayRef<Value> lbs, ArrayRef<Value> ubs,
                                    ArrayRef<Value> steps);
/// Constructs a new scf::ForOp and captures the associated induction
/// variable. A Value pointer is passed as the first argument and is the
/// *only* way to capture the loop induction variable.
LoopBuilder makeLoopBuilder(Value *iv, Value lb, Value ub, Value step,
                            MutableArrayRef<Value> iterArgsHandles,
                            ValueRange iterArgsInitValues);
LoopBuilder makeLoopBuilder(Value *iv, Value lb, Value ub, Value step,
                            MutableArrayRef<Value> iterArgsHandles,
                            ValueRange iterArgsInitValues);
inline LoopBuilder makeLoopBuilder(Value *iv, Value lb, Value ub, Value step) {
  return makeLoopBuilder(iv, lb, ub, step, MutableArrayRef<Value>{}, {});
}

/// Helper class to sugar building scf.parallel loop nests from lower/upper
/// bounds and step sizes.
class ParallelLoopNestBuilder {
public:
  ParallelLoopNestBuilder(MutableArrayRef<Value> ivs, ArrayRef<Value> lbs,
                          ArrayRef<Value> ubs, ArrayRef<Value> steps);

  void operator()(function_ref<void(void)> fun = nullptr);

private:
  SmallVector<LoopBuilder, 4> loops;
};

/// Helper class to sugar building scf.for loop nests from ranges.
/// This is similar to edsc::AffineLoopNestBuilder except it operates on
/// scf.for.
class LoopNestBuilder {
public:
  LoopNestBuilder(Value *iv, Value lb, Value ub, Value step);
  LoopNestBuilder(Value *iv, Value lb, Value ub, Value step,
                  MutableArrayRef<Value> iterArgsHandles,
                  ValueRange iterArgsInitValues);
  LoopNestBuilder(MutableArrayRef<Value> ivs, ArrayRef<Value> lbs,
                  ArrayRef<Value> ubs, ArrayRef<Value> steps);
  Operation::result_range operator()(std::function<void(void)> fun = nullptr);

private:
  SmallVector<LoopBuilder, 4> loops;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_SCF_EDSC_BUILDERS_H_
