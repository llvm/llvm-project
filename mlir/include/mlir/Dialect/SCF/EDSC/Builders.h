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

/// Adapters for building loop nests using the builder and the location stored
/// in ScopedContext. Actual builders are in scf::buildLoopNest.
scf::LoopNest loopNestBuilder(ValueRange lbs, ValueRange ubs,
                                 ValueRange steps,
                                 function_ref<void(ValueRange)> fun = nullptr);
scf::LoopNest loopNestBuilder(Value lb, Value ub, Value step,
                                 function_ref<void(Value)> fun = nullptr);
scf::LoopNest loopNestBuilder(
    Value lb, Value ub, Value step, ValueRange iterArgInitValues,
    function_ref<scf::ValueVector(Value, ValueRange)> fun = nullptr);
scf::LoopNest loopNestBuilder(
    ValueRange lbs, ValueRange ubs, ValueRange steps,
    ValueRange iterArgInitValues,
    function_ref<scf::ValueVector(ValueRange, ValueRange)> fun = nullptr);

/// Adapters for building if conditions using the builder and the location
/// stored in ScopedContext. 'thenBody' is mandatory, 'elseBody' can be omitted
/// if the condition should not have an 'else' part.
/// When `ifOp` is specified, the scf::IfOp is captured. This is particularly
/// convenient for 0-result conditions.
ValueRange conditionBuilder(TypeRange results, Value condition,
                            function_ref<scf::ValueVector()> thenBody,
                            function_ref<scf::ValueVector()> elseBody = nullptr,
                            scf::IfOp *ifOp = nullptr);
ValueRange conditionBuilder(Value condition, function_ref<void()> thenBody,
                            function_ref<void()> elseBody = nullptr,
                            scf::IfOp *ifOp = nullptr);

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_SCF_EDSC_BUILDERS_H_
