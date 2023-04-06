//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_ARITH_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class DataFlowSolver;

namespace arith {

#define GEN_PASS_DECL
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
#define GEN_PASS_DECL_ARITHINTRANGEOPTS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"

class WideIntEmulationConverter;

/// Create a pass to bufferize Arith ops.
std::unique_ptr<Pass> createArithBufferizePass();

/// Create a pass to bufferize arith.constant ops.
std::unique_ptr<Pass> createConstantBufferizePass(uint64_t alignment = 0);

/// Adds patterns to emulate wide Arith and Function ops over integer
/// types into supported ones. This is done by splitting original power-of-two
/// i2N integer types into two iN halves.
void populateArithWideIntEmulationPatterns(
    WideIntEmulationConverter &typeConverter, RewritePatternSet &patterns);

/// Add patterns to expand Arith ceil/floor division ops.
void populateCeilFloorDivExpandOpsPatterns(RewritePatternSet &patterns);

/// Add patterns to expand Arith bf16 patterns to lower level bitcasts/shifts.
void populateExpandBFloat16Patterns(RewritePatternSet &patterns);

/// Add patterns to expand Arith ops.
void populateArithExpandOpsPatterns(RewritePatternSet &patterns);

/// Create a pass to legalize Arith ops.
std::unique_ptr<Pass> createArithExpandOpsPass();

/// Create a pass to replace signed ops with unsigned ones where they are proven
/// equivalent.
std::unique_ptr<Pass> createArithUnsignedWhenEquivalentPass();

/// Add patterns for int range based optimizations.
void populateIntRangeOptimizationsPatterns(RewritePatternSet &patterns,
                                           DataFlowSolver &solver);

/// Create a pass which do optimizations based on integer range analysis.
std::unique_ptr<Pass> createIntRangeOptimizationsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"

} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_TRANSFORMS_PASSES_H_
