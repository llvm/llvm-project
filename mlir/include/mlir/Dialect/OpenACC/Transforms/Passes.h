//===- Passes.h - OpenACC Passes Construction and Registration ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

namespace acc {

class OpenACCSupport;

#define GEN_PASS_DECL
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// ACCSpecializeForDevice patterns
//===----------------------------------------------------------------------===//

/// Populates all patterns for device specialization.
/// In specialized device code (such as specialized acc routine), many ACC
/// operations do not make sense because they are host-side constructs. This
/// function adds patterns to remove or transform them.
void populateACCSpecializeForDevicePatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// ACCSpecializeForHost patterns
//===----------------------------------------------------------------------===//

/// Populates patterns for converting orphan ACC operations to host.
/// All patterns check that the operation is NOT inside or associated with a
/// compute region before converting.
/// @param enableLoopConversion Whether to convert orphan acc.loop operations.
void populateACCOrphanToHostPatterns(RewritePatternSet &patterns,
                                     OpenACCSupport &accSupport,
                                     bool enableLoopConversion = true);

/// Populates all patterns for host fallback path (when `if` clause evaluates
/// to false). In this mode, ALL ACC operations should be converted or removed.
/// @param enableLoopConversion Whether to convert orphan acc.loop operations.
void populateACCHostFallbackPatterns(RewritePatternSet &patterns,
                                     OpenACCSupport &accSupport,
                                     bool enableLoopConversion = true);

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES_H
