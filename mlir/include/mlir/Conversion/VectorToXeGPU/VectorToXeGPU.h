//===- VectorToXeGPU.h - Convert vector to XeGPU dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H
#define MLIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class DialectRegistry;
class Operation;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOXEGPU
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the vector to XeGPU ops.
void populateVectorToXeGPUConversionPatterns(RewritePatternSet &patterns);

namespace xegpu {

/// Attaches the Mem2Reg interface external models used by whole-buffer
/// promotion to `memref.alloc`, `vector.transfer_read` and
/// `vector.transfer_write`. Intended to be called from a pass's
/// `getDependentDialects` so the models are visible only within pipelines that
/// contain that pass.
void registerWholeBufferPromotionExternalModels(DialectRegistry &registry);

/// Promotes every `memref.alloc` under `scopeOp` that is only ever accessed as
/// a whole buffer through `vector.transfer_read`/`vector.transfer_write` into a
/// single vector SSA value, reusing the upstream Mem2Reg driver (which threads
/// the value through `scf.for` as an iter_arg/result when the accesses live in
/// a loop). Allocations larger than `maxPromotedBytes`, or with any other use,
/// are left untouched. Requires the external models above to be registered on
/// `scopeOp`'s context.
void promoteWholeBufferAllocs(Operation *scopeOp, uint64_t maxPromotedBytes);

} // namespace xegpu
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H
