//===- VectorToGPU.h - Convert vector to GPU dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOGPU_VECTORTOGPU_H
#define MLIR_CONVERSION_VECTORTOGPU_VECTORTOGPU_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class LogicalResult;
class MLIRContext;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOGPU
#include "mlir/Conversion/Passes.h.inc"

/// Patterns to transform vector ops into a canonical form to convert to MMA
/// matrix operations. If `useNvGpu` is true, then the patterns will populated
/// will prepare for conversion to `nvgpu` mma operations rather than the `gpu`
/// dialect WMMA operations.
void populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns,
                                        bool useNvGpu = false);

/// Convert vector ops to MMA matrix operations nested under `rootOp`. This will
/// convert slice of operations that can be legally converted to MMA operations.
/// The rest of the vector operations are left untouched.
LogicalResult convertVectorToMMAOps(RewriterBase &rewriter, Operation *rootOp);

/// Convert vector ops ops nested under `rootOp` to vector and GPU operaitons
/// compatible with the `nvvm.mma.sync` lowering path. This will convert a slice
/// of operations that can be legally lowered on this path while the rest of
/// the vector operations are left untouched.
LogicalResult convertVectorToNVVMCompatibleMMASync(RewriterBase &rewriter,
                                                   Operation *rootOp);

/// Convert from vector to GPU ops.
std::unique_ptr<Pass> createConvertVectorToGPUPass(bool useNvGpu = false);

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOGPU_VECTORTOGPU_H
