//===- VectorToNVGPU.h - Convert vector to NVGPU dialect ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTONVGPU_VECTORTONVGPU_H
#define MLIR_CONVERSION_VECTORTONVGPU_VECTORTONVGPU_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTONVGPUPASS
#include "mlir/Conversion/Passes.h.inc"

/// Populate patterns that prepare vector operations for lowering to
/// `nvgpu.mma.sync` operations.
void populatePrepareVectorToNVMMAPatterns(RewritePatternSet &patterns);

/// Convert vector ops nested under `rootOp` to vector and GPU operations
/// compatible with the `nvvm.mma.sync` lowering path. This will convert a slice
/// of operations that can be legally lowered on this path while the rest of
/// the vector operations are left untouched.
LogicalResult convertVectorToNVVMCompatibleMMASync(RewriterBase &rewriter,
                                                   Operation *rootOp);
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTONVGPU_VECTORTONVGPU_H
