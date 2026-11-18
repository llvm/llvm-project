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
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOXEGPU
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the vector to XeGPU ops.
void populateVectorToXeGPUConversionPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns that bring vector ops into a form the conversion
/// patterns lower directly: gather/scatter ops flattened to 1-D operands are
/// un-flattened back to N-D, and transfer_read broadcast permutation maps are
/// replaced by a transfer of the data plus the vector ops that expand it.
/// Apply these before the conversion patterns.
void populateVectorToXeGPULegalizationPatterns(RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H
