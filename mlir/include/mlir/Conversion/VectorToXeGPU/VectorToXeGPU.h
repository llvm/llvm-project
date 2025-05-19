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

/// Create a pass to convert ops from vector to XeGPU.
std::unique_ptr<Pass> createConvertVectorToXeGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOXEGPU_VECTORTOXEGPU_H
