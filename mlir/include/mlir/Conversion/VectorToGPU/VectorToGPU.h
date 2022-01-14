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
class MLIRContext;
class Pass;
class FuncOp;
class RewritePatternSet;

/// Patterns to transform vector ops into a canonical form to convert to MMA
/// matrix operations.
void populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns);

/// Convert vector ops to MMA matrix operations. This will convert slice of
/// operations that can be legally converted to MMA operations. The rest of the
/// vector operations are left untouched.
void convertVectorToMMAOps(FuncOp funcOp);

/// Convert from vector to GPU ops.
std::unique_ptr<Pass> createConvertVectorToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOGPU_VECTORTOGPU_H
