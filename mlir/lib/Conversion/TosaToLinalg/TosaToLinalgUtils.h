//===- TosaToLinalgUtils.h - Shared TOSA to Linalg lowering utilities
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_TOSATOLINALG_TOSATOLINALGUTILS_H
#define MLIR_LIB_CONVERSION_TOSATOLINALG_TOSATOLINALGUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
class PatternRewriter;

namespace tosa {
struct MatMulTOperands {
  Value a;
  Value b;
  Value init;
};

/// Broadcast the batch dimensions of the ranked operands and create a
/// zero-filled result tensor for either MATMUL_T lowering pattern.
MatMulTOperands prepareMatMulOperands(PatternRewriter &rewriter, Location loc,
                                      Value a, Value b,
                                      RankedTensorType outputType,
                                      bool transposeB);
} // namespace tosa
} // namespace mlir

#endif // MLIR_LIB_CONVERSION_TOSATOLINALG_TOSATOLINALGUTILS_H
