//===- TensorToLinalg.h - Tensor to Linalg Patterns -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H
#define MLIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

/// Appends to a pattern list additional patterns for translating tensor ops
/// to Linalg ops.
void populateTensorToLinalgPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H
