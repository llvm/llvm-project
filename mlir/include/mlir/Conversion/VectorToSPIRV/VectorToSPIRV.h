//=- VectorToSPIRV.h - Vector to SPIR-V Patterns ------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Vector dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INCLUDE_MLIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRV_H
#define MLIR_INCLUDE_MLIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRV_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating Vector Ops to
/// SPIR-V ops.
void populateVectorToSPIRVPatterns(MLIRContext *context,
                                   SPIRVTypeConverter &typeConverter,
                                   OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_INCLUDE_MLIR_CONVERSION_VECTORTOSPIRV_VECTORTOSPIRV_H
