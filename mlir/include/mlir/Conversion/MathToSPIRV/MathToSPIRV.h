//===- MathToSPIRV.h - Math to SPIR-V Patterns ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Math dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATHTOSPIRV_MATHTOSPIRV_H
#define MLIR_CONVERSION_MATHTOSPIRV_MATHTOSPIRV_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating Math ops
/// to SPIR-V ops.
void populateMathToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                 RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOSPIRV_MATHTOSPIRV_H
