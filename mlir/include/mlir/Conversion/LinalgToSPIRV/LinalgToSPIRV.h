//===- LinalgToSPIRV.h - Linalg to SPIR-V Patterns --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Linalg dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOSPIRV_LINALGTOSPIRV_H
#define MLIR_CONVERSION_LINALGTOSPIRV_LINALGTOSPIRV_H

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating Linalg ops to
/// SPIR-V ops.
void populateLinalgToSPIRVPatterns(MLIRContext *context,
                                   SPIRVTypeConverter &typeConverter,
                                   OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOSPIRV_LINALGTOSPIRV_H
