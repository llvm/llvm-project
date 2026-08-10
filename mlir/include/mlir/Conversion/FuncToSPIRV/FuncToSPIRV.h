//===- FuncToSPIRV.h - Func to SPIR-V Patterns ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Func dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FUNCTOSPIRV_FUNCTOSPIRV_H
#define MLIR_CONVERSION_FUNCTOSPIRV_FUNCTOSPIRV_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating Func ops
/// to SPIR-V ops. Also adds the patterns to legalize ops not directly
/// translated to SPIR-V dialect.
void populateFuncToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                 RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_FUNCTOSPIRV_FUNCTOSPIRV_H
