//===- StandardToStandard.h - Std intra-dialect conversion  -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files contains patterns for lowering within the Standard dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSTANDARD_STANDARDTOSTANDARD_H_
#define MLIR_CONVERSION_STANDARDTOSTANDARD_STANDARDTOSTANDARD_H_

namespace mlir {

// Forward declarations.
class MLIRContext;
class OwningRewritePatternList;
class TypeConverter;

/// Add a pattern to the given pattern list to convert the operand and result
/// types of a CallOp with the given type converter.
void populateCallOpTypeConversionPattern(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx,
                                         TypeConverter &converter);

} // end namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSTANDARD_STANDARDTOSTANDARD_H_
