//===- FuncConversions.h - Patterns for converting std.funcs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files contains patterns for converting standard functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_TRANSFORMS_FUNCCONVERSIONS_H_
#define MLIR_DIALECT_STANDARDOPS_TRANSFORMS_FUNCCONVERSIONS_H_

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

/// Add a pattern to the given pattern list to rewrite branch operations and
/// `return` to use operands that have been legalized by the conversion
/// framework. This can only be done if the branch operation implements the
/// BranchOpInterface. Only needed for partial conversions.
void populateBranchOpInterfaceAndReturnOpTypeConversionPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx,
    TypeConverter &converter);
} // end namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_TRANSFORMS_FUNCCONVERSIONS_H_
