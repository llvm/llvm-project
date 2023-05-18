//===- Syntax.h - Custom syntax for Linalg transform ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMOPS_SYNTAX_H
#define MLIR_DIALECT_LINALG_TRANSFORMOPS_SYNTAX_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class ParseResult;
class OpAsmParser;
class OpAsmPrinter;
class Type;
class TypeRange;
class Operation;

/// Parses a single non-function type or a function type with at least one
/// argument. This allows for the following syntax:
///
///   - type: just the argument type;
///   - `(` type `)` `->` type: one argument and one result type;
///   - `(` type `)` `->` `(` comma-separated-type-list `)`: one argument and
///     multiple result types.
///
/// Unlike FunctionType, this allows and requires one to omit the parens around
/// the argument type in absence of result types, and does not accept the
/// trailing `-> ()` construct, which makes the syntax nicer for operations.
ParseResult parseSemiFunctionType(OpAsmParser &parser, Type &argumentType,
                                  Type &resultType);
ParseResult parseSemiFunctionType(OpAsmParser &parser, Type &argumentType,
                                  SmallVectorImpl<Type> &resultTypes);

/// Prints argument and result types in a syntax similar to that of FunctionType
/// but allowing and requiring one to omit the parens around the argument type
/// in absence of result types, and without the trailing `-> ()`.
void printSemiFunctionType(OpAsmPrinter &printer, Operation *op,
                           Type argumentType, TypeRange resultType);
void printSemiFunctionType(OpAsmPrinter &printer, Operation *op,
                           Type argumentType, Type resultType);
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_SYNTAX_H
