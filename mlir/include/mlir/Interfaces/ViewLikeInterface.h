//===- ViewLikeInterface.h - View-like operations interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for view-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
#define MLIR_INTERFACES_VIEWLIKEINTERFACE_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {

class OffsetSizeAndStrideOpInterface;

namespace detail {

LogicalResult verifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op);

bool sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface a, OffsetSizeAndStrideOpInterface b,
    llvm::function_ref<bool(OpFoldResult, OpFoldResult)> cmp);

} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ViewLikeInterface.h.inc"

namespace mlir {

/// Printer hook for custom directive in assemblyFormat.
///
///   custom<DynamicIndexList>($values, $integers)
///
/// where `values` is of ODS type `Variadic<Index>` and `integers` is of ODS
/// type `I64ArrayAttr`. Prints a list with either (1) the static integer value
/// in `integers` is `dynVal` or (2) the next value otherwise. This allows
/// idiomatic printing of mixed value and integer attributes in a list. E.g.
/// `[%arg0, 7, 42, %arg42]`.
void printDynamicIndexList(OpAsmPrinter &printer, Operation *op,
                           OperandRange values, ArrayRef<int64_t> integers);

/// Pasrer hook for custom directive in assemblyFormat.
///
///   custom<DynamicIndexList>($values, $integers)
///
/// where `values` is of ODS type `Variadic<Index>` and `integers` is of ODS
/// type `I64ArrayAttr`. Parse a mixed list with either (1) static integer
/// values or (2) SSA values. Fill `integers` with the integer ArrayAttr, where
/// `dynVal` encodes the position of SSA values. Add the parsed SSA values
/// to `values` in-order.
//
/// E.g. after parsing "[%arg0, 7, 42, %arg42]":
///   1. `result` is filled with the i64 ArrayAttr "[`dynVal`, 7, 42, `dynVal`]"
///   2. `ssa` is filled with "[%arg0, %arg1]".
ParseResult
parseDynamicIndexList(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
                      DenseI64ArrayAttr &integers);

/// Verify that a the `values` has as many elements as the number of entries in
/// `attr` for which `isDynamic` evaluates to true.
LogicalResult verifyListOfOperandsOrIntegers(Operation *op, StringRef name,
                                             unsigned expectedNumElements,
                                             ArrayRef<int64_t> attr,
                                             ValueRange values);

} // namespace mlir

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
