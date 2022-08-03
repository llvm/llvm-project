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

/// Auxiliary range data structure to unpack the offset, size and stride
/// operands into a list of triples. Such a list can be more convenient to
/// manipulate.
struct Range {
  OpFoldResult offset;
  OpFoldResult size;
  OpFoldResult stride;
};

/// Return a vector of OpFoldResults given the special value
/// that indicates whether of the value is dynamic or not.
SmallVector<OpFoldResult, 4> getMixedValues(ArrayAttr staticValues,
                                            ValueRange dynamicValues,
                                            int64_t dynamicValueIndicator);

/// Return a vector of all the static and dynamic offsets/strides.
SmallVector<OpFoldResult, 4> getMixedStridesOrOffsets(ArrayAttr staticValues,
                                                      ValueRange dynamicValues);

/// Return a vector of all the static and dynamic sizes.
SmallVector<OpFoldResult, 4> getMixedSizes(ArrayAttr staticValues,
                                           ValueRange dynamicValues);

/// Decompose a vector of mixed static or dynamic values into the corresponding
/// pair of arrays. This is the inverse function of `getMixedValues`.
std::pair<ArrayAttr, SmallVector<Value>>
decomposeMixedValues(Builder &b,
                     const SmallVectorImpl<OpFoldResult> &mixedValues,
                     const int64_t dynamicValueIndicator);

/// Decompose a vector of mixed static and dynamic strides/offsets into the
/// corresponding pair of arrays. This is the inverse function of
/// `getMixedStridesOrOffsets`.
std::pair<ArrayAttr, SmallVector<Value>> decomposeMixedStridesOrOffsets(
    OpBuilder &b, const SmallVectorImpl<OpFoldResult> &mixedValues);

/// Decompose a vector of mixed static or dynamic strides/offsets into the
/// corresponding pair of arrays. This is the inverse function of
/// `getMixedSizes`.
std::pair<ArrayAttr, SmallVector<Value>>
decomposeMixedSizes(OpBuilder &b,
                    const SmallVectorImpl<OpFoldResult> &mixedValues);

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
///   custom<OperandsOrIntegersOffsetsOrStridesList>($values, $integers)
///
/// where `values` is of ODS type `Variadic<Index>` and `integers` is of ODS
/// type `I64ArrayAttr`.  for use in in assemblyFormat. Prints a list with
/// either (1) the static integer value in `integers` if the value is
/// ShapedType::kDynamicStrideOrOffset or (2) the next value otherwise.  This
/// allows idiomatic printing of mixed value and integer attributes in a
/// list. E.g. `[%arg0, 7, 42, %arg42]`.
void printOperandsOrIntegersOffsetsOrStridesList(OpAsmPrinter &printer,
                                                 Operation *op,
                                                 OperandRange values,
                                                 ArrayAttr integers);

/// Printer hook for custom directive in assemblyFormat.
///
///   custom<OperandsOrIntegersSizesList>($values, $integers)
///
/// where `values` is of ODS type `Variadic<Index>` and `integers` is of ODS
/// type `I64ArrayAttr`.  for use in in assemblyFormat. Prints a list with
/// either (1) the static integer value in `integers` if the value is
/// ShapedType::kDynamicSize or (2) the next value otherwise.  This
/// allows idiomatic printing of mixed value and integer attributes in a
/// list. E.g. `[%arg0, 7, 42, %arg42]`.
void printOperandsOrIntegersSizesList(OpAsmPrinter &printer, Operation *op,
                                      OperandRange values, ArrayAttr integers);

/// Pasrer hook for custom directive in assemblyFormat.
///
///   custom<OperandsOrIntegersOffsetsOrStridesList>($values, $integers)
///
/// where `values` is of ODS type `Variadic<Index>` and `integers` is of ODS
/// type `I64ArrayAttr`.  for use in in assemblyFormat. Parse a mixed list with
/// either (1) static integer values or (2) SSA values.  Fill `integers` with
/// the integer ArrayAttr, where ShapedType::kDynamicStrideOrOffset encodes the
/// position of SSA values. Add the parsed SSA values to `values` in-order.
//
/// E.g. after parsing "[%arg0, 7, 42, %arg42]":
///   1. `result` is filled with the i64 ArrayAttr "[`dynVal`, 7, 42, `dynVal`]"
///   2. `ssa` is filled with "[%arg0, %arg1]".
ParseResult parseOperandsOrIntegersOffsetsOrStridesList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    ArrayAttr &integers);

/// Pasrer hook for custom directive in assemblyFormat.
///
///   custom<OperandsOrIntegersSizesList>($values, $integers)
///
/// where `values` is of ODS type `Variadic<Index>` and `integers` is of ODS
/// type `I64ArrayAttr`.  for use in in assemblyFormat. Parse a mixed list with
/// either (1) static integer values or (2) SSA values.  Fill `integers` with
/// the integer ArrayAttr, where ShapedType::kDynamicSize encodes the
/// position of SSA values. Add the parsed SSA values to `values` in-order.
//
/// E.g. after parsing "[%arg0, 7, 42, %arg42]":
///   1. `result` is filled with the i64 ArrayAttr "[`dynVal`, 7, 42, `dynVal`]"
///   2. `ssa` is filled with "[%arg0, %arg1]".
ParseResult parseOperandsOrIntegersSizesList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    ArrayAttr &integers);

/// Verify that a the `values` has as many elements as the number of entries in
/// `attr` for which `isDynamic` evaluates to true.
LogicalResult verifyListOfOperandsOrIntegers(
    Operation *op, StringRef name, unsigned expectedNumElements, ArrayAttr attr,
    ValueRange values, llvm::function_ref<bool(int64_t)> isDynamic);

} // namespace mlir

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
