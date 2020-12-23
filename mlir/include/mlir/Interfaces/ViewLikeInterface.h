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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
/// Auxiliary range data structure to unpack the offset, size and stride
/// operands into a list of triples. Such a list can be more convenient to
/// manipulate.
struct Range {
  Value offset;
  Value size;
  Value stride;
};

class OffsetSizeAndStrideOpInterface;
LogicalResult verify(OffsetSizeAndStrideOpInterface op);
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ViewLikeInterface.h.inc"

namespace mlir {
/// Print a list with either (1) the static integer value in `arrayAttr` if
/// `isDynamic` evaluates to false or (2) the next value otherwise.
/// This allows idiomatic printing of mixed value and integer attributes in a
/// list. E.g. `[%arg0, 7, 42, %arg42]`.
void printListOfOperandsOrIntegers(OpAsmPrinter &p, ValueRange values,
                                   ArrayAttr arrayAttr,
                                   llvm::function_ref<bool(int64_t)> isDynamic);

/// Print part of an op of the form:
/// ```
///   <optional-offset-prefix>`[` offset-list `]`
///   <optional-size-prefix>`[` size-list `]`
///   <optional-stride-prefix>[` stride-list `]`
/// ```
void printOffsetsSizesAndStrides(
    OpAsmPrinter &p, OffsetSizeAndStrideOpInterface op,
    StringRef offsetPrefix = "", StringRef sizePrefix = " ",
    StringRef stridePrefix = " ",
    ArrayRef<StringRef> elidedAttrs =
        OffsetSizeAndStrideOpInterface::getSpecialAttrNames());

/// Parse a mixed list with either (1) static integer values or (2) SSA values.
/// Fill `result` with the integer ArrayAttr named `attrName` where `dynVal`
/// encode the position of SSA values. Add the parsed SSA values to `ssa`
/// in-order.
//
/// E.g. after parsing "[%arg0, 7, 42, %arg42]":
///   1. `result` is filled with the i64 ArrayAttr "[`dynVal`, 7, 42, `dynVal`]"
///   2. `ssa` is filled with "[%arg0, %arg1]".
ParseResult
parseListOfOperandsOrIntegers(OpAsmParser &parser, OperationState &result,
                              StringRef attrName, int64_t dynVal,
                              SmallVectorImpl<OpAsmParser::OperandType> &ssa);

/// Parse trailing part of an op of the form:
/// ```
///   <optional-offset-prefix>`[` offset-list `]`
///   <optional-size-prefix>`[` size-list `]`
///   <optional-stride-prefix>[` stride-list `]`
/// ```
/// Each entry in the offset, size and stride list either resolves to an integer
/// constant or an operand of index type.
/// Constants are added to the `result` as named integer array attributes with
/// name `OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName()` (resp.
/// `getStaticSizesAttrName()`, `getStaticStridesAttrName()`).
///
/// Append the number of offset, size and stride operands to `segmentSizes`
/// before adding it to `result` as the named attribute:
/// `OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()`.
///
/// Offset, size and stride operands resolution occurs after `preResolutionFn`
/// to give a chance to leading operands to resolve first, after parsing the
/// types.
ParseResult parseOffsetsSizesAndStrides(
    OpAsmParser &parser, OperationState &result, ArrayRef<int> segmentSizes,
    llvm::function_ref<ParseResult(OpAsmParser &, OperationState &)>
        preResolutionFn = nullptr,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalOffsetPrefix =
        nullptr,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalSizePrefix =
        nullptr,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalStridePrefix =
        nullptr);
/// `preResolutionFn`-less version of `parseOffsetsSizesAndStrides`.
ParseResult parseOffsetsSizesAndStrides(
    OpAsmParser &parser, OperationState &result, ArrayRef<int> segmentSizes,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalOffsetPrefix =
        nullptr,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalSizePrefix =
        nullptr,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalStridePrefix =
        nullptr);

/// Verify that a the `values` has as many elements as the number of entries in
/// `attr` for which `isDynamic` evaluates to true.
LogicalResult verifyListOfOperandsOrIntegers(
    Operation *op, StringRef name, unsigned expectedNumElements, ArrayAttr attr,
    ValueRange values, llvm::function_ref<bool(int64_t)> isDynamic);

} // namespace mlir

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
