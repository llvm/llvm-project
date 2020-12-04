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

} // namespace mlir

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
