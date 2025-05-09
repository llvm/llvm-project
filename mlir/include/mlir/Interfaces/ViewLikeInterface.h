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
#include "mlir/IR/PatternMatch.h"

namespace mlir {

class OffsetSizeAndStrideOpInterface;

namespace detail {

LogicalResult verifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op);

bool sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface a, OffsetSizeAndStrideOpInterface b,
    llvm::function_ref<bool(OpFoldResult, OpFoldResult)> cmp);

/// Helper method to compute the number of dynamic entries of `staticVals`,
/// up to `idx`.
unsigned getNumDynamicEntriesUpToIdx(ArrayRef<int64_t> staticVals,
                                     unsigned idx);

} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ViewLikeInterface.h.inc"

namespace mlir {

/// Result for slice bounds verification;
struct SliceBoundsVerificationResult {
  /// If set to "true", the slice bounds verification was successful.
  bool isValid;
  /// An error message that can be printed during op verification.
  std::string errorMessage;
};

/// Verify that the offsets/sizes/strides-style access into the given shape
/// is in-bounds. Only static values are verified. If `generateErrorMessage`
/// is set to "true", an error message is produced that can be printed by the
///  op verifier.
SliceBoundsVerificationResult
verifyInBoundsSlice(ArrayRef<int64_t> shape, ArrayRef<int64_t> staticOffsets,
                    ArrayRef<int64_t> staticSizes,
                    ArrayRef<int64_t> staticStrides,
                    bool generateErrorMessage = false);
SliceBoundsVerificationResult verifyInBoundsSlice(
    ArrayRef<int64_t> shape, ArrayRef<OpFoldResult> mixedOffsets,
    ArrayRef<OpFoldResult> mixedSizes, ArrayRef<OpFoldResult> mixedStrides,
    bool generateErrorMessage = false);

/// Pattern to rewrite dynamic offsets/sizes/strides of view/slice-like ops as
/// constant arguments. This pattern assumes that the op has a suitable builder
/// that takes a result type, a "source" operand and mixed offsets, sizes and
/// strides.
///
/// `OpType` is the type of op to which this pattern is applied. `ResultTypeFn`
/// returns the new result type of the op, based on the new offsets, sizes and
/// strides. `CastOpFunc` is used to generate a cast op if the result type of
/// the op has changed.
template <typename OpType, typename ResultTypeFn, typename CastOpFunc>
class OpWithOffsetSizesAndStridesConstantArgumentFolder final
    : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(op.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());

    // No constant operands were folded, just return;
    if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) &&
        failed(foldDynamicIndexList(mixedSizes, /*onlyNonNegative=*/true)) &&
        failed(foldDynamicIndexList(mixedStrides)))
      return failure();

    // Pattern does not apply if the produced op would not verify.
    SliceBoundsVerificationResult sliceResult = verifyInBoundsSlice(
        cast<ShapedType>(op.getSource().getType()).getShape(), mixedOffsets,
        mixedSizes, mixedStrides);
    if (!sliceResult.isValid)
      return failure();

    // Compute the new result type.
    auto resultType =
        ResultTypeFn()(op, mixedOffsets, mixedSizes, mixedStrides);
    if (!resultType)
      return failure();

    // Create the new op in canonical form.
    auto newOp =
        rewriter.create<OpType>(op.getLoc(), resultType, op.getSource(),
                                mixedOffsets, mixedSizes, mixedStrides);
    CastOpFunc()(rewriter, op, newOp);

    return success();
  }
};

/// Printer hooks for custom directive in assemblyFormat.
///
///   custom<DynamicIndexList>($values, $integers)
///   custom<DynamicIndexList>($values, $integers, type($values))
///
/// where `values` is of ODS type `Variadic<*>` and `integers` is of ODS type
/// `I64ArrayAttr`. Print a list where each element is either:
///    1. the static integer value in `integers`, if it's not `kDynamic` or,
///    2. the next value in `values`, otherwise.
///
/// If `valueTypes` is provided, the corresponding type of each dynamic value is
/// printed. Otherwise, the type is not printed. Each type must match the type
/// of the corresponding value in `values`. `valueTypes` is redundant for
/// printing as we can retrieve the types from the actual `values`. However,
/// `valueTypes` is needed for parsing and we must keep the API symmetric for
/// parsing and printing. The type for integer elements is `i64` by default and
/// never printed.
///
/// Integer indices can also be scalable in the context of scalable vectors,
/// denoted by square brackets (e.g., "[2, [4], 8]"). For each value in
/// `integers`, the corresponding `bool` in `scalableFlags` encodes whether it's
/// a scalable index. If `scalableFlags` is empty then assume that all indices
/// are non-scalable.
///
/// Examples:
///
///   * Input: `integers = [kDynamic, 7, 42, kDynamic]`,
///            `values = [%arg0, %arg42]` and
///            `valueTypes = [index, index]`
///     prints:
///       `[%arg0 : index, 7, 42, %arg42 : i32]`
///
///   * Input: `integers = [kDynamic, 7, 42, kDynamic]`,
///            `values = [%arg0, %arg42]` and
///            `valueTypes = []`
///     prints:
///       `[%arg0, 7, 42, %arg42]`
///
///   * Input: `integers = [2, 4, 8]`,
///            `values = []` and
///            `scalableFlags = [false, true, false]`
///     prints:
///       `[2, [4], 8]`
///
void printDynamicIndexList(
    OpAsmPrinter &printer, Operation *op, OperandRange values,
    ArrayRef<int64_t> integers, ArrayRef<bool> scalableFlags,
    TypeRange valueTypes = TypeRange(),
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square);
inline void printDynamicIndexList(
    OpAsmPrinter &printer, Operation *op, OperandRange values,
    ArrayRef<int64_t> integers, TypeRange valueTypes = TypeRange(),
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {
  return printDynamicIndexList(printer, op, values, integers,
                               /*scalableFlags=*/{}, valueTypes, delimiter);
}

/// Parser hooks for custom directive in assemblyFormat.
///
///   custom<DynamicIndexList>($values, $integers)
///   custom<DynamicIndexList>($values, $integers, type($values))
///
/// where `values` is of ODS type `Variadic<*>` and `integers` is of ODS
/// type `I64ArrayAttr`. Parse a mixed list where each element is either a
/// static integer or an SSA value. Fill `integers` with the integer ArrayAttr,
/// where `kDynamic` encodes the position of SSA values. Add the parsed SSA
/// values to `values` in-order.
///
/// If `valueTypes` is provided, fill it with the types corresponding to each
/// value in `values`. Otherwise, the caller must handle the types and parsing
/// will fail if the type of the value is found (e.g., `[%arg0 : index, 3, %arg1
/// : index]`).
///
/// Integer indices can also be scalable in the context of scalable vectors,
/// denoted by square brackets (e.g., "[2, [4], 8]"). For each value in
/// `integers`, the corresponding `bool` in `scalableFlags` encodes whether it's
/// a scalable index.
///
/// Examples:
///
///   * After parsing "[%arg0 : index, 7, 42, %arg42 : i32]":
///       1. `result` is filled with `[kDynamic, 7, 42, kDynamic]`
///       2. `values` is filled with "[%arg0, %arg1]".
///       3. `scalableFlags` is filled with `[false, true, false]`.
///
///   * After parsing `[2, [4], 8]`:
///       1. `result` is filled with `[2, 4, 8]`
///       2. `values` is empty.
///       3. `scalableFlags` is filled with `[false, true, false]`.
///
ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, DenseBoolArrayAttr &scalableFlags,
    SmallVectorImpl<Type> *valueTypes = nullptr,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square);
inline ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, SmallVectorImpl<Type> *valueTypes = nullptr,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {
  DenseBoolArrayAttr scalableFlags;
  return parseDynamicIndexList(parser, values, integers, scalableFlags,
                               valueTypes, delimiter);
}

/// Verify that a the `values` has as many elements as the number of entries in
/// `attr` for which `isDynamic` evaluates to true.
LogicalResult verifyListOfOperandsOrIntegers(Operation *op, StringRef name,
                                             unsigned expectedNumElements,
                                             ArrayRef<int64_t> attr,
                                             ValueRange values);

} // namespace mlir

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
