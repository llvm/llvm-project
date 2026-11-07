//===- InferIntRangeCommon.cpp - Inference for common ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares implementations of range inference for operations that are
// common to both the `arith` and `index` dialects to facilitate reuse.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_UTILS_INFERINTRANGECOMMON_H
#define MLIR_INTERFACES_UTILS_INFERINTRANGECOMMON_H

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include <optional>

namespace mlir {
class AffineExpr;
class ShapedDimOpInterface;

namespace intrange {
/// Function that performs inference on an array of integer ranges.
using InferRangeFn =
    std::function<ConstantIntRanges(ArrayRef<ConstantIntRanges>)>;

/// Function that performs inference on an array of integer range lattices.
using InferIntegerValueRangeFn =
    std::function<IntegerValueRange(ArrayRef<IntegerValueRange>)>;

enum class OverflowFlags : uint32_t {
  None = 0,
  Nsw = 1,
  Nuw = 2,
  LLVM_MARK_AS_BITMASK_ENUM(Nuw)
};

/// Function that performs inference on an array of `ConstantIntRanges` while
/// taking special overflow behavior into account.
using InferRangeWithOvfFlagsFn = function_ref<ConstantIntRanges(
    ArrayRef<ConstantIntRanges>, std::optional<unsigned>, OverflowFlags)>;

/// Independently zero-extend the unsigned values and sign-extend the signed
/// values in `range` to `destWidth` bits, returning the resulting range.
ConstantIntRanges extRange(const ConstantIntRanges &range, unsigned destWidth);

/// Use the unsigned values in `range` to zero-extend it to `destWidth`.
ConstantIntRanges extUIRange(const ConstantIntRanges &range,
                             unsigned destWidth);

/// Use the signed values in `range` to sign-extend it to `destWidth`.
ConstantIntRanges extSIRange(const ConstantIntRanges &range,
                             unsigned destWidth);

/// Truncate `range` to `destWidth` bits, taking care to handle cases such as
/// the truncation of [255, 256] to i8 not being a uniform range.
ConstantIntRanges truncRange(const ConstantIntRanges &range,
                             unsigned destWidth);

/// Infer a constant after conversion to the runtime result width.
ConstantIntRanges inferConstant(const APInt &value,
                                std::optional<unsigned> bitwidth);

/// Infer an index cast with the supplied runtime source and result widths.
IntegerValueRange inferIndexCast(const IntegerValueRange &range,
                                 std::optional<unsigned> srcWidth,
                                 std::optional<unsigned> destWidth,
                                 bool isSigned);

/// Infer arithmetic bounds at the supplied runtime operand width.
/// A missing width gives maximal bounds in index attribute storage.
ConstantIntRanges inferAdd(ArrayRef<ConstantIntRanges> argRanges,
                           std::optional<unsigned> bitwidth,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferSub(ArrayRef<ConstantIntRanges> argRanges,
                           std::optional<unsigned> bitwidth,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferMul(ArrayRef<ConstantIntRanges> argRanges,
                           std::optional<unsigned> bitwidth,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferDivS(ArrayRef<ConstantIntRanges> argRanges,
                            std::optional<unsigned> bitwidth);

ConstantIntRanges inferDivU(ArrayRef<ConstantIntRanges> argRanges,
                            std::optional<unsigned> bitwidth);

ConstantIntRanges inferCeilDivS(ArrayRef<ConstantIntRanges> argRanges,
                                std::optional<unsigned> bitwidth);

ConstantIntRanges inferCeilDivU(ArrayRef<ConstantIntRanges> argRanges,
                                std::optional<unsigned> bitwidth);

ConstantIntRanges inferFloorDivS(ArrayRef<ConstantIntRanges> argRanges,
                                 std::optional<unsigned> bitwidth);

ConstantIntRanges inferRemS(ArrayRef<ConstantIntRanges> argRanges,
                            std::optional<unsigned> bitwidth);

ConstantIntRanges inferRemU(ArrayRef<ConstantIntRanges> argRanges,
                            std::optional<unsigned> bitwidth);

ConstantIntRanges inferMaxS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMaxU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMinS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMinU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferAnd(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferOr(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferXor(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferShl(ArrayRef<ConstantIntRanges> argRanges,
                           std::optional<unsigned> bitwidth,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferShrS(ArrayRef<ConstantIntRanges> argRanges,
                            std::optional<unsigned> bitwidth);

ConstantIntRanges inferShrU(ArrayRef<ConstantIntRanges> argRanges,
                            std::optional<unsigned> bitwidth);

/// Copy of the enum from `arith` and `index` to allow the common integer range
/// infrastructure to not depend on either dialect.
enum class CmpPredicate : uint64_t {
  eq,
  ne,
  slt,
  sle,
  sgt,
  sge,
  ult,
  ule,
  ugt,
  uge,
};

/// Infer a comparison result at the supplied runtime operand width.
IntegerValueRange inferCmp(CmpPredicate pred,
                           ArrayRef<IntegerValueRange> argRanges,
                           std::optional<unsigned> bitwidth);

/// Returns a boolean value if `pred` is statically true or false for
/// anypossible inputs falling within `lhs` and `rhs`, and std::nullopt if the
/// value of the predicate cannot be determined.
std::optional<bool> evaluatePred(CmpPredicate pred,
                                 const ConstantIntRanges &lhs,
                                 const ConstantIntRanges &rhs);

/// Returns the integer range for the result of a `ShapedDimOpInterface` given
/// the optional inferred ranges for the `dimension` index `maybeDim`. When a
/// dynamic dimension is encountered, returns [0, signed_max(type(result))].
ConstantIntRanges
inferShapedDimOpInterface(ShapedDimOpInterface op,
                          const IntegerValueRange &maybeDim,
                          std::optional<unsigned> indexBitwidth);

/// Infer the integer range for an affine expression given ranges for its
/// dimensions and symbols.
ConstantIntRanges inferAffineExpr(AffineExpr expr,
                                  ArrayRef<ConstantIntRanges> dimRanges,
                                  ArrayRef<ConstantIntRanges> symbolRanges,
                                  std::optional<unsigned> indexBitwidth);
} // namespace intrange
} // namespace mlir

#endif // MLIR_INTERFACES_UTILS_INFERINTRANGECOMMON_H
