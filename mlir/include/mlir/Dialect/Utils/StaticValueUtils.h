//===- StaticValueUtils.h - Utilities for static values ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for dealing with static values, e.g.,
// converting back and forth between Value and OpFoldResult. Such functionality
// is used in multiple dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_STATICVALUEUTILS_H
#define MLIR_DIALECT_UTILS_STATICVALUEUTILS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// Represents a range (offset, size, and stride) where each element of the
/// triple may be dynamic or static.
struct Range {
  OpFoldResult offset;
  OpFoldResult size;
  OpFoldResult stride;
};

/// Given an array of Range values, return a tuple of (offset vector, sizes
/// vector, and strides vector) formed by separating out the individual elements
/// of each range.
std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
           SmallVector<OpFoldResult>>
getOffsetsSizesAndStrides(ArrayRef<Range> ranges);

/// Helper function to dispatch an OpFoldResult into `staticVec` if:
///   a) it is an IntegerAttr
/// In other cases, the OpFoldResult is dispached to the `dynamicVec`.
/// In such dynamic cases, a copy of the `sentinel` value is also pushed to
/// `staticVec`. This is useful to extract mixed static and dynamic entries that
/// come from an AttrSizedOperandSegments trait.
void dispatchIndexOpFoldResult(OpFoldResult ofr,
                               SmallVectorImpl<Value> &dynamicVec,
                               SmallVectorImpl<int64_t> &staticVec,
                               int64_t sentinel);

/// Helper function to dispatch multiple OpFoldResults according to the behavior
/// of `dispatchIndexOpFoldResult(OpFoldResult ofr` for a single OpFoldResult.
void dispatchIndexOpFoldResults(ArrayRef<OpFoldResult> ofrs,
                                SmallVectorImpl<Value> &dynamicVec,
                                SmallVectorImpl<int64_t> &staticVec,
                                int64_t sentinel);

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr);

/// Given a value, try to extract a constant Attribute. If this fails, return
/// the original value.
OpFoldResult getAsOpFoldResult(Value val);

/// Given an array of values, try to extract a constant Attribute from each
/// value. If this fails, return the original value.
SmallVector<OpFoldResult> getAsOpFoldResult(ValueRange values);

/// Convert `arrayAttr` to a vector of OpFoldResult.
SmallVector<OpFoldResult> getAsOpFoldResult(ArrayAttr arrayAttr);

/// If ofr is a constant integer or an IntegerAttr, return the integer.
Optional<int64_t> getConstantIntValue(OpFoldResult ofr);

/// Return true if `ofr` is constant integer equal to `value`.
bool isConstantIntValue(OpFoldResult ofr, int64_t value);

/// Return true if ofr1 and ofr2 are the same integer constant attribute values
/// or the same SSA value.
/// Ignore integer bitwitdh and type mismatch that come from the fact there is
/// no IndexAttr and that IndexType have no bitwidth.
bool isEqualConstantIntOrValue(OpFoldResult ofr1, OpFoldResult ofr2);

/// Helper function to convert a vector of `OpFoldResult`s into a vector of
/// `Value`s. For each `OpFoldResult` in `valueOrAttrVec` return the fold result
/// if it casts to  a `Value` or create an index-type constant if it casts to
/// `IntegerAttr`. No other attribute types are supported.
SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                               ArrayRef<OpFoldResult> valueOrAttrVec);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_STATICVALUEUTILS_H
