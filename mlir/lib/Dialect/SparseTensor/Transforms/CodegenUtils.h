//===- CodegenUtils.h - Utilities for generating MLIR -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for generating MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/Enums.h"
#include "mlir/ExecutionEngine/SparseTensorUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir {
class Location;
class Type;
class Value;

namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ExecutionEngine/SparseTensorUtils helper functions.
//===----------------------------------------------------------------------===//

/// Converts an overhead storage bitwidth to its internal type-encoding.
OverheadType overheadTypeEncoding(unsigned width);

/// Converts an overhead storage type to its internal type-encoding.
OverheadType overheadTypeEncoding(Type tp);

/// Converts the internal type-encoding for overhead storage to an mlir::Type.
Type getOverheadType(Builder &builder, OverheadType ot);

/// Returns the OverheadType for pointer overhead storage.
OverheadType pointerOverheadTypeEncoding(const SparseTensorEncodingAttr &enc);

/// Returns the OverheadType for index overhead storage.
OverheadType indexOverheadTypeEncoding(const SparseTensorEncodingAttr &enc);

/// Returns the mlir::Type for pointer overhead storage.
Type getPointerOverheadType(Builder &builder,
                            const SparseTensorEncodingAttr &enc);

/// Returns the mlir::Type for index overhead storage.
Type getIndexOverheadType(Builder &builder,
                          const SparseTensorEncodingAttr &enc);

/// Convert OverheadType to its function-name suffix.
StringRef overheadTypeFunctionSuffix(OverheadType ot);

/// Converts an overhead storage type to its function-name suffix.
StringRef overheadTypeFunctionSuffix(Type overheadTp);

/// Converts a primary storage type to its internal type-encoding.
PrimaryType primaryTypeEncoding(Type elemTp);

/// Convert PrimaryType to its function-name suffix.
StringRef primaryTypeFunctionSuffix(PrimaryType pt);

/// Converts a primary storage type to its function-name suffix.
StringRef primaryTypeFunctionSuffix(Type elemTp);

/// Converts the IR's dimension level type to its internal type-encoding.
DimLevelType dimLevelTypeEncoding(SparseTensorEncodingAttr::DimLevelType dlt);

//===----------------------------------------------------------------------===//
// Misc code generators.
//
// TODO: both of these should move upstream to their respective classes.
// Once RFCs have been created for those changes, list them here.
//===----------------------------------------------------------------------===//

/// Generates a 1-valued attribute of the given type.  This supports
/// all the same types as `getZeroAttr`; however, unlike `getZeroAttr`,
/// for unsupported types we raise `llvm_unreachable` rather than
/// returning a null attribute.
Attribute getOneAttr(Builder &builder, Type tp);

/// Generates the comparison `v != 0` where `v` is of numeric type.
/// For floating types, we use the "unordered" comparator (i.e., returns
/// true if `v` is NaN).
Value genIsNonzero(OpBuilder &builder, Location loc, Value v);

//===----------------------------------------------------------------------===//
// Constant generators.
//
// All these functions are just wrappers to improve code legibility;
// therefore, we mark them as `inline` to avoid introducing any additional
// overhead due to the legibility.
//
// TODO: Ideally these should move upstream, so that we don't
// develop a design island.  However, doing so will involve
// substantial design work.  For related prior discussion, see
// <https://llvm.discourse.group/t/evolving-builder-apis-based-on-lessons-learned-from-edsc/879>
//===----------------------------------------------------------------------===//

/// Generates a 0-valued constant of the given type.  In addition to
/// the scalar types (`ComplexType`, ``FloatType`, `IndexType`, `IntegerType`),
/// this also works for `RankedTensorType` and `VectorType` (for which it
/// generates a constant `DenseElementsAttr` of zeros).
inline Value constantZero(OpBuilder &builder, Location loc, Type tp) {
  if (auto ctp = tp.dyn_cast<ComplexType>()) {
    auto zeroe = builder.getZeroAttr(ctp.getElementType());
    auto zeroa = builder.getArrayAttr({zeroe, zeroe});
    return builder.create<complex::ConstantOp>(loc, tp, zeroa);
  }
  return builder.create<arith::ConstantOp>(loc, tp, builder.getZeroAttr(tp));
}

/// Generates a 1-valued constant of the given type.  This supports all
/// the same types as `constantZero`.
inline Value constantOne(OpBuilder &builder, Location loc, Type tp) {
  if (auto ctp = tp.dyn_cast<ComplexType>()) {
    auto zeroe = builder.getZeroAttr(ctp.getElementType());
    auto onee = getOneAttr(builder, ctp.getElementType());
    auto zeroa = builder.getArrayAttr({onee, zeroe});
    return builder.create<complex::ConstantOp>(loc, tp, zeroa);
  }
  return builder.create<arith::ConstantOp>(loc, tp, getOneAttr(builder, tp));
}

/// Generates a constant of `index` type.
inline Value constantIndex(OpBuilder &builder, Location loc, int64_t i) {
  return builder.create<arith::ConstantIndexOp>(loc, i);
}

/// Generates a constant of `i32` type.
inline Value constantI32(OpBuilder &builder, Location loc, int32_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 32);
}

/// Generates a constant of `i16` type.
inline Value constantI16(OpBuilder &builder, Location loc, int16_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 16);
}

/// Generates a constant of `i8` type.
inline Value constantI8(OpBuilder &builder, Location loc, int8_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 8);
}

/// Generates a constant of `i1` type.
inline Value constantI1(OpBuilder &builder, Location loc, bool b) {
  return builder.create<arith::ConstantIntOp>(loc, b, 1);
}

/// Generates a constant of the given `Action`.
inline Value constantAction(OpBuilder &builder, Location loc, Action action) {
  return constantI32(builder, loc, static_cast<uint32_t>(action));
}

/// Generates a constant of the internal type-encoding for overhead storage.
inline Value constantOverheadTypeEncoding(OpBuilder &builder, Location loc,
                                          unsigned width) {
  return constantI32(builder, loc,
                     static_cast<uint32_t>(overheadTypeEncoding(width)));
}

/// Generates a constant of the internal type-encoding for pointer
/// overhead storage.
inline Value constantPointerTypeEncoding(OpBuilder &builder, Location loc,
                                         const SparseTensorEncodingAttr &enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getPointerBitWidth());
}

/// Generates a constant of the internal type-encoding for index overhead
/// storage.
inline Value constantIndexTypeEncoding(OpBuilder &builder, Location loc,
                                       const SparseTensorEncodingAttr &enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getIndexBitWidth());
}

/// Generates a constant of the internal type-encoding for primary storage.
inline Value constantPrimaryTypeEncoding(OpBuilder &builder, Location loc,
                                         Type elemTp) {
  return constantI32(builder, loc,
                     static_cast<uint32_t>(primaryTypeEncoding(elemTp)));
}

/// Generates a constant of the internal dimension level type encoding.
inline Value
constantDimLevelTypeEncoding(OpBuilder &builder, Location loc,
                             SparseTensorEncodingAttr::DimLevelType dlt) {
  return constantI8(builder, loc,
                    static_cast<uint8_t>(dimLevelTypeEncoding(dlt)));
}

/// Helper method to translate indices during a reshaping operation.
void translateIndicesArray(OpBuilder &builder, Location loc,
                           ArrayRef<ReassociationIndices> reassociation,
                           ValueRange srcIndices, ArrayRef<Value> srcShape,
                           ArrayRef<Value> dstShape,
                           SmallVectorImpl<Value> &dstIndices);
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_
