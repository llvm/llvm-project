//===- SparseTensor.h - Sparse tensor dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.h.inc"

#include "mlir/Dialect/SparseTensor/IR/SparseTensorOpsDialect.h.inc"

namespace mlir {
namespace sparse_tensor {

/// Convenience method to get a sparse encoding attribute from a type.
/// Returns null-attribute for any type without an encoding.
SparseTensorEncodingAttr getSparseTensorEncoding(Type type);

/// Returns true iff the given type is a type for a COO tensor with the last
/// dimension level type being unique.
bool isUniqueCOOType(RankedTensorType tp);

//
// Dimension level types.
//

// MSVC does not allow this function to be constexpr, because
// `SparseTensorEncodingAttr::operator bool` isn't declared constexpr.
// And therefore all functions calling it cannot be constexpr either.
// TODO: since Clang does allow these to be constexpr, perhaps we should
// define a macro to abstract over `inline` vs `constexpr` annotations.
inline DimLevelType getDimLevelType(const SparseTensorEncodingAttr &enc,
                                    uint64_t d) {
  if (enc) {
    auto types = enc.getDimLevelType();
    assert(d < types.size() && "Dimension out of bounds");
    return types[d];
  }
  return DimLevelType::Dense; // unannotated tensor is dense
}

inline DimLevelType getDimLevelType(RankedTensorType type, uint64_t d) {
  return getDimLevelType(getSparseTensorEncoding(type), d);
}

/// Convenience function to test for dense dimension (0 <= d < rank).
inline bool isDenseDim(RankedTensorType type, uint64_t d) {
  return isDenseDLT(getDimLevelType(type, d));
}

/// Convenience function to test for compressed dimension (0 <= d < rank).
inline bool isCompressedDim(RankedTensorType type, uint64_t d) {
  return isCompressedDLT(getDimLevelType(type, d));
}

/// Convenience function to test for singleton dimension (0 <= d < rank).
inline bool isSingletonDim(RankedTensorType type, uint64_t d) {
  return isSingletonDLT(getDimLevelType(type, d));
}

/// Convenience function to test for dense dimension (0 <= d < rank).
inline bool isDenseDim(SparseTensorEncodingAttr enc, uint64_t d) {
  return isDenseDLT(getDimLevelType(enc, d));
}

/// Convenience function to test for compressed dimension (0 <= d < rank).
inline bool isCompressedDim(SparseTensorEncodingAttr enc, uint64_t d) {
  return isCompressedDLT(getDimLevelType(enc, d));
}

/// Convenience function to test for singleton dimension (0 <= d < rank).
inline bool isSingletonDim(SparseTensorEncodingAttr enc, uint64_t d) {
  return isSingletonDLT(getDimLevelType(enc, d));
}

//
// Dimension level properties.
//

/// Convenience function to test for ordered property in the
/// given dimension (0 <= d < rank).
inline bool isOrderedDim(RankedTensorType type, uint64_t d) {
  return isOrderedDLT(getDimLevelType(type, d));
}

/// Convenience function to test for unique property in the
/// given dimension (0 <= d < rank).
inline bool isUniqueDim(RankedTensorType type, uint64_t d) {
  return isUniqueDLT(getDimLevelType(type, d));
}

//
// Reordering.
//

uint64_t toOrigDim(const SparseTensorEncodingAttr &enc, uint64_t d);
uint64_t toStoredDim(const SparseTensorEncodingAttr &enc, uint64_t d);

/// Convenience method to translate the given stored dimension
/// to the original dimension (0 <= d < rank).
uint64_t toOrigDim(RankedTensorType type, uint64_t d);

/// Convenience method to translate the given original dimension
/// to the stored dimension (0 <= d < rank).
uint64_t toStoredDim(RankedTensorType type, uint64_t d);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
