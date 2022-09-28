//===- SparseTensor.h - Sparse tensor dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_

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

//
// Dimension level types.
//

bool isDenseDim(SparseTensorEncodingAttr::DimLevelType dltp);
bool isCompressedDim(SparseTensorEncodingAttr::DimLevelType dltp);
bool isSingletonDim(SparseTensorEncodingAttr::DimLevelType dltp);

/// Convenience method to test for dense dimension (0 <= d < rank).
bool isDenseDim(RankedTensorType type, uint64_t d);

/// Convenience method to test for compressed dimension (0 <= d < rank).
bool isCompressedDim(RankedTensorType type, uint64_t d);

/// Convenience method to test for singleton dimension (0 <= d < rank).
bool isSingletonDim(RankedTensorType type, uint64_t d);

//
// Dimension level properties.
//

bool isOrderedDim(SparseTensorEncodingAttr::DimLevelType dltp);
bool isUniqueDim(SparseTensorEncodingAttr::DimLevelType dltp);

/// Convenience method to test for ordered property in the
/// given dimension (0 <= d < rank).
bool isOrderedDim(RankedTensorType type, uint64_t d);

/// Convenience method to test for unique property in the
/// given dimension (0 <= d < rank).
bool isUniqueDim(RankedTensorType type, uint64_t d);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
