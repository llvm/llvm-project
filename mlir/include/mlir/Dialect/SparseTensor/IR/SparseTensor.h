//===- SparseTensor.h - Sparse tensor dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
//
// Type aliases to help code be more self-documenting. Unfortunately
// these are not type-checked, so they only provide documentation rather
// than doing anything to prevent mixups.
//
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sparse_tensor {

/// The type of dimension identifiers and dimension-ranks.
using Dimension = uint64_t;

/// The type of level identifiers and level-ranks.
using Level = uint64_t;

/// The type for individual components of a compile-time shape,
/// including the value `ShapedType::kDynamic` (for shapes).
using Size = int64_t;

} // namespace sparse_tensor
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen-defined classes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.h.inc"

#include "mlir/Dialect/SparseTensor/IR/SparseTensorOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Additional convenience methods.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sparse_tensor {

/// Convenience method to abbreviate casting `getType()`.
template <typename T>
inline RankedTensorType getRankedTensorType(T &&t) {
  assert(static_cast<bool>(std::forward<T>(t)) &&
         "getRankedTensorType got null argument");
  return dyn_cast<RankedTensorType>(std::forward<T>(t).getType());
}

/// Convenience method to abbreviate casting `getType()`.
template <typename T>
inline MemRefType getMemRefType(T &&t) {
  assert(static_cast<bool>(std::forward<T>(t)) &&
         "getMemRefType got null argument");
  return cast<MemRefType>(std::forward<T>(t).getType());
}

/// Convenience method to get a sparse encoding attribute from a type.
/// Returns null-attribute for any type without an encoding.
SparseTensorEncodingAttr getSparseTensorEncoding(Type type);

/// Returns true iff MLIR operand has any sparse operand.
inline bool hasAnySparseOperand(Operation *op) {
  return llvm::any_of(op->getOperands().getTypes(), [](Type t) {
    return getSparseTensorEncoding(t) != nullptr;
  });
}

/// Returns true iff MLIR operand has any sparse result.
inline bool hasAnySparseResult(Operation *op) {
  return llvm::any_of(op->getResults().getTypes(), [](Type t) {
    return getSparseTensorEncoding(t) != nullptr;
  });
}

/// Returns true iff MLIR operand has any sparse operand or result.
inline bool hasAnySparseOperandOrResult(Operation *op) {
  return hasAnySparseOperand(op) || hasAnySparseResult(op);
}

/// Returns true iff MLIR operation has any sparse tensor with non-identity
/// dim2lvl maps.
bool hasAnyNonIdentityOperandsOrResults(Operation *op);

//
// Inference.
//

/// Given the dimToLvl map, infers the lvlToDim map, or returns
/// empty Affine map when inference fails.
AffineMap inferLvlToDim(AffineMap dimToLvl, MLIRContext *context);

/// Returns the lvlToDim map for the given dimToLvl map specific
/// to the block sparse cases.
/// Asserts on failure (so only use when known to succeed).
AffineMap inverseBlockSparsity(AffineMap dimToLvl, MLIRContext *context);

/// Given the dimToLvl map, returns the block sizes in a vector.
/// For instance, a 2x3 block will return [2, 3]. Unblocked dimension i
/// will return 0, and i floordiv 1, i mod 1 will return 1. Therefore,
/// the example below will return [0, 1].
/// map = ( i, j ) ->
///       ( i : dense,
///         j floordiv 1 : compressed,
///         j mod 1      : dense
///       )
/// Only valid block sparsity will be accepted.
SmallVector<unsigned> getBlockSize(AffineMap dimToLvl);

/// Given the dimToLvl map, returns if it's block sparsity.
bool isBlockSparsity(AffineMap dimToLvl);

//
// Reordering.
//

/// Convenience method to translate the given level to the corresponding
/// dimension.
/// Requires: `enc` has a permuted dim2lvl map and `0 <= l < lvlRank`.
Dimension toDim(SparseTensorEncodingAttr enc, Level l);

/// Convenience method to translate the given dimension to the corresponding
/// level.
/// Requires: `enc` has a permuted dim2lvl map and `0 <= d < dimRank`.
Level toLvl(SparseTensorEncodingAttr enc, Dimension d);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
