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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
//
// Type aliases to help code be more self-documenting.  Unfortunately
// these are not type-checked, so they only provide documentation rather
// than doing anything to prevent mixups.
//
// We must include these here (rather than in "SparseTensorType.h")
// because they are used by methods declared in the tablegen files.
//
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sparse_tensor {

/// The type of dimension identifiers, and dimension-ranks.  We use the
/// same type for both identifiers and ranks because the latter are used
/// mainly for ordering-comparisons against the former (just like how the
/// one-past-the-end iterators are used).
using Dimension = uint64_t;

/// The type of level identifiers, and level-ranks.  We use the same
/// type for both identifiers and ranks because the latter are used
/// mainly for ordering-comparisons against the former (just like how
/// the one-past-the-end iterators are used).
using Level = uint64_t;

/// The type for individual components of a compile-time shape.  We avoid
/// calling this "size" because we use the term "sizes" to indicate the
/// actual run-time sizes, whereas this type also allows the value
/// `ShapedType::kDynamic`.
using DynSize = int64_t;

/// The type for individual components of a compile-time shape which
/// are known not to be `ShapedType::kDynamic`.
using StaticSize = int64_t;

} // namespace sparse_tensor
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen-defined classes
//===----------------------------------------------------------------------===//

// We must include Enums.h.inc before AttrDefs.h.inc due to dependency between
// StorageSpecifierKindAttr and StorageSpeciferKind Enum.

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

// NOTE: `Value::getType` doesn't check for null before trying to
// dereference things.  Therefore we check, because an assertion-failure
// is easier to debug than a segfault.  Presumably other `T::getType`
// methods are similarly susceptible.

/// Convenience method to abbreviate casting `getType()`.
template <typename T>
inline RankedTensorType getRankedTensorType(T &&t) {
  assert(static_cast<bool>(std::forward<T>(t)) &&
         "getRankedTensorType got null argument");
  return cast<RankedTensorType>(std::forward<T>(t).getType());
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

/// Convenience method to query whether a given DLT needs both position and
/// coordinates array or only coordinates array.
constexpr inline bool isDLTWithPos(DimLevelType dlt) {
  return isCompressedWithHiDLT(dlt) || isCompressedDLT(dlt);
}
constexpr inline bool isDLTWithCrd(DimLevelType dlt) {
  return isSingletonDLT(dlt) || isCompressedWithHiDLT(dlt) ||
         isCompressedDLT(dlt);
}

/// Returns true iff the given sparse tensor encoding attribute has a trailing
/// COO region starting at the given level.
bool isCOOType(SparseTensorEncodingAttr enc, Level startLvl, bool isUnique);

/// Returns true iff the given type is a COO type where the last level
/// is unique.
bool isUniqueCOOType(Type tp);

/// Returns the starting level for a trailing COO region that spans
/// at least two levels.  If no such COO region is found, then returns
/// the level-rank.
Level getCOOStart(SparseTensorEncodingAttr enc);

/// Helpers to setup a COO type.
RankedTensorType getCOOFromTypeWithOrdering(RankedTensorType src,
                                            AffineMap ordering, bool ordered);

RankedTensorType getCOOFromType(RankedTensorType src, bool ordered);

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

//
// Reordering.
//

// This CPP guard is to disable deprecation warnings for the LLVM
// build-bot, while making it easy to re-enable it for local development.
#if 0
#define DEPRECATED                                                             \
  LLVM_DEPRECATED("The toOrigDim/toStoredDim functions are deprecated "        \
                  "because they only work for permutations; therefore any "    \
                  "code using them cannot support non-permutations.",          \
                  "")
#else
#define DEPRECATED
#endif

/// [deprecated] Convenience method to translate the given level to the
/// corresponding dimension.  Requires: `0 <= l < lvlRank`.
DEPRECATED Dimension toOrigDim(SparseTensorEncodingAttr enc, Level l);
DEPRECATED Dimension toOrigDim(RankedTensorType type, Level l);

/// [deprecated] Convenience method to translate the given dimension to
/// the corresponding level.  Requires: `0 <= d < dimRank`.
DEPRECATED Level toStoredDim(SparseTensorEncodingAttr enc, Dimension d);
DEPRECATED Level toStoredDim(RankedTensorType type, Dimension d);

#undef DEPRECATED

namespace detail {
Type getIntegerOrIndexType(MLIRContext *ctx, unsigned bitwidth);
} // namespace detail

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSOR_H_
