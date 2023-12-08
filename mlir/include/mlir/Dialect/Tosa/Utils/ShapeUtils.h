//===-- ShapeUtils.h - TOSA shape support declarations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class declarations for shape utilities meant to assist shape propagation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_UTILS_SHAPEUTILS_H
#define MLIR_DIALECT_TOSA_UTILS_SHAPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tosa {
/// Statically known information for a particular Value.
///
/// This struct currently tracks only information relevant for tensor/array-like
/// shaped types. It is fine to associate a `ValueKnowledge` with a non-shaped
/// type as long as it is in the default "no knowledge" state returned by
/// `getPessimisticValueState`. The important invariant is that we cannot
/// claim to know something about a value which is false.
///
/// This class could also be called "dataflow facts", "lattice value", etc.
struct ValueKnowledge {
  ValueKnowledge() = delete;
  ValueKnowledge(bool hasRank, llvm::ArrayRef<int64_t> newSizes, Type dtype)
      : hasError(false), hasRank(hasRank), dtype(dtype) {
    sizes.reserve(newSizes.size());
    for (auto size : newSizes)
      sizes.push_back(size);
  }

  operator bool() const { return !hasError; }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type) {
    ValueKnowledge result = getPessimisticValueState();
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      if (shapedType.hasRank()) {
        result.hasRank = true;
        result.sizes.reserve(shapedType.getRank());
        for (auto dim : shapedType.getShape())
          result.sizes.push_back(dim);
      }
      result.dtype = shapedType.getElementType();
    }
    return result;
  }

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState() {
    return ValueKnowledge(false, {}, Type());
  }

  ShapedTypeComponents getShapedTypeComponents() const {
    return hasRank ? ShapedTypeComponents(sizes) : ShapedTypeComponents();
  }

  Type getType() const {
    if (hasRank)
      return RankedTensorType::get(llvm::ArrayRef(sizes), dtype);
    return UnrankedTensorType::get(dtype);
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return hasRank == rhs.hasRank && sizes == rhs.sizes && dtype == rhs.dtype;
  }

  // Given two pieces of static knowledge, calculate conservatively the
  // information we can be sure about.
  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result = getPessimisticValueState();
    result.hasError = true;

    if (!lhs || !rhs || lhs.dtype != rhs.dtype)
      return result;

    result.hasError = false;
    result.dtype = lhs.dtype;

    if (!lhs.hasRank && !rhs.hasRank)
      return result;

    if (!rhs.hasRank) {
      result.hasRank = true;
      result.sizes = lhs.sizes;
      return result;
    }

    if (!lhs.hasRank) {
      result.hasRank = true;
      result.sizes = rhs.sizes;
      return result;
    }

    if (lhs.sizes.size() != rhs.sizes.size())
      return result;

    result.hasRank = true;
    result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
    for (auto i : llvm::seq<unsigned>(0, result.sizes.size())) {
      int64_t lhsSize = lhs.sizes[i];
      int64_t rhsSize = rhs.sizes[i];
      int64_t &resultSize = result.sizes[i];
      if (lhsSize == ShapedType::kDynamic) {
        resultSize = rhsSize;
      } else if (rhsSize == ShapedType::kDynamic) {
        resultSize = lhsSize;
      } else if (lhsSize == rhsSize) {
        resultSize = lhsSize;
      } else {
        result.hasError = true;
      }
    }

    return result;
  }

  // Given to types, generate a new ValueKnowledge that meets to cover both
  // cases. E.g. if the rank of the LHS and RHS differ, the resulting tensor
  // has unknown rank.
  static ValueKnowledge meet(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    ValueKnowledge result = getPessimisticValueState();
    result.hasError = true;

    if (!lhs || !rhs || lhs.dtype != rhs.dtype)
      return result;

    result.hasError = false;
    result.dtype = lhs.dtype;

    if (!lhs.hasRank || !rhs.hasRank) {
      result.hasRank = false;
      return result;
    }

    if (lhs.sizes.size() != rhs.sizes.size()) {
      result.hasRank = false;
      return result;
    }

    result.hasRank = true;
    result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
    for (int i = 0, e = lhs.sizes.size(); i < e; i++) {
      if (lhs.sizes[i] == rhs.sizes[i]) {
        result.sizes[i] = lhs.sizes[i];
      }
    }

    return result;
  }

  // Whether the value information has an error.
  bool hasError;
  // Whether the value has known rank.
  bool hasRank;
  // If `hasRank`, the sizes along each rank. Unknown sizes are represented as
  // `ShapedType::kDynamic`.
  llvm::SmallVector<int64_t> sizes;
  // The dtype of a tensor.
  // This is equal to nullptr if we don't know that it is a specific concrete
  // type.
  Type dtype;
};
} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_UTILS_SHAPEUTILS_H
