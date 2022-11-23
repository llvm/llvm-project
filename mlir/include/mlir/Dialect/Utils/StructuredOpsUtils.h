//===- StructuredOpsUtils.h - Utilities used by structured ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file define utilities that operate on builtin types and are
// useful across multiple dialects that use structured ops abstractions. These
// abstractions consist of define custom operations that encode and transport
// information about their semantics (e.g. type of iterators like parallel,
// reduction, etc..) as attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H
#define MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/Utils/DialectUtilsEnums.h.inc"

namespace mlir {

class OpBuilder;
class TypeRange;
class ValueRange;

/// Tests whether the given maps describe a row major matmul. The test is
/// permutation-invariant. Note that this only checks the affine maps from an
/// operation, so does not perform any checks on the math being performed within
/// the reduction.
bool isRowMajorMatmul(ArrayAttr indexingMaps);

/// Tests whether the given maps describe a column major matmul. The test is
/// permutation-invariant. Note that this only checks the affine maps from an
/// operation, so does not perform any checks on the math being performed within
/// the reduction.
bool isColumnMajorMatmul(ArrayAttr indexingMaps);

/// Tests whether the given maps describe a row major batch matmul. The test is
/// permutation-invariant. Note that this only checks the affine maps from an
/// operation, so does not perform any checks on the math being performed within
/// the reduction.
bool isRowMajorBatchMatmul(ArrayAttr indexingMaps);

/// Return positions in `iteratorTypes` that match `iteratorTypeName`.
inline void findPositionsOfType(ArrayRef<utils::IteratorType> iteratorTypes,
                                utils::IteratorType iteratorTypeName,
                                SmallVectorImpl<unsigned> &res) {
  for (const auto &en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == iteratorTypeName)
      res.push_back(en.index());
  }
}

/// Helper StructuredGenerator class to manipulate and rewrite ops with
/// `StructuredOpInterface`. This is templated for now because VectorOps do not
/// yet implement the StructuredOpInterface itself.
template <typename StructuredOpInterface, typename IteratorTypeT>
class StructuredGenerator {
public:
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;

  struct IteratorType {
    IteratorType(IteratorTypeT iter) : iter(iter) {}
    bool isOfType(IteratorTypeT expectedIter) const {
      return expectedIter == iter;
    }
    IteratorTypeT iter;
  };
  struct Par : public IteratorType {
    Par() : IteratorType(IteratorTypeT::parallel) {}
  };
  struct Red : public IteratorType {
    Red() : IteratorType(IteratorTypeT::reduction) {}
  };

  StructuredGenerator(OpBuilder &builder, StructuredOpInterface op)
      : builder(builder), ctx(op.getContext()), loc(op.getLoc()),
        iterators(op.getIteratorTypesArray()), maps(op.getIndexingMapsArray()),
        op(op) {}

  bool iters(ArrayRef<IteratorType> its) {
    if (its.size() != iterators.size())
      return false;
    for (int i = 0, e = its.size(); i != e; ++i) {
      if (!its[i].isOfType(iterators[i]))
        return false;
    }
    return true;
  }

  bool layout(MapList l) {
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    return maps == infer(l);
  }

protected:
  OpBuilder &builder;
  MLIRContext *ctx;
  Location loc;
  SmallVector<IteratorTypeT> iterators;
  SmallVector<AffineMap, 4> maps;
  Operation *op;
};

// Clone the current operation with the operands. This is used to abstract away
// the optional underlying region creation.
Operation *clone(OpBuilder &b, Operation *op, TypeRange newResultTypes,
                 ValueRange newOperands);

// Clone the current operation with the operands but leave the regions empty.
Operation *cloneWithoutRegions(OpBuilder &b, Operation *op,
                               TypeRange newResultTypes,
                               ValueRange newOperands);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H
