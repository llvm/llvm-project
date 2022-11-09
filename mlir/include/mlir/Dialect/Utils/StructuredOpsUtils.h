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
#include "llvm/ADT/StringRef.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/Utils/DialectUtilsEnums.h.inc"

namespace mlir {

class OpBuilder;

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

/// Use to encode that a particular iterator type has parallel semantics.
constexpr StringRef getParallelIteratorTypeName() { return "parallel"; }

/// Use to encode that a particular iterator type has reduction semantics.
constexpr StringRef getReductionIteratorTypeName() { return "reduction"; }

/// Use to encode that a particular iterator type has window semantics.
constexpr StringRef getWindowIteratorTypeName() { return "window"; }

/// Use to encode that a particular iterator type has window semantics.
inline ArrayRef<StringRef> getAllIteratorTypeNames() {
  static constexpr StringRef names[3] = {getParallelIteratorTypeName(),
                                         getReductionIteratorTypeName(),
                                         getWindowIteratorTypeName()};
  return llvm::makeArrayRef(names);
}

/// Returns the iterator of a certain type.
inline unsigned getNumIterators(StringRef name,
                                ArrayRef<StringRef> iteratorTypes) {
  auto names = getAllIteratorTypeNames();
  (void)names;
  assert(llvm::is_contained(names, name));
  return llvm::count(iteratorTypes, name);
}

inline unsigned getNumIterators(ArrayRef<StringRef> iteratorTypes) {
  unsigned res = 0;
  for (auto n : getAllIteratorTypeNames())
    res += getNumIterators(n, iteratorTypes);
  return res;
}

/// Return positions in `iteratorTypes` that match `iteratorTypeName`.
inline void findPositionsOfType(ArrayRef<StringRef> iteratorTypes,
                                StringRef iteratorTypeName,
                                SmallVectorImpl<unsigned> &res) {
  for (const auto &en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == iteratorTypeName)
      res.push_back(en.index());
  }
}

/// Helper StructuredGenerator class to manipulate and rewrite ops with
/// `StructuredOpInterface`. This is templated for now because VectorOps do not
/// yet implement the StructuredOpInterface itself.
template <typename StructuredOpInterface>
class StructuredGenerator {
public:
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;

  struct IteratorType {
    IteratorType(StringRef strRef) : strRef(strRef) {}
    bool isOfType(StringRef typeName) const { return typeName == strRef; }
    StringRef strRef;
  };
  struct Par : public IteratorType {
    Par() : IteratorType(getParallelIteratorTypeName()) {}
  };
  struct Red : public IteratorType {
    Red() : IteratorType(getReductionIteratorTypeName()) {}
  };
  struct Win : public IteratorType {
    Win() : IteratorType(getWindowIteratorTypeName()) {}
  };

  StructuredGenerator(OpBuilder &builder, StructuredOpInterface op)
      : builder(builder), ctx(op.getContext()), loc(op.getLoc()),
        iterators(op.getIteratorTypeNames()), maps(op.getIndexingMapsArray()),
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
  SmallVector<StringRef> iterators;
  SmallVector<AffineMap, 4> maps;
  Operation *op;
};

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H
