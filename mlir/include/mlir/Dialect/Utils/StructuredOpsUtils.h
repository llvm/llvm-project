//===- StructuredOpsUtils.h - Utilities used by structured ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file define utilities that operate on standard types and are
// useful across multiple dialects that use structured ops abstractions. These
// abstractions consist of define custom operations that encode and transport
// information about their semantics (e.g. type of iterators like parallel,
// reduction, etc..) as attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H
#define MLIR_DIALECT_UTILS_STRUCTUREDOPSUTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

inline bool isRowMajorMatmul(ArrayAttr indexingMaps) {
  auto context = indexingMaps.getContext();
  AffineExpr m, n, k;
  bindDims(context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, context));
  auto maps = ArrayAttr::get({mapA, mapB, mapC}, context);
  return indexingMaps == maps;
}

inline bool isColumnMajorMatmul(ArrayAttr indexingMaps) {
  auto context = indexingMaps.getContext();
  AffineExpr m, n, k;
  bindDims(context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {n, m}, context));
  auto maps = ArrayAttr::get({mapA, mapB, mapC}, context);
  return indexingMaps == maps;
}

/// Attribute name for the AffineArrayAttr which encodes the relationship
/// between a structured op iterators' and its operands.
constexpr StringRef getIndexingMapsAttrName() { return "indexing_maps"; }

/// Attribute name for the StrArrayAttr which encodes the type of a structured
/// op's iterators.
constexpr StringRef getIteratorTypesAttrName() { return "iterator_types"; }

/// Attribute name for the IntegerAttr which encodes the number of input buffer
/// arguments.
constexpr StringRef getArgsInAttrName() { return "args_in"; }

/// Attribute name for the IntegerAttr which encodes the number of input buffer
/// arguments.
constexpr StringRef getArgsOutAttrName() { return "args_out"; }

/// Attribute name for the StringAttr which encodes an optional documentation
/// string of the structured op.
constexpr StringRef getDocAttrName() { return "doc"; }

/// Attribute name for the StrArrayAttr which encodes the external library
/// function that implements the structured op.
constexpr StringRef getLibraryCallAttrName() { return "library_call"; }

/// Attribute name for the StrArrayAttr which encodes the value of strides.
constexpr StringRef getStridesAttrName() { return "strides"; }

/// Attribute name for the StrArrayAttr which encodes the value of dilations.
constexpr StringRef getDilationsAttrName() { return "dilations"; }

/// Attribute name for the StrArrayAttr which encodes the value of paddings.
constexpr StringRef getPaddingAttrName() { return "padding"; }

/// Use to encode that a particular iterator type has parallel semantics.
constexpr StringRef getParallelIteratorTypeName() { return "parallel"; }
bool isParallelIterator(Attribute attr) {
  auto strAttr = attr.dyn_cast_or_null<StringAttr>();
  return strAttr && strAttr.getValue() == getParallelIteratorTypeName();
}

/// Use to encode that a particular iterator type has reduction semantics.
constexpr StringRef getReductionIteratorTypeName() { return "reduction"; }
bool isReductionIterator(Attribute attr) {
  auto strAttr = attr.dyn_cast_or_null<StringAttr>();
  return strAttr && strAttr.getValue() == getReductionIteratorTypeName();
}

/// Use to encode that a particular iterator type has window semantics.
constexpr StringRef getWindowIteratorTypeName() { return "window"; }
bool isWindowIterator(Attribute attr) {
  auto strAttr = attr.dyn_cast_or_null<StringAttr>();
  return strAttr && strAttr.getValue() == getWindowIteratorTypeName();
}

/// Use to encode that a particular iterator type has window semantics.
inline ArrayRef<StringRef> getAllIteratorTypeNames() {
  static constexpr StringRef names[3] = {getParallelIteratorTypeName(),
                                         getReductionIteratorTypeName(),
                                         getWindowIteratorTypeName()};
  return llvm::makeArrayRef(names);
}

/// Returns the iterator of a certain type.
inline unsigned getNumIterators(StringRef name, ArrayAttr iteratorTypes) {
  auto names = getAllIteratorTypeNames();
  (void)names;
  assert(llvm::is_contained(names, name));
  return llvm::count_if(iteratorTypes, [name](Attribute a) {
    return a.cast<StringAttr>().getValue() == name;
  });
}

inline unsigned getNumIterators(ArrayAttr iteratorTypes) {
  unsigned res = 0;
  for (auto n : getAllIteratorTypeNames())
    res += getNumIterators(n, iteratorTypes);
  return res;
}

/// Typed representation for loop type strings.
enum class IteratorType { Parallel, Reduction };

inline StringRef toString(IteratorType t) {
  switch (t) {
  case IteratorType::Parallel:
    return getParallelIteratorTypeName();
  case IteratorType::Reduction:
    return getReductionIteratorTypeName();
  }
  llvm_unreachable("Unsupported IteratorType");
}

} // end namespace mlir

#endif // MLIR_UTILS_STRUCTUREDOPSUTILS_H
