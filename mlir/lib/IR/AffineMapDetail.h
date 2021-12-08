//===- AffineMapDetail.h - MLIR Affine Map details Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of AffineMap.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINEMAPDETAIL_H_
#define AFFINEMAPDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace detail {

struct AffineMapStorage : public StorageUniquer::BaseStorage {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>>;

  unsigned numDims;
  unsigned numSymbols;

  /// The affine expressions for this (multi-dimensional) map.
  /// TODO: use trailing objects for this.
  ArrayRef<AffineExpr> results;

  MLIRContext *context;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numDims && std::get<1>(key) == numSymbols &&
           std::get<2>(key) == results;
  }

  // Constructs an AffineMapStorage from a key. The context must be set by the
  // caller.
  static AffineMapStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *res = new (allocator.allocate<AffineMapStorage>()) AffineMapStorage();
    res->numDims = std::get<0>(key);
    res->numSymbols = std::get<1>(key);
    res->results = allocator.copyInto(std::get<2>(key));
    return res;
  }
};

} // namespace detail
} // namespace mlir

#endif // AFFINEMAPDETAIL_H_
