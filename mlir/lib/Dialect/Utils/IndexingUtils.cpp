//===- IndexingUtils.cpp - Helpers related to index computations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/IndexingUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

int64_t mlir::linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(offsets.size() == basis.size());
  int64_t linearIndex = 0;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex += offsets[idx] * basis[idx];
  return linearIndex;
}

llvm::SmallVector<int64_t, 4> mlir::delinearize(ArrayRef<int64_t> sliceStrides,
                                                int64_t index) {
  int64_t rank = sliceStrides.size();
  SmallVector<int64_t, 4> vectorOffsets(rank);
  for (int64_t r = 0; r < rank; ++r) {
    assert(sliceStrides[r] > 0);
    vectorOffsets[r] = index / sliceStrides[r];
    index %= sliceStrides[r];
  }
  return vectorOffsets;
}

llvm::SmallVector<int64_t, 4> mlir::getI64SubArray(ArrayAttr arrayAttr,
                                                   unsigned dropFront,
                                                   unsigned dropBack) {
  assert(arrayAttr.size() > dropFront + dropBack && "Out of bounds");
  auto range = arrayAttr.getAsRange<IntegerAttr>();
  SmallVector<int64_t, 4> res;
  res.reserve(arrayAttr.size() - dropFront - dropBack);
  for (auto it = range.begin() + dropFront, eit = range.end() - dropBack;
       it != eit; ++it)
    res.push_back((*it).getValue().getSExtValue());
  return res;
}

mlir::AffineExpr mlir::getLinearAffineExpr(ArrayRef<int64_t> basis,
                                           mlir::Builder &b) {
  AffineExpr resultExpr = b.getAffineDimExpr(0);
  resultExpr = resultExpr * basis[0];
  for (unsigned i = 1; i < basis.size(); i++)
    resultExpr = resultExpr + b.getAffineDimExpr(i) * basis[i];
  return resultExpr;
}

llvm::SmallVector<mlir::AffineExpr, 4>
mlir::getDelinearizedAffineExpr(mlir::ArrayRef<int64_t> strides, Builder &b) {
  AffineExpr resultExpr = b.getAffineDimExpr(0);
  int64_t rank = strides.size();
  SmallVector<AffineExpr, 4> vectorOffsets(rank);
  vectorOffsets[0] = resultExpr.floorDiv(strides[0]);
  resultExpr = resultExpr % strides[0];
  for (unsigned i = 1; i < rank; i++) {
    vectorOffsets[i] = resultExpr.floorDiv(strides[i]);
    resultExpr = resultExpr % strides[i];
  }
  return vectorOffsets;
}
