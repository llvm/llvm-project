//===- StructuredOpsUtils.cpp - Utilities used by structured ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/Dialect/Utils/DialectUtilsEnums.cpp.inc"

using namespace mlir;

bool mlir::isRowMajorMatmul(ArrayAttr indexingMaps) {
  if (indexingMaps.size() != 3)
    return false;

  auto map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  auto map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  auto map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }

  // Extract dimensions for MxK * KxN -> MxN
  AffineExpr m = map2.getResult(0);
  AffineExpr n = map2.getResult(1);
  AffineExpr k = map0.getResult(1);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

bool mlir::isColumnMajorMatmul(ArrayAttr indexingMaps) {
  if (indexingMaps.size() != 3)
    return false;

  auto map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  auto map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  auto map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }

  // Extract dimensions for KxM * NxK -> NxM
  AffineExpr n = map2.getResult(0);
  AffineExpr m = map2.getResult(1);
  AffineExpr k = map0.getResult(0);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, m}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {n, m}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

bool mlir::isRowMajorBatchMatmul(ArrayAttr indexingMaps) {
  if (indexingMaps.size() != 3)
    return false;

  auto map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  auto map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  auto map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 3 || map1.getNumResults() != 3 ||
      map2.getNumResults() != 3 || map0.getNumInputs() != 4 ||
      map1.getNumInputs() != 4 || map2.getNumInputs() != 4) {
    return false;
  }

  // Extract dimensions for BxMxK * BxKxN -> BxMxN
  AffineExpr b = map2.getResult(0);
  AffineExpr m = map2.getResult(1);
  AffineExpr n = map2.getResult(2);
  AffineExpr k = map0.getResult(2);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {b, m, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {b, k, n}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {b, m, n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

bool mlir::isVecmat(ArrayAttr indexingMaps) {
  if (indexingMaps.size() != 3)
    return false;
  auto map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  auto map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  auto map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 1 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 1 || map0.getNumInputs() != 2 ||
      map1.getNumInputs() != 2 || map2.getNumInputs() != 2) {
    return false;
  }

  // Extract dimensions for K * KxN -> N
  AffineExpr k = map0.getResult(0);
  AffineExpr n = map2.getResult(0);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k, n}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

bool mlir::isMatvec(ArrayAttr indexingMaps) {
  if (indexingMaps.size() != 3)
    return false;
  auto map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  auto map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  auto map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 1 ||
      map2.getNumResults() != 1 || map0.getNumInputs() != 2 ||
      map1.getNumInputs() != 2 || map2.getNumInputs() != 2) {
    return false;
  }

  // Extract dimensions for N*K * K -> N
  AffineExpr k = map1.getResult(0);
  AffineExpr n = map2.getResult(0);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {n, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

bool mlir::isBatchMatvec(ArrayAttr indexingMaps) {
  if (indexingMaps.size() != 3)
    return false;
  auto map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  auto map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  auto map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 3 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }

  // Extract dimensions for B*N*K * B*K -> B*N
  AffineExpr b = map0.getResult(0);
  AffineExpr k = map1.getResult(1);
  AffineExpr n = map2.getResult(1);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {b, n, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {b, k}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {b, n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

Operation *mlir::clone(OpBuilder &b, Operation *op, TypeRange newResultTypes,
                       ValueRange newOperands) {
  IRMapping bvm;
  OperationState state(op->getLoc(), op->getName(), newOperands, newResultTypes,
                       op->getAttrs());
  for (Region &r : op->getRegions())
    r.cloneInto(state.addRegion(), bvm);
  return b.create(state);
}

Operation *mlir::cloneWithoutRegions(OpBuilder &b, Operation *op,
                                     TypeRange newResultTypes,
                                     ValueRange newOperands) {
  OperationState state(op->getLoc(), op->getName(), newOperands, newResultTypes,
                       op->getAttrs());
  for (size_t cnt = 0, e = op->getNumRegions(); cnt < e; ++cnt)
    state.addRegion();
  return b.create(state);
}

SmallVector<NamedAttribute>
mlir::getPrunedAttributeList(Operation *op, ArrayRef<StringRef> elidedAttrs) {
  llvm::StringSet<> elidedAttrsSet;
  elidedAttrsSet.insert(elidedAttrs.begin(), elidedAttrs.end());
  SmallVector<NamedAttribute> attrs;
  for (auto attr : op->getAttrs()) {
    if (elidedAttrsSet.count(attr.getName()))
      continue;
    attrs.push_back(attr);
  }
  return attrs;
}
