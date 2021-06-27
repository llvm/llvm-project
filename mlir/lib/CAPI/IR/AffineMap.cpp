//===- AffineMap.cpp - C API for MLIR Affine Maps -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineExpr.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/AffineMap.h"

// TODO: expose the C API related to `AffineExpr` and mutable affine map.

using namespace mlir;

MlirContext mlirAffineMapGetContext(MlirAffineMap affineMap) {
  return wrap(unwrap(affineMap).getContext());
}

bool mlirAffineMapEqual(MlirAffineMap a1, MlirAffineMap a2) {
  return unwrap(a1) == unwrap(a2);
}

void mlirAffineMapPrint(MlirAffineMap affineMap, MlirStringCallback callback,
                        void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  unwrap(affineMap).print(stream);
}

void mlirAffineMapDump(MlirAffineMap affineMap) { unwrap(affineMap).dump(); }

MlirAffineMap mlirAffineMapEmptyGet(MlirContext ctx) {
  return wrap(AffineMap::get(unwrap(ctx)));
}

MlirAffineMap mlirAffineMapZeroResultGet(MlirContext ctx, intptr_t dimCount,
                                         intptr_t symbolCount) {
  return wrap(AffineMap::get(dimCount, symbolCount, unwrap(ctx)));
}

MlirAffineMap mlirAffineMapGet(MlirContext ctx, intptr_t dimCount,
                               intptr_t symbolCount, intptr_t nAffineExprs,
                               MlirAffineExpr *affineExprs) {
  SmallVector<AffineExpr, 4> exprs;
  ArrayRef<AffineExpr> exprList = unwrapList(nAffineExprs, affineExprs, exprs);
  return wrap(AffineMap::get(dimCount, symbolCount, exprList, unwrap(ctx)));
}

MlirAffineMap mlirAffineMapConstantGet(MlirContext ctx, int64_t val) {
  return wrap(AffineMap::getConstantMap(val, unwrap(ctx)));
}

MlirAffineMap mlirAffineMapMultiDimIdentityGet(MlirContext ctx,
                                               intptr_t numDims) {
  return wrap(AffineMap::getMultiDimIdentityMap(numDims, unwrap(ctx)));
}

MlirAffineMap mlirAffineMapMinorIdentityGet(MlirContext ctx, intptr_t dims,
                                            intptr_t results) {
  return wrap(AffineMap::getMinorIdentityMap(dims, results, unwrap(ctx)));
}

MlirAffineMap mlirAffineMapPermutationGet(MlirContext ctx, intptr_t size,
                                          unsigned *permutation) {
  return wrap(AffineMap::getPermutationMap(
      llvm::makeArrayRef(permutation, static_cast<size_t>(size)), unwrap(ctx)));
}

bool mlirAffineMapIsIdentity(MlirAffineMap affineMap) {
  return unwrap(affineMap).isIdentity();
}

bool mlirAffineMapIsMinorIdentity(MlirAffineMap affineMap) {
  return unwrap(affineMap).isMinorIdentity();
}

bool mlirAffineMapIsEmpty(MlirAffineMap affineMap) {
  return unwrap(affineMap).isEmpty();
}

bool mlirAffineMapIsSingleConstant(MlirAffineMap affineMap) {
  return unwrap(affineMap).isSingleConstant();
}

int64_t mlirAffineMapGetSingleConstantResult(MlirAffineMap affineMap) {
  return unwrap(affineMap).getSingleConstantResult();
}

intptr_t mlirAffineMapGetNumDims(MlirAffineMap affineMap) {
  return unwrap(affineMap).getNumDims();
}

intptr_t mlirAffineMapGetNumSymbols(MlirAffineMap affineMap) {
  return unwrap(affineMap).getNumSymbols();
}

intptr_t mlirAffineMapGetNumResults(MlirAffineMap affineMap) {
  return unwrap(affineMap).getNumResults();
}

MlirAffineExpr mlirAffineMapGetResult(MlirAffineMap affineMap, intptr_t pos) {
  return wrap(unwrap(affineMap).getResult(static_cast<unsigned>(pos)));
}

intptr_t mlirAffineMapGetNumInputs(MlirAffineMap affineMap) {
  return unwrap(affineMap).getNumInputs();
}

bool mlirAffineMapIsProjectedPermutation(MlirAffineMap affineMap) {
  return unwrap(affineMap).isProjectedPermutation();
}

bool mlirAffineMapIsPermutation(MlirAffineMap affineMap) {
  return unwrap(affineMap).isPermutation();
}

MlirAffineMap mlirAffineMapGetSubMap(MlirAffineMap affineMap, intptr_t size,
                                     intptr_t *resultPos) {
  SmallVector<unsigned, 8> pos;
  pos.reserve(size);
  for (intptr_t i = 0; i < size; ++i)
    pos.push_back(static_cast<unsigned>(resultPos[i]));
  return wrap(unwrap(affineMap).getSubMap(pos));
}

MlirAffineMap mlirAffineMapGetMajorSubMap(MlirAffineMap affineMap,
                                          intptr_t numResults) {
  return wrap(unwrap(affineMap).getMajorSubMap(numResults));
}

MlirAffineMap mlirAffineMapGetMinorSubMap(MlirAffineMap affineMap,
                                          intptr_t numResults) {
  return wrap(unwrap(affineMap).getMinorSubMap(numResults));
}

MlirAffineMap mlirAffineMapReplace(MlirAffineMap affineMap,
                                   MlirAffineExpr expression,
                                   MlirAffineExpr replacement,
                                   intptr_t numResultDims,
                                   intptr_t numResultSyms) {
  return wrap(unwrap(affineMap).replace(unwrap(expression), unwrap(replacement),
                                        numResultDims, numResultSyms));
}

void mlirAffineMapCompressUnusedSymbols(
    MlirAffineMap *affineMaps, intptr_t size, void *result,
    void (*populateResult)(void *res, intptr_t idx, MlirAffineMap m)) {
  SmallVector<AffineMap> maps;
  for (intptr_t idx = 0; idx < size; ++idx)
    maps.push_back(unwrap(affineMaps[idx]));
  intptr_t idx = 0;
  for (auto m : mlir::compressUnusedSymbols(maps))
    populateResult(result, idx++, wrap(m));
}
