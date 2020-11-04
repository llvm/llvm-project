//===- AffineMap.cpp - C API for MLIR Affine Maps -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/AffineMap.h"

// TODO: expose the C API related to `AffineExpr` and mutable affine map.

using namespace mlir;

MlirContext mlirAffineMapGetContext(MlirAffineMap affineMap) {
  return wrap(unwrap(affineMap).getContext());
}

int mlirAffineMapEqual(MlirAffineMap a1, MlirAffineMap a2) {
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

MlirAffineMap mlirAffineMapGet(MlirContext ctx, intptr_t dimCount,
                               intptr_t symbolCount) {
  return wrap(AffineMap::get(dimCount, symbolCount, unwrap(ctx)));
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

int mlirAffineMapIsIdentity(MlirAffineMap affineMap) {
  return unwrap(affineMap).isIdentity();
}

int mlirAffineMapIsMinorIdentity(MlirAffineMap affineMap) {
  return unwrap(affineMap).isMinorIdentity();
}

int mlirAffineMapIsEmpty(MlirAffineMap affineMap) {
  return unwrap(affineMap).isEmpty();
}

int mlirAffineMapIsSingleConstant(MlirAffineMap affineMap) {
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

intptr_t mlirAffineMapGetNumInputs(MlirAffineMap affineMap) {
  return unwrap(affineMap).getNumInputs();
}

int mlirAffineMapIsProjectedPermutation(MlirAffineMap affineMap) {
  return unwrap(affineMap).isProjectedPermutation();
}

int mlirAffineMapIsPermutation(MlirAffineMap affineMap) {
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
