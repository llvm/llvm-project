//===- Dominance.cpp - C API for Dominance Analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dominance.h"
#include "mlir/CAPI/Dominance.h"
#include "mlir/CAPI/IR.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DominanceInfo API
//===----------------------------------------------------------------------===//

MlirDominanceInfo mlirDominanceInfoCreate(MlirOperation op) {
  return wrap(new DominanceInfo(unwrap(op)));
}

void mlirDominanceInfoDestroy(MlirDominanceInfo info) { delete unwrap(info); }

bool mlirDominanceInfoProperlyDominatesOperation(MlirDominanceInfo info,
                                                 MlirOperation a,
                                                 MlirOperation b) {
  return unwrap(info)->properlyDominates(unwrap(a), unwrap(b));
}

bool mlirDominanceInfoDominatesOperation(MlirDominanceInfo info,
                                         MlirOperation a, MlirOperation b) {
  return unwrap(info)->dominates(unwrap(a), unwrap(b));
}

bool mlirDominanceInfoValueProperlyDominates(MlirDominanceInfo info,
                                             MlirValue a, MlirOperation b) {
  return unwrap(info)->properlyDominates(unwrap(a), unwrap(b));
}

bool mlirDominanceInfoValueDominates(MlirDominanceInfo info, MlirValue a,
                                     MlirOperation b) {
  return unwrap(info)->dominates(unwrap(a), unwrap(b));
}

bool mlirDominanceInfoProperlyDominatesBlock(MlirDominanceInfo info,
                                             MlirBlock a, MlirBlock b) {
  return unwrap(info)->properlyDominates(unwrap(a), unwrap(b));
}

bool mlirDominanceInfoDominatesBlock(MlirDominanceInfo info, MlirBlock a,
                                     MlirBlock b) {
  return unwrap(info)->dominates(unwrap(a), unwrap(b));
}

MlirBlock mlirDominanceInfoFindNearestCommonDominator(MlirDominanceInfo info,
                                                      MlirBlock a,
                                                      MlirBlock b) {
  return wrap(unwrap(info)->findNearestCommonDominator(unwrap(a), unwrap(b)));
}

bool mlirDominanceInfoIsReachableFromEntry(MlirDominanceInfo info,
                                           MlirBlock block) {
  return unwrap(info)->isReachableFromEntry(unwrap(block));
}

void mlirDominanceInfoInvalidate(MlirDominanceInfo info) {
  unwrap(info)->invalidate();
}

//===----------------------------------------------------------------------===//
// PostDominanceInfo API
//===----------------------------------------------------------------------===//

MlirPostDominanceInfo mlirPostDominanceInfoCreate(MlirOperation op) {
  return wrap(new PostDominanceInfo(unwrap(op)));
}

void mlirPostDominanceInfoDestroy(MlirPostDominanceInfo info) {
  delete unwrap(info);
}

bool mlirPostDominanceInfoProperlyPostDominatesOperation(
    MlirPostDominanceInfo info, MlirOperation a, MlirOperation b) {
  return unwrap(info)->properlyPostDominates(unwrap(a), unwrap(b));
}

bool mlirPostDominanceInfoPostDominatesOperation(MlirPostDominanceInfo info,
                                                 MlirOperation a,
                                                 MlirOperation b) {
  return unwrap(info)->postDominates(unwrap(a), unwrap(b));
}

bool mlirPostDominanceInfoProperlyPostDominatesBlock(MlirPostDominanceInfo info,
                                                     MlirBlock a, MlirBlock b) {
  return unwrap(info)->properlyPostDominates(unwrap(a), unwrap(b));
}

bool mlirPostDominanceInfoPostDominatesBlock(MlirPostDominanceInfo info,
                                             MlirBlock a, MlirBlock b) {
  return unwrap(info)->postDominates(unwrap(a), unwrap(b));
}

void mlirPostDominanceInfoInvalidate(MlirPostDominanceInfo info) {
  unwrap(info)->invalidate();
}
