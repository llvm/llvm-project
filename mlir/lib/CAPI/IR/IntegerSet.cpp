//===- IntegerSet.cpp - C API for MLIR Integer Sets -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IntegerSet.h"
#include "mlir-c/AffineExpr.h"
#include "mlir/CAPI/AffineExpr.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/IntegerSet.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/IntegerSet.h"

using namespace mlir;

MlirContext mlirIntegerSetGetContext(MlirIntegerSet set) {
  return wrap(unwrap(set).getContext());
}

bool mlirIntegerSetEqual(MlirIntegerSet s1, MlirIntegerSet s2) {
  return unwrap(s1) == unwrap(s2);
}

void mlirIntegerSetPrint(MlirIntegerSet set, MlirStringCallback callback,
                         void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  unwrap(set).print(stream);
}

void mlirIntegerSetDump(MlirIntegerSet set) { unwrap(set).dump(); }

MlirIntegerSet mlirIntegerSetEmptyGet(MlirContext context, intptr_t numDims,
                                      intptr_t numSymbols) {
  return wrap(IntegerSet::getEmptySet(static_cast<unsigned>(numDims),
                                      static_cast<unsigned>(numSymbols),
                                      unwrap(context)));
}

MlirIntegerSet mlirIntegerSetGet(MlirContext context, intptr_t numDims,
                                 intptr_t numSymbols, intptr_t numConstraints,
                                 const MlirAffineExpr *constraints,
                                 const bool *eqFlags) {
  SmallVector<AffineExpr> mlirConstraints;
  (void)unwrapList(static_cast<size_t>(numConstraints), constraints,
                   mlirConstraints);
  return wrap(IntegerSet::get(
      static_cast<unsigned>(numDims), static_cast<unsigned>(numSymbols),
      mlirConstraints,
      llvm::makeArrayRef(eqFlags, static_cast<size_t>(numConstraints))));
}

MlirIntegerSet
mlirIntegerSetReplaceGet(MlirIntegerSet set,
                         const MlirAffineExpr *dimReplacements,
                         const MlirAffineExpr *symbolReplacements,
                         intptr_t numResultDims, intptr_t numResultSymbols) {
  SmallVector<AffineExpr> mlirDims, mlirSymbols;
  (void)unwrapList(unwrap(set).getNumDims(), dimReplacements, mlirDims);
  (void)unwrapList(unwrap(set).getNumSymbols(), symbolReplacements,
                   mlirSymbols);
  return wrap(unwrap(set).replaceDimsAndSymbols(
      mlirDims, mlirSymbols, static_cast<unsigned>(numResultDims),
      static_cast<unsigned>(numResultSymbols)));
}

bool mlirIntegerSetIsCanonicalEmpty(MlirIntegerSet set) {
  return unwrap(set).isEmptyIntegerSet();
}

intptr_t mlirIntegerSetGetNumDims(MlirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumDims());
}

intptr_t mlirIntegerSetGetNumSymbols(MlirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumSymbols());
}

intptr_t mlirIntegerSetGetNumInputs(MlirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumInputs());
}

intptr_t mlirIntegerSetGetNumConstraints(MlirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumConstraints());
}

intptr_t mlirIntegerSetGetNumEqualities(MlirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumEqualities());
}

intptr_t mlirIntegerSetGetNumInequalities(MlirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumInequalities());
}

MlirAffineExpr mlirIntegerSetGetConstraint(MlirIntegerSet set, intptr_t pos) {
  return wrap(unwrap(set).getConstraint(static_cast<unsigned>(pos)));
}

bool mlirIntegerSetIsConstraintEq(MlirIntegerSet set, intptr_t pos) {
  return unwrap(set).isEq(pos);
}
