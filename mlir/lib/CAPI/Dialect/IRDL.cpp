//===- IRDL.cpp - C Interface for IRDL dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/IRDL.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IRDL, irdl, mlir::irdl::IRDLDialect)

MlirLogicalResult mlirLoadIRDLDialects(MlirModule module) {
  return wrap(mlir::irdl::loadDialects(unwrap(module)));
}

//===----------------------------------------------------------------------===//
// VariadicityAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirIRDLVariadicityAttrGet(MlirContext ctx, MlirStringRef value) {
  return wrap(mlir::irdl::VariadicityAttr::get(
      unwrap(ctx), mlir::irdl::symbolizeVariadicity(unwrap(value)).value()));
}

MlirStringRef mlirIRDLVariadicityAttrGetName(void) {
  return wrap(mlir::irdl::VariadicityAttr::name);
}

//===----------------------------------------------------------------------===//
// VariadicityArrayAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirIRDLVariadicityArrayAttrGet(MlirContext ctx, intptr_t nValues,
                                              MlirAttribute const *values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  llvm::ArrayRef<mlir::Attribute> unwrappedAttrs =
      unwrapList(nValues, values, attrs);

  llvm::SmallVector<mlir::irdl::VariadicityAttr> variadicities;
  for (auto attr : unwrappedAttrs)
    variadicities.push_back(llvm::cast<mlir::irdl::VariadicityAttr>(attr));

  return wrap(
      mlir::irdl::VariadicityArrayAttr::get(unwrap(ctx), variadicities));
}

MlirStringRef mlirIRDLVariadicityArrayAttrGetName(void) {
  return wrap(mlir::irdl::VariadicityArrayAttr::name);
}
