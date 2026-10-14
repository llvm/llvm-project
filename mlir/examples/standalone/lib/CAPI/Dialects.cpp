//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/Dialects.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneTypes.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Standalone, standalone,
                                      mlir::standalone::StandaloneDialect)

MlirType mlirStandaloneCustomTypeGet(MlirContext ctx, MlirStringRef value) {
  return wrap(mlir::standalone::CustomType::get(unwrap(ctx), unwrap(value)));
}

bool mlirStandaloneTypeIsACustomType(MlirType t) {
  return llvm::isa<mlir::standalone::CustomType>(unwrap(t));
}

MlirTypeID mlirStandaloneCustomTypeGetTypeID() {
  return wrap(mlir::standalone::CustomType::getTypeID());
}
