//===- DialectHandle.cpp - C Interface for MLIR Dialect Operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Registration.h"

static inline const MlirDialectRegistrationHooks *
unwrap(MlirDialectHandle handle) {
  return (const MlirDialectRegistrationHooks *)handle.ptr;
}

MlirStringRef mlirDialectHandleGetNamespace(MlirDialectHandle handle) {
  return unwrap(handle)->getNamespaceHook();
}

void mlirDialectHandleRegisterDialect(MlirDialectHandle handle,
                                      MlirContext ctx) {
  unwrap(handle)->registerHook(ctx);
}

MlirDialect mlirDialectHandleLoadDialect(MlirDialectHandle handle,
                                         MlirContext ctx) {
  return unwrap(handle)->loadHook(ctx);
}
