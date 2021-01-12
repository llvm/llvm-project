//===- Registration.h - C API Registration implementation  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_REGISTRATION_H
#define MLIR_CAPI_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

//===----------------------------------------------------------------------===//
// Corrolary to MLIR_DECLARE_CAPI_DIALECT_REGISTRATION that defines an impl.
// Takes the same name passed to the above and the fully qualified class name
// of the dialect class.
//===----------------------------------------------------------------------===//

#define MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Name, Namespace, ClassName)      \
  void mlirContextRegister##Name##Dialect(MlirContext context) {               \
    unwrap(context)->getDialectRegistry().insert<ClassName>();                 \
  }                                                                            \
  MlirDialect mlirContextLoad##Name##Dialect(MlirContext context) {            \
    return wrap(unwrap(context)->getOrLoadDialect<ClassName>());               \
  }                                                                            \
  MlirStringRef mlir##Name##DialectGetNamespace() {                            \
    return wrap(ClassName::getDialectNamespace());                             \
  }                                                                            \
  const MlirDialectRegistrationHooks *mlirGetDialectHooks__##Namespace##__() { \
    static MlirDialectRegistrationHooks hooks = {                              \
        mlirContextRegister##Name##Dialect, mlirContextLoad##Name##Dialect,    \
        mlir##Name##DialectGetNamespace};                                      \
    return &hooks;                                                             \
  }

#endif // MLIR_CAPI_REGISTRATION_H
