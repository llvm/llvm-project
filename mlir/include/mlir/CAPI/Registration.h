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
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

//===----------------------------------------------------------------------===//
// Corrolary to MLIR_DECLARE_CAPI_DIALECT_REGISTRATION that defines an impl.
// Takes the same name passed to the above and the fully qualified class name
// of the dialect class.
//===----------------------------------------------------------------------===//

/// Hooks for dynamic discovery of dialects.
typedef void (*MlirDialectRegistryInsertDialectHook)(
    MlirDialectRegistry registry);
typedef MlirDialect (*MlirContextLoadDialectHook)(MlirContext context);
typedef MlirStringRef (*MlirDialectGetNamespaceHook)();

/// Structure of dialect registration hooks.
struct MlirDialectRegistrationHooks {
  MlirDialectRegistryInsertDialectHook insertHook;
  MlirContextLoadDialectHook loadHook;
  MlirDialectGetNamespaceHook getNamespaceHook;
};
typedef struct MlirDialectRegistrationHooks MlirDialectRegistrationHooks;

#define MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Name, Namespace, ClassName)      \
  static void mlirDialectRegistryInsert##Name##Dialect(                        \
      MlirDialectRegistry registry) {                                          \
    unwrap(registry)->insert<ClassName>();                                     \
  }                                                                            \
  static MlirDialect mlirContextLoad##Name##Dialect(MlirContext context) {     \
    return wrap(unwrap(context)->getOrLoadDialect<ClassName>());               \
  }                                                                            \
  static MlirStringRef mlir##Name##DialectGetNamespace() {                     \
    return wrap(ClassName::getDialectNamespace());                             \
  }                                                                            \
  MlirDialectHandle mlirGetDialectHandle__##Namespace##__() {                  \
    static MlirDialectRegistrationHooks hooks = {                              \
        mlirDialectRegistryInsert##Name##Dialect,                              \
        mlirContextLoad##Name##Dialect, mlir##Name##DialectGetNamespace};      \
    return MlirDialectHandle{&hooks};                                          \
  }

#endif // MLIR_CAPI_REGISTRATION_H
