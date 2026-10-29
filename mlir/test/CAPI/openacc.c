//===- openacc.c - OpenACC dialect registration tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-openacc-test | FileCheck %s

#include "mlir-c/Dialect/OpenACC.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include <stdio.h>

int main(void) {
  for (int useRegistry = 0; useRegistry != 2; ++useRegistry) {
    MlirContext context = mlirContextCreate();
    MlirDialectHandle acc = mlirGetDialectHandle__acc__();
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), context);
    if (useRegistry)
      mlirDialectHandleRegisterDialect(acc, context);
    else
      mlirDialectHandleLoadDialect(acc, context);

    // Verifying the copyin requires the standard MemRef pointer model even
    // though the input has no MemRef operations.
    MlirModule module = mlirModuleCreateParse(
        context,
        mlirStringRefCreateFromCString(
            "module { func.func @f(%arg : memref<f32>) { "
            "%0 = acc.copyin varPtr(%arg : memref<f32>) -> memref<f32> "
            "return } }"));
    if (mlirModuleIsNull(module) ||
        !mlirOperationVerify(mlirModuleGetOperation(module)))
      return 1;

    MlirAttribute space = mlirAttributeParseGet(
        context,
        mlirStringRefCreateFromCString("#gpu.address_space<workgroup>"));
    MlirAttribute dim = mlirAttributeParseGet(
        context, mlirStringRefCreateFromCString("#acc.par_dim<thread_y>"));
    MlirType token = mlirTypeParseGet(
        context, mlirStringRefCreateFromCString("!gpu.async.token"));
    if (mlirAttributeIsNull(space) || mlirAttributeIsNull(dim) ||
        mlirTypeIsNull(token))
      return 2;
    mlirModuleDestroy(module);
    mlirContextDestroy(context);
  }
  // CHECK: OpenACC C API registration passed
  puts("OpenACC C API registration passed");
  return 0;
}
