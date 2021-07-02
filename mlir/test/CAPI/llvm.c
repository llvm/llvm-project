//===- llvm.c - Test of llvm APIs -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-llvm-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/BuiltinTypes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CHECK-LABEL: testTypeCreation()
static void testTypeCreation(MlirContext ctx) {
  fprintf(stderr, "testTypeCreation()\n");
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);

  const char *i32p_text = "!llvm.ptr<i32>";
  MlirType i32p = mlirLLVMPointerTypeGet(i32, 0);
  MlirType i32p_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32p_text));
  // CHECK: !llvm.ptr<i32>: 1
  fprintf(stderr, "%s: %d\n", i32p_text, mlirTypeEqual(i32p, i32p_ref));

  const char *i32p4_text = "!llvm.ptr<i32, 4>";
  MlirType i32p4 = mlirLLVMPointerTypeGet(i32, 4);
  MlirType i32p4_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32p4_text));
  // CHECK: !llvm.ptr<i32, 4>: 1
  fprintf(stderr, "%s: %d\n", i32p4_text, mlirTypeEqual(i32p4, i32p4_ref));
}

int main() {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
  mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("llvm"));
  testTypeCreation(ctx);
  mlirContextDestroy(ctx);
  return 0;
}

