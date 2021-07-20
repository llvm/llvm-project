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
  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);

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

  const char *voidt_text = "!llvm.void";
  MlirType voidt = mlirLLVMVoidTypeGet(ctx);
  MlirType voidt_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(voidt_text));
  // CHECK: !llvm.void: 1
  fprintf(stderr, "%s: %d\n", voidt_text, mlirTypeEqual(voidt, voidt_ref));

  const char *i32_4_text = "!llvm.array<4xi32>";
  MlirType i32_4 = mlirLLVMArrayTypeGet(i32, 4);
  MlirType i32_4_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_4_text));
  // CHECK: !llvm.array<4xi32>: 1
  fprintf(stderr, "%s: %d\n", i32_4_text, mlirTypeEqual(i32_4, i32_4_ref));

  const char *i8_i32_i64_text = "!llvm.func<i8 (i32, i64)>";
  const MlirType i32_i64_arr[] = {i32, i64};
  MlirType i8_i32_i64 = mlirLLVMFunctionTypeGet(i8, 2, i32_i64_arr, false);
  MlirType i8_i32_i64_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i8_i32_i64_text));
  // CHECK: !llvm.func<i8 (i32, i64)>: 1
  fprintf(stderr, "%s: %d\n", i8_i32_i64_text,
          mlirTypeEqual(i8_i32_i64, i8_i32_i64_ref));

  const char *i32_i64_s_text = "!llvm.struct<(i32, i64)>";
  MlirType i32_i64_s = mlirLLVMStructTypeLiteralGet(ctx, 2, i32_i64_arr, false);
  MlirType i32_i64_s_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_i64_s_text));
  // CHECK: !llvm.struct<(i32, i64)>: 1
  fprintf(stderr, "%s: %d\n", i32_i64_s_text,
          mlirTypeEqual(i32_i64_s, i32_i64_s_ref));
}

int main() {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
  mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("llvm"));
  testTypeCreation(ctx);
  mlirContextDestroy(ctx);
  return 0;
}

