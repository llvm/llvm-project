//===- llvm.c - Test of llvm APIs -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-llvm-test 2>&1 | FileCheck %s

#include "llvm-c/Core.h"
#include "llvm-c/Support.h"
#include "llvm-c/Types.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Target/LLVMIR.h"

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

  const char *ptr_text = "!llvm.ptr";
  MlirType ptr = mlirLLVMPointerTypeGet(ctx, 0);
  MlirType ptr_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(ptr_text));
  // CHECK: !llvm.ptr: 1
  fprintf(stderr, "%s: %d\n", ptr_text, mlirTypeEqual(ptr, ptr_ref));

  const char *ptr_addr_text = "!llvm.ptr<42>";
  MlirType ptr_addr = mlirLLVMPointerTypeGet(ctx, 42);
  MlirType ptr_addr_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(ptr_addr_text));
  // CHECK: !llvm.ptr<42>: 1
  fprintf(stderr, "%s: %d\n", ptr_addr_text,
          mlirTypeEqual(ptr_addr, ptr_addr_ref));

  const char *voidt_text = "!llvm.void";
  MlirType voidt = mlirLLVMVoidTypeGet(ctx);
  MlirType voidt_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(voidt_text));
  // CHECK: !llvm.void: 1
  fprintf(stderr, "%s: %d\n", voidt_text, mlirTypeEqual(voidt, voidt_ref));

  const char *i32_4_text = "!llvm.array<4 x i32>";
  MlirType i32_4 = mlirLLVMArrayTypeGet(i32, 4);
  MlirType i32_4_ref =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_4_text));
  // CHECK: !llvm.array<4 x i32>: 1
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

// CHECK-LABEL: testToLLVMIR()
static void testToLLVMIR(MlirContext ctx) {
  fprintf(stderr, "testToLLVMIR()\n");
  LLVMContextRef llvmCtx = LLVMContextCreate();

  const char *moduleString = "llvm.func @add(%arg0: i64, %arg1: i64) -> i64 { \
                                %0 = llvm.add %arg0, %arg1  : i64 \
                                llvm.return %0 : i64 \
                             }";

  mlirRegisterAllLLVMTranslations(ctx);

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));

  MlirOperation operation = mlirModuleGetOperation(module);

  LLVMModuleRef llvmModule = mlirTranslateModuleToLLVMIR(operation, llvmCtx);

  // clang-format off
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK-NEXT: source_filename = "LLVMDialectModule"
  // CHECK: declare ptr @malloc(i64 %0)
  // CHECK: declare void @free(ptr %0)
  // CHECK: define i64 @add(i64 %0, i64 %1) {
  // CHECK-NEXT:   %3 = add i64 %0, %1
  // CHECK-NEXT:   ret i64 %3
  // CHECK-NEXT: }
  // clang-format on
  LLVMDumpModule(llvmModule);

  LLVMDisposeModule(llvmModule);
  mlirModuleDestroy(module);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
  mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("llvm"));
  testTypeCreation(ctx);
  testToLLVMIR(ctx);
  mlirContextDestroy(ctx);
  return 0;
}
