//===- translation.c - Test MLIR Target translations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-translation-test 2>&1 | FileCheck %s

#include "llvm-c/Core.h"
#include "llvm-c/Support.h"
#include "llvm-c/Types.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"
#include "mlir-c/Target/LLVMIR.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  // CHECK: define i64 @add(i64 %[[arg1:.*]], i64 %[[arg2:.*]]) {
  // CHECK-NEXT:   %[[arg3:.*]] = add i64 %[[arg1]], %[[arg2]]
  // CHECK-NEXT:   ret i64 %[[arg3]]
  // CHECK-NEXT: }
  // clang-format on
  LLVMDumpModule(llvmModule);

  LLVMDisposeModule(llvmModule);
  mlirModuleDestroy(module);
  LLVMContextDispose(llvmCtx);
}

// CHECK-LABEL: testTypeToFromLLVMIRTranslator
static void testTypeToFromLLVMIRTranslator(MlirContext ctx) {
  fprintf(stderr, "testTypeToFromLLVMIRTranslator\n");
  LLVMContextRef llvmCtx = LLVMContextCreate();

  LLVMTypeRef llvmTy = LLVMInt32TypeInContext(llvmCtx);
  MlirTypeFromLLVMIRTranslator fromLLVMTranslator =
      mlirTypeFromLLVMIRTranslatorCreate(ctx);
  MlirType mlirTy =
      mlirTypeFromLLVMIRTranslatorTranslateType(fromLLVMTranslator, llvmTy);
  // CHECK: i32
  mlirTypeDump(mlirTy);

  MlirTypeToLLVMIRTranslator toLLVMTranslator =
      mlirTypeToLLVMIRTranslatorCreate(llvmCtx);
  LLVMTypeRef llvmTy2 =
      mlirTypeToLLVMIRTranslatorTranslateType(toLLVMTranslator, mlirTy);
  // CHECK: i32
  LLVMDumpType(llvmTy2);
  fprintf(stderr, "\n");

  mlirTypeFromLLVMIRTranslatorDestroy(fromLLVMTranslator);
  mlirTypeToLLVMIRTranslatorDestroy(toLLVMTranslator);
  LLVMContextDispose(llvmCtx);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
  mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("llvm"));
  testToLLVMIR(ctx);
  testTypeToFromLLVMIRTranslator(ctx);
  mlirContextDestroy(ctx);
  return 0;
}
