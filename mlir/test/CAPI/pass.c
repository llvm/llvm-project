/*===- pass.c - Simple test of C APIs -------------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: mlir-capi-pass-test 2>&1 | FileCheck %s
 */

#include "mlir-c/Pass.h"
#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir-c/Transforms.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void testRunPassOnModule() {
  MlirContext ctx = mlirContextCreate();
  mlirRegisterAllDialects(ctx);

  MlirModule module =
      mlirModuleCreateParse(ctx,
                            // clang-format off
"func @foo(%arg0 : i32) -> i32 {                                            \n"
"  %res = addi %arg0, %arg0 : i32                                           \n"
"  return %res : i32                                                        \n"
"}");
  // clang-format on
  if (mlirModuleIsNull(module))
    exit(EXIT_FAILURE);

  // Run the print-op-stats pass on the top-level module:
  // CHECK-LABEL: Operations encountered:
  // CHECK: func              , 1
  // CHECK: module_terminator , 1
  // CHECK: std.addi          , 1
  // CHECK: std.return        , 1
  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirPass printOpStatPass = mlirCreateTransformsPrintOpStats();
    mlirPassManagerAddOwnedPass(pm, printOpStatPass);
    MlirLogicalResult success = mlirPassManagerRun(pm, module);
    if (mlirLogicalResultIsFailure(success))
      exit(EXIT_FAILURE);
    mlirPassManagerDestroy(pm);
  }
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

void testRunPassOnNestedModule() {
  MlirContext ctx = mlirContextCreate();
  mlirRegisterAllDialects(ctx);

  MlirModule module =
      mlirModuleCreateParse(ctx,
                            // clang-format off
"func @foo(%arg0 : i32) -> i32 {                                            \n"
"  %res = addi %arg0, %arg0 : i32                                           \n"
"  return %res : i32                                                        \n"
"}                                                                          \n"
"module {                                                                   \n"
"  func @bar(%arg0 : f32) -> f32 {                                          \n"
"    %res = addf %arg0, %arg0 : f32                                         \n"
"    return %res : f32                                                      \n"
"  }                                                                        \n"
"}");
  // clang-format on
  if (mlirModuleIsNull(module))
    exit(1);

  // Run the print-op-stats pass on functions under the top-level module:
  // CHECK-LABEL: Operations encountered:
  // CHECK-NOT: module_terminator
  // CHECK: func              , 1
  // CHECK: std.addi          , 1
  // CHECK: std.return        , 1
  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirOpPassManager nestedFuncPm = mlirPassManagerGetNestedUnder(
        pm, mlirStringRefCreateFromCString("func"));
    MlirPass printOpStatPass = mlirCreateTransformsPrintOpStats();
    mlirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);
    MlirLogicalResult success = mlirPassManagerRun(pm, module);
    if (mlirLogicalResultIsFailure(success))
      exit(2);
    mlirPassManagerDestroy(pm);
  }
  // Run the print-op-stats pass on functions under the nested module:
  // CHECK-LABEL: Operations encountered:
  // CHECK-NOT: module_terminator
  // CHECK: func              , 1
  // CHECK: std.addf          , 1
  // CHECK: std.return        , 1
  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirOpPassManager nestedModulePm = mlirPassManagerGetNestedUnder(
        pm, mlirStringRefCreateFromCString("module"));
    MlirOpPassManager nestedFuncPm = mlirOpPassManagerGetNestedUnder(
        nestedModulePm, mlirStringRefCreateFromCString("func"));
    MlirPass printOpStatPass = mlirCreateTransformsPrintOpStats();
    mlirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);
    MlirLogicalResult success = mlirPassManagerRun(pm, module);
    if (mlirLogicalResultIsFailure(success))
      exit(2);
    mlirPassManagerDestroy(pm);
  }

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main() {
  testRunPassOnModule();
  testRunPassOnNestedModule();
  return 0;
}
