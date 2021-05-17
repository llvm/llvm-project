//===- execution_engine.c - Test for the C bindings for the MLIR JIT-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-capi-execution-engine-test 2>&1 | FileCheck %s
 */

#include "mlir-c/Conversion.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lowerModuleToLLVM(MlirContext ctx, MlirModule module) {
  MlirPassManager pm = mlirPassManagerCreate(ctx);
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertStandardToLLVM());
  MlirLogicalResult status = mlirPassManagerRun(pm, module);
  if (mlirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Unexpected failure running pass pipeline\n");
    exit(2);
  }
  mlirPassManagerDestroy(pm);
}

// CHECK-LABEL: Running test 'testSimpleExecution'
void testSimpleExecution() {
  MlirContext ctx = mlirContextCreate();
  mlirRegisterAllDialects(ctx);
  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
               // clang-format off
"module {                                                                   \n"
"  func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {     \n"
"    %res = std.addi %arg0, %arg0 : i32                                     \n"
"    return %res : i32                                                      \n"
"  }                                                                        \n"
"}"));
  // clang-format on
  lowerModuleToLLVM(ctx, module);
  mlirRegisterAllLLVMTranslations(ctx);
  MlirExecutionEngine jit = mlirExecutionEngineCreate(module, /*optLevel=*/2);
  if (mlirExecutionEngineIsNull(jit)) {
    fprintf(stderr, "Execution engine creation failed");
    exit(2);
  }
  int input = 42;
  int result = -1;
  void *args[2] = {&input, &result};
  if (mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
          jit, mlirStringRefCreateFromCString("add"), args))) {
    fprintf(stderr, "Execution engine creation failed");
    abort();
  }
  // CHECK: Input: 42 Result: 84
  printf("Input: %d Result: %d\n", input, result);
  mlirExecutionEngineDestroy(jit);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main() {

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define TEST(test)                                                             \
  printf("Running test '" STRINGIFY(test) "'\n");                              \
  test();

  TEST(testSimpleExecution);
  return 0;
}
