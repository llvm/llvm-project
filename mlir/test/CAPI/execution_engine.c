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
/* REQUIRES: host-supports-jit
 */

#include "mlir-c/Conversion.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

void lowerModuleToLLVM(MlirContext ctx, MlirModule module) {
  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirOpPassManager opm = mlirPassManagerGetNestedUnder(
      pm, mlirStringRefCreateFromCString("func.func"));
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
  mlirOpPassManagerAddOwnedPass(
      opm, mlirCreateConversionArithToLLVMConversionPass());
  MlirLogicalResult status =
      mlirPassManagerRunOnOp(pm, mlirModuleGetOperation(module));
  if (mlirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Unexpected failure running pass pipeline\n");
    exit(2);
  }
  mlirPassManagerDestroy(pm);
}

// CHECK-LABEL: Running test 'testSimpleExecution'
void testSimpleExecution(void) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
               // clang-format off
"module {                                                                    \n"
"  func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {     \n"
"    %res = arith.addi %arg0, %arg0 : i32                                        \n"
"    return %res : i32                                                           \n"
"  }                                                                             \n"
"}"));
  // clang-format on
  lowerModuleToLLVM(ctx, module);
  mlirRegisterAllLLVMTranslations(ctx);
  MlirExecutionEngine jit = mlirExecutionEngineCreate(
      module, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL,
      /*enableObjectDump=*/false);
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

// CHECK-LABEL: Running test 'testOmpCreation'
void testOmpCreation(void) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
               // clang-format off
"module {                                                                       \n"
"  func.func @main() attributes { llvm.emit_c_interface } {                     \n"
"    %0 = arith.constant 0 : i32                                                \n"
"    %1 = arith.constant 1 : i32                                                \n"
"    %2 = arith.constant 2 : i32                                                \n"
"    omp.parallel {                                                             \n"
"      omp.wsloop for (%3) : i32 = (%0) to (%2) step (%1) {                     \n"
"        omp.yield                                                              \n"
"      }                                                                        \n"
"      omp.terminator                                                           \n"
"    }                                                                          \n"
"    llvm.return                                                                \n"
"  }                                                                            \n"
"}                                                                              \n"
      ));
  // clang-format on
  lowerModuleToLLVM(ctx, module);

  // At this point all operations in the MLIR module have been lowered to the
  // 'llvm' dialect except 'omp' operations. The goal of this test is
  // guaranteeing that the execution engine C binding has registered OpenMP
  // translations and therefore does not fail when it encounters 'omp' ops.
  // We don't attempt to run the engine, since that would force us to link
  // against the OpenMP library.
  MlirExecutionEngine jit = mlirExecutionEngineCreate(
      module, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL,
      /*enableObjectDump=*/false);
  if (mlirExecutionEngineIsNull(jit)) {
    fprintf(stderr, "Engine creation failed with OpenMP");
    exit(2);
  }
  // CHECK: Engine creation succeeded with OpenMP
  printf("Engine creation succeeded with OpenMP\n");
  mlirExecutionEngineDestroy(jit);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main(void) {

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define TEST(test)                                                             \
  printf("Running test '" STRINGIFY(test) "'\n");                              \
  test();

  TEST(testSimpleExecution);
  TEST(testOmpCreation);
  return 0;
}
