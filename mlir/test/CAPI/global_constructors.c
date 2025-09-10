//===- global_constructors.c - Test JIT with the global constructors ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: target=aarch64{{.*}}, target=arm64{{.*}}
/* RUN: mlir-capi-global-constructors-test 2>&1 | FileCheck %s
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

// Helper variable to track callback invocations
static int initCnt = 0;

// Callback function that will be called during JIT initialization
static void initCallback(void) { initCnt += 1; }

// CHECK-LABEL: Running test 'testGlobalCtorJitCallback'
void testGlobalCtorJitCallback(void) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  // Create module with global constructor that calls our callback
  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
               // clang-format off
"module {                                                                       \n"
"  llvm.mlir.global_ctors ctors = [@ctor], priorities = [0 : i32], data = [#llvm.zero] \n"
"  llvm.func @ctor() {                                                          \n"
"    func.call @init_callback() : () -> ()                                      \n"
"    llvm.return                                                                \n"
"  }                                                                            \n"
"  func.func private @init_callback() attributes { llvm.emit_c_interface }      \n"
"}                                                                              \n"
               // clang-format on
               ));

  lowerModuleToLLVM(ctx, module);
  mlirRegisterAllLLVMTranslations(ctx);

  // Create execution engine with initialization disabled
  MlirExecutionEngine jit = mlirExecutionEngineCreate(
      module, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL,
      /*enableObjectDump=*/false);

  if (mlirExecutionEngineIsNull(jit)) {
    fprintf(stderr, "Execution engine creation failed");
    exit(2);
  }

  // Register callback symbol before initialization
  mlirExecutionEngineRegisterSymbol(
      jit, mlirStringRefCreateFromCString("_mlir_ciface_init_callback"),
      (void *)(uintptr_t)initCallback);

  mlirExecutionEngineInitialize(jit);

  // CHECK: Init count: 1
  printf("Init count: %d\n", initCnt);

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
  TEST(testGlobalCtorJitCallback);
  return 0;
}
