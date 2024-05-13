//===- standalone-cap-demo.c - Simple demo of C-API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: standalone-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "Standalone-c/Dialects.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"

static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  // TODO: Create the dialect handles for the builtin dialects and avoid this.
  // This adds dozens of MB of binary size over just the standalone dialect.
  registerAllUpstreamDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__standalone__(), ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString("%0 = arith.constant 2 : i32\n"
                                          "%1 = standalone.foo %0 : i32\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 2 : i32
  // CHECK: standalone.foo %[[C]] : i32
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
