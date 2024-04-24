//===- irdl.c - Test of IRDL dialect C API --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-irdl-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/IRDL.h"
#include "mlir-c/IR.h"

#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testLoadDialect
int testLoadDialect(MlirContext ctx) {
  fprintf(stderr, "testLoadDialect\n");

  const char *moduleString = "irdl.dialect @cmath {"
                             "  irdl.type @complex {"
                             "    %0 = irdl.is f32"
                             "    %1 = irdl.is f64"
                             "    %2 = irdl.any_of(%0, %1)"
                             "    irdl.parameters(%2)"
                             "  }"
                             "  irdl.operation @mul {"
                             "    %0 = irdl.is f32"
                             "    %1 = irdl.is f64"
                             "    %2 = irdl.any_of(%0, %1)"
                             "    %3 = irdl.parametric @complex<%2>"
                             "    irdl.operands(%3, %3)"
                             "    irdl.results(%3)"
                             "  }"
                             "}";

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));

  if (mlirModuleIsNull(module))
    return 1;

  MlirLogicalResult result = mlirIRDLLoadDialects(module);
  if (mlirLogicalResultIsFailure(result))
    return 2;

  if (!mlirContextIsRegisteredOperation(
          ctx, mlirStringRefCreateFromCString("cmath.mul")))
    return 3;

  mlirModuleDestroy(module);

  return 0;
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  if (mlirContextIsNull(ctx))
    return 1;
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__irdl__(), ctx);
  int result = testLoadDialect(ctx);
  mlirContextDestroy(ctx);
  if (result)
    return result;

  return EXIT_SUCCESS;
}
