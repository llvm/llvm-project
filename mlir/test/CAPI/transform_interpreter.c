//===- transform_interpreter.c - Test of the Transform interpreter C API --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-transform-interpreter-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/Dialect/Transform/Interpreter.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <stdio.h>
#include <stdlib.h>

int testApplyNamedSequence(MlirContext ctx) {
  fprintf(stderr, "%s\n", __func__);

  const char module[] =
      "module attributes {transform.with_named_sequence} {"
      "  transform.named_sequence @__transform_main(%root: !transform.any_op) {"
      "    transform.print %root { name = \"from interpreter\" }: "
      "!transform.any_op"
      "    transform.yield"
      "  }"
      "}";

  MlirStringRef moduleStringRef = mlirStringRefCreateFromCString(module);
  MlirStringRef nameStringRef = mlirStringRefCreateFromCString("inline-module");

  MlirOperation root =
      mlirOperationCreateParse(ctx, moduleStringRef, nameStringRef);
  if (mlirOperationIsNull(root))
    return 1;
  MlirBlock body = mlirRegionGetFirstBlock(mlirOperationGetRegion(root, 0));
  MlirOperation entry = mlirBlockGetFirstOperation(body);

  MlirTransformOptions options = mlirTransformOptionsCreate();
  mlirTransformOptionsEnableExpensiveChecks(options, true);
  mlirTransformOptionsEnforceSingleTopLevelTransformOp(options, true);

  MlirLogicalResult result =
      mlirTransformApplyNamedSequence(root, entry, root, options);
  mlirTransformOptionsDestroy(options);
  mlirOperationDestroy(root);
  if (mlirLogicalResultIsFailure(result))
    return 2;

  return 0;
}
// CHECK-LABEL: testApplyNamedSequence
// CHECK: from interpreter
// CHECK: transform.named_sequence @__transform_main
// CHECK:   transform.print %arg0
// CHECK:   transform.yield

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
  int result = testApplyNamedSequence(ctx);
  mlirContextDestroy(ctx);
  if (result)
    return result;

  return EXIT_SUCCESS;
}
