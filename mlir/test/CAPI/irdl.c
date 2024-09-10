//===- irdl.c - Test for the C bindings for IRDL registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-capi-irdl-test 2>&1 | FileCheck %s
 */

#include "mlir-c/Dialect/IRDL.h"
#include "mlir-c/IR.h"

const char irdlDialect[] = "\
  irdl.dialect @foo {\
    irdl.operation @op {\
      %i32 = irdl.is i32\
      irdl.results(%i32)\
    }\
  }\
  irdl.dialect @bar {\
    irdl.operation @op {\
      %i32 = irdl.is i32\
      irdl.operands(%i32)\
    }\
  }";

// CHECK:      module {
// CHECK-NEXT:   %[[RES:.*]] = "foo.op"() : () -> i32
// CHECK-NEXT:   "bar.op"(%[[RES]]) :  (i32) -> ()
// CHECK-NEXT: }
const char newDialectUsage[] = "\
  module {\
    %res = \"foo.op\"() : () -> i32\
    \"bar.op\"(%res) : (i32) -> ()\
  }";

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__irdl__(), ctx);

  MlirModule dialectDecl =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(irdlDialect));

  mlirLoadIRDLDialects(dialectDecl);
  mlirModuleDestroy(dialectDecl);

  MlirModule usingModule = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(newDialectUsage));

  mlirOperationDump(mlirModuleGetOperation(usingModule));

  mlirModuleDestroy(usingModule);
  mlirContextDestroy(ctx);
  return 0;
}
