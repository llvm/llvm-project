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
#include "mlir-c/Support.h"

const char irdlDialect[] = "\
  irdl.dialect @foo {\
    irdl.operation @op {\
      %i32 = irdl.is i32\
      irdl.results(baz: %i32)\
    }\
  }\
  irdl.dialect @bar {\
    irdl.operation @op {\
      %i32 = irdl.is i32\
      irdl.operands(baz: %i32)\
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

void testVariadicityAttributes(MlirContext ctx) {
  MlirAttribute variadicitySingle =
      mlirIRDLVariadicityAttrGet(ctx, mlirStringRefCreateFromCString("single"));

  // CHECK: #irdl<variadicity single>
  mlirAttributeDump(variadicitySingle);

  MlirAttribute variadicityOptional = mlirIRDLVariadicityAttrGet(
      ctx, mlirStringRefCreateFromCString("optional"));

  // CHECK: #irdl<variadicity optional>
  mlirAttributeDump(variadicityOptional);

  MlirAttribute variadicityVariadic = mlirIRDLVariadicityAttrGet(
      ctx, mlirStringRefCreateFromCString("variadic"));

  // CHECK: #irdl<variadicity variadic>
  mlirAttributeDump(variadicityVariadic);

  MlirAttribute variadicities[] = {variadicitySingle, variadicityOptional,
                                   variadicityVariadic};
  MlirAttribute variadicityArray =
      mlirIRDLVariadicityArrayAttrGet(ctx, 3, variadicities);

  // CHECK: #irdl<variadicity_array[ single, optional,  variadic]>
  mlirAttributeDump(variadicityArray);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__irdl__(), ctx);

  // Test loading an IRDL dialect and using it.
  MlirModule dialectDecl =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(irdlDialect));

  mlirLoadIRDLDialects(dialectDecl);
  mlirModuleDestroy(dialectDecl);

  MlirModule usingModule = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(newDialectUsage));

  mlirOperationDump(mlirModuleGetOperation(usingModule));

  mlirModuleDestroy(usingModule);

  // Test variadicity attributes.
  testVariadicityAttributes(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
