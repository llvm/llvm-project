//===- extensible_dialect.c - Test C API for extensible dialect creation
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-capi-extensible-dialect-test 2>&1 | FileCheck %s
 */

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/ExtensibleDialect.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <stdio.h>

static MlirStringRef strref(const char *s) {
  return mlirStringRefCreateFromCString(s);
}

static MlirLogicalResult successTypeVerify(intptr_t nParams,
                                           MlirAttribute const *params,
                                           void *userData) {
  (void)nParams;
  (void)params;
  (void)userData;
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult successAttrVerify(intptr_t nParams,
                                           MlirAttribute const *params,
                                           void *userData) {
  (void)nParams;
  (void)params;
  (void)userData;
  return mlirLogicalResultSuccess();
}

void testDynamicDialectCreation(MlirContext ctx) {
  // Create a dynamic dialect.
  MlirDynamicDialect testDialect =
      mlirDynamicDialectCreate(ctx, strref("test_dyn"));
  MlirDialect dialect = mlirDynamicDialectAsDialect(testDialect);

  // CHECK: test_dyn is extensible: 1
  fprintf(stderr, "test_dyn is extensible: %d\n",
          mlirDialectIsAExtensibleDialect(dialect));
}

void testDynamicOpCreation(MlirContext ctx) {
  MlirDynamicDialect testDialect =
      mlirDynamicDialectCreate(ctx, strref("testop"));

  // Define and register a simple op with no callbacks.
  MlirDynamicOpDefinitionCallbacks opCallbacks = {NULL, NULL, NULL, NULL};
  MlirDynamicOpDefinition opDef = mlirDynamicOpDefinitionCreate(
      testDialect, strref("my_op"), opCallbacks, NULL);
  mlirDynamicDialectRegisterOp(testDialect, opDef);

  // Parse and dump IR that uses this op.
  const char *irStr = "module { \"testop.my_op\"() : () -> () }";
  MlirModule module = mlirModuleCreateParse(ctx, strref(irStr));
  if (mlirModuleIsNull(module)) {
    fprintf(stderr, "failed to parse module\n");
    return;
  }

  // CHECK:      module {
  // CHECK-NEXT:   "testop.my_op"() : () -> ()
  // CHECK-NEXT: }
  mlirOperationDump(mlirModuleGetOperation(module));
  mlirModuleDestroy(module);
}

void testDynamicTypeCreation(MlirContext ctx) {
  MlirDynamicDialect testDialect =
      mlirDynamicDialectCreate(ctx, strref("testtype"));

  // Define and register a dynamic type.
  MlirDynamicTypeDefinitionCallbacks typeCallbacks = {NULL, NULL,
                                                      successTypeVerify};
  MlirDynamicTypeDefinition typeDef = mlirDynamicTypeDefinitionCreate(
      testDialect, strref("my_type"), typeCallbacks, NULL);
  mlirDynamicDialectRegisterType(testDialect, typeDef);

  // Look it up to verify registration worked.
  MlirDialect dialect = mlirDynamicDialectAsDialect(testDialect);
  MlirDynamicTypeDefinition lookedUp =
      mlirExtensibleDialectLookupTypeDefinition(dialect, strref("my_type"));

  // CHECK: type lookup succeeded: 1
  fprintf(stderr, "type lookup succeeded: %d\n", lookedUp.ptr != NULL);

  // Create an instance with no parameters.
  MlirType dynType = mlirDynamicTypeGet(lookedUp, NULL, 0);

  // CHECK: is dynamic type: 1
  fprintf(stderr, "is dynamic type: %d\n", mlirTypeIsADynamicType(dynType));

  // CHECK: dynamic type num params: 0
  fprintf(stderr, "dynamic type num params: %ld\n",
          (long)mlirDynamicTypeGetNumParams(dynType));
}

void testDynamicAttrCreation(MlirContext ctx) {
  MlirDynamicDialect testDialect =
      mlirDynamicDialectCreate(ctx, strref("testattr"));

  // Define and register a dynamic attribute.
  MlirDynamicAttrDefinitionCallbacks attrCallbacks = {NULL, NULL,
                                                      successAttrVerify};
  MlirDynamicAttrDefinition attrDef = mlirDynamicAttrDefinitionCreate(
      testDialect, strref("my_attr"), attrCallbacks, NULL);
  mlirDynamicDialectRegisterAttr(testDialect, attrDef);

  // Look it up to verify registration worked.
  MlirDialect dialect = mlirDynamicDialectAsDialect(testDialect);
  MlirDynamicAttrDefinition lookedUp =
      mlirExtensibleDialectLookupAttrDefinition(dialect, strref("my_attr"));

  // CHECK: attr lookup succeeded: 1
  fprintf(stderr, "attr lookup succeeded: %d\n", lookedUp.ptr != NULL);

  // Create an instance with no parameters.
  MlirAttribute dynAttr = mlirDynamicAttrGet(lookedUp, NULL, 0);

  // CHECK: is dynamic attr: 1
  fprintf(stderr, "is dynamic attr: %d\n",
          mlirAttributeIsADynamicAttr(dynAttr));

  // CHECK: dynamic attr num params: 0
  fprintf(stderr, "dynamic attr num params: %ld\n",
          (long)mlirDynamicAttrGetNumParams(dynAttr));
}

void testDynamicTypeWithParams(MlirContext ctx) {
  MlirDynamicDialect testDialect =
      mlirDynamicDialectCreate(ctx, strref("testparams"));

  // Define and register a parameterized type.
  MlirDynamicTypeDefinitionCallbacks typeCallbacks = {NULL, NULL,
                                                      successTypeVerify};
  MlirDynamicTypeDefinition typeDef = mlirDynamicTypeDefinitionCreate(
      testDialect, strref("pair"), typeCallbacks, NULL);
  mlirDynamicDialectRegisterType(testDialect, typeDef);

  MlirDialect dialect = mlirDynamicDialectAsDialect(testDialect);
  MlirDynamicTypeDefinition lookedUp =
      mlirExtensibleDialectLookupTypeDefinition(dialect, strref("pair"));

  // Create an instance with two integer attribute parameters.
  MlirAttribute params[2];
  params[0] = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), 42);
  params[1] = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), 7);
  MlirType dynType = mlirDynamicTypeGet(lookedUp, params, 2);

  // CHECK: parameterized type num params: 2
  fprintf(stderr, "parameterized type num params: %ld\n",
          (long)mlirDynamicTypeGetNumParams(dynType));
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirContextSetAllowUnregisteredDialects(ctx, true);

  testDynamicDialectCreation(ctx);
  testDynamicOpCreation(ctx);
  testDynamicTypeCreation(ctx);
  testDynamicAttrCreation(ctx);
  testDynamicTypeWithParams(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
