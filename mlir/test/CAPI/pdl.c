//===- pdl.c - Test of PDL dialect C API ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-pdl-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/PDL.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testAttributeType
void testAttributeType(MlirContext ctx) {
  fprintf(stderr, "testAttributeType\n");

  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!pdl.attribute"));
  MlirType constructedType = mlirPDLAttributeTypeGet(ctx);

  assert(!mlirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 1
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 1
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.attribute
  mlirTypeDump(parsedType);
  // CHECK: !pdl.attribute
  mlirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testOperationType
void testOperationType(MlirContext ctx) {
  fprintf(stderr, "testOperationType\n");

  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!pdl.operation"));
  MlirType constructedType = mlirPDLOperationTypeGet(ctx);

  assert(!mlirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 1 
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 1
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.operation
  mlirTypeDump(parsedType);
  // CHECK: !pdl.operation
  mlirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testRangeType
void testRangeType(MlirContext ctx) {
  fprintf(stderr, "testRangeType\n");

  MlirType typeType = mlirPDLTypeTypeGet(ctx);
  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!pdl.range<type>"));
  MlirType constructedType = mlirPDLRangeTypeGet(typeType);
  MlirType elementType = mlirPDLRangeTypeGetElementType(constructedType);

  assert(!mlirTypeIsNull(typeType) && "couldn't get PDLTypeType");
  assert(!mlirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 1 
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 1 
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));
  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(typeType, elementType));

  // CHECK: !pdl.range<type>
  mlirTypeDump(parsedType);
  // CHECK: !pdl.range<type>
  mlirTypeDump(constructedType);
  // CHECK: !pdl.type
  mlirTypeDump(elementType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testTypeType
void testTypeType(MlirContext ctx) {
  fprintf(stderr, "testTypeType\n");

  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!pdl.type"));
  MlirType constructedType = mlirPDLTypeTypeGet(ctx);

  assert(!mlirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 1 
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 1
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.type
  mlirTypeDump(parsedType);
  // CHECK: !pdl.type
  mlirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testValueType
void testValueType(MlirContext ctx) {
  fprintf(stderr, "testValueType\n");

  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!pdl.value"));
  MlirType constructedType = mlirPDLValueTypeGet(ctx);

  assert(!mlirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 1
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      mlirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      mlirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      mlirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      mlirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      mlirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 1
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      mlirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.value
  mlirTypeDump(parsedType);
  // CHECK: !pdl.value
  mlirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__pdl__(), ctx);
  testAttributeType(ctx);
  testOperationType(ctx);
  testRangeType(ctx);
  testTypeType(ctx);
  testValueType(ctx);
  mlirContextDestroy(ctx);
  return EXIT_SUCCESS;
}
