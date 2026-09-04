//===- transform.c - Test of Transform dialect C API ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-transform-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testAnyOpType
void testAnyOpType(MlirContext ctx) {
  fprintf(stderr, "testAnyOpType\n");

  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!transform.any_op"));
  MlirType constructedType = mlirTransformAnyOpTypeGet(ctx);

  assert(!mlirTypeIsNull(parsedType) && "couldn't parse AnyOpType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct AnyOpType");

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));

  // CHECK: parsedType isa AnyOpType: 1
  fprintf(stderr, "parsedType isa AnyOpType: %d\n",
          mlirTypeIsATransformAnyOpType(parsedType));
  // CHECK: parsedType isa OperationType: 0
  fprintf(stderr, "parsedType isa OperationType: %d\n",
          mlirTypeIsATransformOperationType(parsedType));

  // CHECK: !transform.any_op
  mlirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testOperationType
void testOperationType(MlirContext ctx) {
  fprintf(stderr, "testOperationType\n");

  MlirType parsedType = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!transform.op<\"foo.bar\">"));
  MlirType constructedType = mlirTransformOperationTypeGet(
      ctx, mlirStringRefCreateFromCString("foo.bar"));

  assert(!mlirTypeIsNull(parsedType) && "couldn't parse AnyOpType");
  assert(!mlirTypeIsNull(constructedType) && "couldn't construct AnyOpType");

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(parsedType, constructedType));

  // CHECK: parsedType isa AnyOpType: 0
  fprintf(stderr, "parsedType isa AnyOpType: %d\n",
          mlirTypeIsATransformAnyOpType(parsedType));
  // CHECK: parsedType isa OperationType: 1
  fprintf(stderr, "parsedType isa OperationType: %d\n",
          mlirTypeIsATransformOperationType(parsedType));

  // CHECK: operation name equal: 1
  MlirStringRef operationName =
      mlirTransformOperationTypeGetOperationName(constructedType);
  fprintf(stderr, "operation name equal: %d\n",
          mlirStringRefEqual(operationName,
                             mlirStringRefCreateFromCString("foo.bar")));

  // CHECK: !transform.op<"foo.bar">
  mlirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

typedef struct {
  intptr_t numEffects;
  intptr_t numReads;
  intptr_t numWrites;
} MemoryEffectCallbackData;

static void collectMemoryEffects(intptr_t numEffects,
                                 MlirMemoryEffectInstance *effects,
                                 void *userData) {
  MemoryEffectCallbackData *data = (MemoryEffectCallbackData *)userData;
  data->numEffects += numEffects;
  MlirTypeID readID = mlirMemoryEffectGetEffectID(mlirMemoryEffectsReadGet());
  MlirTypeID writeID = mlirMemoryEffectGetEffectID(mlirMemoryEffectsWriteGet());
  for (intptr_t i = 0; i < numEffects; ++i) {
    MlirTypeID effectID = mlirMemoryEffectGetEffectID(
        mlirMemoryEffectInstanceGetEffect(effects[i]));
    data->numReads += mlirTypeIDEqual(effectID, readID);
    data->numWrites += mlirTypeIDEqual(effectID, writeID);
  }
}

// CHECK-LABEL: testMemoryEffectHelpers
void testMemoryEffectHelpers(void) {
  fprintf(stderr, "testMemoryEffectHelpers\n");

  MemoryEffectCallbackData modifies = {0};
  mlirTransformModifiesPayload(collectMemoryEffects, &modifies);
  // CHECK: modifies payload: 2 effects, 1 read, 1 write
  fprintf(stderr,
          "modifies payload: %" PRIdPTR " effects, %" PRIdPTR " read, %" PRIdPTR
          " write\n",
          modifies.numEffects, modifies.numReads, modifies.numWrites);

  MemoryEffectCallbackData reads = {0};
  mlirTransformOnlyReadsPayload(collectMemoryEffects, &reads);
  // CHECK: only reads payload: 1 effects, 1 read, 0 write
  fprintf(stderr,
          "only reads payload: %" PRIdPTR " effects, %" PRIdPTR
          " read, %" PRIdPTR " write\n",
          reads.numEffects, reads.numReads, reads.numWrites);

  fprintf(stderr, "\n\n");
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
  testAnyOpType(ctx);
  testOperationType(ctx);
  testMemoryEffectHelpers();
  mlirContextDestroy(ctx);
  return EXIT_SUCCESS;
}
