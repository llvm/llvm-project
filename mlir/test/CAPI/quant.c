//===- quant.c - Test of Quant dialect C API ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-quant-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/Quant.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testTypeHierarchy
static void testTypeHierarchy(MlirContext ctx) {
  fprintf(stderr, "testTypeHierarchy\n");

  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType any = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>"));
  MlirType uniform =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(
                                "!quant.uniform<i8<-8:7>:f32, 0.99872:127>"));
  MlirType perAxis = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString(
               "!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>"));
  MlirType calibrated = mlirTypeParseGet(
      ctx,
      mlirStringRefCreateFromCString("!quant.calibrated<f32<-0.998:1.2321>>"));

  // The parser itself is checked in C++ dialect tests.
  assert(!mlirTypeIsNull(any) && "couldn't parse AnyQuantizedType");
  assert(!mlirTypeIsNull(uniform) && "couldn't parse UniformQuantizedType");
  assert(!mlirTypeIsNull(perAxis) &&
         "couldn't parse UniformQuantizedPerAxisType");
  assert(!mlirTypeIsNull(calibrated) &&
         "couldn't parse CalibratedQuantizedType");

  // CHECK: i8 isa QuantizedType: 0
  fprintf(stderr, "i8 isa QuantizedType: %d\n", mlirTypeIsAQuantizedType(i8));
  // CHECK: any isa QuantizedType: 1
  fprintf(stderr, "any isa QuantizedType: %d\n", mlirTypeIsAQuantizedType(any));
  // CHECK: uniform isa QuantizedType: 1
  fprintf(stderr, "uniform isa QuantizedType: %d\n",
          mlirTypeIsAQuantizedType(uniform));
  // CHECK: perAxis isa QuantizedType: 1
  fprintf(stderr, "perAxis isa QuantizedType: %d\n",
          mlirTypeIsAQuantizedType(perAxis));
  // CHECK: calibrated isa QuantizedType: 1
  fprintf(stderr, "calibrated isa QuantizedType: %d\n",
          mlirTypeIsAQuantizedType(calibrated));

  // CHECK: any isa AnyQuantizedType: 1
  fprintf(stderr, "any isa AnyQuantizedType: %d\n",
          mlirTypeIsAAnyQuantizedType(any));
  // CHECK: uniform isa UniformQuantizedType: 1
  fprintf(stderr, "uniform isa UniformQuantizedType: %d\n",
          mlirTypeIsAUniformQuantizedType(uniform));
  // CHECK: perAxis isa UniformQuantizedPerAxisType: 1
  fprintf(stderr, "perAxis isa UniformQuantizedPerAxisType: %d\n",
          mlirTypeIsAUniformQuantizedPerAxisType(perAxis));
  // CHECK: calibrated isa CalibratedQuantizedType: 1
  fprintf(stderr, "calibrated isa CalibratedQuantizedType: %d\n",
          mlirTypeIsACalibratedQuantizedType(calibrated));

  // CHECK: perAxis isa UniformQuantizedType: 0
  fprintf(stderr, "perAxis isa UniformQuantizedType: %d\n",
          mlirTypeIsAUniformQuantizedType(perAxis));
  // CHECK: uniform isa CalibratedQuantizedType: 0
  fprintf(stderr, "uniform isa CalibratedQuantizedType: %d\n",
          mlirTypeIsACalibratedQuantizedType(uniform));
  fprintf(stderr, "\n");
}

// CHECK-LABEL: testAnyQuantizedType
void testAnyQuantizedType(MlirContext ctx) {
  fprintf(stderr, "testAnyQuantizedType\n");

  MlirType anyParsed = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>"));

  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType f32 = mlirF32TypeGet(ctx);
  MlirType any =
      mlirAnyQuantizedTypeGet(mlirQuantizedTypeGetSignedFlag(), i8, f32, -8, 7);

  // CHECK: flags: 1
  fprintf(stderr, "flags: %u\n", mlirQuantizedTypeGetFlags(any));
  // CHECK: signed: 1
  fprintf(stderr, "signed: %u\n", mlirQuantizedTypeIsSigned(any));
  // CHECK: storage type: i8
  fprintf(stderr, "storage type: ");
  mlirTypeDump(mlirQuantizedTypeGetStorageType(any));
  fprintf(stderr, "\n");
  // CHECK: expressed type: f32
  fprintf(stderr, "expressed type: ");
  mlirTypeDump(mlirQuantizedTypeGetExpressedType(any));
  fprintf(stderr, "\n");
  // CHECK: storage min: -8
  fprintf(stderr, "storage min: %" PRId64 "\n",
          mlirQuantizedTypeGetStorageTypeMin(any));
  // CHECK: storage max: 7
  fprintf(stderr, "storage max: %" PRId64 "\n",
          mlirQuantizedTypeGetStorageTypeMax(any));
  // CHECK: storage width: 8
  fprintf(stderr, "storage width: %u\n",
          mlirQuantizedTypeGetStorageTypeIntegralWidth(any));
  // CHECK: quantized element type: !quant.any<i8<-8:7>:f32>
  fprintf(stderr, "quantized element type: ");
  mlirTypeDump(mlirQuantizedTypeGetQuantizedElementType(any));
  fprintf(stderr, "\n");

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(anyParsed, any));
  // CHECK: !quant.any<i8<-8:7>:f32>
  mlirTypeDump(any);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testUniformType
void testUniformType(MlirContext ctx) {
  fprintf(stderr, "testUniformType\n");

  MlirType uniformParsed =
      mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(
                                "!quant.uniform<i8<-8:7>:f32, 0.99872:127>"));

  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType f32 = mlirF32TypeGet(ctx);
  MlirType uniform = mlirUniformQuantizedTypeGet(
      mlirQuantizedTypeGetSignedFlag(), i8, f32, 0.99872, 127, -8, 7);

  // CHECK: scale: 0.998720
  fprintf(stderr, "scale: %lf\n", mlirUniformQuantizedTypeGetScale(uniform));
  // CHECK: zero point: 127
  fprintf(stderr, "zero point: %" PRId64 "\n",
          mlirUniformQuantizedTypeGetZeroPoint(uniform));
  // CHECK: fixed point: 0
  fprintf(stderr, "fixed point: %d\n",
          mlirUniformQuantizedTypeIsFixedPoint(uniform));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(uniform, uniformParsed));
  // CHECK: !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
  mlirTypeDump(uniform);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testUniformPerAxisType
void testUniformPerAxisType(MlirContext ctx) {
  fprintf(stderr, "testUniformPerAxisType\n");

  MlirType perAxisParsed = mlirTypeParseGet(
      ctx, mlirStringRefCreateFromCString(
               "!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>"));

  MlirType i8 = mlirIntegerTypeGet(ctx, 8);
  MlirType f32 = mlirF32TypeGet(ctx);
  double scales[] = {200.0, 0.99872};
  int64_t zeroPoints[] = {0, 120};
  MlirType perAxis = mlirUniformQuantizedPerAxisTypeGet(
      mlirQuantizedTypeGetSignedFlag(), i8, f32,
      /*nDims=*/2, scales, zeroPoints,
      /*quantizedDimension=*/1,
      mlirQuantizedTypeGetDefaultMinimumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8),
      mlirQuantizedTypeGetDefaultMaximumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8));

  // CHECK: num dims: 2
  fprintf(stderr, "num dims: %" PRIdPTR "\n",
          mlirUniformQuantizedPerAxisTypeGetNumDims(perAxis));
  // CHECK: scale 0: 200.000000
  fprintf(stderr, "scale 0: %lf\n",
          mlirUniformQuantizedPerAxisTypeGetScale(perAxis, 0));
  // CHECK: scale 1: 0.998720
  fprintf(stderr, "scale 1: %lf\n",
          mlirUniformQuantizedPerAxisTypeGetScale(perAxis, 1));
  // CHECK: zero point 0: 0
  fprintf(stderr, "zero point 0: %" PRId64 "\n",
          mlirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 0));
  // CHECK: zero point 1: 120
  fprintf(stderr, "zero point 1: %" PRId64 "\n",
          mlirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 1));
  // CHECK: quantized dim: 1
  fprintf(stderr, "quantized dim: %" PRId32 "\n",
          mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(perAxis));
  // CHECK: fixed point: 0
  fprintf(stderr, "fixed point: %d\n",
          mlirUniformQuantizedPerAxisTypeIsFixedPoint(perAxis));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(perAxis, perAxisParsed));
  // CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>
  mlirTypeDump(perAxis);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testCalibratedType
void testCalibratedType(MlirContext ctx) {
  fprintf(stderr, "testCalibratedType\n");

  MlirType calibratedParsed = mlirTypeParseGet(
      ctx,
      mlirStringRefCreateFromCString("!quant.calibrated<f32<-0.998:1.2321>>"));

  MlirType f32 = mlirF32TypeGet(ctx);
  MlirType calibrated = mlirCalibratedQuantizedTypeGet(f32, -0.998, 1.2321);

  // CHECK: min: -0.998000
  fprintf(stderr, "min: %lf\n", mlirCalibratedQuantizedTypeGetMin(calibrated));
  // CHECK: max: 1.232100
  fprintf(stderr, "max: %lf\n", mlirCalibratedQuantizedTypeGetMax(calibrated));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirTypeEqual(calibrated, calibratedParsed));
  // CHECK: !quant.calibrated<f32<-0.998:1.232100e+00>>
  mlirTypeDump(calibrated);
  fprintf(stderr, "\n\n");
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__quant__(), ctx);
  testTypeHierarchy(ctx);
  testAnyQuantizedType(ctx);
  testUniformType(ctx);
  testUniformPerAxisType(ctx);
  testCalibratedType(ctx);
  mlirContextDestroy(ctx);
  return EXIT_SUCCESS;
}
