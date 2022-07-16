//===- sparse_tensor.c - Test of sparse_tensor APIs -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-sparse-tensor-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CHECK-LABEL: testRoundtripEncoding()
static int testRoundtripEncoding(MlirContext ctx) {
  fprintf(stderr, "testRoundtripEncoding()\n");
  // clang-format off
  const char *originalAsm =
    "#sparse_tensor.encoding<{ "
    "dimLevelType = [ \"dense\", \"compressed\", \"singleton\"], "
    "dimOrdering = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, "
    "pointerBitWidth = 32, indexBitWidth = 64 }>";
  // clang-format on
  MlirAttribute originalAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString(originalAsm));
  // CHECK: isa: 1
  fprintf(stderr, "isa: %d\n",
          mlirAttributeIsASparseTensorEncodingAttr(originalAttr));
  MlirAffineMap dimOrdering =
      mlirSparseTensorEncodingAttrGetDimOrdering(originalAttr);
  // CHECK: (d0, d1, d2) -> (d0, d1, d2)
  mlirAffineMapDump(dimOrdering);
  // CHECK: level_type: 0
  // CHECK: level_type: 1
  // CHECK: level_type: 2
  int numLevelTypes = mlirSparseTensorEncodingGetNumDimLevelTypes(originalAttr);
  enum MlirSparseTensorDimLevelType *levelTypes =
      malloc(sizeof(enum MlirSparseTensorDimLevelType) * numLevelTypes);
  for (int i = 0; i < numLevelTypes; ++i) {
    levelTypes[i] =
        mlirSparseTensorEncodingAttrGetDimLevelType(originalAttr, i);
    fprintf(stderr, "level_type: %d\n", levelTypes[i]);
  }
  // CHECK: pointer: 32
  int pointerBitWidth =
      mlirSparseTensorEncodingAttrGetPointerBitWidth(originalAttr);
  fprintf(stderr, "pointer: %d\n", pointerBitWidth);
  // CHECK: index: 64
  int indexBitWidth =
      mlirSparseTensorEncodingAttrGetIndexBitWidth(originalAttr);
  fprintf(stderr, "index: %d\n", indexBitWidth);

  MlirAttribute newAttr = mlirSparseTensorEncodingAttrGet(
      ctx, numLevelTypes, levelTypes, dimOrdering, pointerBitWidth,
      indexBitWidth);
  mlirAttributeDump(newAttr); // For debugging filecheck output.
  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirAttributeEqual(originalAttr, newAttr));

  free(levelTypes);
  return 0;
}

int main() {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sparse_tensor__(),
                                   ctx);
  if (testRoundtripEncoding(ctx))
    return 1;

  mlirContextDestroy(ctx);
  return 0;
}
