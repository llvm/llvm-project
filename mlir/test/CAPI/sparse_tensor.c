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
    "dimLevelType = [ \"dense\", \"compressed\", \"compressed\"], "
    "dimOrdering = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, "
    "higherOrdering = affine_map<(d0, d1)[s0] -> (s0, d0, d1)>, "
    "posWidth = 32, crdWidth = 64 }>";
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
  MlirAffineMap higherOrdering =
      mlirSparseTensorEncodingAttrGetHigherOrdering(originalAttr);
  // CHECK: (d0, d1)[s0] -> (s0, d0, d1)
  mlirAffineMapDump(higherOrdering);
  // CHECK: level_type: 4
  // CHECK: level_type: 8
  // CHECK: level_type: 8
  int lvlRank = mlirSparseTensorEncodingGetLvlRank(originalAttr);
  enum MlirSparseTensorDimLevelType *levelTypes =
      malloc(sizeof(enum MlirSparseTensorDimLevelType) * lvlRank);
  for (int l = 0; l < lvlRank; ++l) {
    levelTypes[l] =
        mlirSparseTensorEncodingAttrGetDimLevelType(originalAttr, l);
    fprintf(stderr, "level_type: %d\n", levelTypes[l]);
  }
  // CHECK: posWidth: 32
  int posWidth = mlirSparseTensorEncodingAttrGetPosWidth(originalAttr);
  fprintf(stderr, "posWidth: %d\n", posWidth);
  // CHECK: crdWidth: 64
  int crdWidth = mlirSparseTensorEncodingAttrGetCrdWidth(originalAttr);
  fprintf(stderr, "crdWidth: %d\n", crdWidth);

  MlirAttribute newAttr =
      mlirSparseTensorEncodingAttrGet(ctx, lvlRank, levelTypes, dimOrdering,
                                      higherOrdering, posWidth, crdWidth);
  mlirAttributeDump(newAttr); // For debugging filecheck output.
  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", mlirAttributeEqual(originalAttr, newAttr));

  free(levelTypes);
  return 0;
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sparse_tensor__(),
                                   ctx);
  if (testRoundtripEncoding(ctx))
    return 1;

  mlirContextDestroy(ctx);
  return 0;
}
