// Forward path `named -> category -> generic`
// RUN: mlir-opt %s -linalg-morph-ops=named-to-category | FileCheck %s  --check-prefix=NAMED_TO_CATEGORY

// RUN: mlir-opt %s -linalg-morph-ops=named-to-category |  \
// RUN:   mlir-opt -linalg-morph-ops=category-to-generic | FileCheck %s  --check-prefix=CATEGORY_TO_GENERIC

func.func @exp(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %exp = linalg.exp ins(%A : tensor<16x8xf32>) outs(%B :  tensor<16x8xf32>) -> tensor<16x8xf32>
  return %exp :  tensor<16x8xf32>
}
// NAMED_TO_CATEGORY: linalg.elementwise
// NAMED_TO_CATEGORY-NOT: linalg.exp

// CATEGORY_TO_GENERIC: linalg.generic
// CATEGORY_TO_GENERIC-NOT: linalg.elementwise
