// Forward path `named -> category -> generic`
// RUN: mlir-opt %s -linalg-morph-ops=named-to-category | FileCheck %s  --check-prefix=NAMED_TO_CATEGORY
// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic |  FileCheck %s  --check-prefix=NAMED_TO_GENERIC
// RUN: mlir-opt %s -linalg-morph-ops=named-to-category |  \
// RUN:   mlir-opt %s -linalg-morph-ops=category-to-generic | FileCheck %s  --check-prefix=CATEGORY_TO_GENERIC
//
// Backward path `named <- category <- generic`
// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic |  mlir-opt %s -linalg-morph-ops=generic-to-named | \
// RUN:   FileCheck %s  --check-prefix=GENERIC_TO_NAMED

func.func @exp(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %exp = linalg.exp ins(%A : tensor<16x8xf32>) outs(%B :  tensor<16x8xf32>) -> tensor<16x8xf32>
  return %exp :  tensor<16x8xf32>
}
// NAMED_TO_CATEGORY: linalg.elementwise
// NAMED_TO_CATEGORY-NOT: linalg.exp

// NAMED_TO_GENERIC: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.exp

// CATEGORY_TO_GENERIC: linalg.generic
// CATEGORY_TO_GENERIC-NOT: linalg.elementwise

// GENERIC_TO_NAMED: linalg.exp
// GENERIC_TO_NAMED-NOT: linalg.generic
