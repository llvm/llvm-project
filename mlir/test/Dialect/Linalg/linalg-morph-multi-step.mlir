// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic |  FileCheck %s  --check-prefix=NAMED_TO_GENERIC
// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic |  mlir-opt %s -linalg-morph-ops=generic-to-named | \
// RUN:   FileCheck %s  --check-prefix=ROUND_TRIP

func.func @exp(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %exp = linalg.exp ins(%A : tensor<16x8xf32>) outs(%B :  tensor<16x8xf32>) -> tensor<16x8xf32>
  return %exp :  tensor<16x8xf32>
}

// NAMED_TO_GENERIC: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.exp

// ROUND_TRIP: linalg.exp
// ROUND_TRIP-NOT: linalg.generic
