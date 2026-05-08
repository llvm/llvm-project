// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic |  FileCheck %s  --check-prefix=NAMED_TO_GENERIC
// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic |  mlir-opt -linalg-morph-ops=generic-to-named | \
// RUN:   FileCheck %s  --check-prefix=ROUND_TRIP

func.func @unary_ops(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %exp = linalg.exp ins(%A : tensor<16x8xf32>) outs(%B :  tensor<16x8xf32>) -> tensor<16x8xf32>
  %log = linalg.log ins(%exp : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %abs = linalg.abs ins(%log : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %ceil = linalg.ceil ins(%abs : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %floor = linalg.floor ins(%ceil : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %negf = linalg.negf ins(%floor : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %recip = linalg.reciprocal ins(%negf : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %round = linalg.round ins(%recip : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %sqrt = linalg.sqrt ins(%round : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %rsqrt = linalg.rsqrt ins(%sqrt : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %square = linalg.square ins(%rsqrt : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %tanh = linalg.tanh ins(%square : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %erf = linalg.erf ins(%tanh : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  return %erf :  tensor<16x8xf32>
}

// NAMED_TO_GENERIC-COUNT-13: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.exp
// NAMED_TO_GENERIC-NOT: linalg.log
// NAMED_TO_GENERIC-NOT: linalg.abs
// NAMED_TO_GENERIC-NOT: linalg.ceil
// NAMED_TO_GENERIC-NOT: linalg.floor
// NAMED_TO_GENERIC-NOT: linalg.negf
// NAMED_TO_GENERIC-NOT: linalg.reciprocal
// NAMED_TO_GENERIC-NOT: linalg.round
// NAMED_TO_GENERIC-NOT: linalg.sqrt
// NAMED_TO_GENERIC-NOT: linalg.rsqrt
// NAMED_TO_GENERIC-NOT: linalg.square
// NAMED_TO_GENERIC-NOT: linalg.tanh
// NAMED_TO_GENERIC-NOT: linalg.erf

// ROUND_TRIP: linalg.exp
// ROUND_TRIP: linalg.log
// ROUND_TRIP: linalg.abs
// ROUND_TRIP: linalg.ceil
// ROUND_TRIP: linalg.floor
// ROUND_TRIP: linalg.negf
// ROUND_TRIP: linalg.reciprocal
// ROUND_TRIP: linalg.round
// ROUND_TRIP: linalg.sqrt
// ROUND_TRIP: linalg.rsqrt
// ROUND_TRIP: linalg.square
// ROUND_TRIP: linalg.tanh
// ROUND_TRIP: linalg.erf
// ROUND_TRIP-NOT: linalg.generic
