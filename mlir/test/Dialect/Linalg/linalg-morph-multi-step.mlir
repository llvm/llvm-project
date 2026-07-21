// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic -split-input-file | \
// RUN:   FileCheck %s  --check-prefix=ALL,NAMED_TO_GENERIC
// RUN: mlir-opt %s -linalg-morph-ops=named-to-generic -split-input-file | \
// RUN:   mlir-opt -linalg-morph-ops=generic-to-named -split-input-file | \
// RUN:     FileCheck %s  --check-prefix=ALL,ROUND_TRIP

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

// ALL-LABEL: unary_ops

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

// -----

func.func @binary_ops_int(%A: tensor<?x?xi32>, %B: tensor<?x?xi32>,
                          %Out: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.add ins(%A, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %1 = linalg.sub ins(%0, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = linalg.mul ins(%1, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %3 = linalg.div ins(%2, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %4 = linalg.div_unsigned ins(%3, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %5 = linalg.max ins(%4, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %6 = linalg.min ins(%5, %B : tensor<?x?xi32>, tensor<?x?xi32>)
                  outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %6 : tensor<?x?xi32>
}

// ALL-LABEL: binary_ops_int

// NAMED_TO_GENERIC-COUNT-7: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.add
// NAMED_TO_GENERIC-NOT: linalg.sub
// NAMED_TO_GENERIC-NOT: linalg.mul
// NAMED_TO_GENERIC-NOT: linalg.div
// NAMED_TO_GENERIC-NOT: linalg.div_unsigned
// NAMED_TO_GENERIC-NOT: linalg.max
// NAMED_TO_GENERIC-NOT: linalg.min

// ROUND_TRIP: linalg.add
// ROUND_TRIP: linalg.sub
// ROUND_TRIP: linalg.mul
// ROUND_TRIP: linalg.div
// ROUND_TRIP: linalg.div_unsigned
// ROUND_TRIP: linalg.max
// ROUND_TRIP: linalg.min
// ROUND_TRIP-NOT: linalg.generic

// -----

func.func @binary_ops_float(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                            %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.add ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.sub ins(%0, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.mul ins(%1, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.div ins(%2, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.max ins(%3, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = linalg.min ins(%4, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = linalg.powf ins(%5, %B : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}

// ALL-LABEL: binary_ops_float

// NAMED_TO_GENERIC-COUNT-7: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.add
// NAMED_TO_GENERIC-NOT: linalg.sub
// NAMED_TO_GENERIC-NOT: linalg.mul
// NAMED_TO_GENERIC-NOT: linalg.div
// NAMED_TO_GENERIC-NOT: linalg.max
// NAMED_TO_GENERIC-NOT: linalg.min
// NAMED_TO_GENERIC-NOT: linalg.powf

// ROUND_TRIP: linalg.add
// ROUND_TRIP: linalg.sub
// ROUND_TRIP: linalg.mul
// ROUND_TRIP: linalg.div
// ROUND_TRIP: linalg.max
// ROUND_TRIP: linalg.min
// ROUND_TRIP: linalg.powf
// ROUND_TRIP-NOT: linalg.generic

// -----

func.func @binary_ops_complex(%A: tensor<?x?xcomplex<f32>>, %B: tensor<?x?xcomplex<f32>>,
                              %Out: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>> {
  %0 = linalg.add ins(%A, %B : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>)
                  outs(%Out : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  %1 = linalg.sub ins(%0, %B : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>)
                  outs(%Out : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  %2 = linalg.mul ins(%1, %B : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>)
                  outs(%Out : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  %3 = linalg.div ins(%2, %B : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>)
                  outs(%Out : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  return %3 : tensor<?x?xcomplex<f32>>
}

// ALL-LABEL: binary_ops_complex

// NAMED_TO_GENERIC-COUNT-4: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.add
// NAMED_TO_GENERIC-NOT: linalg.sub
// NAMED_TO_GENERIC-NOT: linalg.mul
// NAMED_TO_GENERIC-NOT: linalg.div

// ROUND_TRIP: linalg.add
// ROUND_TRIP: linalg.sub
// ROUND_TRIP: linalg.mul
// ROUND_TRIP: linalg.div
// ROUND_TRIP-NOT: linalg.generic

// -----

func.func @binary_ops_bool(%A: tensor<?x?xi1>, %B: tensor<?x?xi1>,
                           %Out: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = linalg.add ins(%A, %B : tensor<?x?xi1>, tensor<?x?xi1>)
                  outs(%Out : tensor<?x?xi1>) -> tensor<?x?xi1>
  %1 = linalg.mul ins(%0, %B : tensor<?x?xi1>, tensor<?x?xi1>)
                  outs(%Out : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %1 : tensor<?x?xi1>
}

// ALL-LABEL: binary_ops_bool

// NAMED_TO_GENERIC-COUNT-2: linalg.generic
// NAMED_TO_GENERIC-NOT: linalg.add
// NAMED_TO_GENERIC-NOT: linalg.mul

// ROUND_TRIP: linalg.add
// ROUND_TRIP: linalg.mul
// ROUND_TRIP-NOT: linalg.generic
