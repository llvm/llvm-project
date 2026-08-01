// Category to named conversion and roundtrip (output is identical).
// RUN: mlir-opt %s -linalg-morph-ops=category-to-named -split-input-file | \
// RUN:   FileCheck %s
// RUN: mlir-opt %s -linalg-morph-ops=category-to-named -split-input-file | \
// RUN:   mlir-opt -linalg-morph-ops=named-to-category -split-input-file | \
// RUN:   mlir-opt -linalg-morph-ops=category-to-named -split-input-file | \
// RUN:     FileCheck %s

func.func @unary_ops(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>) -> tensor<16x8xf32> {
  %exp = linalg.elementwise kind=#linalg.elementwise_kind<exp>
    ins(%A : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %log = linalg.elementwise kind=#linalg.elementwise_kind<log>
    ins(%exp : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %abs = linalg.elementwise kind=#linalg.elementwise_kind<abs>
    ins(%log : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %ceil = linalg.elementwise kind=#linalg.elementwise_kind<ceil>
    ins(%abs : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %floor = linalg.elementwise kind=#linalg.elementwise_kind<floor>
    ins(%ceil : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %negf = linalg.elementwise kind=#linalg.elementwise_kind<negf>
    ins(%floor : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %recip = linalg.elementwise kind=#linalg.elementwise_kind<reciprocal>
    ins(%negf : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %round = linalg.elementwise kind=#linalg.elementwise_kind<round>
    ins(%recip : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %sqrt = linalg.elementwise kind=#linalg.elementwise_kind<sqrt>
    ins(%round : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %rsqrt = linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
    ins(%sqrt : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %square = linalg.elementwise kind=#linalg.elementwise_kind<square>
    ins(%rsqrt : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %tanh = linalg.elementwise kind=#linalg.elementwise_kind<tanh>
    ins(%square : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  %erf = linalg.elementwise kind=#linalg.elementwise_kind<erf>
    ins(%tanh : tensor<16x8xf32>) outs(%B : tensor<16x8xf32>) -> tensor<16x8xf32>
  return %erf : tensor<16x8xf32>
}

// CHECK-LABEL: unary_ops
// CHECK-SAME: %[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>)
// CHECK-NOT: linalg.elementwise
// CHECK: %[[EXP:.+]] = linalg.exp
// CHECK-SAME: ins(%[[A]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[LOG:.+]] = linalg.log
// CHECK-SAME: ins(%[[EXP]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[ABS:.+]] = linalg.abs
// CHECK-SAME: ins(%[[LOG]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[CEIL:.+]] = linalg.ceil
// CHECK-SAME: ins(%[[ABS]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[FLOOR:.+]] = linalg.floor
// CHECK-SAME: ins(%[[CEIL]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[NEGF:.+]] = linalg.negf
// CHECK-SAME: ins(%[[FLOOR]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[RECIP:.+]] = linalg.reciprocal
// CHECK-SAME: ins(%[[NEGF]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[ROUND:.+]] = linalg.round
// CHECK-SAME: ins(%[[RECIP]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[SQRT:.+]] = linalg.sqrt
// CHECK-SAME: ins(%[[ROUND]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[RSQRT:.+]] = linalg.rsqrt
// CHECK-SAME: ins(%[[SQRT]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[SQUARE:.+]] = linalg.square
// CHECK-SAME: ins(%[[RSQRT]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[TANH:.+]] = linalg.tanh
// CHECK-SAME: ins(%[[SQUARE]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: linalg.erf
// CHECK-SAME: ins(%[[TANH]] : tensor<16x8xf32>)
// CHECK-SAME: outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>

// -----

func.func @binary_ops_int(%A: tensor<?x?xi32>, %B: tensor<?x?xi32>,
                          %Out: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%A, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<sub>
    ins(%0, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%1, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %3 = linalg.elementwise kind=#linalg.elementwise_kind<div>
    ins(%2, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %4 = linalg.elementwise kind=#linalg.elementwise_kind<div_unsigned>
    ins(%3, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %5 = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%4, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  %6 = linalg.elementwise kind=#linalg.elementwise_kind<min_signed>
    ins(%5, %B : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%Out : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %6 : tensor<?x?xi32>
}

// CHECK-LABEL: binary_ops_int
// CHECK-SAME: %[[A:.+]]: tensor<?x?xi32>, %[[B:.+]]: tensor<?x?xi32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xi32>)
// CHECK-NOT: linalg.elementwise
// CHECK: %[[ADD:.+]] = linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK: %[[SUB:.+]] = linalg.sub
// CHECK-SAME: ins(%[[ADD]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK: %[[MUL:.+]] = linalg.mul
// CHECK-SAME: ins(%[[SUB]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK: %[[DIV:.+]] = linalg.div
// CHECK-SAME: ins(%[[MUL]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK: %[[DIVU:.+]] = linalg.div_unsigned
// CHECK-SAME: ins(%[[DIV]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK: %[[MAX:.+]] = linalg.max
// CHECK-SAME: ins(%[[DIVU]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK: linalg.min
// CHECK-SAME: ins(%[[MAX]], %[[B]] : tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi32>) -> tensor<?x?xi32>

// -----

func.func @binary_ops_float(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                            %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elementwise kind=#linalg.elementwise_kind<sub>
    ins(%0, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%1, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.elementwise kind=#linalg.elementwise_kind<div>
    ins(%2, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%3, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = linalg.elementwise kind=#linalg.elementwise_kind<min_signed>
    ins(%4, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = linalg.elementwise kind=#linalg.elementwise_kind<powf>
    ins(%5, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}

// CHECK-LABEL: binary_ops_float
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK-NOT: linalg.elementwise
// CHECK: %[[ADD:.+]] = linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[SUB:.+]] = linalg.sub
// CHECK-SAME: ins(%[[ADD]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MUL:.+]] = linalg.mul
// CHECK-SAME: ins(%[[SUB]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[DIV:.+]] = linalg.div
// CHECK-SAME: ins(%[[MUL]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MAX:.+]] = linalg.max
// CHECK-SAME: ins(%[[DIV]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MIN:.+]] = linalg.min
// CHECK-SAME: ins(%[[MAX]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: linalg.powf
// CHECK-SAME: ins(%[[MIN]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @ternary_select(%A: tensor<?x?xi1>, %B: tensor<?x?xf32>,
                          %C: tensor<?x?xf32>,
                          %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise kind=#linalg.elementwise_kind<select>
    ins(%A, %B, %C : tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: ternary_select
// CHECK-SAME: %[[A:.+]]: tensor<?x?xi1>, %[[B:.+]]: tensor<?x?xf32>, %[[C:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK-NOT: linalg.elementwise
// CHECK: linalg.select
// CHECK-SAME: ins(%[[A]], %[[B]], %[[C]] : tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// Non-identity indexing maps: should NOT be converted to named op.
func.func @non_identity_maps(%A: tensor<?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise
    kind=#linalg.elementwise_kind<exp>
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]
    ins(%A : tensor<?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: non_identity_maps
// CHECK-SAME: %[[A:.+]]: tensor<?xf32>, %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME: ins(%[[A]] : tensor<?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.exp

// -----

// Kinds without named op equivalent: should NOT be converted.
func.func @no_named_op(%A: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise kind=#linalg.elementwise_kind<sin>
    ins(%A : tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: no_named_op
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sin>
// CHECK-SAME: ins(%[[A]] : tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
