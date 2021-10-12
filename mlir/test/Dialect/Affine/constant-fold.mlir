// RUN: mlir-opt -test-constant-fold -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @affine_apply
func @affine_apply(%variable : index) -> (index, index, index) {
  %c177 = arith.constant 177 : index
  %c211 = arith.constant 211 : index
  %N = arith.constant 1075 : index

  // CHECK:[[C1159:%.+]] = arith.constant 1159 : index
  // CHECK:[[C1152:%.+]] = arith.constant 1152 : index
  %x0 = affine.apply affine_map<(d0, d1)[S0] -> ( (d0 + 128 * S0) floordiv 128 + d1 mod 128)>
           (%c177, %c211)[%N]
  %x1 = affine.apply affine_map<(d0, d1)[S0] -> (128 * (S0 ceildiv 128))>
           (%c177, %c211)[%N]

  // CHECK:[[C42:%.+]] = arith.constant 42 : index
  %y = affine.apply affine_map<(d0) -> (42)> (%variable)

  // CHECK: return [[C1159]], [[C1152]], [[C42]]
  return %x0, %x1, %y : index, index, index
}

// -----

// CHECK: #[[map:.*]] = affine_map<(d0, d1) -> (42, d1)

func @affine_min(%variable: index) -> (index, index) {
  // CHECK: %[[C42:.*]] = arith.constant 42
  %c42 = arith.constant 42 : index
  %c44 = arith.constant 44 : index
  // Partial folding will use a different map.
  // CHECK: %[[r:.*]] = affine.min #[[map]](%[[C42]], %{{.*}})
  %0 = affine.min affine_map<(d0, d1) -> (d0, d1)>(%c42, %variable)

  // Full folding will remove the operation entirely.
  // CHECK-NOT: affine.min
  %1 = affine.min affine_map<(d0, d1) -> (d0, d1)>(%c42, %c44)

  // CHECK: return %[[r]], %[[C42]]
  return %0, %1 : index, index
}

// -----

// CHECK: #[[map:.*]] = affine_map<(d0, d1) -> (42, d1)

func @affine_min(%variable: index) -> (index, index) {
  // CHECK: %[[C42:.*]] = arith.constant 42
  %c42 = arith.constant 42 : index
  // CHECK: %[[C44:.*]] = arith.constant 44
  %c44 = arith.constant 44 : index
  // Partial folding will use a different map.
  // CHECK: %[[r:.*]] = affine.max #[[map]](%[[C42]], %{{.*}})
  %0 = affine.max affine_map<(d0, d1) -> (d0, d1)>(%c42, %variable)

  // Full folding will remove the operation entirely.
  // CHECK-NOT: affine.max
  %1 = affine.max affine_map<(d0, d1) -> (d0, d1)>(%c42, %c44)

  // CHECK: return %[[r]], %[[C44]]
  return %0, %1 : index, index
}
