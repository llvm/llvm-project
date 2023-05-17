// RUN: mlir-opt --tosa-make-broadcastable %s | FileCheck %s

// -----
// CHECK-LABEL: broadcast0
func.func @test_broadcast0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  //  CHECK-NOT: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----
// CHECK-LABEL: broadcast1
func.func @test_broadcast1(%arg0: tensor<1xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x1xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// -----
// CHECK-LABEL: broadcast2
func.func @test_broadcast2(%arg0: tensor<2x1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// -----
// CHECK-LABEL: broadcast3
func.func @test_broadcast3(%arg0: tensor<2x1x1x1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1x1x1xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1x1x1xf32>, tensor<1xf32>) -> tensor<2x1x1x1xf32>
  return %0 : tensor<2x1x1x1xf32>
}

// -----
// CHECK-LABEL: broadcast4
func.func @test_broadcast4(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x1x2xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1x1x2xf32>, tensor<1xf32>) -> tensor<1x1x1x2xf32>
  return %0 : tensor<1x1x1x2xf32>
}

// -----
// CHECK-LABEL: broadcast5
func.func @test_broadcast5(%arg0: tensor<1x1x2x1xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x2x1xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1x2x1xf32>, tensor<1xf32>) -> tensor<1x1x2x1xf32>
  return %0 : tensor<1x1x2x1xf32>
}

// -----
// CHECK-LABEL: broadcast6
func.func @test_broadcast6(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<1xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast7
func.func @test_broadcast7(%arg0: tensor<17x16x1x14xf32>, %arg1: tensor<1x1xf32>) -> tensor<17x16x1x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x1x14xf32>, tensor<1x1xf32>) -> tensor<17x16x1x14xf32>
  return %0 : tensor<17x16x1x14xf32>
}

// -----
// CHECK-LABEL: broadcast8
func.func @test_broadcast8(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<1x1xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<1x1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast9
func.func @test_broadcast9(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x1xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 15, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<15x1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast10
func.func @test_broadcast10(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x14xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 15, 14>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast13
func.func @test_broadcast13(%arg0: tensor<1xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast14
func.func @test_broadcast14(%arg0: tensor<1x1xf32>, %arg1: tensor<17x16x1x14xf32>) -> tensor<17x16x1x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<17x16x1x14xf32>) -> tensor<17x16x1x14xf32>
  return %0 : tensor<17x16x1x14xf32>
}

// -----
// CHECK-LABEL: broadcast15
func.func @test_broadcast15(%arg0: tensor<1x1xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast16
func.func @test_broadcast16(%arg0: tensor<15x1xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 15, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<15x1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast17
func.func @test_broadcast17(%arg0: tensor<15x14xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 15, 14>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<15x14xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast18
func.func @test_broadcast18(%arg0: tensor<14x1xf32>, %arg1: tensor<1x15xf32>) -> tensor<14x15xf32> {
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<14x1xf32>, tensor<1x15xf32>) -> tensor<14x15xf32>
  return %0 : tensor<14x15xf32>
}

// -----
// CHECK-LABEL: broadcast19
func.func @test_broadcast19(%arg0: tensor<64x64x1xf32>, %arg1: tensor<1x17xf32>) -> (tensor<64x64x17xf32> ) {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 17>}
  // CHECK: %[[VAR1:.*]] = "tosa.sub"(%arg0, %[[VAR0]])
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<64x64x1xf32>, tensor<1x17xf32>) -> tensor<64x64x17xf32>
  return %0 : tensor<64x64x17xf32>
}

// -----
// CHECK-LABEL: broadcast20
func.func @test_broadcast20(%arg0: tensor<3x3x4x1xf32>, %arg1: tensor<4x5xf32>) -> (tensor<3x3x4x5xf32> ) {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 4, 5>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%arg0, %[[VAR0]])
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<3x3x4x1xf32>, tensor<4x5xf32>) -> tensor<3x3x4x5xf32>
  return %0 : tensor<3x3x4x5xf32>
}

// -----
// CHECK-LABEL: broadcast_mul
func.func @test_broadcast_mul(%arg0: tensor<15x14xi32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 15, 14>}
  // CHECK: %[[VAR1:.*]] = "tosa.mul"(%[[VAR0]], %arg1)
  %0 = "tosa.mul"(%arg0, %arg1) <{shift = 1 : i32 }> : (tensor<15x14xi32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}

// -----
// CHECK-LABEL: broadcast_arithmetic_right_shift
func.func @test_broadcast_arithmetic_right_shift(%arg0: tensor<15x14xi32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 15, 14>}
  // CHECK: %[[VAR1:.*]] = "tosa.arithmetic_right_shift"(%[[VAR0]], %arg1)
  %0 = "tosa.arithmetic_right_shift"(%arg0, %arg1) <{ round = true }> : (tensor<15x14xi32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}

// -----
// CHECK-LABEL: broadcast_scalar
func.func @test_broadcast_scalar(%arg0: tensor<i32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAR1:.*]] = "tosa.add"(%[[VAR0]], %arg1)
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<i32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}

// -----
// CHECK-LABEL: broadcast_select_both_input
func.func @test_broadcast_select_both_input(%arg0: tensor<1x16x16xi1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1x16x16xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%arg0, %[[VAL_0]], %[[VAL_1]])
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x16x16xi1>, tensor<f32>, tensor<f32>) -> tensor<1x16x16xf32>
  return %0 : tensor<1x16x16xf32>
}

// -----
// CHECK-LABEL: broadcast_select_one_input
func.func @test_broadcast_select_one_input(%arg0: tensor<17x16x15x14xi1>, %arg1: tensor<17x16x15x14xf32>, %arg2: tensor<f32>) -> tensor<17x16x15x14xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAL_1:.*]] = "tosa.select"(%arg0, %arg1, %[[VAL_0]])
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<17x16x15x14xi1>, tensor<17x16x15x14xf32>, tensor<f32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast_select_predicate
func.func @test_broadcast_select_predicate(%arg0: tensor<i1>, %arg1: tensor<1x32x32x8xf32>, %arg2: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAL_1:.*]] = "tosa.select"(%[[VAL_0]], %arg1, %arg2)
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: broadcast_select_abc
func.func @test_broadcast_select_abc(%arg0: tensor<i1>, %arg1: tensor<32x8xf32>, %arg2: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 32, 8>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%[[VAL_0]], %[[VAL_1]], %arg2)
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<32x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: broadcast_select_acb
func.func @test_broadcast_select_acb(%arg0: tensor<i1>, %arg1: tensor<1x32x32x8xf32>, %arg2: tensor<32x8xf32>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 32, 8>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%[[VAL_0]], %arg1, %[[VAL_1]])
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<1x32x32x8xf32>, tensor<32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: broadcast_select_bac
func.func @test_broadcast_select_bac(%arg0: tensor<32x8xi1>, %arg1: tensor<f32>, %arg2: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 32, 8>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%[[VAL_0]], %[[VAL_1]], %arg2)
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<32x8xi1>, tensor<f32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: broadcast_select_bca
func.func @test_broadcast_select_bca(%arg0: tensor<32x8xi1>, %arg1: tensor<1x32x32x8xf32>, %arg2: tensor<i1>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 1, 32, 8>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%[[VAL_0]], %arg1, %[[VAL_1]])
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<32x8xi1>, tensor<1x32x32x8xf32>, tensor<i1>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: broadcast_select_cab
func.func @test_broadcast_select_cab(%arg0: tensor<1x32x32x8xi1>, %arg1: tensor<f32>, %arg2: tensor<32x8xf32>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 32, 8>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%arg0, %[[VAL_0]], %[[VAL_1]])
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x32x32x8xi1>, tensor<f32>, tensor<32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: broadcast_select_cba
func.func @test_broadcast_select_cba(%arg0: tensor<1x32x32x8xi1>, %arg1: tensor<32x8xf32>, %arg2: tensor<i1>) -> tensor<1x32x32x8xf32> {
  // CHECK-DAG: %[[VAL_0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 1, 32, 8>}
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[VAL_2:.*]] = "tosa.select"(%arg0, %[[VAL_0]], %[[VAL_1]])
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x32x32x8xi1>, tensor<32x8xf32>, tensor<i1>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}
