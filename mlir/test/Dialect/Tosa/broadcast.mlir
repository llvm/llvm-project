// RUN: mlir-opt --tosa-make-broadcastable %s | FileCheck %s

// -----
// CHECK-LABEL: broadcast0
func @test_broadcast0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  //  CHECK-NOT: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----
// CHECK-LABEL: broadcast1
func @test_broadcast1(%arg0: tensor<1xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x1xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// -----
// CHECK-LABEL: broadcast2
func @test_broadcast2(%arg0: tensor<2x1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// -----
// CHECK-LABEL: broadcast3
func @test_broadcast3(%arg0: tensor<2x1x1x1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1x1x1xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1x1x1xf32>, tensor<1xf32>) -> tensor<2x1x1x1xf32>
  return %0 : tensor<2x1x1x1xf32>
}

// -----
// CHECK-LABEL: broadcast4
func @test_broadcast4(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x1x2xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1x1x2xf32>, tensor<1xf32>) -> tensor<1x1x1x2xf32>
  return %0 : tensor<1x1x1x2xf32>
}

// -----
// CHECK-LABEL: broadcast5
func @test_broadcast5(%arg0: tensor<1x1x2x1xf32>, %arg1: tensor<1xf32>) -> tensor<1x1x2x1xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1x2x1xf32>, tensor<1xf32>) -> tensor<1x1x2x1xf32>
  return %0 : tensor<1x1x2x1xf32>
}

// -----
// CHECK-LABEL: broadcast6
func @test_broadcast6(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<1xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast7
func @test_broadcast7(%arg0: tensor<17x16x1x14xf32>, %arg1: tensor<1x1xf32>) -> tensor<17x16x1x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x1x14xf32>, tensor<1x1xf32>) -> tensor<17x16x1x14xf32>
  return %0 : tensor<17x16x1x14xf32>
}

// -----
// CHECK-LABEL: broadcast8
func @test_broadcast8(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<1x1xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<1x1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast9
func @test_broadcast9(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x1xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<15x1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast10
func @test_broadcast10(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x14xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast13
func @test_broadcast13(%arg0: tensor<1xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast14
func @test_broadcast14(%arg0: tensor<1x1xf32>, %arg1: tensor<17x16x1x14xf32>) -> tensor<17x16x1x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<17x16x1x14xf32>) -> tensor<17x16x1x14xf32>
  return %0 : tensor<17x16x1x14xf32>
}

// -----
// CHECK-LABEL: broadcast15
func @test_broadcast15(%arg0: tensor<1x1xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast16
func @test_broadcast16(%arg0: tensor<15x1xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<15x1xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast17
func @test_broadcast17(%arg0: tensor<15x14xf32>, %arg1: tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32> {
  //  CHECK: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<15x14xf32>, tensor<17x16x15x14xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}

// -----
// CHECK-LABEL: broadcast18
func @test_broadcast18(%arg0: tensor<14x1xf32>, %arg1: tensor<1x15xf32>) -> tensor<14x15xf32> {
  //  CHECK: add
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<14x1xf32>, tensor<1x15xf32>) -> tensor<14x15xf32>
  return %0 : tensor<14x15xf32>
}

// -----
// CHECK-LABEL: broadcast_mul
func @test_broadcast_mul(%arg0: tensor<15x14xi32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  //  CHECK: reshape
  %0 = "tosa.mul"(%arg0, %arg1) {shift = 1 : i32 } : (tensor<15x14xi32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}

// -----
// CHECK-LABEL: broadcast_arithmetic_right_shift
func @test_broadcast_arithmetic_right_shift(%arg0: tensor<15x14xi32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  //  CHECK: reshape
  %0 = "tosa.arithmetic_right_shift"(%arg0, %arg1) { round = true } : (tensor<15x14xi32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}

// -----
// CHECK-LABEL: broadcast_scalar
func @test_broadcast_scalar(%arg0: tensor<i32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  //  CHECK-NEXT: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<i32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}
