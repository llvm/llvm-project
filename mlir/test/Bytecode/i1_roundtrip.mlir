// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @test_i1_splat_true
func.func @test_i1_splat_true() -> tensor<100xi1> {
// CHECK: arith.constant dense<true> : tensor<100xi1>
  %0 = arith.constant dense<true> : tensor<100xi1>
  return %0 : tensor<100xi1>
}


// CHECK-LABEL: func.func @test_i1_splat_false
func.func @test_i1_splat_false() -> tensor<100xi1> {
// CHECK: arith.constant dense<false> : tensor<100xi1>
  %0 = arith.constant dense<false> : tensor<100xi1>
  return %0 : tensor<100xi1>
}


// CHECK-LABEL: func.func @test_8xi1_splat_true
func.func @test_8xi1_splat_true() -> tensor<8xi1> {
// CHECK: arith.constant dense<true> : tensor<8xi1>
  %0 = arith.constant dense<true> : tensor<8xi1>
  return %0 : tensor<8xi1>
}

// CHECK-LABEL: func.func @test_8xi1_splat_false
func.func @test_8xi1_splat_false() -> tensor<8xi1> {
// CHECK: arith.constant dense<false> : tensor<8xi1>
  %0 = arith.constant dense<false> : tensor<8xi1>
  return %0 : tensor<8xi1>
}

// CHECK-LABEL: func.func @test_i8_mixed()
func.func @test_i8_mixed() {
  // CHECK: arith.constant dense<[true, false, true, false, true, false, true, false]> : tensor<8xi1>
  %0 = arith.constant dense<[true, false, true, false, true, false, true, false]> : tensor<8xi1>
  return
}

// CHECK-LABEL: func.func @test_i9_mixed()
func.func @test_i9_mixed() {
  // CHECK: arith.constant dense<[true, false, true, false, true, false, true, false, true]> : tensor<9xi1>
  %0 = arith.constant dense<[true, false, true, false, true, false, true, false, true]> : tensor<9xi1>
  return
}
