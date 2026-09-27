// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -canonicalize | FileCheck %s --check-prefix=CHECK-FOLD
// RUN: mlir-opt %s -emit-bytecode | mlir-opt -canonicalize | FileCheck %s --check-prefix=CHECK-FOLD

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

// Test that the in-memory representation of i1 values is correctly handled
// during bytecode roundtrip (must be unpacked to 0x01 not 0xFF).
// See llvm/llvm-project#186178.
func.func public @test_in_memory_repr() -> (tensor<32xi32> {jax.result_info = "result"}) {
  // CHECK-FOLD: dense<1> : tensor<32xi32>
  %cst = arith.constant dense<true> : tensor<32xi1>
  %0 = arith.extui %cst : tensor<32xi1> to tensor<32xi32>
  return %0 : tensor<32xi32>
}
