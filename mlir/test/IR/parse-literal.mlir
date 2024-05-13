// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: @parse_i64_tensor
func.func @parse_i64_tensor() -> tensor<4xi64> {
  // CHECK: dense<255> : tensor<4xi64>
  %0 = arith.constant dense<"0xFF00000000000000FF00000000000000FF00000000000000FF00000000000000"> : tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK-LABEL: @parse_i32_tensor
func.func @parse_i32_tensor() -> tensor<8xi32> {
  // CHECK: dense<255> : tensor<8xi32>
  %0 = arith.constant dense<"0xFF000000FF000000FF000000FF000000FF000000FF000000FF000000FF000000"> : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: @parse_i16_tensor
func.func @parse_i16_tensor() -> tensor<16xi16> {
  // CHECK: dense<255> : tensor<16xi16>
  %0 = arith.constant dense<"0xFF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00"> : tensor<16xi16>
  return %0 : tensor<16xi16>
}

// CHECK-LABEL: @parse_i8_tensor
func.func @parse_i8_tensor() -> tensor<32xi8> {
  // CHECK: dense<15> : tensor<32xi8>
  %0 = arith.constant dense<"0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F"> : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: @parse_i4_tensor
func.func @parse_i4_tensor() -> tensor<32xi4> {
  // CHECK: dense<-1> : tensor<32xi4>
  %0 = arith.constant dense<"0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F"> : tensor<32xi4>
  return %0 : tensor<32xi4>
}

// CHECK-LABEL: @parse_i1_tensor
func.func @parse_i1_tensor() -> tensor<256xi1> {
  // CHECK: dense<"0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F"> : tensor<256xi1>
  %0 = arith.constant dense<"0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F"> : tensor<256xi1>
  return %0 : tensor<256xi1>
}
