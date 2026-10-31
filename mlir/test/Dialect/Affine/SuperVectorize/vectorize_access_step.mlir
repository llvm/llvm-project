// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0 vectorize-reductions=true" | FileCheck %s
// RUN: mlir-opt %s -affine-loop-normalize="promote-single-iter=false" -affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0 vectorize-reductions=true" | FileCheck %s

// CHECK-LABEL: func.func @unit_address_step
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
func.func @unit_address_step(%input: memref<32xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 16 iter_args(%sum = %zero) -> i32 {
    %value = affine.load %input[%index] : memref<32xi32>
    %next = arith.addi %sum, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @invariant_quotient_offset
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
func.func @invariant_quotient_offset(%input: memref<?xi32>, %offset: index) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 16 iter_args(%sum = %zero) -> i32 {
    %value = affine.load %input[%index + symbol(%offset) floordiv 3] : memref<?xi32>
    %next = arith.addi %sum, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @double_address_step
// CHECK-NOT: vector.transfer_read
// CHECK: affine.load
// CHECK-NOT: vector.transfer_read
// CHECK: return
func.func @double_address_step(%input: memref<32xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 8 iter_args(%sum = %zero) -> i32 {
    %value = affine.load %input[2 * %index] : memref<32xi32>
    %next = arith.addi %sum, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @normalized_partial_group
// CHECK-NOT: vector.transfer_read
// CHECK: affine.load
// CHECK-NOT: vector.transfer_read
// CHECK: return
func.func @normalized_partial_group(%input: memref<32xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 11 step 2 iter_args(%sum = %zero) -> i32 {
    %value = affine.load %input[%index] : memref<32xi32>
    %next = arith.addi %sum, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @modulo_address
// CHECK-NOT: vector.transfer_read
// CHECK: affine.load
// CHECK-NOT: vector.transfer_read
// CHECK: return
func.func @modulo_address(%input: memref<32xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 8 iter_args(%sum = %zero) -> i32 {
    %value = affine.load %input[%index mod 3] : memref<32xi32>
    %next = arith.addi %sum, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @derived_modulo_address
// CHECK-NOT: vector.transfer_read
// CHECK: affine.load
// CHECK-NOT: vector.transfer_read
// CHECK: return
func.func @derived_modulo_address(%input: memref<32xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 8 iter_args(%sum = %zero) -> i32 {
    %address = affine.apply affine_map<(d0) -> (d0 mod 3)>(%index)
    %value = affine.load %input[%address] : memref<32xi32>
    %next = arith.addi %sum, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @negative_address_step
// CHECK-NOT: vector.transfer_write
// CHECK: affine.store
// CHECK-NOT: vector.transfer_write
// CHECK: return
func.func @negative_address_step(%output: memref<32xi32>) {
  %zero = arith.constant 0 : i32
  affine.for %index = 0 to 16 {
    affine.store %zero, %output[31 - %index] : memref<32xi32>
  }
  return
}