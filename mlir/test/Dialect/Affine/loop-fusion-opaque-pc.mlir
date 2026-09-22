// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer maximal}))' | FileCheck %s

// Opaque func.call on a producer-consumer memref must block fusion.
// CHECK-LABEL: func @fusion_blocked_by_opaque_call_on_pc_memref
func.func private @opaque(memref<32xf64>)
func.func @fusion_blocked_by_opaque_call_on_pc_memref(
    %in: memref<32xf64>, %comm: memref<32xf64>, %out: memref<32xf64>) {
  affine.for %i = 0 to 16 {
    %a = affine.load %in[%i] : memref<32xf64>
    %b = arith.addf %a, %a : f64
    affine.store %b, %comm[%i] : memref<32xf64>
    func.call @opaque(%comm) : (memref<32xf64>) -> ()
  }
  affine.for %j = 0 to 16 {
    %c = affine.load %comm[%j] : memref<32xf64>
    %d = arith.addf %c, %c : f64
    affine.store %d, %out[%j] : memref<32xf64>
  }
  // CHECK: affine.for
  // CHECK: func.call @opaque
  // CHECK: affine.for
  return
}

// CHECK-LABEL: func @fusion_still_happens_without_opaque_effect
func.func @fusion_still_happens_without_opaque_effect(
    %in: memref<32xf64>, %comm: memref<32xf64>, %out: memref<32xf64>) {
  affine.for %i = 0 to 16 {
    %a = affine.load %in[%i] : memref<32xf64>
    %b = arith.addf %a, %a : f64
    affine.store %b, %comm[%i] : memref<32xf64>
  }
  affine.for %j = 0 to 16 {
    %c = affine.load %comm[%j] : memref<32xf64>
    %d = arith.addf %c, %c : f64
    affine.store %d, %out[%j] : memref<32xf64>
  }
  // CHECK: affine.for
  // CHECK-NOT: affine.for
  return
}

// A call on an unrelated memref must not blanket-block PC fusion of %comm.
// CHECK-LABEL: func @call_on_unrelated_memref_does_not_block_pc_fusion
func.func @call_on_unrelated_memref_does_not_block_pc_fusion(
    %in: memref<32xf64>, %comm: memref<32xf64>,
    %scratch: memref<32xf64>, %out: memref<32xf64>) {
  affine.for %i = 0 to 16 {
    %a = affine.load %in[%i] : memref<32xf64>
    %b = arith.addf %a, %a : f64
    affine.store %b, %comm[%i] : memref<32xf64>
    func.call @opaque(%scratch) : (memref<32xf64>) -> ()
  }
  affine.for %j = 0 to 16 {
    %c = affine.load %comm[%j] : memref<32xf64>
    %d = arith.addf %c, %c : f64
    affine.store %d, %out[%j] : memref<32xf64>
  }
  // CHECK: affine.for
  // CHECK: func.call @opaque
  // CHECK-NOT: affine.for
  return
}
