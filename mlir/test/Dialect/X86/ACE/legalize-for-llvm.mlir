// RUN: mlir-opt %s -convert-vector-to-llvm="enable-x86" | mlir-opt | FileCheck %s

// CHECK-LABEL: ace_mulf8_bf8(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4mxbf8ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_mulf8_bf8(%lhs: vector<64xf8E5M2>, %rhs: vector<64xf8E5M2>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  %2 = x86.ace.tile_mulf %lhs, %rhs, %1, 0 : vector<64xf8E5M2>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: ace_mulf8_hf8(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4mxhf8ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_mulf8_hf8(%lhs: vector<64xf8E4M3FN>, %rhs: vector<64xf8E4M3FN>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  %2 = x86.ace.tile_mulf %lhs, %rhs, %1, 1 : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: ace_mulf8_bf8xhf8(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4mxbhf8ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_mulf8_bf8xhf8(%lhs: vector<64xf8E5M2>, %rhs: vector<64xf8E4M3FN>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  %2 = x86.ace.tile_mulf %lhs, %rhs, %1, 2 : vector<64xf8E5M2>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: ace_mulf8_hf8xbf8(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4mxhbf8ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_mulf8_hf8xbf8(%lhs: vector<64xf8E4M3FN>, %rhs: vector<64xf8E5M2>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  %2 = x86.ace.tile_mulf %lhs, %rhs, %1, 3 : vector<64xf8E4M3FN>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: ace_mulf_bf16(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top2bf16ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_mulf_bf16(%lhs: vector<32xbf16>, %rhs: vector<32xbf16>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  %2 = x86.ace.tile_mulf %lhs, %rhs, %1 : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: ace_muli_uu(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4buud.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_muli_uu(%lhs: vector<64xi8>, %rhs: vector<64xi8>, %arg1: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
  %2 = x86.ace.tile_muli %lhs zext, %rhs zext, %1 : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xi32>, !x86.amx.tile<16x16xi32>
  return 
}

// CHECK-LABEL: ace_muli_ss(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4bssd.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_muli_ss(%lhs: vector<64xi8>, %rhs: vector<64xi8>, %arg1: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
  %2 = x86.ace.tile_muli %lhs, %rhs, %1 : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xi32>, !x86.amx.tile<16x16xi32>
  return
}

// CHECK-LABEL: ace_muli_us(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4busd.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_muli_us(%lhs: vector<64xi8>, %rhs: vector<64xi8>, %arg1: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
  %2 = x86.ace.tile_muli %lhs zext, %rhs, %1 : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xi32>, !x86.amx.tile<16x16xi32>
  return
}

// CHECK-LABEL: ace_muli_su(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.acev1.top4bsud.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @ace_muli_su(%lhs: vector<64xi8>, %rhs: vector<64xi8>, %arg1: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
  %2 = x86.ace.tile_muli %lhs, %rhs zext, %1 : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %arg1[%0, %0], %2 : memref<?x?xi32>, !x86.amx.tile<16x16xi32>
  return
}
