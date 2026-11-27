// RUN: mlir-opt %s --convert-vector-to-llvm="enable-x86" --convert-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | FileCheck %s

// CHECK-LABEL: define void @ace_tile_mulf_bf16
func.func @ace_tile_mulf_bf16(%vecA: vector<32xbf16>, %vecB: vector<32xbf16>,
    %idx: index, %out: memref<?x?xf32>)
{
  // CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %acc = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  // CHECK: call x86_amx @llvm.x86.acev1.top2bf16ps.internal
  %tRes = x86.ace.tile_mulf %vecA, %vecB, %acc
    : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
  // CHECK: call void @llvm.x86.tilestored64.internal
  x86.amx.tile_store %out[%idx, %idx], %tRes : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: define void @ace_tile_mulf8
func.func @ace_tile_mulf8(
    %hA: vector<64xf8E4M3FN>, %hB: vector<64xf8E4M3FN>,
    %bA: vector<64xf8E5M2>, %bB: vector<64xf8E5M2>,
    %idx: index, %out: memref<?x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  // CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %acc = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
  // CHECK: call x86_amx @llvm.x86.acev1.top4mxhf8ps.internal
  // CHECK: call x86_amx @llvm.x86.acev1.top4mxhbf8ps.internal
  // CHECK: call x86_amx @llvm.x86.acev1.top4mxbhf8ps.internal
  // CHECK: call x86_amx @llvm.x86.acev1.top4mxbf8ps.internal
  %tRes = x86.ace.tile_mulf %hA, %hB, %acc, 0
    : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
  %tRes1 = x86.ace.tile_mulf %hA, %bB, %acc, 1
    : vector<64xf8E4M3FN>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
  %tRes2 = x86.ace.tile_mulf %bA, %hB, %acc, 2
    : vector<64xf8E5M2>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
  %tRes3 = x86.ace.tile_mulf %bA, %bB, %acc, 3
    : vector<64xf8E5M2>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
  // CHECK-COUNT-4: call void @llvm.x86.tilestored64.internal
  x86.amx.tile_store %out[%idx, %c0], %tRes : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %out[%idx, %c16], %tRes1 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %out[%c0, %idx], %tRes2 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  x86.amx.tile_store %out[%c16, %idx], %tRes3 : memref<?x?xf32>, !x86.amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: define void @ace_tile_muli
func.func @ace_tile_muli(%vecA: vector<64xi8>, %vecB: vector<64xi8>,
    %matC: memref<?x?xi32>, %idx: index, %out: memref<?x?xi8>)
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  // CHECK: call x86_amx @llvm.x86.tileloadd64.internal
  %acc = x86.amx.tile_load %matC[%idx, %idx] : memref<?x?xi32> into !x86.amx.tile<16x16xi32>
  // CHECK: call x86_amx @llvm.x86.acev1.top4buud.internal
  // CHECK: call x86_amx @llvm.x86.acev1.top4bssd.internal
  // CHECK: call x86_amx @llvm.x86.acev1.top4busd.internal
  // CHECK: call x86_amx @llvm.x86.acev1.top4bsud.internal
  %res = x86.ace.tile_muli %vecA zext, %vecB zext, %acc
    : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  %res1 = x86.ace.tile_muli %vecA, %vecB, %acc
    : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  %res2 = x86.ace.tile_muli %vecA zext, %vecB, %acc
    : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  %res3 = x86.ace.tile_muli %vecA, %vecB zext, %acc
    : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
  // CHECK-COUNT-4: call void @llvm.x86.tilestored64.internal
  x86.amx.tile_store %out[%c0, %c0], %res : memref<?x?xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %out[%c0, %c16], %res1 : memref<?x?xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %out[%c16, %c0], %res2 : memref<?x?xi8>, !x86.amx.tile<16x16xi32>
  x86.amx.tile_store %out[%c16, %c16], %res3 : memref<?x?xi8>, !x86.amx.tile<16x16xi32>
  return
}
