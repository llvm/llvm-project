// RUN: mlir-opt %s --convert-vector-to-llvm="enable-amx" --convert-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | FileCheck %s

// CHECK-LABEL: define void @amx_tile_zero
func.func @amx_tile_zero(%out: memref<?x?xf32>, %idx: index)
{
  // CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  // CHECK: call void @llvm.x86.tilestored64.internal
  %zero = amx.tile_zero : !amx.tile<16x16xf32>
  amx.tile_store %out[%idx, %idx], %zero : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: define void @amx_tile_load_store
func.func @amx_tile_load_store(%base: memref<?x?xi8>, %out: memref<?x?xi8>,
    %idx: index)
{
  // CHECK: call x86_amx @llvm.x86.tileloadd64.internal
  // CHECK: call void @llvm.x86.tilestored64.internal
  %val = amx.tile_load %base[%idx, %idx] : memref<?x?xi8> into !amx.tile<16x64xi8>
  amx.tile_store %out[%idx, %idx], %val : memref<?x?xi8>, !amx.tile<16x64xi8>
  return
}

// CHECK-LABEL: define void @amx_tile_load_store_strided
func.func @amx_tile_load_store_strided(%base: memref<?xi8>, %out: memref<?xi8>,
    %idx: index, %stride: index)
{
  // CHECK: call x86_amx @llvm.x86.tileloadd64.internal
  // CHECK: call void @llvm.x86.tilestored64.internal
  %val = amx.tile_load %base[%idx], %stride
    : memref<?xi8> into !amx.tile<16x64xi8>
  amx.tile_store %out[%idx], %val, %stride
    : memref<?xi8>, !amx.tile<16x64xi8>
  return
}

// CHECK-LABEL: define void @amx_tile_mulf_bf16
func.func @amx_tile_mulf_bf16(
    %matA: memref<?x?xbf16>, %matB: memref<?x?xbf16>, %idx: index,
    %out: memref<?x?xf32>)
{
  // CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %acc = amx.tile_zero : !amx.tile<16x16xf32>
  // CHECK-COUNT-2: call x86_amx @llvm.x86.tileloadd64.internal
  %tA = amx.tile_load %matA[%idx, %idx] : memref<?x?xbf16> into !amx.tile<16x32xbf16>
  %tB = amx.tile_load %matB[%idx, %idx] : memref<?x?xbf16> into !amx.tile<16x32xbf16>
  // CHECK: call x86_amx @llvm.x86.tdpbf16ps.internal
  %tRes = amx.tile_mulf %tA, %tB, %acc
    : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
  // CHECK: call void @llvm.x86.tilestored64.internal
  amx.tile_store %out[%idx, %idx], %tRes : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: define void @amx_tile_mulf_f16
func.func @amx_tile_mulf_f16(
    %matA: memref<?x?xf16>, %matB: memref<?x?xf16>, %idx: index,
    %out: memref<?x?xf32>)
{
  // CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %acc = amx.tile_zero : !amx.tile<16x16xf32>
  // CHECK-COUNT-2: call x86_amx @llvm.x86.tileloadd64.internal
  %tA = amx.tile_load %matA[%idx, %idx] : memref<?x?xf16> into !amx.tile<16x32xf16>
  %tB = amx.tile_load %matB[%idx, %idx] : memref<?x?xf16> into !amx.tile<16x32xf16>
  // CHECK: call x86_amx @llvm.x86.tdpfp16ps.internal
  %tRes = amx.tile_mulf %tA, %tB, %acc
    : !amx.tile<16x32xf16>, !amx.tile<16x32xf16>, !amx.tile<16x16xf32>
    // CHECK: call void @llvm.x86.tilestored64.internal
  amx.tile_store %out[%idx, %idx], %tRes : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: define void @amx_tile_muli
func.func @amx_tile_muli(%matA: memref<?x?xi8>, %matB: memref<?x?xi8>,
    %matC: memref<?x?xi32>, %idx: index, %out: memref<?x?xi8>)
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  // CHECK-COUNT-3: call x86_amx @llvm.x86.tileloadd64.internal
  %tA = amx.tile_load %matA[%idx, %idx] : memref<?x?xi8> into !amx.tile<16x64xi8>
  %tB = amx.tile_load %matB[%idx, %idx] : memref<?x?xi8> into !amx.tile<16x64xi8>
  %acc = amx.tile_load %matC[%idx, %idx] : memref<?x?xi32> into !amx.tile<16x16xi32>
  // CHECK: call x86_amx @llvm.x86.tdpbuud.internal
  // CHECK: call x86_amx @llvm.x86.tdpbssd.internal
  // CHECK: call x86_amx @llvm.x86.tdpbusd.internal
  // CHECK: call x86_amx @llvm.x86.tdpbsud.internal
  %res = amx.tile_muli %tA zext, %tB zext, %acc
    : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  %res1 = amx.tile_muli %tA, %tB, %acc
    : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  %res2 = amx.tile_muli %tA zext, %tB, %acc
    : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  %res3 = amx.tile_muli %tA, %tB zext, %acc
    : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  // CHECK-COUNT-4: call void @llvm.x86.tilestored64.internal
  amx.tile_store %out[%c0, %c0], %res : memref<?x?xi8>, !amx.tile<16x16xi32>
  amx.tile_store %out[%c0, %c16], %res1 : memref<?x?xi8>, !amx.tile<16x16xi32>
  amx.tile_store %out[%c16, %c0], %res2 : memref<?x?xi8>, !amx.tile<16x16xi32>
  amx.tile_store %out[%c16, %c16], %res3 : memref<?x?xi8>, !amx.tile<16x16xi32>
  return
}

// CHECK-LABEL: define void @amx_tile_type_through_cf
func.func @amx_tile_type_through_cf(%src: memref<?x?xi8>, %out: memref<?x?xi8>,
    %idx: index, %cond: i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: call x86_amx @llvm.x86.tileloadd64.internal
  %0 = amx.tile_load %src[%idx, %idx] : memref<?x?xi8> into !amx.tile<16x64xi8>
  cf.br ^bb3(%0 : !amx.tile<16x64xi8>)
^bb2:  // pred: ^bb0
  // CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %1 = amx.tile_zero : !amx.tile<16x64xi8>
  cf.br ^bb3(%1 : !amx.tile<16x64xi8>)
^bb3(%2: !amx.tile<16x64xi8>):  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // pred: ^bb3
  // CHECK: call void @llvm.x86.tilestored64.internal
  amx.tile_store %out[%idx, %idx], %2 : memref<?x?xi8>, !amx.tile<16x64xi8>
  return
}
