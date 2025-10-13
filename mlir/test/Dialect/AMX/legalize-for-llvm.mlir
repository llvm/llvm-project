// RUN: mlir-opt %s -convert-vector-to-llvm="enable-amx" | mlir-opt | FileCheck %s

// CHECK-LABEL: muli(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tdpbuud.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tdpbssd.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tdpbusd.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tdpbsud.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @muli(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_zero : !amx.tile<16x64xi8>
  %2 = amx.tile_load %arg0[%0, %0] : memref<?x?xi8> into !amx.tile<16x64xi8>
  %3 = amx.tile_load %arg1[%0, %0] : memref<?x?xi32> into !amx.tile<16x16xi32>
  %4 = amx.tile_muli %1 zext, %2 zext, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  amx.tile_store %arg1[%0, %0], %4 : memref<?x?xi32>, !amx.tile<16x16xi32>
  %5 = amx.tile_muli %1, %2, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  amx.tile_store %arg1[%0, %0], %5 : memref<?x?xi32>, !amx.tile<16x16xi32>
  %6 = amx.tile_muli %1 zext, %2, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  amx.tile_store %arg1[%0, %0], %6 : memref<?x?xi32>, !amx.tile<16x16xi32>
  %7 = amx.tile_muli %1, %2 zext, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  amx.tile_store %arg1[%0, %0], %7  : memref<?x?xi32>, !amx.tile<16x16xi32>
  return
}

// CHECK-LABEL: mulbf16(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tdpbf16ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @mulbf16(%arg0: memref<?x?xbf16>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_zero : !amx.tile<16x32xbf16>
  %2 = amx.tile_load %arg0[%0, %0] : memref<?x?xbf16> into !amx.tile<16x32xbf16>
  %3 = amx.tile_load %arg1[%0, %0] : memref<?x?xf32> into !amx.tile<16x16xf32>
  %4 = amx.tile_mulf %1, %2, %3 : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
  amx.tile_store %arg1[%0, %0], %4 : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: mulfp16(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tdpfp16ps.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"
func.func @mulfp16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_zero : !amx.tile<16x32xf16>
  %2 = amx.tile_load %arg0[%0, %0] : memref<?x?xf16> into !amx.tile<16x32xf16>
  %3 = amx.tile_load %arg1[%0, %0] : memref<?x?xf32> into !amx.tile<16x16xf32>
  %4 = amx.tile_mulf %1, %2, %3 : !amx.tile<16x32xf16>, !amx.tile<16x32xf16>, !amx.tile<16x16xf32>
  amx.tile_store %arg1[%0, %0], %4 : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

/// Intrinsics require stride in number of bytes.
// CHECK-LABEL: strides_implicit(
// CHECK: %[[LOAD_STRIDE_1:.+]] = llvm.mlir.constant(32 : i64) : i64
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[LOAD_STRIDE_1]]
// CHECK: %[[LOAD_STRIDE_2:.+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[LOAD_STRIDE_2]]
// CHECK: llvm.extractvalue %{{.+}}[4, 0]
// CHECK: %[[LOAD_BUF_STRIDE:.+]] = llvm.extractvalue %{{.+}}[4, 0]
// CHECK: %[[LOAD_STRIDE_SCALE:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[LOAD_STRIDE_3:.+]] = llvm.mul %[[LOAD_STRIDE_SCALE]], %[[LOAD_BUF_STRIDE]]
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[LOAD_STRIDE_3]]
// CHECK: %[[STORE_STRIDE_1:.+]] = llvm.mlir.constant(32 : i64) : i64
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STORE_STRIDE_1]]
// CHECK: %[[STORE_STRIDE_2:.+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STORE_STRIDE_2]]
// CHECK: llvm.extractvalue %{{.+}}[4, 0]
// CHECK: %[[STORE_BUF_STRIDE:.+]] = llvm.extractvalue %{{.+}}[4, 0]
// CHECK: %[[STORE_STRIDE_SCALE:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[STORE_STRIDE_3:.+]] = llvm.mul %[[STORE_STRIDE_SCALE]], %[[STORE_BUF_STRIDE]]
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STORE_STRIDE_3]]
func.func @strides_implicit(%arg0: memref<16x32xi8>,
    %arg1: memref<32x32xbf16, strided<[64, 1]>>,
    %arg2: memref<16x32xf32, strided<[?, 1]>>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x32xi8> into !amx.tile<16x32xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<32x32xbf16, strided<[64, 1]>> into !amx.tile<16x32xbf16>
  %3 = amx.tile_load %arg2[%0, %0] : memref<16x32xf32, strided<[?, 1]>> into !amx.tile<16x16xf32>
  amx.tile_store %arg0[%0, %0], %1 : memref<16x32xi8>, !amx.tile<16x32xi8>
  amx.tile_store %arg1[%0, %0], %2 : memref<32x32xbf16, strided<[64, 1]>>, !amx.tile<16x32xbf16>
  amx.tile_store %arg2[%0, %0], %3 : memref<16x32xf32, strided<[?, 1]>>, !amx.tile<16x16xf32>
  return
}

/// Intrinsics require stride in number of bytes.
// CHECK-LABEL: strides_explicit(
// CHECK-SAME:    %[[STRIDE:.+]]: index
// CHECK-DAG: %[[STRIDE_I64:.+]] = builtin.unrealized_conversion_cast %[[STRIDE]] : index to i64
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C64_I64:.+]] = builtin.unrealized_conversion_cast %[[C64]] : index to i64
// CHECK: %[[LOAD_STRIDE_SCALE_1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[LOAD_STRIDE_1:.+]] = llvm.mul %[[LOAD_STRIDE_SCALE_1]], %[[STRIDE_I64]]
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[LOAD_STRIDE_1]]
// CHECK: %[[LOAD_STRIDE_SCALE_2:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[LOAD_STRIDE_2:.+]] = llvm.mul %[[LOAD_STRIDE_SCALE_2]], %[[STRIDE_I64]]
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[LOAD_STRIDE_2]]
// CHECK: %[[LOAD_STRIDE_SCALE_3:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[LOAD_STRIDE_3:.+]] = llvm.mul %[[LOAD_STRIDE_SCALE_3]], %[[C64_I64]]
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[LOAD_STRIDE_3]]
// CHECK: %[[STORE_STRIDE_SCALE_1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[STORE_STRIDE_1:.+]] = llvm.mul %[[STORE_STRIDE_SCALE_1]], %[[STRIDE_I64]]
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STORE_STRIDE_1]]
// CHECK: %[[STORE_STRIDE_SCALE_2:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[STORE_STRIDE_2:.+]] = llvm.mul %[[STORE_STRIDE_SCALE_2]], %[[STRIDE_I64]]
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STORE_STRIDE_2]]
// CHECK: %[[STORE_STRIDE_SCALE_3:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[STORE_STRIDE_3:.+]] = llvm.mul %[[STORE_STRIDE_SCALE_3]], %[[C64_I64]]
// CHECK: llvm.call_intrinsic "llvm.x86.tilestored64.internal"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STORE_STRIDE_3]]
func.func @strides_explicit(%stride: index,
    %arg0: memref<?xi8>,
    %arg1: memref<16x32xbf16>,
    %arg2: memref<32x32xf32, strided<[64, 1]>>) {
  %0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %1 = amx.tile_load %arg0[%0], %stride : memref<?xi8> into !amx.tile<16x32xi8>
  %2 = amx.tile_load %arg1[%0, %0], %stride : memref<16x32xbf16> into !amx.tile<16x32xbf16>
  %3 = amx.tile_load %arg2[%0, %0], %c64 : memref<32x32xf32, strided<[64, 1]>> into !amx.tile<16x16xf32>
  amx.tile_store %arg0[%0], %1, %stride : memref<?xi8>, !amx.tile<16x32xi8>
  amx.tile_store %arg1[%0, %0], %2, %stride : memref<16x32xbf16>, !amx.tile<16x32xbf16>
  amx.tile_store %arg2[%0, %0], %3, %c64 : memref<32x32xf32, strided<[64, 1]>>, !amx.tile<16x16xf32>
  return
}
