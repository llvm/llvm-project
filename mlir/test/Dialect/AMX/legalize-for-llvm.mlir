// RUN: mlir-opt %s -convert-vector-to-llvm="enable-amx" | mlir-opt | FileCheck %s

// CHECK-LABEL: muli(
// CHECK: amx.tilezero
// CHECK: amx.tileloadd64
// CHECK: amx.tileloadd64
// CHECK: amx.tdpbuud
// CHECK: amx.tilestored64
// CHECK: amx.tdpbssd
// CHECK: amx.tilestored64
// CHECK: amx.tdpbusd
// CHECK: amx.tilestored64
// CHECK: amx.tdpbsud
// CHECK: amx.tilestored64
func.func @muli(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_zero : vector<16x64xi8>
  %2 = amx.tile_load %arg0[%0, %0] : memref<?x?xi8> into vector<16x64xi8>
  %3 = amx.tile_load %arg1[%0, %0] : memref<?x?xi32> into vector<16x16xi32>
  %4 = amx.tile_muli %1 zext, %2 zext, %3 : vector<16x64xi8>, vector<16x64xi8>, vector<16x16xi32>
  amx.tile_store %arg1[%0, %0], %4 : memref<?x?xi32>, vector<16x16xi32>
  %5 = amx.tile_muli %1, %2, %3 : vector<16x64xi8>, vector<16x64xi8>, vector<16x16xi32>
  amx.tile_store %arg1[%0, %0], %5 : memref<?x?xi32>, vector<16x16xi32>
  %6 = amx.tile_muli %1 zext, %2, %3 : vector<16x64xi8>, vector<16x64xi8>, vector<16x16xi32>
  amx.tile_store %arg1[%0, %0], %6 : memref<?x?xi32>, vector<16x16xi32>
  %7 = amx.tile_muli %1, %2 zext, %3 : vector<16x64xi8>, vector<16x64xi8>, vector<16x16xi32>
  amx.tile_store %arg1[%0, %0], %7  : memref<?x?xi32>, vector<16x16xi32>
  return
}

// CHECK-LABEL: mulf(
// CHECK: amx.tilezero
// CHECK: amx.tileloadd64
// CHECK: amx.tileloadd64
// CHECK: amx.tdpbf16ps
// CHECK: amx.tilestored64
func.func @mulf(%arg0: memref<?x?xbf16>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_zero : vector<16x32xbf16>
  %2 = amx.tile_load %arg0[%0, %0] : memref<?x?xbf16> into vector<16x32xbf16>
  %3 = amx.tile_load %arg1[%0, %0] : memref<?x?xf32> into vector<16x16xf32>
  %4 = amx.tile_mulf %1, %2, %3 : vector<16x32xbf16>, vector<16x32xbf16>, vector<16x16xf32>
  amx.tile_store %arg1[%0, %0], %4 : memref<?x?xf32>, vector<16x16xf32>
  return
}

// CHECK-LABEL: strides(
// CHECK: %[[CST_64_1:.+]] = llvm.mlir.constant(64 : i64) : i64
// CHECK: "amx.tileloadd64"(%{{.+}}, %{{.+}}, %{{.+}}, %[[CST_64_1]]
// CHECK: %[[CST_128_1:.+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK: "amx.tileloadd64"(%{{.+}}, %{{.+}}, %{{.+}}, %[[CST_128_1]]
// CHECK: llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.extractvalue %{{.+}}[4, 0]
// CHECK: %[[STRIDE_1:.+]] = llvm.mul
// CHECK: "amx.tileloadd64"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STRIDE_1]]
// CHECK: %[[CST_64_2:.+]] = llvm.mlir.constant(64 : i64) : i64
// CHECK: "amx.tilestored64"(%{{.+}}, %{{.+}}, %{{.+}}, %[[CST_64_2]]
// CHECK: %[[CST_128_2:.+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK: "amx.tilestored64"(%{{.+}}, %{{.+}}, %{{.+}}, %[[CST_128_2]]
// CHECK: llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.extractvalue %{{.+}}[4, 0]
// CHECK: %[[STRIDE_2:.+]] = llvm.mul
// CHECK: "amx.tilestored64"(%{{.+}}, %{{.+}}, %{{.+}}, %[[STRIDE_2]]
func.func @strides(%arg0: memref<16x32xbf16>, %arg1: memref<16x32xbf16, strided<[64, 1]>>, %arg2: memref<16x32xbf16, strided<[?, 1]>>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x32xbf16> into vector<16x32xbf16>
  %2 = amx.tile_load %arg1[%0, %0] : memref<16x32xbf16, strided<[64, 1]>> into vector<16x32xbf16>
  %3 = amx.tile_load %arg2[%0, %0] : memref<16x32xbf16, strided<[?, 1]>> into vector<16x32xbf16>
  amx.tile_store %arg0[%0, %0], %3 : memref<16x32xbf16>, vector<16x32xbf16>
  amx.tile_store %arg1[%0, %0], %1 : memref<16x32xbf16, strided<[64, 1]>>, vector<16x32xbf16>
  amx.tile_store %arg2[%0, %0], %2 : memref<16x32xbf16, strided<[?, 1]>>, vector<16x32xbf16>
  return
}
