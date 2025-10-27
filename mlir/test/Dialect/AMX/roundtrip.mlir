// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

// CHECK-LABEL: tloadstore
// CHECK:      %[[x:.*]] = amx.tile_load %{{.*}}[%{{.*}}], %{{.*}} :
// CHECK-SAME:   memref<?xbf16> into !amx.tile<16x32xbf16>
// CHECK:      %[[y:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} :
// CHECK-SAME:   memref<?x?xbf16> into !amx.tile<16x32xbf16>
// CHECK:      %[[z:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}] :
// CHECK-SAME:   memref<?x?xbf16, strided<[64, 1]>> into !amx.tile<16x32xbf16>
// CHECK:      amx.tile_store %{{.*}}[%{{.*}}], %[[z]], %{{.*}} :
// CHECK-SAME:   memref<?xbf16>, !amx.tile<16x32xbf16>
// CHECK:      amx.tile_store %{{.*}}[%{{.*}}, %{{.*}}], %[[x]], %{{.*}} :
// CHECK-SAME:   memref<?x?xbf16>, !amx.tile<16x32xbf16>
// CHECK:      amx.tile_store %{{.*}}[%{{.*}}, %{{.*}}], %[[y]] :
// CHECK-SAME:   memref<?x?xbf16, strided<[64, 1]>>, !amx.tile<16x32xbf16>
func.func @tloadstore(%stride: index,
    %arg0: memref<?xbf16>,
    %arg1: memref<?x?xbf16>,
    %arg2: memref<?x?xbf16, strided<[64, 1]>>) {
  %0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %1 = amx.tile_load %arg0[%0], %stride : memref<?xbf16> into !amx.tile<16x32xbf16>
  %2 = amx.tile_load %arg1[%0, %0], %stride : memref<?x?xbf16> into !amx.tile<16x32xbf16>
  %3 = amx.tile_load %arg2[%0, %0] : memref<?x?xbf16, strided<[64, 1]>> into !amx.tile<16x32xbf16>
  amx.tile_store %arg0[%0], %3, %stride : memref<?xbf16>, !amx.tile<16x32xbf16>
  amx.tile_store %arg1[%0, %0], %1, %stride : memref<?x?xbf16>, !amx.tile<16x32xbf16>
  amx.tile_store %arg2[%0, %0], %2 : memref<?x?xbf16, strided<[64, 1]>>, !amx.tile<16x32xbf16>
  return
}

// CHECK-LABEL: tzero
// CHECK: amx.tile_zero : !amx.tile<16x16xbf16>
// CHECK: amx.tile_store %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<?x?xbf16>, !amx.tile<16x16xbf16>
func.func @tzero(%arg0: memref<?x?xbf16>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_zero : !amx.tile<16x16xbf16>
  amx.tile_store %arg0[%0, %0], %1 : memref<?x?xbf16>, !amx.tile<16x16xbf16>
  return
}

// CHECK-LABEL: tmulf
// CHECK: %[[x:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xbf16> into !amx.tile<16x32xbf16>
// CHECK: %[[z:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32> into !amx.tile<16x16xf32>
// CHECK: %[[m:.*]] = amx.tile_mulf %[[x]], %[[x]], %[[z]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK: amx.tile_store %{{.*}}[%{{.*}}, %{{.*}}], %[[m]] : memref<?x?xf32>, !amx.tile<16x16xf32>
func.func @tmulf(%arg0: memref<?x?xbf16>, %arg1: memref<?x?xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<?x?xbf16> into !amx.tile<16x32xbf16>
  %2 = amx.tile_load %arg1[%0, %0] : memref<?x?xf32> into !amx.tile<16x16xf32>
  %3 = amx.tile_mulf %1, %1, %2 : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
  amx.tile_store %arg1[%0, %0], %3 : memref<?x?xf32>, !amx.tile<16x16xf32>
  return
}

// CHECK-LABEL: tmuli
// CHECK: %[[x:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xi8> into !amx.tile<16x64xi8>
// CHECK: %[[y:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xi8> into !amx.tile<16x64xi8>
// CHECK: %[[z:.*]] = amx.tile_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xi32> into !amx.tile<16x16xi32>
// CHECK: %[[m:.*]] = amx.tile_muli %[[x]] zext, %[[y]] zext, %[[z]] : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
// CHECK: amx.tile_store %{{.*}}[%{{.*}}, %{{.*}}], %[[m]] : memref<?x?xi32>, !amx.tile<16x16xi32>
// Verify the parsing/printing of the sign-extension annotation.
// CHECK: amx.tile_muli %{{.*}}, %{{.*}} zext, %{{.*}}
// CHECK: amx.tile_muli %{{.*}} zext, %{{.*}}, %{{.*}}
// CHECK: amx.tile_muli %{{.*}}, %{{.*}}, %{{.*}}
func.func @tmuli(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi8>, %arg2: memref<?x?xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<?x?xi8> into !amx.tile<16x64xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<?x?xi8> into !amx.tile<16x64xi8>
  %3 = amx.tile_load %arg2[%0, %0] : memref<?x?xi32> into !amx.tile<16x16xi32>
  %4 = amx.tile_muli %1 zext, %2 zext, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<?x?xi32>, !amx.tile<16x16xi32>
  // Verify the various `zext` combinations.
  %5 = amx.tile_muli %1, %2 zext, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  %6 = amx.tile_muli %1 zext, %2, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  %7 = amx.tile_muli %1, %2, %3 : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
  return
}
