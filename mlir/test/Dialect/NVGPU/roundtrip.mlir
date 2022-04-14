// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ldmatrix(
func @ldmatrix(%arg0: memref<?x?xf16, 3>, %x: index, %y: index) {
//      CHECK: nvgpu.ldmatrix %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK-SAME: {numTiles = 4 : i32, transpose = false} : memref<?x?xf16, 3> -> vector<4x2xf16>
  %l = nvgpu.ldmatrix %arg0[%x, %y] {numTiles = 4 : i32, transpose = false} :
    memref<?x?xf16, 3> -> vector<4x2xf16>
  return
}

// CHECK-LABEL: func @mma_sync(
func @mma_sync(%arg0: vector<4x2xf16>,
               %arg1: vector<2x2xf16>,
               %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
//       CHECK: nvgpu.mma.sync(%{{.*}}, %{{.*}}, %{{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
  %d = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
  return %d : vector<2x2xf16>
}
