// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ldmatrix(
func @ldmatrix(%arg0: memref<?x?xf16, 3>, %x: index, %y: index) {
//      CHECK: nvgpu.ldmatrix %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK-SAME: {numTiles = 4 : i32, transpose = false} : memref<?x?xf16, 3> -> vector<4x2xf16>
  %l = nvgpu.ldmatrix %arg0[%x, %y] {numTiles = 4 : i32, transpose = false} :
    memref<?x?xf16, 3> -> vector<4x2xf16>
  return
}
