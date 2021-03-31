// RUN: mlir-opt -pass-pipeline="gpu.module(lower-affine)" %s | FileCheck %s

#map0gpufunc = affine_map<(d0) -> (d0)>
gpu.module @kernels {
  gpu.func @foo(%arg0 : index, %arg1 : memref<?xf32>) -> f32 {
    %0 = affine.apply #map0gpufunc(%arg0)
    %1 = memref.load %arg1[%0] : memref<?xf32>
    gpu.return %1 : f32
  }

//      CHECK: gpu.func
// CHECK-SAME: %[[ARG0:.*]]: index
//  CHECK-NOT:   affine.apply
//      CHECK:   memref.load %{{.*}}[%[[ARG0]]]
}
