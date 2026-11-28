// RUN: tr-opt %s --tr-tile-linalg=tile-sizes=128,128 | FileCheck %s --check-prefix=T128
// RUN: tr-opt %s --tr-tile-linalg=tile-sizes=64,128 | FileCheck %s --check-prefix=T64
// RUN: tr-opt %s --tr-tile-linalg=tile-sizes=32,128 | FileCheck %s --check-prefix=T32

// Milestone 10: tile a row reduction (parallel, reduction) with scf.for.
// Original Linalg is the function body below. No GPU thread mapping.
//
// iterator_types = ["parallel", "reduction"]
//   dim 0 = parallel  (rows)
//   dim 1 = reduction (K)

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @row(%in: memref<128x128xf32>, %out: memref<128xf32>) {
  linalg.generic {
      indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "reduction"]}
    ins(%in : memref<128x128xf32>)
    outs(%out : memref<128xf32>) {
  ^bb0(%a: f32, %b: f32):
    %0 = arith.addf %a, %b : f32
    linalg.yield %0 : f32
  }
  return
}

// 128x128: one-trip loops on both dims. Parallel = outer, reduction = inner.
// T128: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// T128: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// T128: memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [128, 128] [1, 1]
// T128: linalg.generic
// T128-SAME: iterator_types = ["parallel", "reduction"]

// 64x128: two tiles along the parallel (row) dimension.
// T64: %[[C64:.*]] = arith.constant 64 : index
// T64: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C64]]
// T64: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// T64: memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [64, 128] [1, 1]
// T64: linalg.generic
// T64-SAME: iterator_types = ["parallel", "reduction"]

// 32x128: four tiles along the parallel dimension.
// T32: %[[C32:.*]] = arith.constant 32 : index
// T32: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C32]]
// T32: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// T32: memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [32, 128] [1, 1]
// T32: linalg.generic
// T32-SAME: iterator_types = ["parallel", "reduction"]
