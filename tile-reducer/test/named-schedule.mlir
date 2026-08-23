// RUN: tr-opt %s --transform-preload-library=transform-library-paths=%S/../transform/row_sum_schedule.mlir --transform-interpreter=entry-point=row_sum_schedule | FileCheck %s

// Milestone 12: named sequence symbol lookup.
//   - @row_sum_schedule is a public SymbolRefAttr entry point
//   - it includes private @tile_row_reduction (same symbol table)
//   - --transform-interpreter=entry-point=row_sum_schedule does the lookup

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

// CHECK-LABEL: func.func @row
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [64, 128] [1, 1]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
