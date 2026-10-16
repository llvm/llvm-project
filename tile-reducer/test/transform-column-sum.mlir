// RUN: tr-opt %s --transform-interpreter | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @column(%in: memref<128x128xf32>, %out: memref<128xf32>) {
  linalg.generic {
      indexing_maps = [#map, #map1],
      iterator_types = ["reduction", "parallel"]}
    ins(%in : memref<128x128xf32>)
    outs(%out : memref<128xf32>) {
  ^bb0(%a: f32, %b: f32):
    %0 = arith.addf %a, %b : f32
    linalg.yield %0 : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %red tile_sizes [0, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @column
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C64]]
// CHECK: memref.subview
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "parallel"]
