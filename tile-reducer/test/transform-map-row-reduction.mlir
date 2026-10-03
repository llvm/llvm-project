// RUN: tr-opt %s --transform-interpreter | FileCheck %s

// Milestone 13: custom transform.tr.map_row_reduction annotates the
// baseline 8-warp / 32-lane / 4-elem map. No GPU thread IDs yet.

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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %mapped = transform.tr.map_row_reduction %red
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @row
// CHECK: linalg.generic
// CHECK-SAME: tr.elements_per_lane = 4
// CHECK-SAME: tr.warp_size = 32
// CHECK-SAME: tr.warps_per_block = 8
