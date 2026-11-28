// RUN: tr-opt %s --transform-interpreter | FileCheck %s
// RUN: tr-opt %s --transform-interpreter=entry-point=row_sum_schedule | FileCheck %s

// Milestone 12: in-module SymbolRefAttr lookup and visibility.
// @__transform_main and @row_sum_schedule both include private @tile_row.

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
  transform.named_sequence private @tile_row(
      %payload: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %payload
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %red tile_sizes [64, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  transform.named_sequence @row_sum_schedule(
      %payload: !transform.any_op {transform.readonly}) {
    transform.include @tile_row failures(propagate) (%payload)
        : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main(
      %payload: !transform.any_op {transform.readonly}) {
    transform.include @row_sum_schedule failures(propagate) (%payload)
        : (!transform.any_op) -> ()
    transform.yield
  }
}

// CHECK-LABEL: func.func @row
// CHECK: scf.for
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
