// RUN: tr-opt %s --transform-interpreter --verify-diagnostics --split-input-file

// Silenceable failure: column reduction is not a row reduction.

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
    // expected-error @below {{expected row reduction}}
    %mapped = transform.tr.map_row_reduction %red
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Silenceable failure: handle is not a linalg.generic.

func.func @not_generic(%in: memref<128xf32>, %out: memref<128xf32>) {
  linalg.copy ins(%in : memref<128xf32>) outs(%out : memref<128xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected linalg.generic}}
    %mapped = transform.tr.map_row_reduction %copy
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Verifier: mapping parameters must be positive.

func.func @empty() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{mapping parameters must be positive}}
    %mapped = transform.tr.map_row_reduction %red {warps_per_block = 0 : i64}
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
