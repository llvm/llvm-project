// RUN: mlir-opt --allow-unregistered-dialect --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#map = affine_map<()[s0, s1] -> (s1)>
#map1 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map2 = affine_map<() -> (1)>
module {
  func.func @min_max_full_simplify() -> (index, index) {
    %0 = test.value_with_bounds {max = 64 : index, min = 32 : index}
    %1 = test.value_with_bounds {max = 64 : index, min = 32 : index}
    %2 = affine.min #map()[%0, %1]
    // Make compose affine affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%2, %0]
    // No min folder:
    %3 = affine.apply #map1()[%2, %0]
    // Min folder on.
    %4 = affine.apply #map2()
    return %3, %4 : index, index
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["affine.affine_apply"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %folded = transform.test.make_composed_folded_affine_apply %op
        : (!transform.any_op) -> !transform.any_op
    transform.print %folded {name = "folded: " } : !transform.any_op
    transform.yield
  }
}
