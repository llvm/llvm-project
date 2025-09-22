// RUN: mlir-opt --transform-interpreter %s | FileCheck %s

#map = affine_map<()[s0, s1] -> (s0, s1, 128)>
#map1 = affine_map<()[s0, s1] -> (s0 ceildiv 128 + s0 ceildiv s1)>
#map2 = affine_map<()[s0, s1, s2] -> (s0, s1 + s2)>
#map3 = affine_map<()[s0, s1, s2, s3] -> (3 * (s0 ceildiv s3) + s0 ceildiv (s1 + s2))>
#map4 = affine_map<()[s0, s1] -> (s1)>
#map5 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map6 = affine_map<()[s0, s1] -> (s0, s1, -128)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 ceildiv 128 + s0 ceildiv s1)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>

// These test checks the `affine::makeComposedFoldedAffineApply` function when
// `composeAffineMin == true`.

// Check the apply gets simplified.
// CHECK: @apply_simplification
func.func @apply_simplification_1() -> index {
  %0 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %1 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %2 = affine.min #map()[%0, %1]
  // CHECK-NOT: affine.apply
  // CHECK: arith.constant 2 : index
  %3 = affine.apply #map1()[%2, %1]
  return %3 : index
}

// Check the simplification can match non-trivial affine expressions like s1 + s2.
func.func @apply_simplification_2() -> index {
  %0 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %1 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %2 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %3 = affine.min #map2()[%0, %1, %2]
  // CHECK-NOT: affine.apply
  // CHECK: arith.constant 4 : index
  %4 = affine.apply #map3()[%3, %1, %2, %0]
  return %4 : index
}

// Check there's no simplification.
// The apply cannot be simplified because `s1 = %0` doesn't appear in the input min.
// CHECK: @no_simplification_0
func.func @no_simplification_0() -> index {
  // CHECK: %[[V0:.*]] = test.value_with_bounds {max = 64 : index, min = 32 : index}
  // CHECK: %[[V1:.*]] = test.value_with_bounds {max = 64 : index, min = 16 : index}
  // CHECK: %[[V2:.*]] = affine.min #{{.*}}()[%[[V0]], %[[V1]]]
  // CHECK: %[[V3:.*]] = affine.apply #[[MAP5]]()[%[[V2]], %[[V0]]]
  // CHECK: return %[[V3]] : index
  %0 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %1 = test.value_with_bounds {max = 64 : index, min = 16 : index}
  %2 = affine.min #map4()[%0, %1]
  %3 = affine.apply #map5()[%2, %0]
  return %3 : index
}

// The apply cannot be simplified because the min cannot be proven to be greater than 0.
// CHECK: @no_simplification_1
func.func @no_simplification_1() -> index {
  // CHECK: %[[V0:.*]] = test.value_with_bounds {max = 64 : index, min = 32 : index}
  // CHECK: %[[V1:.*]] = test.value_with_bounds {max = 64 : index, min = 16 : index}
  // CHECK: %[[V2:.*]] = affine.min #{{.*}}()[%[[V0]], %[[V1]]]
  // CHECK: %[[V3:.*]] = affine.apply #[[MAP1]]()[%[[V2]], %[[V1]]]
  // CHECK: return %[[V3]] : index
  %0 = test.value_with_bounds {max = 64 : index, min = 32 : index}
  %1 = test.value_with_bounds {max = 64 : index, min = 16 : index}
  %2 = affine.min #map6()[%0, %1]
  %3 = affine.apply #map1()[%2, %1]
  return %3 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.apply"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.test.make_composed_folded_affine_apply %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}
