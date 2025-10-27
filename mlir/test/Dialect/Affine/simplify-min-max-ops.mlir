// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(affine-simplify-min-max))" %s | FileCheck %s

// CHECK-DAG: #[[MAP_0:.*]] = affine_map<()[s0] -> (32, s0)>
// CHECK-DAG: #[[MAP_1:.*]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG: #[[MAP_2:.*]] = affine_map<()[s0] -> (256, s0)>

// CHECK: @min_max_full_simplify
func.func @min_max_full_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  // CHECK: %[[V0:.*]] = test.value_with_bounds {max = 128 : index, min = 0 : index}
  // CHECK: %[[V1:.*]] = test.value_with_bounds {max = 512 : index, min = 256 : index}
  // CHECK-NOT: affine.min
  // CHECK-NOT: affine.max
  // CHECK: return %[[V0]], %[[V1]]
  %r0 = affine.min affine_map<()[s0, s1] -> (s0, 192, s1)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s0, 192, s1)>()[%0, %1]
  return %r0, %r1 : index, index
}

// CHECK: @min_only_simplify
func.func @min_only_simplify() -> (index, index) {
  // CHECK: %[[V0:.*]] = test.value_with_bounds {max = 512 : index, min = 0 : index}
  // CHECK: %[[V1:.*]] = test.value_with_bounds {max = 512 : index, min = 256 : index}
  // CHECK: affine.min #[[MAP_0]]()[%[[V0]]]
  // CHECK: affine.max #[[MAP_1]]()[%[[V0]], %[[V1]]]
  %0 = test.value_with_bounds {max = 512 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s0, 32, s1)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s0, 32, s1)>()[%0, %1]
  return %r0, %r1 : index, index
}

// CHECK: @max_only_simplify
func.func @max_only_simplify() -> (index, index) {
  // CHECK: %[[V0:.*]] = test.value_with_bounds {max = 128 : index, min = 0 : index}
  // CHECK: %[[V1:.*]] = test.value_with_bounds {max = 512 : index, min = 0 : index}
  // CHECK: affine.min #[[MAP_1]]()[%[[V0]], %[[V1]]]
  // CHECK: affine.max #[[MAP_2]]()[%[[V1]]]
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 0 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s0, 256, s1)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s0, 256, s1)>()[%0, %1]
  return %r0, %r1 : index, index
}

// CHECK: @overlapping_constraints
func.func @overlapping_constraints() -> (index, index) {
  %0 = test.value_with_bounds {max = 192 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 384 : index, min = 128 : index}
  %2 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  // CHECK: %[[V0:.*]] = test.value_with_bounds {max = 192 : index, min = 0 : index}
  // CHECK: %[[V1:.*]] = test.value_with_bounds {max = 384 : index, min = 128 : index}
  // CHECK: %[[V2:.*]] = test.value_with_bounds {max = 512 : index, min = 256 : index}
  // CHECK: affine.min #[[MAP_1]]()[%[[V0]], %[[V1]]]
  // CHECK: affine.max #[[MAP_1]]()[%[[V1]], %[[V2]]]
  %r0 = affine.min affine_map<()[s0, s1, s2] -> (s0, s1, s2)>()[%0, %1, %2]
  %r1 = affine.max affine_map<()[s0, s1, s2] -> (s0, s1, s2)>()[%0, %1, %2]
  return %r0, %r1 : index, index
}
