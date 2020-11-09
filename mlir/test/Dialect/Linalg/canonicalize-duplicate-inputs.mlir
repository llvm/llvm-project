// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// Test case: Most basic case. Adding a vector to itself.

#map = affine_map<(d0) -> (d0)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @basic
func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP]], #[[$MAP]]]
  // CHECK:   ^bb0(%[[BBARG:.*]]: f32):
  // CHECK:     addf %[[BBARG]], %[[BBARG]]
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Test case: Different indexing maps mean that args are not redundant, despite
// being the same Value.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @distinct_affine_maps
func @distinct_affine_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Test case: Check rewriting mechanics for mixed redundant and
// non-redundant args.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @mixed_redundant_non_redundant
func @mixed_redundant_non_redundant(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
  // CHECK:   ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32):
  // CHECK:     "test.elementwise_mappable"(%[[BBARG0]], %[[BBARG1]], %[[BBARG0]])
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %1 = "test.elementwise_mappable"(%arg1, %arg2, %arg3) : (f32, f32, f32) -> f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Test case: Check rewriting mechanics for multiple different redundant args.

#map = affine_map<(d0) -> (d0)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @multiple_different_redundant_args
func @multiple_different_redundant_args(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP]], #[[$MAP]], #[[$MAP]]]
  // CHECK:   ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32):
  // CHECK:     "test.elementwise_mappable"(%[[BBARG0]], %[[BBARG1]], %[[BBARG0]], %[[BBARG1]])
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32): 
    %1 = "test.elementwise_mappable"(%arg2, %arg3, %arg4, %arg5) : (f32, f32, f32, f32) -> f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Test case: linalg.indexed_generic.
// Other than the payload argument handling, everything else is the same.

#map = affine_map<(d0) -> (d0)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @indexed_generic
func @indexed_generic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.indexed_generic
  // CHECK:   ^bb0(%{{.*}}: index, %[[BBARG:.*]]: f32):
  // CHECK:     addf %[[BBARG]], %[[BBARG]]
  %0 = linalg.indexed_generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%index: index, %arg1: f32, %arg2: f32):
    %1 = addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
