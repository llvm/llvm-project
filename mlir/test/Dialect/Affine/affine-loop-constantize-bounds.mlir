// RUN: mlir-opt %s -affine-loop-constantize-bounds -split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP_APPLY:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
// CHECK-DAG: #[[$MAP_UB:.+]] = affine_map<()[s0] -> ((-s0 + 9) ceildiv 2)>

// CHECK-LABEL: func @peeling_main_loop

func.func @peeling_main_loop() {
  %c0 = arith.constant 0 :index
  %bound = test.value_with_bounds { min = 0 : index, max = 1 : index}
  affine.for %iv = %bound to 9 step 2 iter_args(%arg = %c0) -> index {
    %sum = arith.addi %arg, %bound : index
    affine.yield %sum : index
  }
  return
}

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[BOUND:.*]] = test.value_with_bounds {max = 1 : index, min = 0 : index}
// CHECK: %[[MAIN_RES:.*]] = affine.for %[[IV_MAIN:.*]] = 0 to 4 iter_args(%[[ARG_MAIN:.*]] = %[[C0]]) -> (index) {
// CHECK:   %{{.*}} = affine.apply #map(%[[IV_MAIN]])[%[[BOUND]]]
// CHECK: }
// CHECK: %[[TAIL_RES:.*]] = affine.for %[[IV_TAIL:.*]] = 4 to #map1()[%[[BOUND]]] iter_args(%[[ARG_TAIL:.*]] = %[[MAIN_RES]]) -> (index) {
// CHECK:   %{{.*}} = affine.apply #map(%[[IV_TAIL]])[%[[BOUND]]]
// CHECK: }

// -----
// CHECK: #map = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
// CHECK-LABEL: func @fully_constantized_no_peeling

func.func @fully_constantized_no_peeling() {
  %c0 = arith.constant 0 :index
  %bound = test.value_with_bounds { min = 0 : index, max = 1 : index}
  affine.for %iv = %bound to 6 step 2 iter_args(%arg = %c0) -> index {
    %sum = arith.addi %arg, %bound : index
    affine.yield %sum : index
  }
  return
}

// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[BOUND:.*]] = test.value_with_bounds {max = 1 : index, min = 0 : index}
// CHECK:   %{{.*}} = affine.for %[[IV:.*]] = 0 to 3 iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:     %{{.*}} = affine.apply #map(%[[IV]])[%[[BOUND]]]
// CHECK:   }
