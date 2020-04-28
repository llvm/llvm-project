// RUN: mlir-opt -test-buffer-placement-preparation -split-input-file %s | FileCheck %s -dump-input-on-failure

// CHECK-LABEL: func @func_signature_conversion
func @func_signature_conversion(%arg0: tensor<4x8xf32>) {
    return
}
// CHECK: ({{.*}}: memref<4x8xf32>) {

// -----

// CHECK-LABEL: func @non_void_to_void_return_op_converter
func @non_void_to_void_return_op_converter(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %arg0 : tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]<[[RANK:.*]]>, %[[RESULT:.*]]: [[TYPE]]<[[RANK]]>) {
// CHECK-NEXT: linalg.copy(%[[ARG0]], %[[RESULT]])
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @func_and_block_signature_conversion
func @func_and_block_signature_conversion(%arg0 : tensor<2xf32>, %cond : i1, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>{
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    br ^exit(%arg0 : tensor<2xf32>)
  ^exit(%arg2: tensor<2xf32>):
    return %arg1 : tensor<4x4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[ARG0_TYPE:.*]], %[[COND:.*]]: i1, %[[ARG1:.*]]: [[ARG1_TYPE:.*]], %[[RESULT:.*]]: [[RESULT_TYPE:.*]]) {
//      CHECK: br ^[[EXIT_BLOCK:.*]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: br ^[[EXIT_BLOCK]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: ^[[EXIT_BLOCK]](%{{.*}}: [[ARG0_TYPE]])
// CHECK-NEXT: linalg.copy(%[[ARG1]], %[[RESULT]])
// CHECK-NEXT: return

// -----

// Test Case: Simple case for checking if BufferAssignmentPlacer creates AllocOps right before GenericOps.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @compute_allocs_position_simple
func @compute_allocs_position_simple(%cond: i1, %arg0: tensor<2xf32>) -> tensor<2xf32>{
    %0 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0 {
    ^bb0(%gen1_arg0: f32):
      %tmp1 = exp %gen1_arg0 : f32
      linalg.yield %tmp1 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    %1 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %0 {
    ^bb0(%gen2_arg0: f32):
      %tmp2 = exp %gen2_arg0 : f32
      linalg.yield %tmp2 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    return %1 : tensor<2xf32>
}
//      CHECK: (%{{.*}}: {{.*}}, %[[ARG0:.*]]: memref<2xf32>,
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ARG0]], %[[FIRST_ALLOC]]
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[FIRST_ALLOC]], %[[SECOND_ALLOC]]

// -----

// Test Case: if-else case for checking if BufferAssignmentPlacer creates AllocOps right before GenericOps.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @compute_allocs_position
func @compute_allocs_position(%cond: i1, %arg0: tensor<2xf32>) -> tensor<2xf32>{
    %0 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0 {
    ^bb0(%gen1_arg0: f32):
      %tmp1 = exp %gen1_arg0 : f32
      linalg.yield %tmp1 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    %1 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %0 {
    ^bb0(%gen2_arg0: f32):
      %tmp2 = exp %gen2_arg0 : f32
      linalg.yield %tmp2 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    %2 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0 {
    ^bb0(%gen3_arg0: f32):
      %tmp3 = exp %gen3_arg0 : f32
      linalg.yield %tmp3 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    %3 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %2 {
    ^bb0(%gen4_arg0: f32):
      %tmp4 = exp %gen4_arg0 : f32
      linalg.yield %tmp4 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    %4 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0 {
    ^bb0(%gen5_arg0: f32):
      %tmp5 = exp %gen5_arg0 : f32
      linalg.yield %tmp5 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    %5 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %4 {
    ^bb0(%gen6_arg0: f32):
      %tmp6 = exp %gen6_arg0 : f32
      linalg.yield %tmp6 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    br ^exit(%arg3, %arg4 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
    %6 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg0 {
    ^bb0(%gen7_arg0: f32):
      %tmp7 = exp %gen7_arg0 : f32
      linalg.yield %tmp7 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    %7 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %6 {
    ^bb0(%gen8_arg0: f32):
      %tmp8 = exp %gen8_arg0 : f32
      linalg.yield %tmp8 : f32
    }: tensor<2xf32> -> tensor<2xf32>
    return %7 : tensor<2xf32>
}
//      CHECK: (%{{.*}}: {{.*}}, %[[ARG0:.*]]: memref<2xf32>,
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ARG0]], %[[ALLOC0]]
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ALLOC0]], %[[ALLOC1]]
//      CHECK: cond_br %{{.*}}, ^[[BB0:.*]]({{.*}}), ^[[BB1:.*]](
// CHECK-NEXT: ^[[BB0]]
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ARG0]], %[[ALLOC2]]
//      CHECK: %[[ALLOC3:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ALLOC2]], %[[ALLOC3]]
//      CHECK: br ^[[EXIT:.*]]({{.*}})
// CHECK-NEXT: ^[[BB1]]
// CHECK-NEXT: %[[ALLOC4:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ARG0]], %[[ALLOC4]]
//      CHECK: %[[ALLOC5:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ALLOC4]], %[[ALLOC5]]
//      CHECK: br ^[[EXIT]]
// CHECK-NEXT: ^[[EXIT]]
// CHECK-NEXT: %[[ALLOC6:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ARG0]], %[[ALLOC6]]
//      CHECK: %[[ALLOC7:.*]] = alloc()
// CHECK-NEXT: linalg.generic {{.*}} %[[ALLOC6]], %[[ALLOC7]]
