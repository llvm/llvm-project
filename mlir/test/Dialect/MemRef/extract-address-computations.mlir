// RUN: mlir-opt -transform-interpreter %s --split-input-file --verify-diagnostics | FileCheck %s

// Simple test: check that we extract the address computation of a load into
// a dedicated subview.
// The resulting load will be loading from the subview and have only indices
// set to zero.

// CHECK-LABEL: @test_load(
// CHECK-SAME: %[[BASE:[^:]*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: return %[[LOADED_VAL]] : f32

// expected-remark @below {{transformed}}
func.func @test_load(%base : memref<2x16x16xf32>, %offset : index) -> f32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %loaded_val = memref.load %base[%offset, %c0, %c8] : memref<2x16x16xf32>
  return %loaded_val : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    // Verify that the returned handle is usable.
    transform.debug.emit_remark_at %0, "transformed" : !transform.any_op
    transform.yield
  }
}

// -----

// Same as previous @test_load but with the nontemporal flag.

// CHECK-LABEL: @test_load_nontemporal(
// CHECK-SAME: %[[BASE:[^:]*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {nontemporal = true} : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: return %[[LOADED_VAL]] : f32
func.func @test_load_nontemporal(%base : memref<2x16x16xf32>, %offset : index) -> f32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %loaded_val = memref.load %base[%offset, %c0, %c8] {nontemporal = true } : memref<2x16x16xf32>
  return %loaded_val : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Simple test: check that we extract the address computation of a store into
// a dedicated subview.
// The resulting store will use the address from the subview and have only
// indices set to zero.

// CHECK-LABEL: @test_store(
// CHECK-SAME: %[[BASE:[^:]*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK-DAG: %[[CF0:.*]] = arith.constant 0.0{{0*e\+00}} : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: memref.store %[[CF0]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: return
func.func @test_store(%base : memref<2x16x16xf32>, %offset : index) -> () {
  %cf0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  memref.store %cf0, %base[%offset, %c0, %c8] : memref<2x16x16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Same as @test_store but check that the nontemporal flag is preserved.

// CHECK-LABEL: @test_store_nontemporal(
// CHECK-SAME: %[[BASE:[^:]*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK-DAG: %[[CF0:.*]] = arith.constant 0.0{{0*e\+00}} : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: memref.store %[[CF0]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {nontemporal = true} : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: return
func.func @test_store_nontemporal(%base : memref<2x16x16xf32>, %offset : index) -> () {
  %cf0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  memref.store %cf0, %base[%offset, %c0, %c8] { nontemporal = true } : memref<2x16x16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----
// For this test, we made the source memref fully dynamic.
// The gist of the check remains the same as the simple test:
// The address computation is extracted into its own subview.
// CHECK-LABEL: @testWithLoop(
// CHECK-SAME: %[[BASE:[^:]*]]: memref
// CHECK:  %[[SUM_ALL:.*]] = arith.constant 0.0{{0*e\+00}} : f32
// CHECK:  %[[C0:.*]] = arith.constant 0 : index
// CHECK:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C2:.*]] = arith.constant 2 : index
// CHECK:  %[[UPPER_BOUND0:.*]] = memref.dim %[[BASE]], %[[C0]] : memref<?x?x?xf32,
// CHECK:  %[[UPPER_BOUND1:.*]] = memref.dim %[[BASE]], %[[C1]] : memref<?x?x?xf32,
// CHECK:  %[[UPPER_BOUND2:.*]] = memref.dim %[[BASE]], %[[C2]] : memref<?x?x?xf32,
// CHECK:  %[[SUM_RES2:.*]] = scf.for %[[IV2:.*]] = %[[C0]] to %[[UPPER_BOUND2]] step %[[C1]] iter_args(%[[SUM_ITER2:.*]] = %[[SUM_ALL]]) -> (f32) {
// CHECK:    %[[SUM_RES1:.*]] = scf.for %[[IV1:.*]] = %[[C0]] to %[[UPPER_BOUND1]] step %[[C1]] iter_args(%[[SUM_ITER1:.*]] = %[[SUM_ITER2]]) -> (f32) {
// CHECK:      %[[SUM_RES0:.*]] = scf.for %[[IV0:.*]] = %[[C0]] to %[[UPPER_BOUND0]] step %[[C1]] iter_args(%[[SUM_ITER0:.*]] = %[[SUM_ITER1]]) -> (f32) {
// CHECK:        %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[IV0]], %[[IV1]], %[[IV2]]] [1, 1, 1] [1, 1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:        %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:        %[[RES:.*]] = arith.addf %[[LOADED_VAL]], %[[SUM_ITER2]] : f32
// CHECK:        scf.yield %[[RES]] : f32
// CHECK:      }
// CHECK:      scf.yield %[[SUM_RES0]] : f32
// CHECK:    }
// CHECK:    scf.yield %[[SUM_RES1]] : f32
// CHECK:  }
// CHECK:  return %[[SUM_RES2]] : f32
func.func @testWithLoop(%base : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>) -> f32 {
  %sum_all = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %upper_bound0 = memref.dim %base, %c0 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %upper_bound1 = memref.dim %base, %c1 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %upper_bound2 = memref.dim %base, %c2 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %sum_res2 = scf.for %iv2 = %c0 to %upper_bound2 step %c1 iter_args(%sum_iter2 = %sum_all) -> (f32) {
    %sum_res1 = scf.for %iv1 = %c0 to %upper_bound1 step %c1 iter_args(%sum_iter1 = %sum_iter2) -> (f32) {
      %sum_res0 = scf.for %iv0 = %c0 to %upper_bound0 step %c1 iter_args(%sum_iter0 = %sum_iter1) -> (f32) {
        %loaded_val = memref.load %base[%iv0, %iv1, %iv2] : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
        %res = arith.addf %loaded_val, %sum_iter2 : f32
        scf.yield %res : f32
      }
      scf.yield %sum_res0 : f32
    }
    scf.yield %sum_res1 : f32
  }
  return %sum_res2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Simple test: check that we extract the address computation of a ldmatrix into
// a dedicated subview.
// The resulting ldmatrix will loaded from with subview and have only indices set
// to zero.
// Also the sizes of the view are adjusted to `original size - offset`.

// CHECK-DAG: #[[$FOUR_MINUS_OFF_MAP:.*]] = affine_map<()[s0] -> (-s0 + 4)>
// CHECK-DAG: #[[$THIRTY_TWO_MINUS_OFF_MAP:.*]] = affine_map<()[s0] -> (-s0 + 32)>
// CHECK-LABEL: @test_ldmatrix(
// CHECK-SAME: %[[BASE:[^:]*]]: memref<{{[^,]*}}, 3>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$FOUR_MINUS_OFF_MAP]]()[%[[DYN_OFFSET0]]]
// CHECK-DAG: %[[DYN_SIZE1:.*]] = affine.apply #[[$THIRTY_TWO_MINUS_OFF_MAP]]()[%[[DYN_OFFSET1]]]
// CHECK-DAG: %[[DYN_SIZE2:.*]] = affine.apply #[[$THIRTY_TWO_MINUS_OFF_MAP]]()[%[[DYN_OFFSET2]]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] [%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]]] [1, 1, 1] : memref<4x32x32xf16, 3> to memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3>
// CHECK: %[[LOADED_VAL:.*]] = nvgpu.ldmatrix %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {numTiles = 4 : i32, transpose = false} : memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3> -> vector<4x2xf16>
// CHECK: return %[[LOADED_VAL]] : vector<4x2xf16>
func.func @test_ldmatrix(%base : memref<4x32x32xf16, 3>,
    %offset0 : index, %offset1: index, %offset2: index)
    -> vector<4x2xf16> {
  %loaded_val = nvgpu.ldmatrix
    %base[%offset0, %offset1, %offset2]
    {numTiles = 4 : i32, transpose = false}
      : memref<4x32x32xf16, 3> -> vector<4x2xf16>
  return %loaded_val : vector<4x2xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Same as test_ldmatrix but with fully dynamic memref.

// CHECK-DAG: #[[$A_MINUS_B_MAP:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-LABEL: @test_ldmatrix(
// CHECK-SAME: %[[BASE:[^:]*]]: memref<{{[^,]*}}, 3>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: {{.*}}, {{.*}}, %[[DYN_SIZES:.*]]:3, {{.*}} = memref.extract_strided_metadata %[[BASE]]
// CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#0, %[[DYN_OFFSET0]]]
// CHECK-DAG: %[[DYN_SIZE1:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#1, %[[DYN_OFFSET1]]]
// CHECK-DAG: %[[DYN_SIZE2:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#2, %[[DYN_OFFSET2]]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] [%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]]] [1, 1, 1] : memref<?x?x?xf16, 3> to memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>, 3>
// CHECK: %[[LOADED_VAL:.*]] = nvgpu.ldmatrix %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {numTiles = 4 : i32, transpose = false} : memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>, 3> -> vector<4x2xf16>
// CHECK: return %[[LOADED_VAL]] : vector<4x2xf16>
func.func @test_ldmatrix(%base : memref<?x?x?xf16, 3>,
    %offset0 : index, %offset1: index, %offset2: index)
    -> vector<4x2xf16> {
  %loaded_val = nvgpu.ldmatrix
    %base[%offset0, %offset1, %offset2]
    {numTiles = 4 : i32, transpose = false}
      : memref<?x?x?xf16, 3> -> vector<4x2xf16>
  return %loaded_val : vector<4x2xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Simple test for vector.transfer_read with fully dynamic memref.
// We also set a permutation map to make sure it is properly preserved.

// CHECK-DAG: #[[$A_MINUS_B_MAP:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-DAG: #[[$PERMUTATION_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: @test_transfer_read_op(
// CHECK-SAME: %[[BASE:[^:]*]]: memref<{{[^,]*}}>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: {{.*}}, {{.*}}, %[[DYN_SIZES:.*]]:3, {{.*}} = memref.extract_strided_metadata %[[BASE]]
// CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#0, %[[DYN_OFFSET0]]]
// CHECK-DAG: %[[DYN_SIZE1:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#1, %[[DYN_OFFSET1]]]
// CHECK-DAG: %[[DYN_SIZE2:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#2, %[[DYN_OFFSET2]]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CF0:.*]] = arith.constant 0.0{{0*e\+00}} : f16
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] [%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]]] [1, 1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>>
// CHECK: %[[LOADED_VAL:.*]] = vector.transfer_read %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]], %[[CF0]] {permutation_map = #[[$PERMUTATION_MAP]]} : memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>>, vector<4x2xf16>
// CHECK: return %[[LOADED_VAL]] : vector<4x2xf16>
func.func @test_transfer_read_op(%base : memref<?x?x?xf16>,
    %offset0 : index, %offset1: index, %offset2: index)
    -> vector<4x2xf16> {
  %cf0 = arith.constant 0.0 : f16
  %loaded_val = vector.transfer_read %base[%offset0, %offset1, %offset2], %cf0 { permutation_map = affine_map<(d0,d1,d2) -> (d2,d0)> } : memref<?x?x?xf16>, vector<4x2xf16>
  return %loaded_val : vector<4x2xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Same as test_transfer_read_op but with tensors.
// Right now this rewrite is not supported but we still shouldn't choke on it.

// CHECK: #[[$PERMUTATION_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: @test_transfer_read_op_with_tensor(
// CHECK-SAME: %[[BASE:[^:]*]]: tensor<{{[^,]*}}>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK: %[[CF0:.*]] = arith.constant 0.0{{0*e\+00}} : f16
// CHECK: %[[LOADED_VAL:.*]] = vector.transfer_read %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]], %[[CF0]] {permutation_map = #[[$PERMUTATION_MAP]]} : tensor<?x?x?xf16>, vector<4x2xf16>
// CHECK: return %[[LOADED_VAL]] : vector<4x2xf16>
func.func @test_transfer_read_op_with_tensor(%base : tensor<?x?x?xf16>,
    %offset0 : index, %offset1: index, %offset2: index)
    -> vector<4x2xf16> {
  %cf0 = arith.constant 0.0 : f16
  %loaded_val = vector.transfer_read %base[%offset0, %offset1, %offset2], %cf0 { permutation_map = affine_map<(d0,d1,d2) -> (d2,d0)> } : tensor<?x?x?xf16>, vector<4x2xf16>
  return %loaded_val : vector<4x2xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Simple test for vector.transfer_write with fully dynamic memref.
// We also set a permutation map to make sure it is properly preserved.

// CHECK-DAG: #[[$A_MINUS_B_MAP:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-DAG: #[[$PERMUTATION_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: @test_transfer_write_op(
// CHECK-SAME: %[[BASE:[^:]*]]: memref<{{[^,]*}}>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: {{.*}}, {{.*}}, %[[DYN_SIZES:.*]]:3, {{.*}} = memref.extract_strided_metadata %[[BASE]]
// CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#0, %[[DYN_OFFSET0]]]
// CHECK-DAG: %[[DYN_SIZE1:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#1, %[[DYN_OFFSET1]]]
// CHECK-DAG: %[[DYN_SIZE2:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#2, %[[DYN_OFFSET2]]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[VCF0:.*]] = arith.constant dense<0.0{{0*e\+00}}> : vector<4x2xf16>
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] [%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]]] [1, 1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>>
// CHECK: vector.transfer_write %[[VCF0]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {permutation_map = #[[$PERMUTATION_MAP]]} : vector<4x2xf16>, memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>>
// CHECK: return
func.func @test_transfer_write_op(%base : memref<?x?x?xf16>,
    %offset0 : index, %offset1: index, %offset2: index) {
  %vcf0 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
  vector.transfer_write %vcf0, %base[%offset0, %offset1, %offset2] { permutation_map = affine_map<(d0,d1,d2) -> (d2,d0)> } : vector<4x2xf16>, memref<?x?x?xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Check that the strides of the original memref are kept.
// Moreover even with non-1 strides the subview should still issue [1,...]
// strides, since this is a multiplication factor.

// CHECK-DAG: #[[$A_MINUS_B_MAP:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-DAG: #[[$PERMUTATION_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: @test_transfer_write_op_with_strides(
// CHECK-SAME: %[[BASE:[^:]*]]: memref<{{[^>]*}}>>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: {{.*}}, {{.*}}, %[[DYN_SIZES:.*]]:3, {{.*}} = memref.extract_strided_metadata %[[BASE]]
// CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#0, %[[DYN_OFFSET0]]]
// CHECK-DAG: %[[DYN_SIZE1:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#1, %[[DYN_OFFSET1]]]
// CHECK-DAG: %[[DYN_SIZE2:.*]] = affine.apply #[[$A_MINUS_B_MAP]]()[%[[DYN_SIZES]]#2, %[[DYN_OFFSET2]]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[VCF0:.*]] = arith.constant dense<0.0{{0*e\+00}}> : vector<4x2xf16>
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] [%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]]] [1, 1, 1] : memref<?x?x?xf16, strided<[329, 26, 12], offset: ?>> to memref<?x?x?xf16, strided<[329, 26, 12], offset: ?>>
// CHECK: vector.transfer_write %[[VCF0]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {permutation_map = #[[$PERMUTATION_MAP]]} : vector<4x2xf16>, memref<?x?x?xf16, strided<[329, 26, 12], offset: ?>>
// CHECK: return
func.func @test_transfer_write_op_with_strides(%base : memref<?x?x?xf16, strided<[329, 26, 12], offset: ?>>,
    %offset0 : index, %offset1: index, %offset2: index) {
  %vcf0 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
  vector.transfer_write %vcf0, %base[%offset0, %offset1, %offset2] { permutation_map = affine_map<(d0,d1,d2) -> (d2,d0)> } : vector<4x2xf16>, memref<?x?x?xf16, strided<[329, 26, 12], offset: ?>>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Same as test_transfer_write_op but with tensors.
// Right now this rewrite is not supported but we still shouldn't choke on it.

// CHECK: #[[$PERMUTATION_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: @test_transfer_write_op_with_tensor(
// CHECK-SAME: %[[BASE:[^:]*]]: tensor<{{[^,]*}}>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: %[[VCF0:.*]] = arith.constant dense<0.0{{0*e\+00}}> : vector<4x2xf16>
// CHECK: %[[RES:.*]] = vector.transfer_write %[[VCF0]], %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] {permutation_map = #[[$PERMUTATION_MAP]]} : vector<4x2xf16>, tensor<?x?x?xf16>
// CHECK: return %[[RES]] : tensor<?x?x?xf16>
func.func @test_transfer_write_op_with_tensor(%base : tensor<?x?x?xf16>,
    %offset0 : index, %offset1: index, %offset2: index) -> tensor<?x?x?xf16> {
  %vcf0 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
  %res = vector.transfer_write %vcf0, %base[%offset0, %offset1, %offset2] { permutation_map = affine_map<(d0,d1,d2) -> (d2,d0)> } : vector<4x2xf16>, tensor<?x?x?xf16>
  return %res : tensor<?x?x?xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

