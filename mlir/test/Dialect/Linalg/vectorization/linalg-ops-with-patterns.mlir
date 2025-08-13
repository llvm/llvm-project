// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: contraction_dot
func.func @contraction_dot(%A: memref<1584xf32>, %B: memref<1584xf32>, %C: memref<f32>) {

// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [0] : vector<1584xf32> to f32
  linalg.dot ins(%A, %B: memref<1584xf32>, memref<1584xf32>)
            outs(%C: memref<f32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.dot"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0  : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: contraction_matvec
func.func @contraction_matvec(%A: memref<1584x1584xf32>, %B: memref<1584xf32>, %C: memref<1584xf32>) {

// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584x1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [1] : vector<1584x1584xf32> to vector<1584xf32>
  linalg.matvec ins(%A, %B: memref<1584x1584xf32>, memref<1584xf32>)
            outs(%C: memref<1584xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matvec"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: contraction_matmul
func.func @contraction_matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584x1584x1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [2] : vector<1584x1584x1584xf32> to vector<1584x1584xf32>
  linalg.matmul ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
            outs(%C: memref<1584x1584xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @float_mixed_precision_matmul
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extf
// CHECK:         vector.contract {{.*}} : vector<1584x1584xbf16>, vector<1584x1584xbf16> into vector<1584x1584xf32>
func.func @float_mixed_precision_matmul(%A: memref<1584x1584xbf16>, %B: memref<1584x1584xbf16>, %C: memref<1584x1584xf32>) {
  linalg.matmul ins(%A, %B: memref<1584x1584xbf16>, memref<1584x1584xbf16>)
            outs(%C: memref<1584x1584xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { fold_type_extensions_into_contract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @vectorization_test_2
func.func @vectorization_test_2(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<8x32x16xf32>
  //       CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [2] : vector<8x32x16xf32> to vector<8x32xf32>
  linalg.matmul
    ins(%A, %B: memref<8x16xf32>, memref<16x32xf32>)
   outs(%C: memref<8x32xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @matmul_tensors
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32>, %[[ARG1:.*]]: tensor<4x12xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: tensor<8x12xf32>) -> tensor<8x12xf32>
func.func @matmul_tensors(
  %arg0: tensor<8x4xf32>, %arg1: tensor<4x12xf32>, %arg2: tensor<8x12xf32>)
    -> tensor<8x12xf32> {
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  //   CHECK-DAG:   %[[V0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x4xf32>, vector<8x12x4xf32>
  //   CHECK-DAG:   %[[V1:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x12xf32>, vector<8x12x4xf32>
  //   CHECK-DAG:   %[[V2:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x12xf32>, vector<8x12xf32>
  //
  // linalg matmul lowers gets expanded to a 3D reduction, canonicalization later
  // convert it to a 2D contract.
  //       CHECK:   %[[MUL:.*]] = arith.mulf %[[V0]], %[[V1]] : vector<8x12x4xf32>
  //       CHECK:   %[[R:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[V2]] [2] : vector<8x12x4xf32> to vector<8x12xf32>
  //       CHECK:   %[[W:.*]] = vector.transfer_write %[[R]], %[[ARG2]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<8x12xf32>, tensor<8x12xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<8x4xf32>, tensor<4x12xf32>)
                     outs(%arg2: tensor<8x12xf32>)
    -> tensor<8x12xf32>
  //       CHECK:   return %[[W]] : tensor<8x12xf32>
  return %0 : tensor<8x12xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: contraction_batch_matmul
func.func @contraction_batch_matmul(%A: memref<1584x1584x1584xf32>, %B: memref<1584x1584x1584xf32>, %C: memref<1584x1584x1584xf32>) {
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1584x1584x1584x1584xf32>
// CHECK: vector.multi_reduction <add>, %{{.*}}, {{.*}} [3] : vector<1584x1584x1584x1584xf32> to vector<1584x1584x1584xf32>
  linalg.batch_matmul
    ins(%A, %B: memref<1584x1584x1584xf32>, memref<1584x1584x1584xf32>)
   outs(%C: memref<1584x1584x1584xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.batch_matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @matmul_as_contract
// CHECK-SAME: %[[A:.*]]: tensor<24x12xf32>
// CHECK-SAME: %[[B:.*]]: tensor<12x25xf32>
// CHECK-SAME: %[[C:.*]]: tensor<24x25xf32>
func.func @matmul_as_contract(%A: tensor<24x12xf32>,
                              %B: tensor<12x25xf32>,
                              %C: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // CHECK: %[[vA:.+]] = vector.transfer_read %[[A]]
  // CHECK: %[[vB:.+]] = vector.transfer_read %[[B]]
  // CHECK: %[[vC:.+]] = vector.transfer_read %[[C]]
  // CHECK: %[[vR:.+]] = vector.contract {{.*}} %[[vA]], %[[vB]], %[[vC]]
  // CHECK: vector.transfer_write %[[vR]], %[[C]]
  %0 = linalg.contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %B : tensor<24x12xf32>, tensor<12x25xf32>)
      outs(%C : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    // TODO: also tests the other available vectorization strategies
    transform.yield
  }
}

// -----

// CHECK-LABEL: @float_mixed_precision_matmul_as_contract
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extf
// CHECK:         vector.contract {{.*}} : vector<24x12xbf16>, vector<12x25xbf16> into vector<24x25xf32>
// CHECK:         vector.transfer_write
func.func @float_mixed_precision_matmul_as_contract(%A: tensor<24x12xbf16>,
                              %B: tensor<12x25xbf16>,
                              %C: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %0 = linalg.contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %B : tensor<24x12xbf16>, tensor<12x25xbf16>)
      outs(%C : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { fold_type_extensions_into_contract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func.func @test_vectorize_fill(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.fill ins(%arg0 : f32) outs(%A : memref<8x16xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func.func @test_vectorize_fill_0d(%A : memref<f32>, %arg0 : f32) {
  // CHECK-SAME: (%[[M:.*]]: memref<f32>, %[[val:.*]]: f32)
  //      CHECK:   %[[VEC:.*]] = vector.broadcast %[[val]] : f32 to vector<f32>
  //      CHECK:   vector.transfer_write %[[VEC]], %[[M]][] : vector<f32>, memref<f32>
  linalg.fill ins(%arg0 : f32) outs(%A : memref<f32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_copy
func.func @test_vectorize_copy(%A : memref<8x16xf32>, %B : memref<8x16xf32>) {
  //       CHECK: %[[V:.*]] = vector.transfer_read {{.*}} : memref<8x16xf32>, vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  memref.copy %A, %B :  memref<8x16xf32> to memref<8x16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_copy_0d
func.func @test_vectorize_copy_0d(%A : memref<f32>, %B : memref<f32>) {
  //  CHECK-SAME: (%[[A:.*]]: memref<f32>, %[[B:.*]]: memref<f32>)
  //       CHECK:   %[[V:.*]] = vector.transfer_read %[[A]][]{{.*}} : memref<f32>, vector<f32>
  //       CHECK:   %[[val:.*]] = vector.extract %[[V]][] : f32 from vector<f32>
  //       CHECK:   %[[VV:.*]] = vector.broadcast %[[val]] : f32 to vector<f32>
  //       CHECK:   vector.transfer_write %[[VV]], %[[B]][] : vector<f32>, memref<f32>
  memref.copy %A, %B :  memref<f32> to memref<f32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_copy_complex
// CHECK-NOT: vector<
func.func @test_vectorize_copy_complex(%A : memref<8x16xcomplex<f32>>, %B : memref<8x16xcomplex<f32>>) {
  memref.copy %A, %B :  memref<8x16xcomplex<f32>> to memref<8x16xcomplex<f32>>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Input identical as the test in vectorization.mlir. Output is different -
// vector sizes are inferred (rather than user-specified) and hence _no_
// masking was used.

func.func @test_vectorize_pack(%arg0: tensor<32x8x16xf32>, %arg1: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x8x16xf32> -> tensor<4x1x32x16x2xf32>
  return %pack : tensor<4x1x32x16x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @test_vectorize_pack(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x8x16xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_2]] {in_bounds = [true, true, true]} : tensor<32x8x16xf32>, vector<32x8x16xf32>
// CHECK:           %[[VAL_5:.*]] = vector.shape_cast %[[VAL_4]] : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
// CHECK:           %[[VAL_6:.*]] = vector.transpose %[[VAL_5]], [1, 3, 0, 4, 2] : vector<32x4x2x1x16xf32> to vector<4x1x32x16x2xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<4x1x32x16x2xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_write %[[VAL_6]], %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true, true, true]} : vector<4x1x32x16x2xf32>, tensor<4x1x32x16x2xf32>
// CHECK:           return %[[VAL_8]] : tensor<4x1x32x16x2xf32>

// -----

func.func @test_vectorize_padded_pack(%arg0: tensor<32x7x15xf32>, %arg1: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
  %pad = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %arg0 padding_value(%pad : f32) inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x7x15xf32> -> tensor<32x4x1x16x2xf32>
  return %pack : tensor<32x4x1x16x2xf32>
}

// CHECK-LABEL:   func.func @test_vectorize_padded_pack(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x7x15xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_2]] {in_bounds = [true, false, false]} : tensor<32x7x15xf32>, vector<32x8x16xf32>
// CHECK:           %[[VAL_5:.*]] = vector.shape_cast %[[VAL_4]] : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
// CHECK:           %[[VAL_6:.*]] = vector.transpose %[[VAL_5]], [0, 1, 3, 4, 2] : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<32x4x1x16x2xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_write %[[VAL_6]], %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true, true, true]} : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
// CHECK:           return %[[VAL_8]] : tensor<32x4x1x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_map(%arg0: memref<64xf32>,
    %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
  linalg.map ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>)
             outs(%arg2 : memref<64xf32>)
    (%in: f32, %in_0: f32) {
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
  return
}
// CHECK-LABEL: func @vectorize_map
// CHECK:         %[[LHS:.*]] = vector.transfer_read
// CHECK-NEXT:    %[[RHS:.*]] = vector.transfer_read
// CHECK-NEXT:    arith.addf %[[LHS]], %[[RHS]] : vector<64xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.map"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_transpose(%arg0: memref<16x32x64xf32>,
                               %arg1: memref<32x64x16xf32>) {
  linalg.transpose ins(%arg0 : memref<16x32x64xf32>)
                   outs(%arg1 : memref<32x64x16xf32>) permutation = [1, 2, 0]
  return
}
// CHECK-LABEL: func @vectorize_transpose
// CHECK:         vector.transpose
// CHECK-SAME:      [1, 2, 0] : vector<16x32x64xf32> to vector<32x64x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_reduce(%arg0: memref<16x32x64xf32>,
                  %arg1: memref<16x64xf32>) {
  linalg.reduce ins(%arg0 : memref<16x32x64xf32>)
                outs(%arg1 : memref<16x64xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  return
}
// CHECK-LABEL: func @vectorize_reduce
// CHECK:         vector.multi_reduction <add>
// CHECK-SAME:    : vector<16x32x64xf32> to vector<16x64xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @vectorization_test
func.func @vectorization_test(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x32x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<8x32x16xf32>
  //       CHECK: %[[ACC:.*]] = vector.transfer_read %{{.*}} : memref<8x32xf32>, vector<8x32xf32>
  //       CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8x32x16xf32>
  //       CHECK: %[[R:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[ACC]] [2] : vector<8x32x16xf32> to vector<8x32xf32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xf32>, memref<8x32xf32>
  linalg.generic #matmul_trait
    ins(%A, %B : memref<8x16xf32>, memref<16x32xf32>)
   outs(%C : memref<8x32xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<() -> ()>

// CHECK-LABEL:   func.func @generic_0d(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<f32>, %[[ARG_1:.*]]: tensor<f32>, %[[ARG_2:.*]]: tensor<f32>)
func.func @generic_0d(%arg0: tensor<f32>, %arg1: tensor<f32>,
                      %arg2: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[PAD:.*]] = ub.poison : f32
// CHECK:           %[[READ_0:.*]] = vector.transfer_read %[[ARG_0]][], %[[PAD]] : tensor<f32>, vector<f32>
// CHECK:           %[[ARG_0_AS_SCALAR:.*]] = vector.extract %[[READ_0]][] : f32 from vector<f32>
// CHECK:           %[[READ_1:.*]] = vector.transfer_read %[[ARG_1]][], %[[PAD]] : tensor<f32>, vector<f32>
// CHECK:           %[[ARG_1_AS_SCALAR:.*]] = vector.extract %[[READ_1]][] : f32 from vector<f32>
// CHECK:           %[[READ_2:.*]] = vector.transfer_read %[[ARG_2]][], %[[PAD]] : tensor<f32>, vector<f32>
// CHECK:           %[[ARG_2_AS_SCALAR:.*]] = vector.extract %[[READ_2]][] : f32 from vector<f32>
// CHECK:           %[[MULF:.*]] = arith.mulf %[[ARG_0_AS_SCALAR]], %[[ARG_1_AS_SCALAR]] : f32
// CHECK:           %[[ADDF:.*]] = arith.addf %[[ARG_2_AS_SCALAR]], %[[MULF]] : f32
// CHECK:           %[[ADDF_BCAST:.*]] = vector.broadcast %[[ADDF]] : f32 to vector<f32>
// CHECK:           vector.transfer_write %[[ADDF_BCAST]], %[[ARG_2]][] : vector<f32>, tensor<f32>
  %res = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = []
  } ins(%arg0, %arg1 : tensor<f32>, tensor<f32>)
    outs(%arg2 : tensor<f32>) {
  ^bb(%a: f32, %b: f32, %c: f32) :
    %d = arith.mulf %a, %b: f32
    %e = arith.addf %c, %d: f32
    linalg.yield %e : f32
  } -> tensor<f32>

  return %res : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#matmul_transpose_out_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (n, m)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @generic_output_transpose
func.func @generic_output_transpose(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                                    %C: memref<32x8xf32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x32x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<8x32x16xf32>
  //       CHECK: %[[ACC:.*]] = vector.transfer_read %{{.*}} : memref<32x8xf32>, vector<8x32xf32>
  //       CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8x32x16xf32>
  //       CHECK: %[[R:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[ACC]] [2] : vector<8x32x16xf32> to vector<8x32xf32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xf32>, memref<32x8xf32>
  linalg.generic #matmul_transpose_out_trait
    ins(%A, %B : memref<8x16xf32>, memref<16x32xf32>)
   outs(%C : memref<32x8xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: func @generic_interchanged_transpose
func.func @generic_interchanged_transpose(%arg0: tensor<12x128x32xf32>) -> tensor<128x12x32xf32> {
  // CHECK: %[[IN:.+]] = vector.transfer_read
  // CHECK: vector.transfer_write %[[IN]], {{.+}} permutation_map = #[[MAP]]
  %0 = tensor.empty() : tensor<128x12x32xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1],
                       iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<12x128x32xf32>)
    outs(%0 : tensor<128x12x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<128x12x32xf32>
  return %1 : tensor<128x12x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @vectorization_test_integer
func.func @vectorization_test_integer(%A: memref<8x16xi32>, %B: memref<16x32xi32>,
                                 %C: memref<8x32xi32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xi32>, vector<8x32x16xi32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xi32>, vector<8x32x16xi32>
  //       CHECK: %[[ACC:.*]] = vector.transfer_read %{{.*}} : memref<8x32xi32>, vector<8x32xi32>
  //       CHECK: %[[MUL:.*]] = arith.muli %{{.*}}, %{{.*}} : vector<8x32x16xi32>
  //       CHECK: vector.multi_reduction <add>, %[[MUL]], %[[ACC]] [2] : vector<8x32x16xi32> to vector<8x32xi32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xi32>, memref<8x32xi32>
  linalg.generic #matmul_trait
    ins(%A, %B : memref<8x16xi32>, memref<16x32xi32>)
   outs(%C : memref<8x32xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32) :
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func @test_vectorize_scalar_input
func.func @test_vectorize_scalar_input(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.generic {
    indexing_maps = [affine_map<(m, n) -> ()>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
   ins(%arg0 : f32)
  outs(%A: memref<8x16xf32>) {
    ^bb(%0: f32, %1: f32) :
      linalg.yield %0 : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_do_not_vectorize_unsupported_element_types
func.func @test_do_not_vectorize_unsupported_element_types(%A : memref<8x16xcomplex<f32>>, %arg0 : complex<f32>) {
  // CHECK-NOT: vector.broadcast
  // CHECK-NOT: vector.transfer_write
  linalg.generic {
    indexing_maps = [affine_map<(m, n) -> ()>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
   ins(%arg0 : complex<f32>)
  outs(%A: memref<8x16xcomplex<f32>>) {
    ^bb(%0: complex<f32>, %1: complex<f32>) :
      linalg.yield %0 : complex<f32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0) -> (d0)>

func.func @vectorize_affine_apply(%arg0: tensor<5xf32>, %arg3: index) -> tensor<5xi32> {
  %0 = tensor.empty() : tensor<5xi32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0],
                       iterator_types = ["parallel"]}
    ins(%arg0 : tensor<5xf32>)
    outs(%0 : tensor<5xi32>) {
  ^bb0(%arg1: f32, %arg2: i32):
    %2 = linalg.index 0 : index
    %11 = affine.apply affine_map<() -> (123)>()
    %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %11)
    %13 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%12)[%arg3]
    %14 = affine.apply affine_map<(d0) -> (d0 + 1)>(%13)
    %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%13, %14, %12)
    %3 = arith.index_cast %15 : index to i32
    linalg.yield %3 : i32
  } -> tensor<5xi32>
  return %1 : tensor<5xi32>
}

// CHECK-LABEL:  func.func @vectorize_affine_apply
// CHECK-SAME: %arg0: tensor<5xf32>
// CHECK-SAME: %[[ARG1:.*]]: index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[123, 124, 125, 126, 127]> : vector<5xindex>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1> : vector<5xindex>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[EMPTY:.*]] = tensor.empty() : tensor<5xi32>
// CHECK:   %[[BCAST:.*]] = vector.broadcast %[[ARG1]] : index to vector<5xindex>
// CHECK:   %[[ADDI_1:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<5xindex>
// CHECK:   %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[CST_0]] : vector<5xindex>
// CHECK:   %[[ADDI_3:.*]] = arith.addi %[[ADDI_1]], %[[ADDI_2]] : vector<5xindex>
// CHECK:   %[[ADDI_4:.*]] = arith.addi %[[ADDI_3]], %[[CST]] : vector<5xindex>
// CHECK:   %[[CAST:.*]] = arith.index_cast %[[ADDI_4]] : vector<5xindex> to vector<5xi32>
// CHECK:   vector.transfer_write %[[CAST]], %[[EMPTY]][%[[C0:.*]]] {in_bounds = [true]} : vector<5xi32>, tensor<5xi32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_trailing_index
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<1x2x4x8xindex>)
func.func @test_vectorize_trailing_index(%arg0: memref<1x2x4x8xindex>) {
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%arg0: memref<1x2x4x8xindex>) {
  ^bb0(%arg1: index):
  //       CHECK:   %[[BCST:.*]] = vector.broadcast %[[CST0]] : vector<8xindex> to vector<1x2x4x8xindex>
  //       CHECK:   vector.transfer_write %[[BCST]], %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {{.*}} : vector<1x2x4x8xindex>, memref<1x2x4x8xindex>
    %0 = linalg.index 3 : index
    linalg.yield %0 : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_inner_index
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<1x2x4x8xindex>)
func.func @test_vectorize_inner_index(%arg0: memref<1x2x4x8xindex>) {
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<[0, 1]> : vector<2xindex>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%arg0: memref<1x2x4x8xindex>) {
  ^bb0(%arg1: index):
  //       CHECK:   %[[BCST:.*]] = vector.broadcast %[[CST0]] : vector<2xindex> to vector<1x8x4x2xindex>
  //       CHECK:   %[[TRAN:.*]] = vector.transpose %[[BCST]], [0, 3, 2, 1] : vector<1x8x4x2xindex> to vector<1x2x4x8xindex>
  //       CHECK:   vector.transfer_write %[[TRAN]], %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {{.*}} : vector<1x2x4x8xindex>, memref<1x2x4x8xindex>
    %0 = linalg.index 1 : index
    linalg.yield %0 : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @generic_vectorize
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<4x256xf32>, %[[ARG1:.*]]: memref<4x256xf32>,
  //  CHECK-SAME:  %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: f32)
func.func @generic_vectorize(%arg0: memref<4x256xf32>,
                        %arg1: memref<4x256xf32>,
                        %arg2: memref<256xf32>, %i: f32) {
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[CST1:.*]] = arith.constant dense<1.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  %c1_f32 = arith.constant 1.0 : f32
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg1, %arg2: memref<4x256xf32>, memref<256xf32>)
  outs(
    %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 :
    memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>,
    memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>,
    memref<4x256xf32>, memref<4x256xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32,
  //       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : memref<256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //       CHECK:   %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : vector<4x256xf32>
    %6 = arith.addf %arg4, %arg6 : f32
  //       CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[V2]], %[[V1]] : vector<4x256xf32>
    %7 = arith.cmpf ogt, %arg3, %arg6 : f32
  //       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
    %8 = arith.constant 2.0 : f32
  //       CHECK:   %[[DIV:.*]] = arith.divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
    %9 = arith.divf %arg5, %i : f32
  //       CHECK:   %[[EXP:.*]] = math.exp2 %[[V3]] : vector<4x256xf32>
    %10 = math.exp2 %arg5 : f32
  //       CHECK:   %[[MUL:.*]] = arith.mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
    %11 = arith.mulf %arg5, %8 : f32
  //       CHECK:   %[[RSQRT:.*]] = math.rsqrt %[[V3]] : vector<4x256xf32>
    %12 = math.rsqrt %arg5 : f32
  //       CHECK:   %[[SEL:.*]] = arith.select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
    %13 = arith.select %7, %arg5, %arg6 : f32
  //       CHECK:   %[[SUB:.*]] = arith.subf %[[V3]], %[[V0]] : vector<4x256xf32>
    %14 = arith.subf %arg5, %arg4 : f32
  //       CHECK:   %[[TAN:.*]] = math.tanh %[[V3]] : vector<4x256xf32>
    %15 = math.tanh %arg5 : f32
  //       CHECK:   vector.transfer_write %[[ADD]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[CST0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[CST1]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[DIV]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[EXP]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[MUL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[RSQRT]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[SEL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[SUB]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[TAN]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @generic_vectorize_tensor
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<4x256xf32>, %[[ARG1:.*]]: tensor<4x256xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: tensor<256xf32>, %[[ARG3:.*]]: f32)
func.func @generic_vectorize_tensor(%arg0: tensor<4x256xf32>,
  %arg1: tensor<4x256xf32>, %arg2: tensor<256xf32>,
  %i: f32) -> (tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>) {
  %c1_f32 = arith.constant 1.0 : f32
  %r:10 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg1, %arg2: tensor<4x256xf32>, tensor<256xf32>)
  outs(
    %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 :
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32,
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[CST1:.*]] = arith.constant dense<1.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  //       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : tensor<256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : vector<4x256xf32>
    %6 = arith.addf %arg4, %arg6 : f32
  //       CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[V2]], %[[V1]] : vector<4x256xf32>
    %7 = arith.cmpf ogt, %arg3, %arg6 : f32
  //       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
    %8 = arith.constant 2.0 : f32
  //       CHECK:   %[[DIV:.*]] = arith.divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
    %9 = arith.divf %arg5, %i : f32
  //       CHECK:   %[[EXP:.*]] = math.exp2 %[[V3]] : vector<4x256xf32>
    %10 = math.exp2 %arg5 : f32
  //       CHECK:   %[[MUL:.*]] = arith.mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
    %11 = arith.mulf %arg5, %8 : f32
  //       CHECK:   %[[RSQRT:.*]] = math.rsqrt %[[V3]] : vector<4x256xf32>
    %12 = math.rsqrt %arg5 : f32
  //       CHECK:   %[[SEL:.*]] = arith.select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
    %13 = arith.select %7, %arg5, %arg6 : f32
  //       CHECK:   %[[SUB:.*]] = arith.subf %[[V3]], %[[V0]] : vector<4x256xf32>
    %14 = arith.subf %arg5, %arg4 : f32
  //       CHECK:   %[[TAN:.*]] = math.tanh %[[V3]] : vector<4x256xf32>
    %15 = math.tanh %arg5 : f32
  //       CHECK:   %[[R0:.*]] = vector.transfer_write %[[ADD]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R1:.*]] = vector.transfer_write %[[CST0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R2:.*]] = vector.transfer_write %[[CST1]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R3:.*]] = vector.transfer_write %[[DIV]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R4:.*]] = vector.transfer_write %[[EXP]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R5:.*]] = vector.transfer_write %[[MUL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R6:.*]] = vector.transfer_write %[[RSQRT]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R7:.*]] = vector.transfer_write %[[SEL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R8:.*]] = vector.transfer_write %[[SUB]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R9:.*]] = vector.transfer_write %[[TAN]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  } -> (tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>)
  //       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]], %[[R8]], %[[R9]] : tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
  return %r#0, %r#1, %r#2, %r#3, %r#4, %r#5, %r#6, %r#7, %r#8, %r#9:
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, 0, 0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0, 0, 0, 0)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1, 0, d0, 0)>
//     CHECK: func @generic_vectorize_broadcast_transpose
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[PV:.*]] = ub.poison : f32
//     CHECK:   %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[PV]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP0]]} : memref<4x4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[PV]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP1]]} : memref<4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V2:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[PV]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP2]]} : memref<4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V3:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[PV]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP3]]} : memref<4x4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[SUB:.*]] = arith.subf %[[V0]], %[[V1]] : vector<4x4x4x4xf32>
//     CHECK:   %[[ADD0:.*]] = arith.addf %[[V2]], %[[SUB]] : vector<4x4x4x4xf32>
//     CHECK:   %[[ADD1:.*]] = arith.addf %[[V3]], %[[ADD0]] : vector<4x4x4x4xf32>
//     CHECK: vector.transfer_write %[[ADD1]], {{.*}} : vector<4x4x4x4xf32>, memref<4x4x4x4xf32>
func.func @generic_vectorize_broadcast_transpose(
  %A: memref<4xf32>, %B: memref<4x4xf32>, %C: memref<4x4x4x4xf32>) {
  linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
                   affine_map<(d0, d1, d2, d3) -> (d0)>,
                   affine_map<(d0, d1, d2, d3) -> (d2)>,
                   affine_map<(d0, d1, d2, d3) -> (d2, d0)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%B, %A, %A, %B: memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4x4xf32>)
  outs(%C : memref<4x4x4x4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %s = arith.subf %arg0, %arg1 : f32
    %a = arith.addf %arg2, %s : f32
    %b = arith.addf %arg3, %a : f32
    linalg.yield %b : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Test different input maps.
#matmul_trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0, 0, 0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (0, d1, 0, d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3, d0)>
//       CHECK: func @vectorization_transpose
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP0]]} : memref<14x7xf32>, vector<7x14x8x16xf32>
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP1]]} : memref<16x14xf32>, vector<7x14x8x16xf32>
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP2]]} : memref<16x14x7x8xf32>, vector<7x14x8x16xf32>
//       CHECK: arith.addf {{.*}} : vector<7x14x8x16xf32>
//       CHECK: arith.addf {{.*}} : vector<7x14x8x16xf32>
//       CHECK: vector.transfer_write {{.*}} : vector<7x14x8x16xf32>, memref<7x14x8x16xf32>
func.func @vectorization_transpose(%A: memref<14x7xf32>, %B: memref<16x14xf32>,
                         %C: memref<16x14x7x8xf32>, %D: memref<7x14x8x16xf32>) {
  linalg.generic #matmul_trait
    ins(%A, %B, %C : memref<14x7xf32>, memref<16x14xf32>, memref<16x14x7x8xf32>)
   outs(%D : memref<7x14x8x16xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32, %d: f32) :
      %e = arith.addf %a, %b: f32
      %f = arith.addf %e, %c: f32
      linalg.yield %f : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @sum_exp
func.func @sum_exp(%input: tensor<4x16x8xf32>, %output: tensor<4x16xf32>)
  -> tensor<4x16xf32>
{
  // CHECK: vector.transfer_read {{.*}} : tensor<4x16x8xf32>, vector<4x16x8xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<4x16xf32>, vector<4x16xf32>
  // CHECK: math.exp {{.*}} : vector<4x16x8xf32>
  // CHECK: vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<4x16x8xf32> to vector<4x16xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4x16xf32>, tensor<4x16xf32>
  // CHECK: return {{.*}} : tensor<4x16xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input : tensor<4x16x8xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %1 = math.exp %arg0 : f32
      %2 = arith.addf %1, %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$M1:.*]] =  affine_map<(d0, d1) -> (d1, d0, 0, 0)>
// CHECK-DAG: #[[$M2:.*]] =  affine_map<(d0, d1) -> (0, 0, d1, d0)>
// CHECK-DAG: #[[$M3:.*]] =  affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @sum_exp_2
func.func @sum_exp_2(%input: tensor<3x2xf32>, %input_2: tensor<5x4xf32>, %output: tensor<5x2xf32>)
  -> tensor<5x2xf32>
{
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M1]]} : tensor<3x2xf32>, vector<2x3x4x5xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M2]]} : tensor<5x4xf32>, vector<2x3x4x5xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #[[$M3]]} : tensor<5x2xf32>, vector<2x5xf32>
  // CHECK: math.exp {{.*}} : vector<2x3x4x5xf32>
  // CHECK: math.exp {{.*}} : vector<2x3x4x5xf32>
  // CHECK: addf {{.*}} : vector<2x3x4x5xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}}, %{{.*}}  [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true, true], permutation_map = #[[$M3]]} : vector<2x5xf32>, tensor<5x2xf32>
  // CHECK: return {{.*}} : tensor<5x2xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d0)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0)>
      ],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]
    } ins(%input, %input_2 : tensor<3x2xf32>, tensor<5x4xf32>) outs(%output : tensor<5x2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %1 = math.exp %arg0 : f32
      %2 = math.exp %arg1 : f32
      %3 = arith.addf %1, %2 : f32
      %4 = arith.addf %3, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_maximumf_2d(
func.func @red_maximumf_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMINF:.+]] = arith.constant dense<-3.402820e+38> : vector<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.multi_reduction <maximumf>, {{.*}}, %[[CMINF]] [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %ident = arith.constant -3.40282e+38 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%ident : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %max = arith.maximumf %in0, %out0 : f32
    linalg.yield %max : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_maxnumf_2d(
func.func @red_maxnumf_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMINF:.+]] = arith.constant dense<-3.402820e+38> : vector<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.multi_reduction <maxnumf>, {{.*}}, %[[CMINF]] [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %ident = arith.constant -3.40282e+38 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%ident : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %max = arith.maxnumf %in0, %out0 : f32
    linalg.yield %max : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_minimumf_2d(
func.func @red_minimumf_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMAXF:.+]] = arith.constant dense<3.402820e+38> : vector<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.multi_reduction <minimumf>, {{.*}}, %[[CMAXF]] [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %maxf32 = arith.constant 3.40282e+38 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%maxf32 : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %min = arith.minimumf %out0, %in0 : f32
    linalg.yield %min : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_minnumf_2d(
func.func @red_minnumf_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMAXF:.+]] = arith.constant dense<3.402820e+38> : vector<4xf32>
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.multi_reduction <minnumf>, {{.*}}, %[[CMAXF]] [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %maxf32 = arith.constant 3.40282e+38 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%maxf32 : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %min = arith.minnumf %out0, %in0 : f32
    linalg.yield %min : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_mul_2d(
func.func @red_mul_2d(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK: tensor.empty() : tensor<4xf32>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.multi_reduction <mul>, {{.*}}, {{.*}} [1] : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<4xf32>
  %ident = arith.constant 1.0 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%ident : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%in0: f32, %out0: f32):
    %mul = arith.mulf %in0, %out0 : f32
    linalg.yield %mul : f32
  } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_or_2d(
func.func @red_or_2d(%arg0: tensor<4x4xi1>) -> tensor<4xi1> {
  // CHECK: tensor.empty() : tensor<4xi1>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xi1>, vector<4x4xi1>
  // CHECK: vector.multi_reduction <or>, {{.*}}, {{.*}} [1] : vector<4x4xi1> to vector<4xi1>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi1>, tensor<4xi1>
  %ident = arith.constant false
  %init = tensor.empty() : tensor<4xi1>
  %fill = linalg.fill ins(%ident : i1) outs(%init : tensor<4xi1>) -> tensor<4xi1>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xi1>) outs(%fill : tensor<4xi1>) {
  ^bb0(%in0: i1, %out0: i1):
    %or = arith.ori %in0, %out0 : i1
    linalg.yield %or : i1
  } -> tensor<4xi1>
  return %red : tensor<4xi1>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_and_2d(
func.func @red_and_2d(%arg0: tensor<4x4xi1>) -> tensor<4xi1> {
  // CHECK: tensor.empty() : tensor<4xi1>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xi1>, vector<4x4xi1>
  // CHECK: vector.multi_reduction <and>, {{.*}}, {{.*}} [1] : vector<4x4xi1> to vector<4xi1>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi1>, tensor<4xi1>
  %ident = arith.constant true
  %init = tensor.empty() : tensor<4xi1>
  %fill = linalg.fill ins(%ident : i1) outs(%init : tensor<4xi1>) -> tensor<4xi1>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xi1>) outs(%fill : tensor<4xi1>) {
  ^bb0(%in0: i1, %out0: i1):
    %and = arith.andi %in0, %out0 : i1
    linalg.yield %and : i1
  } -> tensor<4xi1>
  return %red : tensor<4xi1>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL:   func @red_xor_2d(
func.func @red_xor_2d(%arg0: tensor<4x4xi1>) -> tensor<4xi1> {
  // CHECK: tensor.empty() : tensor<4xi1>
  // CHECK: vector.transfer_read {{.*}} : tensor<4x4xi1>, vector<4x4xi1>
  // CHECK: vector.multi_reduction <xor>, {{.*}}, {{.*}} [1] : vector<4x4xi1> to vector<4xi1>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi1>, tensor<4xi1>
  %ident = arith.constant false
  %init = tensor.empty() : tensor<4xi1>
  %fill = linalg.fill ins(%ident : i1) outs(%init : tensor<4xi1>) -> tensor<4xi1>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                         ins(%arg0 : tensor<4x4xi1>) outs(%fill : tensor<4xi1>) {
  ^bb0(%in0: i1, %out0: i1):
    %xor = arith.xori %in0, %out0 : i1
    linalg.yield %xor : i1
  } -> tensor<4xi1>
  return %red : tensor<4xi1>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$M5:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL:   func @explicit_broadcast(
func.func @explicit_broadcast(%arg0: tensor<4x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x4xf32> {
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #[[$M5]]} : tensor<4x1xf32>, vector<4x4xf32>
  // CHECK: subf {{.*}} : vector<4x4xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, 0)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]}
   ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x1xf32>)
   outs(%fill : tensor<4x4xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %40 = arith.subf %arg7, %arg8 : f32
      linalg.yield %40 : f32
    } -> tensor<4x4xf32>
  return %red : tensor<4x4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$M6:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL:   func @fused_broadcast_red_2d
func.func @fused_broadcast_red_2d(%arg0: tensor<4x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4xf32> {
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #[[$M6]]} : tensor<4x1xf32>, vector<4x4xf32>
  // CHECK: subf {{.*}} : vector<4x4xf32>
  // CHECK: math.exp {{.*}} : vector<4x4xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}}, {{.*}} : vector<4x4xf32> to vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, 0)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x1xf32>)
   outs(%fill : tensor<4xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %40 = arith.subf %arg7, %arg8 : f32
      %41 = math.exp %40 : f32
      %42 = arith.addf %41, %arg9 : f32
      linalg.yield %42 : f32
    } -> tensor<4xf32>
  return %red : tensor<4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op

    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

//  CHECK-LABEL: func @reduce_1d(
//   CHECK-SAME:   %[[A:.*]]: tensor<32xf32>
func.func @reduce_1d(%arg0: tensor<32xf32>) -> tensor<f32> {
  //  CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32

  //      CHECK: %[[init:.*]] = tensor.empty() : tensor<f32>
  %0 = tensor.empty() : tensor<f32>

  %1 = linalg.fill ins(%f0 : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  //      CHECK: %[[r:.*]] = vector.transfer_read %[[A]][%[[C0]]]
  // CHECK-SAME:   : tensor<32xf32>, vector<32xf32>
  //      CHECK: %[[red:.*]] = vector.multi_reduction <add>, %[[r]], %[[F0]] [0]
  // CHECK-SAME:   : vector<32xf32> to f32
  //      CHECK: %[[red_v1:.*]] = vector.broadcast %[[red]] : f32 to vector<f32>
  //      CHECK: %[[res:.*]] = vector.transfer_write %[[red_v1]], %[[init]][]
  // CHECK-SAME:   : vector<f32>, tensor<f32>
  %2 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> ()>],
         iterator_types = ["reduction"]}
         ins(%arg0 : tensor<32xf32>)
         outs(%1 : tensor<f32>) {
    ^bb0(%a: f32, %b: f32):
      %3 = arith.addf %a, %b : f32
      linalg.yield %3 : f32
    } -> tensor<f32>

  return %2 : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}


// -----

// This test checks that vectorization does not occur when an input indexing map
// is not a projected permutation. In the future, this can be converted to a
// positive test when support is added.

// CHECK-LABEL:   func @not_projected_permutation
func.func @not_projected_permutation(%arg0: tensor<8x8xf32>) -> tensor<6x6x3x3xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<6x6x3x3xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<6x6x3x3xf32>) -> tensor<6x6x3x3xf32>
  // CHECK: linalg.generic
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
   iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
   ins(%arg0 : tensor<8x8xf32>)
   outs(%fill : tensor<6x6x3x3xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      linalg.yield %arg7 : f32
    } -> tensor<6x6x3x3xf32>
  return %result : tensor<6x6x3x3xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Check vectorization can handle cases where outputs are a mix of reduced and non-reduced values.
func.func @mixed_parallel_reduced_results(%arg0 : tensor<2x4x8xf32>,
    %arg1 : tensor<2x4xf32>, %arg2 : tensor<2x4x8xf32>, %arg3 : tensor<2x4xf32>) ->
    (tensor<2x4x8xf32>, tensor<2x4xf32>) {
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<2x4x8xf32>, tensor<2x4xf32>)
      outs(%arg2, %arg3 : tensor<2x4x8xf32>, tensor<2x4xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32, %b3 : f32):
      %1 = arith.mulf %b0, %b1 : f32
      %2 = arith.addf %1, %b3 : f32
      linalg.yield %1, %2 : f32, f32
  } -> (tensor<2x4x8xf32>, tensor<2x4xf32>)
  return %0#0, %0#1 : tensor<2x4x8xf32>, tensor<2x4xf32>
}
// CHECK-LABEL: func @mixed_parallel_reduced_results(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x4x8xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x4xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<2x4x8xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<2x4xf32>
//   CHECK-DAG:   %[[V0:.+]] = vector.transfer_read %[[ARG0]]
//   CHECK-DAG:   %[[V1:.+]] = vector.transfer_read %[[ARG1]]
//   CHECK-DAG:   %[[V2:.+]] = vector.transfer_read %[[ARG3]]
//   CHECK-DAG:   %[[MUL:.+]] = arith.mulf %[[V0]], %[[V1]]
//   CHECK-DAG:   %[[ADD:.+]] = vector.multi_reduction <add>, %[[MUL]], %[[V2]]
//   CHECK-DAG:   vector.transfer_write %[[MUL]], %[[ARG2]]
//   CHECK-DAG:   vector.transfer_write %[[ADD]], %[[ARG3]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// This is a regression test. This IR cannot be vectorized, but
// structured.vectorize_children_and_apply_patterns should nevertheless succeed.

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   @not_vectorizable
func.func @not_vectorizable(%arg0: tensor<1x?xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<1x128xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1x128xf32>
  %1 = scf.for %arg5 = %arg2 to %arg1 step %arg3 iter_args(%arg6 = %0) -> (tensor<1x128xf32>) {
    %extracted_slice = tensor.extract_slice %arg6[0, 0] [1, %arg1] [1, 1] : tensor<1x128xf32> to tensor<?xf32>
    %sz0 = tensor.dim %extracted_slice, %c0 : tensor<?xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [1, %sz0] : tensor<?xf32> into tensor<1x?xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, %arg3] [1, %arg2] [1, 1] : tensor<1x?xf32> to tensor<?xf32>
    %extracted_slice_1 = tensor.extract_slice %expanded[0, %arg3] [1, %arg2] [1, 1] : tensor<1x?xf32> to tensor<?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%extracted_slice_0 : tensor<?xf32>) outs(%extracted_slice_1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %2 into %expanded[0, %arg3] [1, %arg2] [1, 1] : tensor<?xf32> into tensor<1x?xf32>
    %collapsed = tensor.collapse_shape %inserted_slice [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    %inserted_slice_2 = tensor.insert_slice %collapsed into %arg6[0, 0] [1, %arg1] [1, 1] : tensor<?xf32> into tensor<1x128xf32>
    scf.yield %inserted_slice_2 : tensor<1x128xf32>
  }
  return %1 : tensor<1x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.vectorize_children_and_apply_patterns %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Regression test: %13 was incorrectly detected as a reduction and
// vectorization failed.

func.func @wrong_reduction_detection(%input: tensor<120x64xf32>) -> tensor<120x64xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst_6 = arith.constant 4.000000e+00 : f32
  %1 = scf.for %arg0 = %c0 to %c64 step %c4 iter_args(%arg1 = %input) -> (tensor<120x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%c0, %arg0] [1, 4] [1, 1] : tensor<120x64xf32> to tensor<1x4xf32>
    %10 = linalg.fill {__internal_linalg_transform__ = "1"} ins(%cst_6 : f32) outs(%extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%10 : tensor<1x4xf32>) {
    ^bb0(%out: f32):
      %12 = linalg.index 0 : index
      %13 = arith.addi %arg0, %12 : index
      %18 = arith.index_cast %13 : index to i32
      %20 = arith.uitofp %18 : i32 to f32
      %67 = arith.mulf %out, %20 : f32
      linalg.yield %67 : f32
    } -> tensor<1x4xf32>
    %inserted_slice = tensor.insert_slice %11 into %arg1[%c0, %arg0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<120x64xf32>
    scf.yield %inserted_slice : tensor<120x64xf32>
  }
  return %1 : tensor<120x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @wrong_reduction_detection
// CHECK:         vector.broadcast
// CHECK:         vector.transfer_write

// -----

// Don't vectorize tensor<0xf32> : (!transform.any_op) -> !transform.any_op
// CHECK-LABEL: @tensor_size0
// CHECK:         linalg.generic
func.func @tensor_size0(%arg0: tensor<0xf32>,
                        %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic
  {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
  iterator_types = ["reduction"]}
  ins(%arg0 : tensor<0xf32>) outs(%arg1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
    %12 = arith.addf %out, %in : f32
    linalg.yield %12 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @zero_dim_tensor(%input: tensor<f32>, %output: tensor<f32>) -> tensor<f32>
{
  %0 = linalg.generic { indexing_maps = [ affine_map<() -> ()>, affine_map<() -> ()> ],
                        iterator_types = [] }
                        ins(%input : tensor<f32>)
                        outs(%output : tensor<f32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %2 = arith.addf %arg0, %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
  return %0 : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @zero_dim_tensor
//       CHECK:     vector.transfer_read {{.*}} : tensor<f32>, vector<f32>
//       CHECK:     vector.extract
//       CHECK:     vector.transfer_read {{.*}} : tensor<f32>, vector<f32>
//       CHECK:     vector.extract
//       CHECK:     arith.addf {{.*}} : f32
//       CHECK:     vector.broadcast %{{.*}} : f32 to vector<f32>
//       CHECK:     vector.transfer_write {{.*}} : vector<f32>, tensor<f32>

// -----

// Make sure we generate the right transfer writes for multi-output generic ops
// with different permutation maps.

func.func @multi_output_generic_different_perm_maps(%in0: tensor<4x1xf32>,
                                                    %out0: tensor<4x1xf32>,
                                                    %out1: tensor<1x4xf32>) -> (tensor<4x1xf32>, tensor<1x4xf32>) {
  %13:2 = linalg.generic {indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>,
                                            affine_map<(d0, d1) -> (d1, d0)>,
                                            affine_map<(d0, d1) -> (d0, d1)> ],
                          iterator_types = ["parallel", "parallel"]}
                          ins(%in0 : tensor<4x1xf32>)
                          outs(%out0, %out1 : tensor<4x1xf32>, tensor<1x4xf32>) {
  ^bb0(%in: f32, %out: f32, %out_2: f32):
    %16 = arith.addf %in, %in : f32
    linalg.yield %16, %16 : f32, f32
  } -> (tensor<4x1xf32>, tensor<1x4xf32>)
  return %13#0, %13#1 : tensor<4x1xf32>, tensor<1x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @multi_output_generic_different_perm_maps
//       CHECK:     %[[VAL_5:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<4x1xf32>, vector<4x1xf32>
//       CHECK:     %[[VAL_6:.*]] = arith.addf %[[VAL_5]], %[[VAL_5]] : vector<4x1xf32>
//       CHECK:     %[[VAL_7:.*]] = vector.transpose %[[VAL_6]], [1, 0] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:     %[[VAL_8:.*]] = vector.transpose %[[VAL_7]], [1, 0] : vector<1x4xf32> to vector<4x1xf32>
//       CHECK:     vector.transfer_write %[[VAL_8]], %{{.*}} {in_bounds = [true, true]} : vector<4x1xf32>, tensor<4x1xf32>
//       CHECK:     vector.transfer_write %[[VAL_7]], %{{.*}} {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>

// -----

// Extracted from: https://github.com/llvm/llvm-project/issues/97247

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

func.func @generic_with_reduction_and_broadcast(%arg0: tensor<1x12x197x197xf32>) -> (tensor<1x12x197x1xf32>) {
  %0 = tensor.empty() : tensor<1x12x197x1xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x12x197x197xf32>) outs(%0 : tensor<1x12x197x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %818 = arith.addf %in, %out : f32
    linalg.yield %818 : f32
  } -> tensor<1x12x197x1xf32>
  return %1 : tensor<1x12x197x1xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$ATTR_32:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL:   func.func @generic_with_reduction_and_broadcast(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<1x12x197x197xf32>) -> tensor<1x12x197x1xf32> {
// CHECK:           %[[VAL_1:.*]] = ub.poison : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x12x197x1xf32>
// CHECK:           %[[VAL_4:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]]], %[[VAL_1]] {in_bounds = [true, true, true, true]} : tensor<1x12x197x197xf32>, vector<1x12x197x197xf32>
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_3]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]]], %[[VAL_1]] {in_bounds = [true, true, true], permutation_map = #[[$ATTR_32]]} : tensor<1x12x197x1xf32>, vector<1x12x197xf32>
// CHECK:           %[[VAL_6:.*]] = vector.multi_reduction <add>, %[[VAL_4]], %[[VAL_5]] [3] : vector<1x12x197x197xf32> to vector<1x12x197xf32>
// CHECK:           %[[VAL_7:.*]] = vector.broadcast %[[VAL_6]] : vector<1x12x197xf32> to vector<1x1x12x197xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transpose %[[VAL_7]], [1, 2, 3, 0] : vector<1x1x12x197xf32> to vector<1x12x197x1xf32>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_write %[[VAL_8]], %[[VAL_3]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] {in_bounds = [true, true, true, true]} : vector<1x12x197x1xf32>, tensor<1x12x197x1xf32>
// CHECK:           return %[[VAL_9]] : tensor<1x12x197x1xf32>

// -----

// CHECK-LABEL: func @float_mixed_precision_matmul_as_generic
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extf
// CHECK:         vector.contract {{.*}} : vector<8x16xbf16>, vector<16x32xbf16> into vector<8x32xf32>
// CHECK:         vector.transfer_write
func.func @float_mixed_precision_matmul_as_generic(%A: memref<8x16xbf16>, %B: memref<16x32xbf16>,
                         %C: memref<8x32xf32>) {
  linalg.generic {
    indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
   ins(%A, %B : memref<8x16xbf16>, memref<16x32xbf16>)
   outs(%C : memref<8x32xf32>) {
    ^bb(%in: bf16, %in_0: bf16, %c: f32) :
      %a = arith.extf %in : bf16 to f32
      %b = arith.extf %in_0 : bf16 to f32
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { fold_type_extensions_into_contract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @integer_mixed_precision_matmul_as_generic
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extsi
// CHECK:         vector.contract {{.*}} : vector<8x16xi8>, vector<16x32xi8> into vector<8x32xi32>
// CHECK:         vector.transfer_write
func.func @integer_mixed_precision_matmul_as_generic(%A: memref<8x16xi8>, %B: memref<16x32xi8>,
                         %C: memref<8x32xi32>) {
  linalg.generic {
    indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
   ins(%A, %B : memref<8x16xi8>, memref<16x32xi8>)
   outs(%C : memref<8x32xi32>) {
    ^bb(%in: i8, %in_0: i8, %c: i32) :
      %a = arith.extsi %in : i8 to i32
      %b = arith.extsi %in_0 : i8 to i32
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { fold_type_extensions_into_contract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

