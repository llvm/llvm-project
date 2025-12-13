// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for vectorizing operations implementing contraction op interface.
/// Ops implementing the contraction interface are vectorized directly to their
/// vector dialect named counterparts.
///----------------------------------------------------------------------------------------

func.func @matmul(%A: tensor<8x4xf32>, %B: tensor<4x16xf32>,
    %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.matmul
    ins(%A, %B : tensor<8x4xf32>, tensor<4x16xf32>)
    outs(%C: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %[[A:.*]]: tensor<8x4xf32>, %[[B:.*]]: tensor<4x16xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<8x16xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.transfer_read %[[A]]{{.*}}: tensor<8x4xf32>, vector<8x4xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.transfer_read %[[B]]{{.*}}: tensor<4x16xf32>, vector<4x16xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.transfer_read %[[C]]{{.*}}: tensor<8x16xf32>, vector<8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x16xf32>, tensor<8x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_dynamic(
// CHECK-SAME:    %[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<?x?xf32>)

/// Get the contraction dimensions
//  CHECK: %[[MATMUL_DIM_M_IDX:.*]] = arith.constant 0 : index
//  CHECK: %[[MATMUL_DIM_M:.*]] = tensor.dim %[[A]], %[[MATMUL_DIM_M_IDX]] : tensor<?x?xf32>
//  CHECK: %[[MATMUL_DIM_N_IDX:.*]] = arith.constant 1 : index
//  CHECK: %[[MATMUL_DIM_N:.*]] = tensor.dim %[[B]], %[[MATMUL_DIM_N_IDX]] : tensor<?x?xf32>
//  CHECK: %[[MATMUL_DIM_K_IDX:.*]] = arith.constant 1 : index
//  CHECK: %[[MATMUL_DIM_K:.*]] = tensor.dim %[[A]], %[[MATMUL_DIM_K_IDX]] : tensor<?x?xf32>

/// Create a mask for the A matrix
//      CHECK: %[[A_OFFSET:.*]] = arith.constant 0 : index
//      CHECK: %[[A_DIM_M_IDX:.*]] = arith.constant 0 : index
//      CHECK: %[[A_DIM_M:.*]] = tensor.dim %[[A]], %[[A_DIM_M_IDX]] : tensor<?x?xf32>
//      CHECK: %[[A_DIM_K_IDX:.*]] = arith.constant 1 : index
//      CHECK: %[[A_DIM_K:.*]] = tensor.dim %[[A]], %[[A_DIM_K_IDX]] : tensor<?x?xf32>
//      CHECK: %[[LOAD_A_MASK:.*]] = vector.create_mask
// CHECK-SAME:   %[[A_DIM_M]], %[[A_DIM_K]] : vector<8x4xi1>
/// Read the A matrix
//      CHECK: %[[LOAD_A:.*]] = vector.mask %[[LOAD_A_MASK]]
// CHECK-SAME:   { vector.transfer_read %[[A]]{{\[}}%[[A_OFFSET]], %[[A_OFFSET]]{{\]}}
// CHECK-SAME:     : tensor<?x?xf32>, vector<8x4xf32> }
// CHECK-SAME:   : vector<8x4xi1> -> vector<8x4xf32>

/// Create a mask for the B matrix
//      CHECK: %[[B_OFFSET:.*]] = arith.constant 0 : index
//      CHECK: %[[B_DIM_K_IDX:.*]] = arith.constant 0 : index
//      CHECK: %[[B_DIM_K:.*]] = tensor.dim %[[B]], %[[B_DIM_K_IDX]] : tensor<?x?xf32>
//      CHECK: %[[B_DIM_N_IDX:.*]] = arith.constant 1 : index
//      CHECK: %[[B_DIM_N:.*]] = tensor.dim %[[B]], %[[B_DIM_N_IDX]] : tensor<?x?xf32>
//      CHECK: %[[LOAD_B_MASK:.*]] = vector.create_mask
// CHECK-SAME:   %[[B_DIM_K]], %[[B_DIM_N]] : vector<4x16xi1>
/// Read the B matrix
//      CHECK: %[[LOAD_B:.*]] = vector.mask %[[LOAD_B_MASK]]
// CHECK-SAME:   { vector.transfer_read %[[B]]{{\[}}%[[B_OFFSET]], %[[B_OFFSET]]{{\]}}
// CHECK-SAME:     : tensor<?x?xf32>, vector<4x16xf32> }
// CHECK-SAME:   : vector<4x16xi1> -> vector<4x16xf32>

/// Create a mask for the C matrix
//      CHECK: %[[C_OFFSET:.*]] = arith.constant 0 : index
//      CHECK: %[[C_DIM_M_IDX:.*]] = arith.constant 0 : index
//      CHECK: %[[C_DIM_M:.*]] = tensor.dim %[[C]], %[[C_DIM_M_IDX]] : tensor<?x?xf32>
//      CHECK: %[[C_DIM_N_IDX:.*]] = arith.constant 1 : index
//      CHECK: %[[C_DIM_N:.*]] = tensor.dim %[[C]], %[[C_DIM_N_IDX]] : tensor<?x?xf32>
//      CHECK: %[[LOAD_C_MASK:.*]] = vector.create_mask
// CHECK-SAME:   %[[C_DIM_M]], %[[C_DIM_N]] : vector<8x16xi1>
/// Read the C matrix
//      CHECK: %[[LOAD_C:.*]] = vector.mask %[[LOAD_C_MASK]]
// CHECK-SAME:   { vector.transfer_read %[[C]]{{\[}}%[[C_OFFSET]], %[[C_OFFSET]]{{\]}}
// CHECK-SAME:     : tensor<?x?xf32>, vector<8x16xf32> }
// CHECK-SAME:   : vector<8x16xi1> -> vector<8x16xf32>

/// Create a mask for the contraction
//      CHECK: %[[CONTRACTION_MASK:.*]] = vector.create_mask
// CHECK-SAME:   %[[MATMUL_DIM_M]], %[[MATMUL_DIM_N]], %[[MATMUL_DIM_K]]
// CHECK-SAME:   : vector<8x16x4xi1>
/// Perform the contraction
//      CHECK: %[[D:.*]] = vector.mask %[[CONTRACTION_MASK]]
// CHECK-SAME:   { vector.contract
// CHECK-SAME:     indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:     kind = #vector.kind<add>
// CHECK-SAME:     %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
// CHECK-SAME:   } : vector<8x16x4xi1> -> vector<8x16xf32>

/// Create a mask for the result
//      CHECK: %[[D_OFFSET:.*]] = arith.constant 0 : index
//      CHECK: %[[D_DIM_M_IDX:.*]] = arith.constant 0 : index
//      CHECK: %[[D_DIM_M:.*]] = tensor.dim %[[C]], %[[D_DIM_M_IDX]] : tensor<?x?xf32>
//      CHECK: %[[D_DIM_N_IDX:.*]] = arith.constant 1 : index
//      CHECK: %[[D_DIM_N:.*]] = tensor.dim %[[C]], %[[D_DIM_N_IDX]] : tensor<?x?xf32>
//      CHECK: %[[LOAD_D_MASK:.*]] = vector.create_mask
// CHECK-SAME:   %[[D_DIM_M]], %[[D_DIM_N]] : vector<8x16xi1>
/// Write the result
//      CHECK: vector.mask %[[LOAD_D_MASK]]
// CHECK-SAME: { vector.transfer_write %[[D]], %[[C]]{{\[}}%[[D_OFFSET]], %[[D_OFFSET]]{{\]}}
// CHECK-SAME:   : vector<8x16xf32>, tensor<?x?xf32> }
// CHECK-SAME: : vector<8x16xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 16, 4]
      {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_dynamic_memref(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
    %C: memref<?x?xf32>) {
  linalg.matmul
    ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
    outs(%C: memref<?x?xf32>)
  return
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_dynamic_memref(
// CHECK-SAME:    %[[A:.*]]: memref<?x?xf32>, %[[B:.*]]: memref<?x?xf32>,
// CHECK-SAME:    %[[C:.*]]: memref<?x?xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[A]]{{.*}}: memref<?x?xf32>, vector<8x4xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[B]]{{.*}}: memref<?x?xf32>, vector<4x16xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[C]]{{.*}}: memref<?x?xf32>, vector<8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.mask{{.*}}{ vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.mask{{.*}}{ vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x16xf32>, memref<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 16, 4]
      {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_dynamic_scalable(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_dynamic_scalable(
// CHECK-SAME:    %[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<?x?xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[A]]{{.*}}: tensor<?x?xf32>, vector<8x4xf32> }
// CHECK-SAME:   : vector<8x4xi1> -> vector<8x4xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[B]]{{.*}}: tensor<?x?xf32>, vector<4x[16]xf32> }
// CHECK-SAME:   : vector<4x[16]xi1> -> vector<4x[16]xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[C]]{{.*}}: tensor<?x?xf32>, vector<8x[16]xf32> }
// CHECK-SAME:   : vector<8x[16]xi1> -> vector<8x[16]xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.mask{{.*}}{ vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
// CHECK-SAME:   } : vector<8x[16]x4xi1> -> vector<8x[16]xf32>
//      CHECK: vector.mask{{.*}}{ vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x[16]xf32>, tensor<?x?xf32> }
// CHECK-SAME:   : vector<8x[16]xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, [16], 4]
      {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_transpose(%A: tensor<4x8xf32>, %B: tensor<16x4xf32>,
    %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.matmul
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>, // transpose A
                     affine_map<(m, n, k) -> (n, k)>, // transpose B
                     affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %B : tensor<4x8xf32>, tensor<16x4xf32>)
    outs(%C: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_transpose(
// CHECK-SAME:    %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<16x4xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<8x16xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.transfer_read %[[A]]{{.*}}: tensor<4x8xf32>, vector<4x8xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.transfer_read %[[B]]{{.*}}: tensor<16x4xf32>, vector<16x4xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.transfer_read %[[C]]{{.*}}: tensor<8x16xf32>, vector<8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x16xf32>, tensor<8x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_dynamic_transpose(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>, // transpose A
                     affine_map<(m, n, k) -> (n, k)>, // transpose B
                     affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_dynamic_transpose(
// CHECK-SAME:    %[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<?x?xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[A]]{{.*}}: tensor<?x?xf32>, vector<4x8xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[B]]{{.*}}: tensor<?x?xf32>, vector<16x4xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.mask{{.*}}{ vector.transfer_read %[[C]]{{.*}}: tensor<?x?xf32>, vector<8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.mask{{.*}}{ vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.mask{{.*}}{ vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x16xf32>, tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 16, 4]
      {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

/// Contractions with arbitrarty broadcasts are not supported in contraction interface
/// vectorization.
/// Dimension broadcasts are expected to be decomposed first which removes ambiguity
/// caused by possible variants of dimensions materialization.
/// For example, whether the below target LHS input layout is (m, k) or (k, m).

func.func @negative_matmul_broadcast(%A: tensor<4xf32>, %B: tensor<4x16xf32>,
    %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  %0 = linalg.matmul
    indexing_maps = [affine_map<(m, n, k) -> (k)>, // broadcast
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %B : tensor<4xf32>, tensor<4x16xf32>)
    outs(%C: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_mixed_precision(%A: tensor<8x4xf16>, %B: tensor<4x16xf16>,
    %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.matmul
    ins(%A, %B : tensor<8x4xf16>, tensor<4x16xf16>)
    outs(%C: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_mixed_precision(
// CHECK-SAME:    %[[A:.*]]: tensor<8x4xf16>, %[[B:.*]]: tensor<4x16xf16>,
// CHECK-SAME:    %[[C:.*]]: tensor<8x16xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.transfer_read %[[A]]{{.*}}: tensor<8x4xf16>, vector<8x4xf16>
//      CHECK: %[[LOAD_B:.*]] = vector.transfer_read %[[B]]{{.*}}: tensor<4x16xf16>, vector<4x16xf16>
//      CHECK: %[[LOAD_C:.*]] = vector.transfer_read %[[C]]{{.*}}: tensor<8x16xf32>, vector<8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x16xf32>, tensor<8x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @batch_matmul(%A: tensor<3x8x4xf32>, %B: tensor<3x4x16xf32>,
    %C: tensor<3x8x16xf32>) -> tensor<3x8x16xf32> {
  %0 = linalg.batch_matmul
    ins(%A, %B : tensor<3x8x4xf32>, tensor<3x4x16xf32>)
    outs(%C: tensor<3x8x16xf32>) -> tensor<3x8x16xf32>
  return %0 : tensor<3x8x16xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @batch_matmul(
// CHECK-SAME:    %[[A:.*]]: tensor<3x8x4xf32>, %[[B:.*]]: tensor<3x4x16xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<3x8x16xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.transfer_read %[[A]]{{.*}}: tensor<3x8x4xf32>, vector<3x8x4xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.transfer_read %[[B]]{{.*}}: tensor<3x4x16xf32>, vector<3x4x16xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.transfer_read %[[C]]{{.*}}: tensor<3x8x16xf32>, vector<3x8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<3x8x16xf32>, tensor<3x8x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.batch_matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @batch_reduce_matmul(%A: tensor<3x8x4xf32>, %B: tensor<3x4x16xf32>,
    %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.batch_reduce_matmul
    ins(%A, %B : tensor<3x8x4xf32>, tensor<3x4x16xf32>)
    outs(%C: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-LABEL: func.func @batch_reduce_matmul(
// CHECK-SAME:    %[[A:.*]]: tensor<3x8x4xf32>, %[[B:.*]]: tensor<3x4x16xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<8x16xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.transfer_read %[[A]]{{.*}}: tensor<3x8x4xf32>, vector<3x8x4xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.transfer_read %[[B]]{{.*}}: tensor<3x4x16xf32>, vector<3x4x16xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.transfer_read %[[C]]{{.*}}: tensor<8x16xf32>, vector<8x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<8x16xf32>, tensor<8x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @contract(%A: tensor<4x8x2xf32>, %B: tensor<8x16x2xf32>,
    %C: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %0 = linalg.contract
    indexing_maps = [affine_map<(m, n, k, kk) -> (m, k, kk)>,
                     affine_map<(m, n, k, kk) -> (k, n, kk)>,
                     affine_map<(m, n, k, kk) -> (m, n)>]
    ins(%A, %B : tensor<4x8x2xf32>, tensor<8x16x2xf32>)
    outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
// CHECK: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-LABEL: func.func @contract(
// CHECK-SAME:    %[[A:.*]]: tensor<4x8x2xf32>, %[[B:.*]]: tensor<8x16x2xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<4x16xf32>)
//      CHECK: %[[LOAD_A:.*]] = vector.transfer_read %[[A]]{{.*}}: tensor<4x8x2xf32>, vector<4x8x2xf32>
//      CHECK: %[[LOAD_B:.*]] = vector.transfer_read %[[B]]{{.*}}: tensor<8x16x2xf32>, vector<8x16x2xf32>
//      CHECK: %[[LOAD_C:.*]] = vector.transfer_read %[[C]]{{.*}}: tensor<4x16xf32>, vector<4x16xf32>
//      CHECK: %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
// CHECK-SAME:   kind = #vector.kind<add>
// CHECK-SAME:   %[[LOAD_A]], %[[LOAD_B]], %[[LOAD_C]]
//      CHECK: vector.transfer_write %[[CONTRACT]], %[[C]]{{.*}}: vector<4x16xf32>, tensor<4x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}

// -----

/// Generic can represent contractions but it does not implement contraction interface.
/// Thus, direct lowering to vector.contract is not supported.
/// Vectorization still works and applies generic rewrite logic.

func.func @negative_generic(%A: tensor<8x4xf32>, %B: tensor<4x16xf32>,
    %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<8x4xf32>, tensor<4x16xf32>)
    outs(%C : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @negative_generic(
// CHECK-NOT: vector.contract
// CHECK: vector.multi_reduction

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 {create_named_contraction} : !transform.any_op
    transform.yield
  }
}
