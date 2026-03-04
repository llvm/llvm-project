// RUN: mlir-opt -transform-preload-library='transform-library-paths=%p/transpose-matmul-a.mlir' -transform-interpreter -split-input-file %s | FileCheck %s --check-prefixes=CHECK,TRANSPOSE-A
// RUN: mlir-opt -transform-preload-library='transform-library-paths=%p/transpose-matmul-b.mlir' -transform-interpreter -split-input-file %s | FileCheck %s --check-prefixes=CHECK,TRANSPOSE-B

// TRANSPOSE-A-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// TRANSPOSE-A-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// TRANSPOSE-A-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// TRANSPOSE-A-DAG: #[[$BMA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// TRANSPOSE-A-DAG: #[[$BMB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// TRANSPOSE-A-DAG: #[[$BMC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// TRANSPOSE-B-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// TRANSPOSE-B-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// TRANSPOSE-B-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// TRANSPOSE-B-DAG: #[[$BMA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// TRANSPOSE-B-DAG: #[[$BMB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// TRANSPOSE-B-DAG: #[[$BMC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL:   func.func @matmul_static(
// CHECK-SAME:                             %[[A:.*]]: tensor<16x8xf32>,
// CHECK-SAME:                             %[[B:.*]]: tensor<8x16xf32>) -> tensor<16x16xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C_INIT:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// TRANSPOSE-A:     %[[A_TRANSP_INIT:.*]] = tensor.empty() : tensor<8x16xf32>
// TRANSPOSE-A:     %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<16x8xf32>) outs(%[[A_TRANSP_INIT]] : tensor<8x16xf32>) permutation = [1, 0]
// TRANSPOSE-A:     %[[C:.*]] = linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A_TRANSP]], %[[B]] : tensor<8x16xf32>, tensor<8x16xf32>) outs(%[[C_ZERO]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// TRANSPOSE-B:     %[[B_TRANSP_INIT:.*]] = tensor.empty() : tensor<16x8xf32>
// TRANSPOSE-B:     %[[B_TRANSP:.*]] = linalg.transpose ins(%[[B]] : tensor<8x16xf32>) outs(%[[B_TRANSP_INIT]] : tensor<16x8xf32>) permutation = [1, 0]
// TRANSPOSE-B:     %[[C:.*]] = linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A]], %[[B_TRANSP]] : tensor<16x8xf32>, tensor<16x8xf32>) outs(%[[C_ZERO]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           return %[[C]] : tensor<16x16xf32>
// CHECK:         }
func.func @matmul_static(%A: tensor<16x8xf32>, %B: tensor<8x16xf32>) -> (tensor<16x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<16x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<16x8xf32>, tensor<8x16xf32>) outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

//-----

// CHECK-LABEL:   func.func @matmul_dynamic(
// CHECK-SAME:                              %[[A:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                              %[[B:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[A_DIM0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[B_DIM1:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C_INIT:.*]] = tensor.empty(%[[A_DIM0]], %[[B_DIM1]]) : tensor<?x?xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// TRANSPOSE-A:     %[[A_DIM1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
// TRANSPOSE-A:     %[[A_TRANSP_INIT:.*]] = tensor.empty(%[[A_DIM1]], %[[A_DIM0]]) : tensor<?x?xf32>
// TRANSPOSE-A:     %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<?x?xf32>) outs(%[[A_TRANSP_INIT]] : tensor<?x?xf32>) permutation = [1, 0]
// TRANSPOSE-A:     %[[C:.*]] = linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A_TRANSP]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[C_ZERO]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// TRANSPOSE-B:     %[[B_DIM0:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?xf32>
// TRANSPOSE-B:     %[[B_TRANSP_INIT:.*]] = tensor.empty(%[[B_DIM1]], %[[B_DIM0]]) : tensor<?x?xf32>
// TRANSPOSE-B:     %[[B_TRANSP:.*]] = linalg.transpose ins(%[[B]] : tensor<?x?xf32>) outs(%[[B_TRANSP_INIT]] : tensor<?x?xf32>) permutation = [1, 0]
// TRANSPOSE-B:     %[[C:.*]] = linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A]], %[[B_TRANSP]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[C_ZERO]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           return %[[C]] : tensor<?x?xf32>
// CHECK:         }
func.func @matmul_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %B, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

//-----

// CHECK-LABEL:   func.func @matmul_mixed(
// CHECK-SAME:                            %[[A:.*]]: tensor<?x8xf32>,
// CHECK-SAME:                            %[[B:.*]]: tensor<8x16xf32>) -> tensor<?x16xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_DIM0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x8xf32>
// CHECK:           %[[C_INIT:.*]] = tensor.empty(%[[A_DIM0]]) : tensor<?x16xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<?x16xf32>) -> tensor<?x16xf32>
// TRANSPOSE-A:     %[[A_TRANSP_INIT:.*]] = tensor.empty(%[[A_DIM0]]) : tensor<8x?xf32>
// TRANSPOSE-A:     %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<?x8xf32>) outs(%[[A_TRANSP_INIT]] : tensor<8x?xf32>) permutation = [1, 0]
// TRANSPOSE-A:     %[[B0:.*]] = linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A_TRANSP]], %[[B]] : tensor<8x?xf32>, tensor<8x16xf32>) outs(%[[C_ZERO]] : tensor<?x16xf32>) -> tensor<?x16xf32>
// TRANSPOSE-B:     %[[B_TRANSP_INIT:.*]] = tensor.empty() : tensor<16x8xf32>
// TRANSPOSE-B:     %[[B_TRANSP:.*]] = linalg.transpose ins(%[[B]] : tensor<8x16xf32>) outs(%[[B_TRANSP_INIT]] : tensor<16x8xf32>) permutation = [1, 0]
// TRANSPOSE-B:     %[[B0:.*]] = linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A]], %[[B_TRANSP]] : tensor<?x8xf32>, tensor<16x8xf32>) outs(%[[C_ZERO]] : tensor<?x16xf32>) -> tensor<?x16xf32>
// CHECK:           return %[[B0]] : tensor<?x16xf32>
// CHECK:         }
func.func @matmul_mixed(%A: tensor<?x8xf32>, %B: tensor<8x16xf32>) -> (tensor<?x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x8xf32>
  %init = tensor.empty(%d0) : tensor<?x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<?x8xf32>, tensor<8x16xf32>) outs(%C : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

//-----

// CHECK-LABEL:   func.func @batch_matmul_static(
// CHECK-SAME:                                   %[[A:.*]]: tensor<2x16x8xf32>,
// CHECK-SAME:                                   %[[B:.*]]: tensor<2x8x16xf32>) -> tensor<2x16x16xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C_INIT:.*]] = tensor.empty() : tensor<2x16x16xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// TRANSPOSE-A:     %[[A_TRANSP_INIT:.*]] = tensor.empty() : tensor<2x8x16xf32>
// TRANSPOSE-A:     %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<2x16x8xf32>) outs(%[[A_TRANSP_INIT]] : tensor<2x8x16xf32>) permutation = [0, 2, 1]
// TRANSPOSE-A:     %[[C:.*]] = linalg.batch_matmul indexing_maps = [#[[$BMA]], #[[$BMB]], #[[$BMC]]] ins(%[[A_TRANSP]], %[[B]] : tensor<2x8x16xf32>, tensor<2x8x16xf32>) outs(%[[C_ZERO]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// TRANSPOSE-B:     %[[B_TRANSP_INIT:.*]] = tensor.empty() : tensor<2x16x8xf32>
// TRANSPOSE-B:     %[[B_TRANSP:.*]] = linalg.transpose ins(%[[B]] : tensor<2x8x16xf32>) outs(%[[B_TRANSP_INIT]] : tensor<2x16x8xf32>) permutation = [0, 2, 1]
// TRANSPOSE-B:     %[[C:.*]] = linalg.batch_matmul indexing_maps = [#[[$BMA]], #[[$BMB]], #[[$BMC]]] ins(%[[A]], %[[B_TRANSP]] : tensor<2x16x8xf32>, tensor<2x16x8xf32>) outs(%[[C_ZERO]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK:           return %[[C]] : tensor<2x16x16xf32>
// CHECK:         }
func.func @batch_matmul_static(%A: tensor<2x16x8xf32>, %B: tensor<2x8x16xf32>) -> (tensor<2x16x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2x16x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %0 = linalg.batch_matmul ins(%A, %B : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%C : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

//-----

// CHECK-LABEL:   func.func @batch_matmul_dynamic(
// CHECK-SAME:                                    %[[A:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                                    %[[B:.*]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[A_DIM0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:           %[[A_DIM1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:           %[[B_DIM2:.*]] = tensor.dim %[[B]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:           %[[C_INIT:.*]] = tensor.empty(%[[A_DIM0]], %[[A_DIM1]], %[[B_DIM2]]) : tensor<?x?x?xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// TRANSPOSE-A:     %[[A_DIM2:.*]] = tensor.dim %[[A]], %[[C2]] : tensor<?x?x?xf32>
// TRANSPOSE-A:     %[[A_TRANSP_INIT:.*]] = tensor.empty(%[[A_DIM0]], %[[A_DIM2]], %[[A_DIM1]]) : tensor<?x?x?xf32>
// TRANSPOSE-A:     %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<?x?x?xf32>) outs(%[[A_TRANSP_INIT]] : tensor<?x?x?xf32>) permutation = [0, 2, 1]
// TRANSPOSE-A:     %[[C:.*]] = linalg.batch_matmul indexing_maps = [#[[$BMA]], #[[$BMB]], #[[$BMC]]] ins(%[[A_TRANSP]], %[[B]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[C_ZERO]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// TRANSPOSE-B:     %[[B_DIM0:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?x?xf32>
// TRANSPOSE-B:     %[[B_DIM1:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?x?xf32>
// TRANSPOSE-B:     %[[B_TRANSP_INIT:.*]] = tensor.empty(%[[B_DIM0]], %[[B_DIM2]], %[[B_DIM1]]) : tensor<?x?x?xf32>
// TRANSPOSE-B:     %[[B_TRANSP:.*]] = linalg.transpose ins(%[[B]] : tensor<?x?x?xf32>) outs(%[[B_TRANSP_INIT]] : tensor<?x?x?xf32>) permutation = [0, 2, 1]
// TRANSPOSE-B:     %[[C:.*]] = linalg.batch_matmul indexing_maps = [#[[$BMA]], #[[$BMB]], #[[$BMC]]] ins(%[[A]], %[[B_TRANSP]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[C_ZERO]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           return %[[C]] : tensor<?x?x?xf32>
// CHECK:         }
func.func @batch_matmul_dynamic(%A: tensor<?x?x?xf32>, %B: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %A, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %B, %c2 : tensor<?x?x?xf32>
  %init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.batch_matmul ins(%A, %B : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%C : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

//-----

// CHECK-LABEL:   func.func @batch_matmul_mixed(
// CHECK-SAME:                                  %[[A:.*]]: tensor<2x?x8xf32>,
// CHECK-SAME:                                  %[[B:.*]]: tensor<2x8x16xf32>) -> tensor<2x?x16xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[A_DIM1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<2x?x8xf32>
// CHECK:           %[[C_INIT:.*]] = tensor.empty(%[[A_DIM1]]) : tensor<2x?x16xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<2x?x16xf32>) -> tensor<2x?x16xf32>
// TRANSPOSE-A:     %[[A_TRANSP_INIT:.*]] = tensor.empty(%[[A_DIM1]]) : tensor<2x8x?xf32>
// TRANSPOSE-A:     %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<2x?x8xf32>) outs(%[[A_TRANSP_INIT]] : tensor<2x8x?xf32>) permutation = [0, 2, 1]
// TRANSPOSE-A:     %[[B0:.*]] = linalg.batch_matmul indexing_maps = [#[[$BMA]], #[[$BMB]], #[[$BMC]]] ins(%[[A_TRANSP]], %[[B]] : tensor<2x8x?xf32>, tensor<2x8x16xf32>) outs(%[[C_ZERO]] : tensor<2x?x16xf32>) -> tensor<2x?x16xf32>
// TRANSPOSE-B:     %[[B_TRANSP_INIT:.*]] = tensor.empty() : tensor<2x16x8xf32>
// TRANSPOSE-B:     %[[B_TRANSP:.*]] = linalg.transpose ins(%[[B]] : tensor<2x8x16xf32>) outs(%[[B_TRANSP_INIT]] : tensor<2x16x8xf32>) permutation = [0, 2, 1]
// TRANSPOSE-B:     %[[B0:.*]] = linalg.batch_matmul indexing_maps = [#[[$BMA]], #[[$BMB]], #[[$BMC]]] ins(%[[A]], %[[B_TRANSP]] : tensor<2x?x8xf32>, tensor<2x16x8xf32>) outs(%[[C_ZERO]] : tensor<2x?x16xf32>) -> tensor<2x?x16xf32>
// CHECK:           return %[[B0]] : tensor<2x?x16xf32>
// CHECK:         }
func.func @batch_matmul_mixed(%A: tensor<2x?x8xf32>, %B: tensor<2x8x16xf32>) -> (tensor<2x?x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d1 = tensor.dim %A, %c1 : tensor<2x?x8xf32>
  %init = tensor.empty(%d1) : tensor<2x?x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<2x?x16xf32>) -> tensor<2x?x16xf32>
  %0 = linalg.batch_matmul ins(%A, %B : tensor<2x?x8xf32>, tensor<2x8x16xf32>) outs(%C : tensor<2x?x16xf32>) -> tensor<2x?x16xf32>
  return %0 : tensor<2x?x16xf32>
}
