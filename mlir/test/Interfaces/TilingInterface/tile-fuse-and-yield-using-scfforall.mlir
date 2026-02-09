// RUN: mlir-opt -transform-interpreter -cse -canonicalize -split-input-file %s | FileCheck %s

func.func @gemm_gemm_fusion_yield_both(%lhs0 : tensor<?x?xf32>, %rhs0 : tensor<?x?xf32>, %rhs1 : tensor<?x?xf32>,
    %init0 : tensor<?x?xf32>, %init1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %lhs0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %rhs0, %c1 : tensor<?x?xf32>
  %fill0 = linalg.fill ins(%cst : f32) outs(%init0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm0 = linalg.matmul
      ins(%lhs0, %rhs0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %d2 = tensor.dim %rhs1, %c1 : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %gemm1 = linalg.matmul
      ins(%gemm0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm0, %gemm1 : tensor<?x?xf32>, tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %mm1, %mm2 = transform.split_handle %matmuls
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_and_yield %mm2 [10] use_forall true
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @gemm_gemm_fusion_yield_both(
// CHECK-SAME:     %[[LHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[INIT0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[RESULT:.+]]:2 = scf.forall (%[[IV:[a-zA-Z0-9]+]]) =
// CHECK-SAME:       shared_outs(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT1]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT0]])
//  CHECK-DAG:     %[[LHS0_TILE:.+]] = tensor.extract_slice %[[LHS0]][%[[IV]], 0]
//  CHECK-DAG:     %[[RHS0_TILE:.+]] = tensor.extract_slice %[[RHS0]][0, 0]
//  CHECK-DAG:     %[[INIT0_TILE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV]], 0]
//      CHECK:     %[[FILL0_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT0_TILE]] :
//      CHECK:     %[[GEMM0_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS0_TILE]], %[[RHS0_TILE]] :
// CHECK-SAME:         outs(%[[FILL0_TILE]] :
//  CHECK-DAG:     %[[RHS1_TILE:.+]] = tensor.extract_slice %[[RHS1]][0, 0]
//  CHECK-DAG:     %[[INIT1_TILE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV]], 0]
//      CHECK:     %[[FILL1_TILE:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT1_TILE]] :
//      CHECK:     %[[GEMM1_TILE:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[GEMM0_TILE]], %[[RHS1_TILE]] :
// CHECK-SAME:         outs(%[[FILL1_TILE]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[GEMM1_TILE]] into %[[ITERARG0]][%[[IV]], 0]
//      CHECK:       tensor.parallel_insert_slice %[[GEMM0_TILE]] into %[[ITERARG1]][%[[IV]], 0]
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0

// -----

func.func @fuse_pack_consumer_into_multi_output_generic(
    %input: tensor<32x1024xf32>) -> (tensor<32x1024xf32>, tensor<2x512x16x2xi8>) {
  %c0_i8 = arith.constant 0 : i8
  %output_f32 = tensor.empty() : tensor<32x1024xf32>
  %output_i8 = tensor.empty() : tensor<32x1024xi8>
  %pack_dest = tensor.empty() : tensor<2x512x16x2xi8>

  %gen:2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<32x1024xf32>)
    outs(%output_f32, %output_i8 : tensor<32x1024xf32>, tensor<32x1024xi8>) {
  ^bb0(%in: f32, %out_f: f32, %out_i: i8):
    %q = arith.fptoui %in : f32 to i8
    linalg.yield %in, %q : f32, i8
  } -> (tensor<32x1024xf32>, tensor<32x1024xi8>)

  %pack = linalg.pack %gen#1
    padding_value(%c0_i8 : i8)
    inner_dims_pos = [0, 1]
    inner_tiles = [16, 2]
    into %pack_dest : tensor<32x1024xi8> -> tensor<2x512x16x2xi8>

  return %gen#0, %pack : tensor<32x1024xf32>, tensor<2x512x16x2xi8>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_and_yield %pack [1] use_forall true
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 16)>
//      CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * -16 + 32, 16)>
//      CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func.func @fuse_pack_consumer_into_multi_output_generic(
// CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<32x1024xf32>)
//  CHECK-DAG:   %[[C0_I8:.+]] = arith.constant 0 : i8
//  CHECK-DAG:   %[[OUTPUT_F32:.+]] = tensor.empty() : tensor<32x1024xf32>
//  CHECK-DAG:   %[[OUTPUT_I8:.+]] = tensor.empty() : tensor<32x1024xi8>
//  CHECK-DAG:   %[[PACK_DEST:.+]] = tensor.empty() : tensor<2x512x16x2xi8>
//      CHECK:   %[[RESULT:.+]]:2 = scf.forall (%[[IV:.+]]) in (2)
// CHECK-SAME:       shared_outs(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[PACK_DEST]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[OUTPUT_F32]])
//      CHECK:     %[[OFFSET:.+]] = affine.apply #[[$MAP0]](%[[IV]])
//      CHECK:     %[[SIZE:.+]] = affine.min #[[$MAP1]](%[[IV]])
//  CHECK-DAG:     %[[INPUT_TILE:.+]] = tensor.extract_slice %[[INPUT]][%[[OFFSET]], 0] [%[[SIZE]], 1024]
//  CHECK-DAG:     %[[F32_TILE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[OFFSET]], 0] [%[[SIZE]], 1024]
//  CHECK-DAG:     %[[I8_TILE:.+]] = tensor.extract_slice %[[OUTPUT_I8]][%[[OFFSET]], 0] [%[[SIZE]], 1024]
//      CHECK:     %[[GENERIC_TILE:.+]]:2 = linalg.generic
// CHECK-SAME:         ins(%[[INPUT_TILE]] :
// CHECK-SAME:         outs(%[[F32_TILE]], %[[I8_TILE]] :
//  CHECK-DAG:     %[[PACK_DEST_TILE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV]], 0, 0, 0] [1, 512, 16, 2]
//      CHECK:     %[[PACK_TILE:.+]] = linalg.pack %[[GENERIC_TILE]]#1
// CHECK-SAME:         padding_value(%[[C0_I8]] : i8)
// CHECK-SAME:         inner_dims_pos = [0, 1] inner_tiles = [16, 2]
// CHECK-SAME:         into %[[PACK_DEST_TILE]]
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[PACK_TILE]] into %[[ITERARG0]][%[[IV]], 0, 0, 0] [1, 512, 16, 2]
//      CHECK:       tensor.parallel_insert_slice %[[GENERIC_TILE]]#0 into %[[ITERARG1]][%[[OFFSET]], 0] [%[[SIZE]], 1024]
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0
