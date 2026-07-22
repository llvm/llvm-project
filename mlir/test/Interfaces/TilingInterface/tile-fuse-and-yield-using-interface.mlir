// RUN: mlir-opt -transform-interpreter -cse -split-input-file %s | FileCheck %s

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
    %a, %b = transform.test.fuse_and_yield %mm2 [10]
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
//      CHECK:   %[[RESULT:.+]]:2 = scf.for %[[IV:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT1]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT0]])
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
//      CHECK:     %[[INSERT0:.+]] = tensor.insert_slice %[[GEMM1_TILE]] into %[[ITERARG0]][%[[IV]], 0]
//      CHECK:     %[[INSERT1:.+]] = tensor.insert_slice %[[GEMM0_TILE]] into %[[ITERARG1]][%[[IV]], 0]
//      CHECK:     scf.yield %[[INSERT0]], %[[INSERT1]]
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0

// -----

func.func @multiple_outputs_fusion_yield_all(%lhs0: tensor<32x32xf32>,
                       %rhs0: tensor<32x32xf32>, %init0: tensor<32x32xf32>, %init1: tensor<32x32xf32>, 
                       %rhs1: tensor<32x32xf32>, %init2: tensor<32x32xf32>) 
                       -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) {
  %out0, %out1 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (j, i)>],
    iterator_types = ["parallel", "parallel"]
  }
  ins(%lhs0, %rhs0: tensor<32x32xf32>, tensor<32x32xf32>)
  outs(%init0, %init1: tensor<32x32xf32>, tensor<32x32xf32>) {
  ^bb0(%0: f32, %1: f32, %2: f32, %3: f32):
    %4 = arith.mulf %0, %1 : f32
    %5 = arith.addf %0, %1 : f32
    linalg.yield %4, %5: f32, f32
  } -> (tensor<32x32xf32>, tensor<32x32xf32>)

  %out3 = linalg.add ins(%out0, %rhs1: tensor<32x32xf32>, tensor<32x32xf32>) outs(%init2: tensor<32x32xf32>) -> tensor<32x32xf32>

  return %out0, %out1, %out3 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %add = transform.structured.match ops{["linalg.add"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_and_yield %add [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @multiple_outputs_fusion_yield_all(
// CHECK-SAME:     %[[LHS0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[RHS0:[a-zA-Z0-9]+]]: tensor<32x32xf32>,
// CHECK-SAME:     %[[INIT0:[a-zA-Z0-9]+]]: tensor<32x32xf32>,
// CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<32x32xf32>,
// CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<32x32xf32>,
// CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<32x32xf32>)
//      CHECK:   %[[RESULT:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ITERARG0:[a-zA-Z0-9]+]] = %[[INIT2]], %[[ITERARG1:[a-zA-Z0-9]+]] = %[[INIT0]], %[[ITERARG2:[a-zA-Z0-9]+]] = %[[INIT1]])
//  CHECK-DAG:     %[[LHS0_TILE:.+]] = tensor.extract_slice %[[LHS0]][%[[IV]], 0]
//  CHECK-DAG:     %[[RHS0_TILE:.+]] = tensor.extract_slice %[[RHS0]][%[[IV]], 0]
//  CHECK-DAG:     %[[INIT0_TILE:.+]] = tensor.extract_slice %[[ITERARG1]][%[[IV]], 0]
//  CHECK-DAG:     %[[INIT1_TILE:.+]] = tensor.extract_slice %[[ITERARG2]][0, %[[IV]]]
//      CHECK:     %[[GENERIC_TILE:.+]]:2 = linalg.generic
// CHECK-SAME:         ins(%[[LHS0_TILE]], %[[RHS0_TILE]] :
// CHECK-SAME:         outs(%[[INIT0_TILE]], %[[INIT1_TILE]] :
//  CHECK-DAG:     %[[RHS1_TILE:.+]] = tensor.extract_slice %[[RHS1]][%[[IV]], 0]
//  CHECK-DAG:     %[[INIT2_TILE:.+]] = tensor.extract_slice %[[ITERARG0]][%[[IV]], 0]
//      CHECK:     %[[ADD_TILE:.+]] = linalg.add
// CHECK-SAME:         ins(%[[GENERIC_TILE]]#0, %[[RHS1_TILE]] :
// CHECK-SAME:         outs(%[[INIT2_TILE]] :
//      CHECK:     %[[INSERT0:.+]] = tensor.insert_slice %[[ADD_TILE]] into %[[ITERARG0]][%[[IV]], 0]
//      CHECK:     %[[INSERT1:.+]] = tensor.insert_slice %[[GENERIC_TILE]]#0 into %[[ITERARG1]][%[[IV]], 0]
//      CHECK:     %[[INSERT2:.+]] = tensor.insert_slice %[[GENERIC_TILE]]#1 into %[[ITERARG2]][0, %[[IV]]]
//      CHECK:     scf.yield %[[INSERT0]], %[[INSERT1]], %[[INSERT2]]
//      CHECK:   return %[[RESULT]]#1, %[[RESULT]]#2, %[[RESULT]]#0

// -----

// Verify that the default FIFO worklist reconstructs the earlier `linalg.fill`
// before the later `linalg.copy`. Their loop results are therefore passed to
// the downstream `linalg.mul` as result #1 and result #2, respectively.

// CHECK-LABEL: func.func @worklist_fifo_order(
// CHECK:         %[[RESULT:.+]]:3 = scf.for
// CHECK:           %[[FILL_TILE:.+]] = linalg.fill
// CHECK:           %[[COPY_TILE:.+]] = linalg.copy
// CHECK:           %[[CONSUMER_TILE:.+]] = linalg.add
// CHECK:           tensor.insert_slice %[[CONSUMER_TILE]]
// CHECK:           tensor.insert_slice %[[FILL_TILE]]
// CHECK:           tensor.insert_slice %[[COPY_TILE]]
// CHECK:           scf.yield
// CHECK:         %[[FINAL:.+]] = linalg.mul
// CHECK-SAME:        ins(%[[RESULT]]#1, %[[RESULT]]#2
// CHECK-SAME:        outs(%[[RESULT]]#0
// CHECK:         return %[[FINAL]]

func.func @worklist_fifo_order(
    %input: tensor<32x32xf32>, %out: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  %copy = linalg.copy ins(%input : tensor<32x32xf32>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  %add = linalg.add
      ins(%fill, %copy : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.mul
      ins(%fill, %copy : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%add : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0 : !transform.any_op {transform.readonly}) {
    %add = transform.structured.match ops{["linalg.add"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.test.fuse_and_yield %add [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Verify that preferring later producers reconstructs the `linalg.copy` before
// the `linalg.fill`. Their loop result numbers are reversed, but the downstream
// `linalg.mul` still receives the fill and copy values in its original order.

// CHECK-LABEL: func.func @worklist_prefer_later_producers(
// CHECK:         %[[RESULT:.+]]:3 = scf.for
// CHECK:           %[[FILL_TILE:.+]] = linalg.fill
// CHECK:           %[[COPY_TILE:.+]] = linalg.copy
// CHECK:           %[[CONSUMER_TILE:.+]] = linalg.add
// CHECK:           tensor.insert_slice %[[CONSUMER_TILE]]
// CHECK:           tensor.insert_slice %[[COPY_TILE]]
// CHECK:           tensor.insert_slice %[[FILL_TILE]]
// CHECK:           scf.yield
// CHECK:         %[[FINAL:.+]] = linalg.mul
// CHECK-SAME:        ins(%[[RESULT]]#2, %[[RESULT]]#1
// CHECK-SAME:        outs(%[[RESULT]]#0
// CHECK:         return %[[FINAL]]


func.func @worklist_prefer_later_producers(
    %input: tensor<32x32xf32>, %out: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  %copy = linalg.copy ins(%input : tensor<32x32xf32>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  %add = linalg.add
      ins(%fill, %copy : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.mul
      ins(%fill, %copy : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%add : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0 : !transform.any_op {transform.readonly}) {
    %add = transform.structured.match ops{["linalg.add"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.test.fuse_and_yield %add [16]
        worklist_prefer_later_producers true
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
