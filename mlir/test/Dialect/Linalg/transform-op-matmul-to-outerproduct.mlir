// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

func.func @matmul_to_outerproduct(%A: memref<3x4xf32>, %B: memref<4x3xf32>, %C: memref<3x3xf32>) {
  linalg.matmul ins(%A, %B: memref<3x4xf32>, memref<4x3xf32>)
            outs(%C: memref<3x3xf32>)
  return
}

// CHECK-LABEL:   func.func @matmul_to_outerproduct(
// CHECK-SAME:      %[[A:.*]]: memref<3x4xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<4x3xf32>,
// CHECK-SAME:      %[[C:.*]]: memref<3x3xf32>) {
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]
// CHECK:           %[[VEC_B:.*]] = vector.transfer_read %[[B]]
// CHECK:           %[[VEC_C:.*]] = vector.transfer_read %[[C]]
// CHECK:           %[[VEC_A_T:.*]] = vector.transpose %[[VEC_A]], [1, 0] : vector<3x4xf32> to vector<4x3xf32>
// CHECK:           %[[A0:.*]] = vector.extract %[[VEC_A_T]][0] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[B0:.*]] = vector.extract %[[VEC_B]][0] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[OP_0:.*]] = vector.outerproduct %[[A0]], %[[B0]], %[[VEC_C]]
// CHECK:           %[[A1:.*]] = vector.extract %[[VEC_A_T]][1] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[B1:.*]] = vector.extract %[[VEC_B]][1] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[OP_1:.*]] = vector.outerproduct %[[A1]], %[[B1]], %[[OP_0]]
// CHECK:           %[[A_2:.*]] = vector.extract %[[VEC_A_T]][2] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[B_2:.*]] = vector.extract %[[VEC_B]][2] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[OP_2:.*]] = vector.outerproduct %[[A_2]], %[[B_2]], %[[OP_1]]
// CHECK:           %[[A_3:.*]] = vector.extract %[[VEC_A_T]][3] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[B_3:.*]] = vector.extract %[[VEC_B]][3] : vector<3xf32> from vector<4x3xf32>
// CHECK:           %[[RES:.*]] = vector.outerproduct %[[A_3]], %[[B_3]], %[[OP_2]]
// CHECK:           vector.transfer_write %[[RES]], %[[C]]{{.*}} : vector<3x3xf32>, memref<3x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op

    // Vectorize: linalg.matmul -> vector.multi_reduction
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %matmul : !transform.any_op

    // vector.multi_reduction --> vector.contract
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.reduction_to_contract
      // Reduce the rank of xfer ops. This transform vector.contract to be more
      // more matmul-like and to enable the lowering to outer product Ops.
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op

    // vector.contract --> vector.outerproduct
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.yield
  }
}
