// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//  CHECK-DAG:  %[[ub:.*]] = ub.poison : vector<8xf32>
//  CHECK-DAG:  %[[ub_0:.*]] = ub.poison : vector<12xf32>
//  CHECK-DAG:  %[[ub_1:.*]] = ub.poison : vector<2x3xf32>
//      CHECK:  %[[a0:.*]] = vector.extract %[[A]][0] : vector<4xf32> from vector<2x4xf32>
//      CHECK:  %[[a1:.*]] = vector.insert_strided_slice %[[a0]], %[[ub]] {offsets = [0], strides = [1]} : vector<4xf32> into vector<8xf32>
//      CHECK:  %[[a2:.*]] = vector.extract %[[A]][1] : vector<4xf32> from vector<2x4xf32>
//      CHECK:  %[[a3:.*]] = vector.insert_strided_slice %[[a2]], %[[a1]] {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
//      CHECK:  %[[b0:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<4x3xf32>
//      CHECK:  %[[b1:.*]] = vector.insert_strided_slice %[[b0]], %[[ub_0]] {offsets = [0], strides = [1]} : vector<3xf32> into vector<12xf32>
//      CHECK:  %[[b2:.*]] = vector.extract %[[B]][1] : vector<3xf32> from vector<4x3xf32>
//      CHECK:  %[[b3:.*]] = vector.insert_strided_slice %[[b2]], %[[b1]] {offsets = [3], strides = [1]} : vector<3xf32> into vector<12xf32>
//      CHECK:  %[[b4:.*]] = vector.extract %[[B]][2] : vector<3xf32> from vector<4x3xf32>
//      CHECK:  %[[b5:.*]] = vector.insert_strided_slice %[[b4]], %[[b3]] {offsets = [6], strides = [1]} : vector<3xf32> into vector<12xf32>
//      CHECK:  %[[b6:.*]] = vector.extract %[[B]][3] : vector<3xf32> from vector<4x3xf32>
//      CHECK:  %[[b7:.*]] = vector.insert_strided_slice %[[b6]], %[[b5]] {offsets = [9], strides = [1]} : vector<3xf32> into vector<12xf32>
//      CHECK:  %[[mm1:.*]] = vector.matrix_multiply %[[a3]], %[[b7]] {lhs_columns = 4 : i32, lhs_rows = 2 : i32, rhs_columns = 3 : i32} : (vector<8xf32>, vector<12xf32>) -> vector<6xf32>
//      CHECK:  %[[mm2:.*]] = vector.extract_strided_slice %[[mm1]] {offsets = [0], sizes = [3], strides = [1]} : vector<6xf32> to vector<3xf32>
//      CHECK:  %[[mm3:.*]] = vector.insert %[[mm2]], %[[ub_1]] [0] : vector<3xf32> into vector<2x3xf32>
//      CHECK:  %[[mm4:.*]] = vector.extract_strided_slice %[[mm1]] {offsets = [3], sizes = [3], strides = [1]} : vector<6xf32> to vector<3xf32>
//      CHECK:  %[[mm5:.*]] = vector.insert %[[mm4]], %[[mm3]] [1] : vector<3xf32> into vector<2x3xf32>
//      CHECK:  %[[mm6:.*]] = arith.addf %[[C]], %[[mm5]] : vector<2x3xf32>
func.func @matmul(%arg0: vector<2x4xf32>,
                  %arg1: vector<4x3xf32>,
                  %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_scalable
// CHECK-NOT: vector.matrix_multiply
func.func @matmul_scalable(%arg0: vector<2x4xf32>,
                           %arg1: vector<4x[3]xf32>,
                           %arg2: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x[3]xf32> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "matmulintrinsics"
    } : !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
