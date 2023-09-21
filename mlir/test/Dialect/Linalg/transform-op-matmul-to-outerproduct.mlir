// RUN: mlir-opt %s -test-transform-dialect-interpreter | FileCheck %s

func.func @outerproduct_matmul(%A: memref<3x3xf32>, %B: memref<3x3xf32>, %C: memref<3x3xf32>) {
  linalg.matmul ins(%A, %B: memref<3x3xf32>, memref<3x3xf32>)
            outs(%C: memref<3x3xf32>)
  return
}

// CHECK-LABEL:   func.func @outerproduct_matmul(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<3x3xf32>, %[[VAL_1:.*]]: memref<3x3xf32>, %[[VAL_2:.*]]: memref<3x3xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<3x3xf32>, vector<3x3xf32>
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<3x3xf32>, vector<3x3xf32>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<3x3xf32>, vector<3x3xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transpose %[[VAL_5]], [1, 0] : vector<3x3xf32> to vector<3x3xf32>
// CHECK:           %[[VAL_9:.*]] = vector.extract %[[VAL_8]][0] : vector<3x3xf32>
// CHECK:           %[[VAL_10:.*]] = vector.extract %[[VAL_6]][0] : vector<3x3xf32>
// CHECK:           %[[VAL_11:.*]] = vector.outerproduct %[[VAL_9]], %[[VAL_10]], %[[VAL_7]] {kind = #vector.kind<add>} : vector<3xf32>, vector<3xf32>
// CHECK:           %[[VAL_12:.*]] = vector.extract %[[VAL_8]][1] : vector<3x3xf32>
// CHECK:           %[[VAL_13:.*]] = vector.extract %[[VAL_6]][1] : vector<3x3xf32>
// CHECK:           %[[VAL_14:.*]] = vector.outerproduct %[[VAL_12]], %[[VAL_13]], %[[VAL_11]] {kind = #vector.kind<add>} : vector<3xf32>, vector<3xf32>
// CHECK:           %[[VAL_15:.*]] = vector.extract %[[VAL_8]][2] : vector<3x3xf32>
// CHECK:           %[[VAL_16:.*]] = vector.extract %[[VAL_6]][2] : vector<3x3xf32>
// CHECK:           %[[VAL_17:.*]] = vector.outerproduct %[[VAL_15]], %[[VAL_16]], %[[VAL_14]] {kind = #vector.kind<add>} : vector<3xf32>, vector<3xf32>
// CHECK:           vector.transfer_write %[[VAL_17]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<3x3xf32>, memref<3x3xf32>
// CHECK:           return
// CHECK:         }

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %2 {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  } : !transform.any_op
}
