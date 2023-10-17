// RUN: mlir-opt %s -sparsification | FileCheck %s

#SM = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)> // A
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "A(i,j) += 2.0 where A(i,j) != 0"
}

module {
  // Example of a semi-ring operation that only adds a
  // constant at stored values (something that would
  // typically not sparsify since it would densify the
  // implicit zeros in the normal case). The sparse
  // compiler should see that this is a "simply dynamic"
  // operation, and the values can be change "in-place".
  //
  // CHECK-LABEL: func.func @add_only_where_nonzero(
  // CHECK-SAME:    %[[VAL_0:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{{.*}}}>> {
  // CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : index
  // CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
  // CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f64
  // CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{{{.*}}}>> to memref<?xindex>
  // CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{{{.*}}}>> to memref<?xf64>
  // CHECK:         scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_3]] {
  // CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_7]]] : memref<?xindex>
  // CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_3]] : index
  // CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<?xindex>
  // CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_3]] {
  // CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xf64>
  // CHECK:             %[[VAL_13:.*]] = arith.addf %[[VAL_12]], %[[VAL_4]] : f64
  // CHECK:             memref.store %[[VAL_13]], %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xf64>
  // CHECK:           } {"Emitted from" = "linalg.generic"}
  // CHECK:         } {"Emitted from" = "linalg.generic"}
  // CHECK:         %[[VAL_14:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{{{.*}}}>>
  // CHECK:         return %[[VAL_14]] : tensor<8x8xf64, #sparse_tensor.encoding<{{{.*}}}>>
  // CHECK:       }
  func.func @add_only_where_nonzero(%argA: tensor<8x8xf64, #SM>) -> tensor<8x8xf64, #SM> {
    %c = arith.constant 2.0 : f64
    %result = linalg.generic #trait
      outs(%argA: tensor<8x8xf64, #SM>) {
        ^bb(%a: f64):
           %u = sparse_tensor.unary %a : f64 to f64
             present={
                ^bb0(%p: f64):
                  %add = arith.addf %p, %c : f64
                  sparse_tensor.yield %add : f64
             }
             absent={}
           linalg.yield %u : f64
    } -> tensor<8x8xf64, #SM>
    return %result : tensor<8x8xf64, #SM>
  }
}
