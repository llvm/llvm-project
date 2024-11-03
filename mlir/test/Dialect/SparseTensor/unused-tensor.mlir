// RUN: mlir-opt %s -sparsification | FileCheck %s

//
// A contrived example where the sparse tensor B is only
// used in the linalg op to determine the number of iterations
// for the k-loop. This is included to make sure the sparse
// compiler still generates the correct loop nest for this case.
//

#SM = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>

#trait = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // S_out
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "C(i,j) = SUM_k A(i,j)"
}

// CHECK-LABEL:   func.func @b_ununsed(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x4xf64>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<8x4xf64, #sparse_tensor.encoding<{{.*}}>>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<2x4xf64>) -> tensor<2x4xf64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<2x4xf64>
// CHECK-DAG:       %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_2]] : memref<2x4xf64>
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_7]] {
// CHECK:             scf.for %[[VAL_11:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_7]] {
// CHECK:               scf.for %[[VAL_12:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_7]] {
// CHECK:                 %[[VAL_13:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_12]]] : memref<2x4xf64>
// CHECK:                 %[[VAL_14:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_10]], %[[VAL_12]]] : memref<2x4xf64>
// CHECK:                 %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f64
// CHECK:                 memref.store %[[VAL_15]], %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_12]]] : memref<2x4xf64>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_9]] : memref<2x4xf64>
// CHECK:           return %[[VAL_16]] : tensor<2x4xf64>
// CHECK:         }
func.func @b_ununsed(%argA: tensor<2x4xf64>,
                     %argB: tensor<8x4xf64, #SM>,
                     %argC: tensor<2x4xf64>) -> tensor<2x4xf64> {
  %result = linalg.generic #trait
    ins(%argA, %argB: tensor<2x4xf64>, tensor<8x4xf64, #SM>)
    outs(%argC: tensor<2x4xf64>) {
      ^bb(%a: f64, %b: f64, %c: f64):
         %0 = arith.addf %c, %a : f64
         linalg.yield %0 : f64
  } -> tensor<2x4xf64>
  return %result : tensor<2x4xf64>
}
