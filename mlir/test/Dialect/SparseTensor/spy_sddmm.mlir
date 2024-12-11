// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

//
// A SDDMM implementation with "spy" function and
// in-place update of the sampling sparse matrix.
//

#SM = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // S
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "S(i,j) += spy[S(i,j)] x SUM_k A(i,k) B(k,j)"
}

// CHECK-LABEL: func.func @sparse_sampled_dd(
// CHECK-SAME:    %[[VAL_0:.*0]]: tensor<8x8xf64>,
// CHECK-SAME:    %[[VAL_1:.*1]]: tensor<8x8xf64>,
// CHECK-SAME:    %[[VAL_2:.*2]]: tensor<8x8xf64, #sparse{{[0-9]*}}>) -> tensor<8x8xf64, #sparse{{[0-9]*}}> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<8x8xf64>
// CHECK-DAG:     %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<8x8xf64>
// CHECK-DAG:     %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_2]] {level = 1 : index} : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_2]] {level = 1 : index} : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_2]] : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK:         scf.for %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:           scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_5]] : index
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_5]] {
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<8x8xf64>
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_12]], %[[VAL_17]]] : memref<8x8xf64>
// CHECK:               %[[VAL_21:.*]] = arith.mulf %[[VAL_19]], %[[VAL_20]] : f64
// CHECK:               %[[VAL_22:.*]] = arith.addf %[[VAL_18]], %[[VAL_21]] : f64
// CHECK:               memref.store %[[VAL_22]], %[[VAL_10]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:         } {"Emitted from" = "linalg.generic"}
// CHECK:         %[[VAL_23:.*]] = sparse_tensor.load %[[VAL_2]] : tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_23]] : tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK:       }
func.func @sparse_sampled_dd(%argA: tensor<8x8xf64>,
                             %argB: tensor<8x8xf64>,
                             %argS: tensor<8x8xf64, #SM>) -> tensor<8x8xf64, #SM> {
  %f0 = arith.constant 0.0 : f64
  %result = linalg.generic #trait_sampled_dense_dense
    ins(%argA, %argB: tensor<8x8xf64>, tensor<8x8xf64>) outs(%argS: tensor<8x8xf64, #SM>) {
      ^bb(%a: f64, %b: f64, %s: f64):
         %u = sparse_tensor.unary %s : f64 to f64
             present={
                ^bb0(%p: f64):
                  %mul = arith.mulf %a, %b : f64
                  sparse_tensor.yield %mul : f64
             }
             absent={}
         %r = sparse_tensor.reduce %s, %u, %f0 : f64 {
              ^bb0(%p: f64, %q: f64):
                %add = arith.addf %p, %q : f64
                sparse_tensor.yield %add : f64
            }
         linalg.yield %r : f64
  } -> tensor<8x8xf64, #SM>
  return %result : tensor<8x8xf64, #SM>
}
