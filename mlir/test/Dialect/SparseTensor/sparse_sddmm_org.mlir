// RUN: mlir-opt %s --pre-sparsification-rewrite --sparse-reinterpret-map --sparsification --cse | FileCheck %s

#SM = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

#trait_matmul = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d1, d0)>,
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>
  ],
  iterator_types = ["reduction", "parallel", "parallel"]
}

#trait_scale = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL:   func.func @sparse_sampled_dd_unfused(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<8x8xf64>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<8x8xf64>) -> tensor<8x8xf64, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_8:.*]] = tensor.empty() : tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_1]] : tensor<8x8xf64> to memref<8x8xf64>
// CHECK-DAG:       %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : tensor<8x8xf64> to memref<8x8xf64>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = scf.for %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_5]] iter_args(%[[VAL_20:.*]] = %[[VAL_8]]) -> (tensor<8x8xf64, #sparse{{[0-9]*}}>) {
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:             %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]] = sparse_tensor.expand %[[VAL_8]] : tensor<8x8xf64, #sparse{{[0-9]*}}> to memref<?xf64>, memref<?xi1>, memref<?xindex>
// CHECK:             %[[VAL_26:.*]] = scf.for %[[VAL_27:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_28:.*]] = %[[VAL_25]]) -> (index) {
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_21]], %[[VAL_27]]] : memref<8x8xf64>
// CHECK:               %[[VAL_30:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:               %[[VAL_31:.*]] = arith.addi %[[VAL_19]], %[[VAL_5]] : index
// CHECK:               %[[VAL_32:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_31]]] : memref<?xindex>
// CHECK:               %[[VAL_33:.*]] = scf.for %[[VAL_34:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_5]] iter_args(%[[VAL_35:.*]] = %[[VAL_28]]) -> (index) {
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_34]]] : memref<?xindex>
// CHECK:                 %[[VAL_37:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_36]]] : memref<?xf64>
// CHECK:                 %[[VAL_38:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_27]], %[[VAL_36]]] : memref<8x8xf64>
// CHECK:                 %[[VAL_39:.*]] = arith.mulf %[[VAL_29]], %[[VAL_38]] : f64
// CHECK:                 %[[VAL_40:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_34]]] : memref<?xf64>
// CHECK:                 %[[VAL_41:.*]] = arith.mulf %[[VAL_39]], %[[VAL_40]] : f64
// CHECK:                 %[[VAL_42:.*]] = arith.addf %[[VAL_37]], %[[VAL_41]] : f64
// CHECK:                 %[[VAL_43:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_36]]] : memref<?xi1>
// CHECK:                 %[[VAL_44:.*]] = arith.cmpi eq, %[[VAL_43]], %[[VAL_6]] : i1
// CHECK:                 %[[VAL_45:.*]] = scf.if %[[VAL_44]] -> (index) {
// CHECK:                   memref.store %[[VAL_7]], %[[VAL_23]]{{\[}}%[[VAL_36]]] : memref<?xi1>
// CHECK:                   memref.store %[[VAL_36]], %[[VAL_24]]{{\[}}%[[VAL_35]]] : memref<?xindex>
// CHECK:                   %[[VAL_46:.*]] = arith.addi %[[VAL_35]], %[[VAL_5]] : index
// CHECK:                   scf.yield %[[VAL_46]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_35]] : index
// CHECK:                 }
// CHECK:                 memref.store %[[VAL_42]], %[[VAL_22]]{{\[}}%[[VAL_36]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_47:.*]] : index
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               scf.yield %[[VAL_48:.*]] : index
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             %[[VAL_49:.*]] = sparse_tensor.compress %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_50:.*]] into %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK:             scf.yield %[[VAL_49]] : tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_51:.*]] = sparse_tensor.load %[[VAL_52:.*]] hasInserts : tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_51]] : tensor<8x8xf64, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @sparse_sampled_dd_unfused(%args: tensor<8x8xf64, #SM>,
                                     %arga: tensor<8x8xf64>,
                                     %argb: tensor<8x8xf64>) -> tensor<8x8xf64, #SM> {
  // Perform dense-dense matrix matrix multiplication.
  %1 = arith.constant dense<0.0> : tensor<8x8xf64>
  %2 = linalg.generic #trait_matmul
    ins(%arga, %argb : tensor<8x8xf64>, tensor<8x8xf64>)
    outs(%1 : tensor<8x8xf64>) {
      ^bb0(%a: f64, %b: f64, %x: f64):
        %p = arith.mulf %a, %b : f64
        %q = arith.addf %x, %p : f64
        linalg.yield %q : f64
  } -> tensor<8x8xf64>
  // Sample the result with elements-wise multiplication with sparse matrix.
  %3 = tensor.empty() : tensor<8x8xf64, #SM>
  %4 = linalg.generic #trait_scale
    ins(%2, %args : tensor<8x8xf64>, tensor<8x8xf64, #SM>)
    outs(%3 : tensor<8x8xf64, #SM>) {
      ^bb0(%t: f64, %s: f64, %x: f64):
        %r = arith.mulf %t, %s : f64
        linalg.yield %r : f64
  } -> tensor<8x8xf64, #SM>
  return %4 : tensor<8x8xf64, #SM>
}
