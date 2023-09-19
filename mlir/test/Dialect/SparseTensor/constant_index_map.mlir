// Reported by https://github.com/llvm/llvm-project/issues/61530

// RUN: mlir-opt %s -sparsification | FileCheck %s

#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (d0)>

#SpVec = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:      %[[VAL_0:.*0]]: tensor<1x77xi1>,
// CHECK-SAME:      %[[VAL_1:.*1]]: tensor<1x77xi1>) -> tensor<77xi1, #{{.*}}> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 77 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = tensor.empty() : tensor<77xi1, #{{.*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<1x77xi1>
// CHECK-DAG:       %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<1x77xi1>
// CHECK:           %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_3]] to %[[VAL_2]] step %[[VAL_4]] iter_args(%[[VAL_10:.*]] = %[[VAL_5]]) -> (tensor<77xi1, #{{.*}}>) {
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_9]]] : memref<1x77xi1>
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_9]]] : memref<1x77xi1>
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = sparse_tensor.insert %[[VAL_13]] into %[[VAL_10]]{{\[}}%[[VAL_9]]] : tensor<77xi1, #{{.*}}>
// CHECK:             scf.yield %[[VAL_14]] : tensor<77xi1, #{{.*}}>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.load %[[VAL_16:.*]] hasInserts : tensor<77xi1, #{{.*}}>
// CHECK:           return %[[VAL_15]] : tensor<77xi1, #{{.*}}>
// CHECK:         }
func.func @main(%arg0: tensor<1x77xi1>, %arg1: tensor<1x77xi1>) -> tensor<77xi1, #SpVec> {
  %0 = tensor.empty() : tensor<77xi1, #SpVec>
  %1 = linalg.generic {
    indexing_maps = [#map1, #map1, #map2],
    iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<1x77xi1>, tensor<1x77xi1>)
    outs(%0 : tensor<77xi1, #SpVec>) {
  ^bb0(%in: i1, %in_0: i1, %out: i1):
    %2 = arith.addi %in, %in_0 : i1
    linalg.yield %2 : i1
  } -> tensor<77xi1, #SpVec>
  return %1 : tensor<77xi1, #SpVec>
}
