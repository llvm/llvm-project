// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#BCSR = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : batch, d1 : dense, d2 : compressed)}>

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x4x2xf32, #sparse{{[0-9]*}}>) -> tensor<8x4x2xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.empty() : tensor<8x4x2xf32>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 2 : index} : tensor<8x4x2xf32, #sparse{{[0-9]*}}> to memref<8x?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 2 : index} : tensor<8x4x2xf32, #sparse{{[0-9]*}}> to memref<8x?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x4x2xf32, #sparse{{[0-9]*}}> to memref<8x?xf32>
// CHECK-DAG:       %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_6]] : memref<8x4x2xf32>
// CHECK-DAG:       linalg.fill ins(%[[VAL_3]] : f32) outs(%[[VAL_10]] : memref<8x4x2xf32>)
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_1]] {
// CHECK:             scf.for %[[VAL_12:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_1]] {
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<8x?xindex>
// CHECK:               %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : index
// CHECK:               %[[VAL_15:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_11]], %[[VAL_14]]] : memref<8x?xindex>
// CHECK:               scf.for %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_1]] {
// CHECK:                 %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_11]], %[[VAL_16]]] : memref<8x?xindex>
// CHECK:                 %[[VAL_18:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_11]], %[[VAL_16]]] : memref<8x?xf32>
// CHECK:                 %[[VAL_19:.*]] = arith.negf %[[VAL_18]] : f32
// CHECK:                 memref.store %[[VAL_19]], %[[VAL_10]]{{\[}}%[[VAL_11]], %[[VAL_12]], %[[VAL_17]]] : memref<8x4x2xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<8x4x2xf32>
// CHECK:           return %[[VAL_20]] : tensor<8x4x2xf32>
// CHECK:         }
func.func @main(%arg0: tensor<8x4x2xf32, #BCSR>) -> tensor<8x4x2xf32> {
  %0 = tensor.empty() : tensor<8x4x2xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  }
  ins(%arg0 : tensor<8x4x2xf32, #BCSR>)
  outs(%0 : tensor<8x4x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.negf %in : f32
    linalg.yield %2 : f32
  } -> tensor<8x4x2xf32>
  return %1 : tensor<8x4x2xf32>
}
