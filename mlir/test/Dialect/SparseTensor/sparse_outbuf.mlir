// RUN: mlir-opt %s -sparsification | FileCheck %s

#SV = #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A (in)
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"]
}

// CHECK-LABEL:   func.func @allout_inplace(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10xi32, #{{.*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<10xi32, #{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<10xi32, #{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10xi32, #{{.*}}> to memref<?xi32>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_1]] : memref<10xf32>
// CHECK:           linalg.fill ins(%[[VAL_3]] : f32) outs(%[[VAL_8]] : memref<10xf32>)
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_4]] {
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK:             %[[VAL_14:.*]] = arith.sitofp %[[VAL_13]] : i32 to f32
// CHECK:             memref.store %[[VAL_14]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<10xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<10xf32>
// CHECK:           return %[[VAL_15]] : tensor<10xf32>
// CHECK:         }
func.func @allout_inplace(%arga: tensor<10xi32, #SV>,
                          %argb: tensor<10xf32>) -> tensor<10xf32> {
  %0 = linalg.generic #trait
  ins(%arga: tensor<10xi32, #SV>)
  outs(%argb: tensor<10xf32>) {
    ^bb(%a: i32, %x : f32):
      %cst = arith.sitofp %a : i32 to f32
      linalg.yield %cst : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL:   func.func @allout_materialize(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10xi32, #{{.*}}>) -> tensor<10xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = bufferization.alloc_tensor() : tensor<10xf32>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<10xi32, #{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<10xi32, #{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10xi32, #{{.*}}> to memref<?xi32>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_4]] : memref<10xf32>
// CHECK:           linalg.fill ins(%[[VAL_2]] : f32) outs(%[[VAL_8]] : memref<10xf32>)
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_3]] {
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK:             %[[VAL_14:.*]] = arith.sitofp %[[VAL_13]] : i32 to f32
// CHECK:             memref.store %[[VAL_14]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<10xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<10xf32>
// CHECK:           return %[[VAL_15]] : tensor<10xf32>
// CHECK:         }
func.func @allout_materialize(%arga: tensor<10xi32, #SV>) -> tensor<10xf32> {
  %m = bufferization.alloc_tensor() : tensor<10xf32>
  %0 = linalg.generic #trait
  ins(%arga: tensor<10xi32, #SV>)
  outs(%m: tensor<10xf32>) {
    ^bb(%a: i32, %x : f32):
      %cst = arith.sitofp %a : i32 to f32
      linalg.yield %cst : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL:   func.func @update_inplace(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10xf32, #{{.*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<10xf32, #{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<10xf32, #{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10xf32, #{{.*}}> to memref<?xf32>
// CHECK:           %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<10xf32>
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] {
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf32>
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<10xf32>
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             memref.store %[[VAL_14]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<10xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<10xf32>
// CHECK:           return %[[VAL_15]] : tensor<10xf32>
// CHECK:         }
func.func @update_inplace(%arga: tensor<10xf32, #SV>,
                          %argb: tensor<10xf32>) -> tensor<10xf32> {
  %0 = linalg.generic #trait
  ins(%arga: tensor<10xf32, #SV>)
  outs(%argb: tensor<10xf32>) {
    ^bb(%a: f32, %x : f32):
      %up = arith.addf %a, %x : f32
      linalg.yield %up : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

