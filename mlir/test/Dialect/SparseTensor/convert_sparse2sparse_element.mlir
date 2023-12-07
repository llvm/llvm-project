// RUN: mlir-opt %s --sparse-tensor-codegen --canonicalize --cse | FileCheck %s

#SparseVector64 = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"],
  posWidth = 64,
  crdWidth = 64
}>

#SparseVector32 = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"],
  posWidth = 32,
  crdWidth = 32
}>


// CHECK-LABEL:   func.func @sparse_convert(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xi64>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xi64>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf32>,
// CHECK-SAME:      %[[VAL_3:.*3]]: !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_5]] : memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = memref.alloc(%[[VAL_6]]) : memref<?xi32>
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_4]] {
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_8]]] : memref<?xi64>
// CHECK:             %[[VAL_10:.*]] = arith.trunci %[[VAL_9]] : i64 to i32
// CHECK:             memref.store %[[VAL_10]], %[[VAL_7]]{{\[}}%[[VAL_8]]] : memref<?xi32>
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = memref.dim %[[VAL_1]], %[[VAL_5]] : memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi32>
// CHECK:           scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_11]] step %[[VAL_4]] {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_13]]] : memref<?xi64>
// CHECK:             %[[VAL_15:.*]] = arith.trunci %[[VAL_14]] : i64 to i32
// CHECK:             memref.store %[[VAL_15]], %[[VAL_12]]{{\[}}%[[VAL_13]]] : memref<?xi32>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = memref.dim %[[VAL_2]], %[[VAL_5]] : memref<?xf32>
// CHECK:           %[[VAL_17:.*]] = memref.alloc(%[[VAL_16]]) : memref<?xf32>
// CHECK:           memref.copy %[[VAL_2]], %[[VAL_17]] : memref<?xf32> to memref<?xf32>
// CHECK:           return %[[VAL_7]], %[[VAL_12]], %[[VAL_17]], %[[VAL_3]] : memref<?xi32>, memref<?xi32>, memref<?xf32>, !sparse_tensor.storage_specifier
// CHECK:         }
func.func @sparse_convert(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-LABEL:   func.func @sparse_convert_value(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xi32>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xi32>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf32>,
// CHECK-SAME:      %[[VAL_3:.*]]: !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_5]] : memref<?xi32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc(%[[VAL_6]]) : memref<?xi32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_7]] : memref<?xi32> to memref<?xi32>
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_1]], %[[VAL_5]] : memref<?xi32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc(%[[VAL_8]]) : memref<?xi32>
// CHECK:           memref.copy %[[VAL_1]], %[[VAL_9]] : memref<?xi32> to memref<?xi32>
// CHECK:           %[[VAL_10:.*]] = memref.dim %[[VAL_2]], %[[VAL_5]] : memref<?xf32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc(%[[VAL_10]]) : memref<?xf64>
// CHECK:           scf.for %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_10]] step %[[VAL_4]] {
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:             %[[VAL_14:.*]] = arith.extf %[[VAL_13]] : f32 to f64
// CHECK:             memref.store %[[VAL_14]], %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<?xf64>
// CHECK:           }
// CHECK:           return %[[VAL_7]], %[[VAL_9]], %[[VAL_11]], %[[VAL_3]] : memref<?xi32>, memref<?xi32>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:         }
func.func @sparse_convert_value(%arg0: tensor<?xf32, #SparseVector32>) -> tensor<?xf64, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector32> to tensor<?xf64, #SparseVector32>
  return %0 : tensor<?xf64, #SparseVector32>
}
