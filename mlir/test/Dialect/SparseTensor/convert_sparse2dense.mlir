// RUN: mlir-opt %s --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL:   func.func @sparse_convert_1d(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<13xi32> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 13 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : i8
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_6]]{{\[}}%[[VAL_3]]] : memref<1xi8>
// CHECK:           %[[VAL_8:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xindex>
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_10]]{{\[}}%[[VAL_3]]] : memref<1xindex>
// CHECK:           %[[VAL_12:.*]] = call @newSparseTensor(%[[VAL_9]], %[[VAL_9]], %[[VAL_7]], %[[VAL_11]], %[[VAL_11]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_13]] : memref<1xindex> to memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<13xi32>
// CHECK:           linalg.fill ins(%[[VAL_2]] : i32) outs(%[[VAL_16]] : memref<13xi32>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_17:.*]] = func.call @getNextI32(%[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<i32>) -> i1
// CHECK:             scf.condition(%[[VAL_17]])
// CHECK:           } do {
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_3]]] : memref<1xindex>
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_15]][] : memref<i32>
// CHECK:             memref.store %[[VAL_19]], %[[VAL_16]]{{\[}}%[[VAL_18]]] : memref<13xi32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorI32(%[[VAL_12]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_20:.*]] = bufferization.to_tensor %[[VAL_16]] : memref<13xi32>
// CHECK:           return %[[VAL_20]] : tensor<13xi32>
// CHECK:         }
func.func @sparse_convert_1d(%arg0: tensor<13xi32, #SparseVector>) -> tensor<13xi32> {
  %0 = sparse_tensor.convert %arg0 : tensor<13xi32, #SparseVector> to tensor<13xi32>
  return %0 : tensor<13xi32>
}

// CHECK-LABEL:   func.func @sparse_convert_1d_dyn(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<?xi32> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_4]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_6:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<1xi8>
// CHECK:           %[[VAL_8:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:           %[[VAL_12:.*]] = call @newSparseTensor(%[[VAL_9]], %[[VAL_9]], %[[VAL_7]], %[[VAL_11]], %[[VAL_11]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]], %[[VAL_1]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_13]] : memref<1xindex> to memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_16:.*]] = memref.alloc(%[[VAL_5]]) : memref<?xi32>
// CHECK:           linalg.fill ins(%[[VAL_2]] : i32) outs(%[[VAL_16]] : memref<?xi32>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_17:.*]] = func.call @getNextI32(%[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<i32>) -> i1
// CHECK:             scf.condition(%[[VAL_17]])
// CHECK:           } do {
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_4]]] : memref<1xindex>
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_15]][] : memref<i32>
// CHECK:             memref.store %[[VAL_19]], %[[VAL_16]]{{\[}}%[[VAL_18]]] : memref<?xi32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorI32(%[[VAL_12]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_20:.*]] = bufferization.to_tensor %[[VAL_16]] : memref<?xi32>
// CHECK:           return %[[VAL_20]] : tensor<?xi32>
// CHECK:         }
func.func @sparse_convert_1d_dyn(%arg0: tensor<?xi32, #SparseVector>) -> tensor<?xi32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xi32, #SparseVector> to tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL:   func.func @sparse_convert_2d(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<2x4xf64> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 4 : i8
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_10]]{{\[}}%[[VAL_6]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_10]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = memref.cast %[[VAL_17]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_20:.*]] = memref.alloc() : memref<2x4xf64>
// CHECK:           linalg.fill ins(%[[VAL_1]] : f64) outs(%[[VAL_20]] : memref<2x4xf64>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_21:.*]] = func.call @getNextF64(%[[VAL_16]], %[[VAL_18]], %[[VAL_19]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:             scf.condition(%[[VAL_21]])
// CHECK:           } do {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_19]][] : memref<f64>
// CHECK:             memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_22]], %[[VAL_23]]] : memref<2x4xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorF64(%[[VAL_16]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_25:.*]] = bufferization.to_tensor %[[VAL_20]] : memref<2x4xf64>
// CHECK:           return %[[VAL_25]] : tensor<2x4xf64>
// CHECK:         }
func.func @sparse_convert_2d(%arg0: tensor<2x4xf64, #SparseMatrix>) -> tensor<2x4xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x4xf64, #SparseMatrix> to tensor<2x4xf64>
  return %0 : tensor<2x4xf64>
}

// CHECK-LABEL:   func.func @sparse_convert_2d_dyn0(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<?x4xf64> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_8]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_10]]{{\[}}%[[VAL_8]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_10]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_12]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_14]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = memref.cast %[[VAL_17]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_20:.*]] = memref.alloc(%[[VAL_9]]) : memref<?x4xf64>
// CHECK:           linalg.fill ins(%[[VAL_1]] : f64) outs(%[[VAL_20]] : memref<?x4xf64>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_21:.*]] = func.call @getNextF64(%[[VAL_16]], %[[VAL_18]], %[[VAL_19]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:             scf.condition(%[[VAL_21]])
// CHECK:           } do {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_19]][] : memref<f64>
// CHECK:             memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_22]], %[[VAL_23]]] : memref<?x4xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorF64(%[[VAL_16]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_25:.*]] = bufferization.to_tensor %[[VAL_20]] : memref<?x4xf64>
// CHECK:           return %[[VAL_25]] : tensor<?x4xf64>
// CHECK:         }
func.func @sparse_convert_2d_dyn0(%arg0: tensor<?x4xf64, #SparseMatrix>) -> tensor<?x4xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x4xf64, #SparseMatrix> to tensor<?x4xf64>
  return %0 : tensor<?x4xf64>
}

// CHECK-LABEL:   func.func @sparse_convert_2d_dyn1(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<2x?xf64> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_8]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_10]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_10]]{{\[}}%[[VAL_8]]] : memref<2xi8>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_12]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_14]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = memref.cast %[[VAL_17]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_20:.*]] = memref.alloc(%[[VAL_9]]) : memref<2x?xf64>
// CHECK:           linalg.fill ins(%[[VAL_1]] : f64) outs(%[[VAL_20]] : memref<2x?xf64>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_21:.*]] = func.call @getNextF64(%[[VAL_16]], %[[VAL_18]], %[[VAL_19]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:             scf.condition(%[[VAL_21]])
// CHECK:           } do {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_19]][] : memref<f64>
// CHECK:             memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_22]], %[[VAL_23]]] : memref<2x?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorF64(%[[VAL_16]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_25:.*]] = bufferization.to_tensor %[[VAL_20]] : memref<2x?xf64>
// CHECK:           return %[[VAL_25]] : tensor<2x?xf64>
// CHECK:         }
func.func @sparse_convert_2d_dyn1(%arg0: tensor<2x?xf64, #SparseMatrix>) -> tensor<2x?xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x?xf64, #SparseMatrix> to tensor<2x?xf64>
  return %0 : tensor<2x?xf64>
}

// CHECK-LABEL:   func.func @sparse_convert_2d_dyn2(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf64> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_7]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_9:.*]] = call @sparseDimSize(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> index
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_10]]{{\[}}%[[VAL_7]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_10]]{{\[}}%[[VAL_6]]] : memref<2xi8>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_12]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_12]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_14]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = memref.cast %[[VAL_17]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_20:.*]] = memref.alloc(%[[VAL_8]], %[[VAL_9]]) : memref<?x?xf64>
// CHECK:           linalg.fill ins(%[[VAL_1]] : f64) outs(%[[VAL_20]] : memref<?x?xf64>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_21:.*]] = func.call @getNextF64(%[[VAL_16]], %[[VAL_18]], %[[VAL_19]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:             scf.condition(%[[VAL_21]])
// CHECK:           } do {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_19]][] : memref<f64>
// CHECK:             memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_22]], %[[VAL_23]]] : memref<?x?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorF64(%[[VAL_16]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_25:.*]] = bufferization.to_tensor %[[VAL_20]] : memref<?x?xf64>
// CHECK:           return %[[VAL_25]] : tensor<?x?xf64>
// CHECK:         }
func.func @sparse_convert_2d_dyn2(%arg0: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x?xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?xf64, #SparseMatrix> to tensor<?x?xf64>
  return %0 : tensor<?x?xf64>
}

// CHECK-LABEL:   func.func @sparse_convert_3d(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>) -> tensor<2x3x4xf64> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 4 : i8
// CHECK:           %[[VAL_11:.*]] = memref.alloca() : memref<3xi8>
// CHECK:           %[[VAL_12:.*]] = memref.cast %[[VAL_11]] : memref<3xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_11]]{{\[}}%[[VAL_6]]] : memref<3xi8>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_11]]{{\[}}%[[VAL_5]]] : memref<3xi8>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_11]]{{\[}}%[[VAL_7]]] : memref<3xi8>
// CHECK:           %[[VAL_13:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_13]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_13]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_13]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_13]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           %[[VAL_15:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_16:.*]] = memref.cast %[[VAL_15]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_15]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_15]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_15]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           %[[VAL_17:.*]] = call @newSparseTensor(%[[VAL_14]], %[[VAL_14]], %[[VAL_12]], %[[VAL_16]], %[[VAL_16]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_18:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_18]] : memref<3xindex> to memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_21:.*]] = memref.alloc() : memref<2x3x4xf64>
// CHECK:           linalg.fill ins(%[[VAL_1]] : f64) outs(%[[VAL_21]] : memref<2x3x4xf64>)
// CHECK:           scf.while : () -> () {
// CHECK:             %[[VAL_22:.*]] = func.call @getNextF64(%[[VAL_17]], %[[VAL_19]], %[[VAL_20]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:             scf.condition(%[[VAL_22]])
// CHECK:           } do {
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_6]]] : memref<3xindex>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<3xindex>
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:             %[[VAL_26:.*]] = memref.load %[[VAL_20]][] : memref<f64>
// CHECK:             memref.store %[[VAL_26]], %[[VAL_21]]{{\[}}%[[VAL_23]], %[[VAL_24]], %[[VAL_25]]] : memref<2x3x4xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorIteratorF64(%[[VAL_17]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[VAL_27:.*]] = bufferization.to_tensor %[[VAL_21]] : memref<2x3x4xf64>
// CHECK:           return %[[VAL_27]] : tensor<2x3x4xf64>
// CHECK:         }
func.func @sparse_convert_3d(%arg0: tensor<2x3x4xf64, #SparseTensor>) -> tensor<2x3x4xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x3x4xf64, #SparseTensor> to tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}
