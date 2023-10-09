// RUN: mlir-opt %s --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL:   func.func @sparse_convert_1d(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?xi32>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_0]], %[[VAL_7]] : tensor<?xi32>
// CHECK:           %[[VAL_9:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_10:.*]] = memref.cast %[[VAL_9]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_9]]{{\[}}%[[VAL_7]]] : memref<1xi8>
// CHECK:           %[[VAL_11:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_12:.*]] = memref.cast %[[VAL_11]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_11]]{{\[}}%[[VAL_7]]] : memref<1xindex>
// CHECK:           %[[VAL_13:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_13]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_13]]{{\[}}%[[VAL_7]]] : memref<1xindex>
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = call @newSparseTensor(%[[VAL_12]], %[[VAL_12]], %[[VAL_10]], %[[VAL_14]], %[[VAL_14]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_15]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_18:.*]] = memref.cast %[[VAL_17]] : memref<1xindex> to memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.alloca() : memref<i32>
// CHECK:           scf.for %[[VAL_20:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_5]] {
// CHECK:             %[[VAL_21:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_20]]] : tensor<?xi32>
// CHECK:             %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_4]] : i32
// CHECK:             scf.if %[[VAL_22]] {
// CHECK:               memref.store %[[VAL_20]], %[[VAL_17]]{{\[}}%[[VAL_7]]] : memref<1xindex>
// CHECK:               memref.store %[[VAL_21]], %[[VAL_19]][] : memref<i32>
// CHECK:               %[[VAL_23:.*]] = func.call @addEltI32(%[[VAL_16]], %[[VAL_19]], %[[VAL_18]], %[[VAL_14]]) : (!llvm.ptr<i8>, memref<i32>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = call @newSparseTensor(%[[VAL_12]], %[[VAL_12]], %[[VAL_10]], %[[VAL_14]], %[[VAL_14]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_1]], %[[VAL_16]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOI32(%[[VAL_16]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_24]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_1d(%arg0: tensor<?xi32>) -> tensor<?xi32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xi32> to tensor<?xi32, #SparseVector>
  return %0 : tensor<?xi32, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_convert_complex(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<100xcomplex<f64>>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 9 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 100 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xi8>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_10]]{{\[}}%[[VAL_6]]] : memref<1xi8>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_6]]] : memref<1xindex>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<1xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<1xindex>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]], %[[VAL_3]], %[[VAL_16]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_18:.*]] = memref.alloca() : memref<1xindex>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_18]] : memref<1xindex> to memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<complex<f64>>
// CHECK:           scf.for %[[VAL_21:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_9]] {
// CHECK:             %[[VAL_22:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_21]]] : tensor<100xcomplex<f64>>
// CHECK:             %[[VAL_23:.*]] = complex.neq %[[VAL_22]], %[[VAL_2]] : complex<f64>
// CHECK:             scf.if %[[VAL_23]] {
// CHECK:               memref.store %[[VAL_21]], %[[VAL_18]]{{\[}}%[[VAL_6]]] : memref<1xindex>
// CHECK:               memref.store %[[VAL_22]], %[[VAL_20]][] : memref<complex<f64>>
// CHECK:               %[[VAL_24:.*]] = func.call @addEltC64(%[[VAL_17]], %[[VAL_20]], %[[VAL_19]], %[[VAL_15]]) : (!llvm.ptr<i8>, memref<complex<f64>>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_13]], %[[VAL_11]], %[[VAL_15]], %[[VAL_15]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]], %[[VAL_1]], %[[VAL_17]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOC64(%[[VAL_17]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_25]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_complex(%arg0: tensor<100xcomplex<f64>>) -> tensor<100xcomplex<f64>, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<100xcomplex<f64>> to tensor<100xcomplex<f64>, #SparseVector>
  return %0 : tensor<100xcomplex<f64>, #SparseVector>
}

// CHECK-LABEL:   func.func @sparse_convert_2d(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x4xf64>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 8 : i8
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_12]]{{\[}}%[[VAL_6]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_11]], %[[VAL_12]]{{\[}}%[[VAL_8]]] : memref<2xi8>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_14]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_16]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_16]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_16]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_19:.*]] = call @newSparseTensor(%[[VAL_15]], %[[VAL_15]], %[[VAL_13]], %[[VAL_17]], %[[VAL_17]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]], %[[VAL_3]], %[[VAL_18]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_20]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_22:.*]] = memref.alloca() : memref<f64>
// CHECK:           scf.for %[[VAL_23:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
// CHECK:             scf.for %[[VAL_24:.*]] = %[[VAL_6]] to %[[VAL_9]] step %[[VAL_8]] {
// CHECK:               %[[VAL_25:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_23]], %[[VAL_24]]] : tensor<2x4xf64>
// CHECK:               %[[VAL_26:.*]] = arith.cmpf une, %[[VAL_25]], %[[VAL_2]] : f64
// CHECK:               scf.if %[[VAL_26]] {
// CHECK:                 memref.store %[[VAL_23]], %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:                 memref.store %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_8]]] : memref<2xindex>
// CHECK:                 memref.store %[[VAL_25]], %[[VAL_22]][] : memref<f64>
// CHECK:                 %[[VAL_27:.*]] = func.call @addEltF64(%[[VAL_19]], %[[VAL_22]], %[[VAL_21]], %[[VAL_17]]) : (!llvm.ptr<i8>, memref<f64>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = call @newSparseTensor(%[[VAL_15]], %[[VAL_15]], %[[VAL_13]], %[[VAL_17]], %[[VAL_17]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]], %[[VAL_1]], %[[VAL_19]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOF64(%[[VAL_19]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_28]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_2d(%arg0: tensor<2x4xf64>) -> tensor<2x4xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x4xf64> to tensor<2x4xf64, #CSR>
  return %0 : tensor<2x4xf64, #CSR>
}

// CHECK-LABEL:   func.func @sparse_constant() -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant dense<[1.000000e+00, 5.000000e+00]> : tensor<2xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant dense<{{\[\[}}0, 0], [1, 6]]> : tensor<2x2xi64>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 7 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_12]]{{\[}}%[[VAL_7]]] : memref<2xi8>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_14]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_16]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_16]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_16]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_19:.*]] = call @newSparseTensor(%[[VAL_15]], %[[VAL_15]], %[[VAL_13]], %[[VAL_17]], %[[VAL_17]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_18]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_20]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_22:.*]] = memref.alloca() : memref<f32>
// CHECK:           scf.for %[[VAL_23:.*]] = %[[VAL_5]] to %[[VAL_11]] step %[[VAL_7]] {
// CHECK:             %[[VAL_24:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_23]], %[[VAL_5]]] : tensor<2x2xi64>
// CHECK:             %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : i64 to index
// CHECK:             %[[VAL_26:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_23]], %[[VAL_7]]] : tensor<2x2xi64>
// CHECK:             %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : i64 to index
// CHECK:             %[[VAL_28:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_23]]] : tensor<2xf32>
// CHECK:             memref.store %[[VAL_25]], %[[VAL_20]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:             memref.store %[[VAL_27]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:             memref.store %[[VAL_28]], %[[VAL_22]][] : memref<f32>
// CHECK:             %[[VAL_29:.*]] = func.call @addEltF32(%[[VAL_19]], %[[VAL_22]], %[[VAL_21]], %[[VAL_17]]) : (!llvm.ptr<i8>, memref<f32>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = call @newSparseTensor(%[[VAL_15]], %[[VAL_15]], %[[VAL_13]], %[[VAL_17]], %[[VAL_17]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_3]], %[[VAL_19]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOF32(%[[VAL_19]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_30]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_constant() -> tensor<8x7xf32, #CSR>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #CSR>
  return %1 : tensor<8x7xf32, #CSR>
}

// CHECK-LABEL:   func.func @sparse_constant_csc() -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant dense<[1.000000e+00, 5.000000e+00]> : tensor<2xf32>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant dense<{{\[\[}}0, 0], [1, 6]]> : tensor<2x2xi64>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 7 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_12]]{{\[}}%[[VAL_7]]] : memref<2xi8>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_14]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_16]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_16]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_16]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_18]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_20]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_20]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_23:.*]] = call @newSparseTensor(%[[VAL_15]], %[[VAL_21]], %[[VAL_13]], %[[VAL_17]], %[[VAL_19]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_22]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_24:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_25:.*]] = memref.cast %[[VAL_24]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_26:.*]] = memref.alloca() : memref<f32>
// CHECK:           scf.for %[[VAL_27:.*]] = %[[VAL_5]] to %[[VAL_11]] step %[[VAL_7]] {
// CHECK:             %[[VAL_28:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_27]], %[[VAL_5]]] : tensor<2x2xi64>
// CHECK:             %[[VAL_29:.*]] = arith.index_cast %[[VAL_28]] : i64 to index
// CHECK:             %[[VAL_30:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_27]], %[[VAL_7]]] : tensor<2x2xi64>
// CHECK:             %[[VAL_31:.*]] = arith.index_cast %[[VAL_30]] : i64 to index
// CHECK:             %[[VAL_32:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_27]]] : tensor<2xf32>
// CHECK:             memref.store %[[VAL_29]], %[[VAL_24]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:             memref.store %[[VAL_31]], %[[VAL_24]]{{\[}}%[[VAL_7]]] : memref<2xindex>
// CHECK:             memref.store %[[VAL_32]], %[[VAL_26]][] : memref<f32>
// CHECK:             %[[VAL_33:.*]] = func.call @addEltF32(%[[VAL_23]], %[[VAL_26]], %[[VAL_25]], %[[VAL_17]]) : (!llvm.ptr<i8>, memref<f32>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = call @newSparseTensor(%[[VAL_15]], %[[VAL_21]], %[[VAL_13]], %[[VAL_17]], %[[VAL_19]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_3]], %[[VAL_23]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOF32(%[[VAL_23]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_34]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_constant_csc() -> tensor<8x7xf32, #CSC>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #CSC>
  return %1 : tensor<8x7xf32, #CSC>
}

// CHECK-LABEL:   func.func @sparse_convert_3d(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?x?xf64>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 4 : i8
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_10]] : tensor<?x?x?xf64>
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_0]], %[[VAL_9]] : tensor<?x?x?xf64>
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_0]], %[[VAL_8]] : tensor<?x?x?xf64>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<3xi8>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<3xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_14]]{{\[}}%[[VAL_10]]] : memref<3xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<3xi8>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_14]]{{\[}}%[[VAL_8]]] : memref<3xi8>
// CHECK:           %[[VAL_16:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_16]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_11]], %[[VAL_16]]{{\[}}%[[VAL_10]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_12]], %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_13]], %[[VAL_16]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:           %[[VAL_19:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_10]]] : memref<3xindex>
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<3xindex>
// CHECK:           %[[VAL_21:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_22:.*]] = memref.cast %[[VAL_21]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_21]]{{\[}}%[[VAL_10]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_21]]{{\[}}%[[VAL_9]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_21]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:           %[[VAL_23:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_24:.*]] = memref.cast %[[VAL_23]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_23]]{{\[}}%[[VAL_10]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_23]]{{\[}}%[[VAL_9]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_23]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:           %[[VAL_25:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_26:.*]] = memref.cast %[[VAL_25]] : memref<3xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_18]], %[[VAL_25]]{{\[}}%[[VAL_10]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_19]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_20]], %[[VAL_25]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.zero : !llvm.ptr<i8>
// CHECK:           %[[VAL_28:.*]] = call @newSparseTensor(%[[VAL_17]], %[[VAL_26]], %[[VAL_15]], %[[VAL_22]], %[[VAL_24]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]], %[[VAL_3]], %[[VAL_27]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_29:.*]] = memref.alloca() : memref<3xindex>
// CHECK:           %[[VAL_30:.*]] = memref.cast %[[VAL_29]] : memref<3xindex> to memref<?xindex>
// CHECK:           %[[VAL_31:.*]] = memref.alloca() : memref<f64>
// CHECK:           scf.for %[[VAL_32:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_9]] {
// CHECK:             scf.for %[[VAL_33:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_9]] {
// CHECK:               scf.for %[[VAL_34:.*]] = %[[VAL_10]] to %[[VAL_13]] step %[[VAL_9]] {
// CHECK:                 %[[VAL_35:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_32]], %[[VAL_33]], %[[VAL_34]]] : tensor<?x?x?xf64>
// CHECK:                 %[[VAL_36:.*]] = arith.cmpf une, %[[VAL_35]], %[[VAL_2]] : f64
// CHECK:                 scf.if %[[VAL_36]] {
// CHECK:                   memref.store %[[VAL_32]], %[[VAL_29]]{{\[}}%[[VAL_10]]] : memref<3xindex>
// CHECK:                   memref.store %[[VAL_33]], %[[VAL_29]]{{\[}}%[[VAL_9]]] : memref<3xindex>
// CHECK:                   memref.store %[[VAL_34]], %[[VAL_29]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:                   memref.store %[[VAL_35]], %[[VAL_31]][] : memref<f64>
// CHECK:                   %[[VAL_37:.*]] = func.call @addEltF64(%[[VAL_28]], %[[VAL_31]], %[[VAL_30]], %[[VAL_22]]) : (!llvm.ptr<i8>, memref<f64>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = call @newSparseTensor(%[[VAL_17]], %[[VAL_26]], %[[VAL_15]], %[[VAL_22]], %[[VAL_24]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]], %[[VAL_1]], %[[VAL_28]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           call @delSparseTensorCOOF64(%[[VAL_28]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_38]] : !llvm.ptr<i8>
// CHECK:         }
func.func @sparse_convert_3d(%arg0: tensor<?x?x?xf64>) -> tensor<?x?x?xf64, #SparseTensor> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64> to tensor<?x?x?xf64, #SparseTensor>
  return %0 : tensor<?x?x?xf64, #SparseTensor>
}
