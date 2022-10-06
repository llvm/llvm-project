// RUN: mlir-opt %s --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

#SparseMatrix_P = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

// CHECK-LABEL: func.func @concat_mix_dense(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<2x4xf64>,
// CHECK-SAME:    %[[TMP_arg1:.*]]: !llvm.ptr<i8>)
// CHECK-DAG:     %[[TMP_c2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[TMP_c6_i32:.*]] = arith.constant 6 : i32
// CHECK-DAG:     %[[TMP_c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[TMP_c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[TMP_c8_i8:.*]] = arith.constant 8 : i8
// CHECK-DAG:     %[[TMP_c3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TMP_cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[TMP_c4:.*]] = arith.constant 4 : index
// CHECK:         %[[TMP_0:.*]] = memref.alloc() : memref<5x4xf64>
// CHECK:         linalg.fill ins(%[[TMP_cst]] : f64) outs(%[[TMP_0]] : memref<5x4xf64>)
// CHECK:         scf.for %[[TMP_arg2:.*]] = %[[TMP_c0]] to %[[TMP_c2]] step %[[TMP_c1]] {
// CHECK:           scf.for %[[TMP_arg3:.*]] = %[[TMP_c0]] to %[[TMP_c4]] step %[[TMP_c1]] {
// CHECK:             %[[TMP_12:.*]] = tensor.extract %[[TMP_arg0]][%[[TMP_arg2]], %[[TMP_arg3]]] : tensor<2x4xf64>
// CHECK:             %[[TMP_13:.*]] = arith.cmpf une, %[[TMP_12]], %[[TMP_cst]] : f64
// CHECK:             scf.if %[[TMP_13]] {
// CHECK:               memref.store %[[TMP_12]], %[[TMP_0]][%[[TMP_arg2]], %[[TMP_arg3]]] : memref<5x4xf64>
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         %[[TMP_1:.*]] = memref.alloca() : memref<2xi8>
// CHECK:         %[[TMP_2:.*]] = memref.cast %[[TMP_1]] : memref<2xi8> to memref<?xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_1]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_1]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:         %[[TMP_3:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_4:.*]] = memref.cast %[[TMP_3]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c3]], %[[TMP_3]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c4]], %[[TMP_3]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_5:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_6:.*]] = memref.cast %[[TMP_5]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c0]], %[[TMP_5]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c1]], %[[TMP_5]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_7:.*]] = call @newSparseTensor(%[[TMP_2]], %[[TMP_4]], %[[TMP_6]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c6_i32]], %[[TMP_arg1]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         %[[TMP_8:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_9:.*]] = memref.cast %[[TMP_8]] : memref<2xindex> to memref<?xindex>
// CHECK:         %[[TMP_10:.*]] = memref.alloca() : memref<f64>
// CHECK:         scf.while : () -> () {
// CHECK:           %[[TMP_12:.*]] = func.call @getNextF64(%[[TMP_7]], %[[TMP_9]], %[[TMP_10]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:           scf.condition(%[[TMP_12]])
// CHECK:         } do {
// CHECK:           %[[TMP_12:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           %[[TMP_13:.*]] = arith.addi %[[TMP_12]], %[[TMP_c2]] : index
// CHECK:           %[[TMP_14:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_15:.*]] = memref.load %[[TMP_10]][] : memref<f64>
// CHECK:           memref.store %[[TMP_15]], %[[TMP_0]][%[[TMP_13]], %[[TMP_14]]] : memref<5x4xf64>
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         call @delSparseTensorCOOF64(%[[TMP_7]]) : (!llvm.ptr<i8>) -> ()
// CHECK:         %[[TMP_11:.*]] = bufferization.to_tensor %[[TMP_0]] : memref<5x4xf64>
// CHECK:         return %[[TMP_11]] : tensor<5x4xf64>
// CHECK:       }
func.func @concat_mix_dense(%arg0: tensor<2x4xf64>, %arg1: tensor<3x4xf64, #SparseMatrix>) -> tensor<5x4xf64> {
  %0 = sparse_tensor.concatenate %arg0, %arg1 {dimension = 0 : index}
       : tensor<2x4xf64>, tensor<3x4xf64, #SparseMatrix> to tensor<5x4xf64>
  return %0 : tensor<5x4xf64>
}

// CHECK-LABEL: func.func @concat_mix_sparse(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<2x4xf64>,
// CHECK-SAME:    %[[TMP_arg1:.*]]: !llvm.ptr<i8>)
// CHECK-DAG:     %[[TMP_c2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[TMP_c2_i32:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[TMP_c6_i32:.*]] = arith.constant 6 : i32
// CHECK-DAG:     %[[TMP_c3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[TMP_cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[TMP_c4_i32:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[TMP_c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[TMP_c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[TMP_c5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[TMP_c4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[TMP_c8_i8:.*]] = arith.constant 8 : i8
// CHECK:         %[[TMP_0:.*]] = memref.alloca() : memref<2xi8>
// CHECK:         %[[TMP_1:.*]] = memref.cast %[[TMP_0]] : memref<2xi8> to memref<?xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_0]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_0]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:         %[[TMP_2:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_3:.*]] = memref.cast %[[TMP_2]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c5]], %[[TMP_2]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c4]], %[[TMP_2]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_4:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_5:.*]] = memref.cast %[[TMP_4]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c0]], %[[TMP_4]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c1]], %[[TMP_4]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_6:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:         %[[TMP_7:.*]] = call @newSparseTensor(%[[TMP_1]], %[[TMP_3]], %[[TMP_5]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c4_i32]], %[[TMP_6]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         %[[TMP_8:.*]] = memref.alloca() : memref<f64>
// CHECK:         %[[TMP_9:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_10:.*]] = memref.cast %[[TMP_9]] : memref<2xindex> to memref<?xindex>
// CHECK:         scf.for %[[TMP_arg2:.*]] = %[[TMP_c0]] to %[[TMP_c2]] step %[[TMP_c1]] {
// CHECK:           scf.for %[[TMP_arg3:.*]] = %[[TMP_c0]] to %[[TMP_c4]] step %[[TMP_c1]] {
// CHECK:             memref.store %[[TMP_arg2]], %[[TMP_9]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:             memref.store %[[TMP_arg3]], %[[TMP_9]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:             %[[TMP_22:.*]] = tensor.extract %[[TMP_arg0]][%[[TMP_arg2]], %[[TMP_arg3]]] : tensor<2x4xf64>
// CHECK:             %[[TMP_23:.*]] = arith.cmpf une, %[[TMP_22]], %[[TMP_cst]] : f64
// CHECK:             scf.if %[[TMP_23]] {
// CHECK:               memref.store %[[TMP_22]], %[[TMP_8]][] : memref<f64>
// CHECK:               %[[TMP_24:.*]] = func.call @addEltF64(%[[TMP_7]], %[[TMP_8]], %[[TMP_10]], %[[TMP_5]]) : (!llvm.ptr<i8>, memref<f64>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         %[[TMP_11:.*]] = memref.alloca() : memref<2xi8>
// CHECK:         %[[TMP_12:.*]] = memref.cast %[[TMP_11]] : memref<2xi8> to memref<?xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_11]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_11]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:         %[[TMP_13:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_14:.*]] = memref.cast %[[TMP_13]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c3]], %[[TMP_13]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c4]], %[[TMP_13]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_15:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_16:.*]] = memref.cast %[[TMP_15]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c0]], %[[TMP_15]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c1]], %[[TMP_15]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_17:.*]] = call @newSparseTensor(%[[TMP_12]], %[[TMP_14]], %[[TMP_16]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c6_i32]], %[[TMP_arg1]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         %[[TMP_18:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_19:.*]] = memref.cast %[[TMP_18]] : memref<2xindex> to memref<?xindex>
// CHECK:         %[[TMP_20:.*]] = memref.alloca() : memref<f64>
// CHECK:         scf.while : () -> () {
// CHECK:           %[[TMP_22:.*]] = func.call @getNextF64(%[[TMP_17]], %[[TMP_19]], %[[TMP_20]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:           scf.condition(%[[TMP_22]])
// CHECK:         } do {
// CHECK:           %[[TMP_22:.*]] = memref.load %[[TMP_18]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           %[[TMP_23:.*]] = arith.addi %[[TMP_22]], %[[TMP_c2]] : index
// CHECK:           %[[TMP_24:.*]] = memref.load %[[TMP_18]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           memref.store %[[TMP_23]], %[[TMP_9]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           memref.store %[[TMP_24]], %[[TMP_9]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_25:.*]] = func.call @addEltF64(%[[TMP_7]], %[[TMP_20]], %[[TMP_10]], %[[TMP_5]]) : (!llvm.ptr<i8>, memref<f64>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         call @delSparseTensorCOOF64(%[[TMP_17]]) : (!llvm.ptr<i8>) -> ()
// CHECK:         %[[TMP_21:.*]] = call @newSparseTensor(%[[TMP_1]], %[[TMP_3]], %[[TMP_5]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c2_i32]], %[[TMP_7]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         call @delSparseTensorCOOF64(%[[TMP_7]]) : (!llvm.ptr<i8>) -> ()
// CHECK:         return %[[TMP_21]] : !llvm.ptr<i8>
// CHECK:       }
func.func @concat_mix_sparse(%arg0: tensor<2x4xf64>, %arg1: tensor<3x4xf64, #SparseMatrix>) -> tensor<5x4xf64, #SparseMatrix> {
  %0 = sparse_tensor.concatenate %arg0, %arg1 {dimension = 0 : index}
       : tensor<2x4xf64>, tensor<3x4xf64, #SparseMatrix> to tensor<5x4xf64, #SparseMatrix>
  return %0 : tensor<5x4xf64, #SparseMatrix>
}

// CHECK-LABEL: func.func @concat_mix_sparse_perm_dim1(
// CHECK-SAME:    %[[TMP_arg0:.*]]: tensor<4x2xf64>,
// CHECK-SAME:    %[[TMP_arg1:.*]]: !llvm.ptr<i8>)
// CHECK-DAG:     %[[TMP_c2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[TMP_c2_i32:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[TMP_c6_i32:.*]] = arith.constant 6 : i32
// CHECK-DAG:     %[[TMP_c3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[TMP_cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[TMP_c4_i32:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[TMP_c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[TMP_c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[TMP_c4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[TMP_c5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[TMP_c8_i8:.*]] = arith.constant 8 : i8
// CHECK:         %[[TMP_0:.*]] = memref.alloca() : memref<2xi8>
// CHECK:         %[[TMP_1:.*]] = memref.cast %[[TMP_0]] : memref<2xi8> to memref<?xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_0]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_0]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:         %[[TMP_2:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_3:.*]] = memref.cast %[[TMP_2]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c4]], %[[TMP_2]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c5]], %[[TMP_2]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_4:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_5:.*]] = memref.cast %[[TMP_4]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c1]], %[[TMP_4]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c0]], %[[TMP_4]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_6:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:         %[[TMP_7:.*]] = call @newSparseTensor(%[[TMP_1]], %[[TMP_3]], %[[TMP_5]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c4_i32]], %[[TMP_6]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         %[[TMP_8:.*]] = memref.alloca() : memref<f64>
// CHECK:         %[[TMP_9:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_10:.*]] = memref.cast %[[TMP_9]] : memref<2xindex> to memref<?xindex>
// CHECK:         scf.for %[[TMP_arg2:.*]] = %[[TMP_c0]] to %[[TMP_c4]] step %[[TMP_c1]] {
// CHECK:           scf.for %[[TMP_arg3:.*]] = %[[TMP_c0]] to %[[TMP_c2]] step %[[TMP_c1]] {
// CHECK:             memref.store %[[TMP_arg2]], %[[TMP_9]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:             memref.store %[[TMP_arg3]], %[[TMP_9]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:             %[[TMP_22:.*]] = tensor.extract %[[TMP_arg0]][%[[TMP_arg2]], %[[TMP_arg3]]] : tensor<4x2xf64>
// CHECK:             %[[TMP_23:.*]] = arith.cmpf une, %[[TMP_22]], %[[TMP_cst]] : f64
// CHECK:             scf.if %[[TMP_23]] {
// CHECK:               memref.store %[[TMP_22]], %[[TMP_8]][] : memref<f64>
// CHECK:               %[[TMP_24:.*]] = func.call @addEltF64(%[[TMP_7]], %[[TMP_8]], %[[TMP_10]], %[[TMP_5]]) : (!llvm.ptr<i8>, memref<f64>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         %[[TMP_11:.*]] = memref.alloca() : memref<2xi8>
// CHECK:         %[[TMP_12:.*]] = memref.cast %[[TMP_11]] : memref<2xi8> to memref<?xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_11]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_11]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:         %[[TMP_13:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_14:.*]] = memref.cast %[[TMP_13]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c4]], %[[TMP_13]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c3]], %[[TMP_13]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_15:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_16:.*]] = memref.cast %[[TMP_15]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c0]], %[[TMP_15]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c1]], %[[TMP_15]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_17:.*]] = call @newSparseTensor(%[[TMP_12]], %[[TMP_14]], %[[TMP_16]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c6_i32]], %[[TMP_arg1]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         %[[TMP_18:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_19:.*]] = memref.cast %[[TMP_18]] : memref<2xindex> to memref<?xindex>
// CHECK:         %[[TMP_20:.*]] = memref.alloca() : memref<f64>
// CHECK:         scf.while : () -> () {
// CHECK:           %[[TMP_22:.*]] = func.call @getNextF64(%[[TMP_17]], %[[TMP_19]], %[[TMP_20]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:           scf.condition(%[[TMP_22]])
// CHECK:         } do {
// CHECK:           %[[TMP_22:.*]] = memref.load %[[TMP_18]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           %[[TMP_23:.*]] = memref.load %[[TMP_18]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_24:.*]] = arith.addi %[[TMP_23]], %[[TMP_c2]] : index
// CHECK:           memref.store %[[TMP_22]], %[[TMP_9]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           memref.store %[[TMP_24]], %[[TMP_9]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_25:.*]] = func.call @addEltF64(%[[TMP_7]], %[[TMP_20]], %[[TMP_10]], %[[TMP_5]]) : (!llvm.ptr<i8>, memref<f64>, memref<?xindex>, memref<?xindex>) -> !llvm.ptr<i8>
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         call @delSparseTensorCOOF64(%[[TMP_17]]) : (!llvm.ptr<i8>) -> ()
// CHECK:         %[[TMP_21:.*]] = call @newSparseTensor(%[[TMP_1]], %[[TMP_3]], %[[TMP_5]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c2_i32]], %[[TMP_7]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         call @delSparseTensorCOOF64(%[[TMP_7]]) : (!llvm.ptr<i8>) -> ()
// CHECK:         return %[[TMP_21]] : !llvm.ptr<i8>
// CHECK:       }
func.func @concat_mix_sparse_perm_dim1(%arg0: tensor<4x2xf64>, %arg1: tensor<4x3xf64, #SparseMatrix_P>) -> tensor<4x5xf64, #SparseMatrix_P> {
  %0 = sparse_tensor.concatenate %arg0, %arg1 {dimension = 1 : index}
       : tensor<4x2xf64>, tensor<4x3xf64, #SparseMatrix_P> to tensor<4x5xf64, #SparseMatrix_P>
  return %0 : tensor<4x5xf64, #SparseMatrix_P>
}

// CHECK-LABEL: func.func @concat_mix_dense_perm_dim1(
// CHECK-SAME:     %[[TMP_arg0:.*]]: tensor<4x2xf64>,
// CHECK-SAME:     %[[TMP_arg1:.*]]: !llvm.ptr<i8>)
// CHECK-DAG:         %[[TMP_c2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[TMP_c6_i32:.*]] = arith.constant 6 : i32
// CHECK-DAG:         %[[TMP_c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:         %[[TMP_c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[TMP_c8_i8:.*]] = arith.constant 8 : i8
// CHECK-DAG:         %[[TMP_c3:.*]] = arith.constant 3 : index
// CHECK-DAG:         %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[TMP_cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:         %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[TMP_c4:.*]] = arith.constant 4 : index
// CHECK:         %[[TMP_0:.*]] = memref.alloc() : memref<4x5xf64>
// CHECK:         linalg.fill ins(%[[TMP_cst]] : f64) outs(%[[TMP_0]] : memref<4x5xf64>)
// CHECK:         scf.for %[[TMP_arg2:.*]] = %[[TMP_c0]] to %[[TMP_c4]] step %[[TMP_c1]] {
// CHECK:           scf.for %[[TMP_arg3:.*]] = %[[TMP_c0]] to %[[TMP_c2]] step %[[TMP_c1]] {
// CHECK:             %[[TMP_12:.*]] = tensor.extract %[[TMP_arg0]][%[[TMP_arg2]], %[[TMP_arg3]]] : tensor<4x2xf64>
// CHECK:             %[[TMP_13:.*]] = arith.cmpf une, %[[TMP_12]], %[[TMP_cst]] : f64
// CHECK:             scf.if %[[TMP_13]] {
// CHECK:               memref.store %[[TMP_12]], %[[TMP_0]][%[[TMP_arg2]], %[[TMP_arg3]]] : memref<4x5xf64>
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         %[[TMP_1:.*]] = memref.alloca() : memref<2xi8>
// CHECK:         %[[TMP_2:.*]] = memref.cast %[[TMP_1]] : memref<2xi8> to memref<?xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_1]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:         memref.store %[[TMP_c8_i8]], %[[TMP_1]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:         %[[TMP_3:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_4:.*]] = memref.cast %[[TMP_3]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c4]], %[[TMP_3]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c3]], %[[TMP_3]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_5:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_6:.*]] = memref.cast %[[TMP_5]] : memref<2xindex> to memref<?xindex>
// CHECK:         memref.store %[[TMP_c0]], %[[TMP_5]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:         memref.store %[[TMP_c1]], %[[TMP_5]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:         %[[TMP_7:.*]] = call @newSparseTensor(%[[TMP_2]], %[[TMP_4]], %[[TMP_6]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c6_i32]], %[[TMP_arg1]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         %[[TMP_8:.*]] = memref.alloca() : memref<2xindex>
// CHECK:         %[[TMP_9:.*]] = memref.cast %[[TMP_8]] : memref<2xindex> to memref<?xindex>
// CHECK:         %[[TMP_10:.*]] = memref.alloca() : memref<f64>
// CHECK:         scf.while : () -> () {
// CHECK:           %[[TMP_12:.*]] = func.call @getNextF64(%[[TMP_7]], %[[TMP_9]], %[[TMP_10]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:           scf.condition(%[[TMP_12]])
// CHECK:         } do {
// CHECK:           %[[TMP_12:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           %[[TMP_13:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_14:.*]] = arith.addi %[[TMP_13]], %[[TMP_c2]] : index
// CHECK:           %[[TMP_15:.*]] = memref.load %[[TMP_10]][] : memref<f64>
// CHECK:           memref.store %[[TMP_15]], %[[TMP_0]][%[[TMP_12]], %[[TMP_14]]] : memref<4x5xf64>
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         call @delSparseTensorCOOF64(%[[TMP_7]]) : (!llvm.ptr<i8>) -> ()
// CHECK:         %[[TMP_11:.*]] = bufferization.to_tensor %[[TMP_0]] : memref<4x5xf64>
// CHECK:         return %[[TMP_11]] : tensor<4x5xf64>
// CHECK:       }
func.func @concat_mix_dense_perm_dim1(%arg0: tensor<4x2xf64>, %arg1: tensor<4x3xf64, #SparseMatrix_P>) -> tensor<4x5xf64> {
  %0 = sparse_tensor.concatenate %arg0, %arg1 {dimension = 1 : index}
       : tensor<4x2xf64>, tensor<4x3xf64, #SparseMatrix_P> to tensor<4x5xf64>
  return %0 : tensor<4x5xf64>
}

// CHECK-LABEL: func.func @concat_mix_dense_perm_dim1_dyn(
// CHECK-SAME:      %[[TMP_arg0:.*]]: tensor<3x2xf64>,
// CHECK-SAME:      %[[TMP_arg1:.*]]: !llvm.ptr<i8>) 
// CHECK-DAG:       %[[TMP_c2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[TMP_c6_i32:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[TMP_c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[TMP_c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[TMP_c8_i8:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[TMP_cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[TMP_c3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK:           %[[TMP_0:.*]] = memref.alloc() : memref<3x5xf64>
// CHECK:           %[[TMP_1:.*]] = memref.cast %[[TMP_0]] : memref<3x5xf64> to memref<?x?xf64>
// CHECK:           linalg.fill ins(%[[TMP_cst]] : f64) outs(%[[TMP_0]] : memref<3x5xf64>)
// CHECK:           scf.for %[[TMP_arg2:.*]] = %[[TMP_c0]] to %[[TMP_c3]] step %[[TMP_c1]] {
// CHECK:             scf.for %[[TMP_arg3:.*]] = %[[TMP_c0]] to %[[TMP_c2]] step %[[TMP_c1]] {
// CHECK:               %[[TMP_13:.*]] = tensor.extract %[[TMP_arg0]][%[[TMP_arg2]], %[[TMP_arg3]]] : tensor<3x2xf64>
// CHECK:               %[[TMP_14:.*]] = arith.cmpf une, %[[TMP_13]], %[[TMP_cst]] : f64
// CHECK:               scf.if %[[TMP_14]] {
// CHECK:                 memref.store %[[TMP_13]], %[[TMP_0]][%[[TMP_arg2]], %[[TMP_arg3]]] : memref<3x5xf64>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TMP_2:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[TMP_3:.*]] = memref.cast %[[TMP_2]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[TMP_c8_i8]], %[[TMP_2]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:           memref.store %[[TMP_c8_i8]], %[[TMP_2]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:           %[[TMP_4:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[TMP_5:.*]] = memref.cast %[[TMP_4]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[TMP_c3]], %[[TMP_4]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           memref.store %[[TMP_c3]], %[[TMP_4]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_6:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[TMP_7:.*]] = memref.cast %[[TMP_6]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[TMP_c0]], %[[TMP_6]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:           memref.store %[[TMP_c1]], %[[TMP_6]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:           %[[TMP_8:.*]] = call @newSparseTensor(%[[TMP_3]], %[[TMP_5]], %[[TMP_7]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c6_i32]], %[[TMP_arg1]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[TMP_9:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[TMP_10:.*]] = memref.cast %[[TMP_9]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[TMP_11:.*]] = memref.alloca() : memref<f64>
// CHECK:           scf.while : () -> () {
// CHECK:             %[[TMP_13:.*]] = func.call @getNextF64(%[[TMP_8]], %[[TMP_10]], %[[TMP_11]]) : (!llvm.ptr<i8>, memref<?xindex>, memref<f64>) -> i1
// CHECK:             scf.condition(%[[TMP_13]])
// CHECK:           } do {
// CHECK:             %[[TMP_13:.*]] = memref.load %[[TMP_9]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:             %[[TMP_14:.*]] = memref.load %[[TMP_9]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:             %[[TMP_15:.*]] = arith.addi %[[TMP_14]], %[[TMP_c2]] : index
// CHECK:             %[[TMP_16:.*]] = memref.load %[[TMP_11]][] : memref<f64>
// CHECK:             memref.store %[[TMP_16]], %[[TMP_0]][%[[TMP_13]], %[[TMP_15]]] : memref<3x5xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           call @delSparseTensorCOOF64(%[[TMP_8]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           %[[TMP_12:.*]] = bufferization.to_tensor %[[TMP_1]] : memref<?x?xf64>
// CHECK:           return %[[TMP_12]] : tensor<?x?xf64>
// CHECK:         }
// CHECK:       }
func.func @concat_mix_dense_perm_dim1_dyn(%arg0: tensor<3x2xf64>, %arg1: tensor<3x3xf64, #SparseMatrix>) -> tensor<?x?xf64> {
  %0 = sparse_tensor.concatenate %arg0, %arg1 {dimension = 1 : index}
       : tensor<3x2xf64>, tensor<3x3xf64, #SparseMatrix> to tensor<?x?xf64>
  return %0 : tensor<?x?xf64>
}
