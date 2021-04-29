// RUN: mlir-opt %s -test-sparsification | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIR
//
// RUN: mlir-opt %s -test-sparsification="lower" --convert-linalg-to-loops | \
// RUN:   FileCheck %s --check-prefix=CHECK-MIR
//
// RUN: mlir-opt %s -test-sparsification="lower" --convert-linalg-to-loops \
// RUN:   --func-bufferize --tensor-constant-bufferize \
// RUN:   --tensor-bufferize --finalizing-bufferize  | \
// RUN:   FileCheck %s --check-prefix=CHECK-LIR
//
// RUN: mlir-opt %s -test-sparsification="lower fast-output" --convert-linalg-to-loops \
// RUN:   --func-bufferize --tensor-constant-bufferize \
// RUN:   --tensor-bufferize --finalizing-bufferize  | \
// RUN:   FileCheck %s --check-prefix=CHECK-FAST

#trait_matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel","reduction"],
  sparse = [
    [ "D", "S" ],  // A
    [ "D" ],       // b
    [ "D" ]        // x (out)
  ],
  sparse_dim_map = [
    affine_map<(i,j) -> (j,i)>,  // A: column-wise
    affine_map<(i)   -> (i)>,    // x
    affine_map<(i)   -> (i)>     // b
  ],
  doc = "x(i) += A(i,j) * b(j)"
}

// CHECK-HIR-LABEL:   func @matvec(
// CHECK-HIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-HIR-SAME:                 %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-HIR-SAME:                 %[[VAL_2:.*]]: tensor<64xf64>) -> tensor<64xf64> {
// CHECK-HIR:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-HIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-HIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-HIR:           %[[VAL_6:.*]] = linalg.sparse_tensor %[[VAL_0]] : !llvm.ptr<i8> to tensor<64x64xf64>
// CHECK-HIR:           %[[VAL_7:.*]] = linalg.sparse_pointers %[[VAL_6]], %[[VAL_5]] : tensor<64x64xf64> to memref<?xindex>
// CHECK-HIR:           %[[VAL_8:.*]] = linalg.sparse_indices %[[VAL_6]], %[[VAL_5]] : tensor<64x64xf64> to memref<?xindex>
// CHECK-HIR:           %[[VAL_9:.*]] = linalg.sparse_values %[[VAL_6]] : tensor<64x64xf64> to memref<?xf64>
// CHECK-HIR:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_1]] : memref<64xf64>
// CHECK-HIR:           %[[VAL_11:.*]] = memref.buffer_cast %[[VAL_2]] : memref<64xf64>
// CHECK-HIR:           %[[VAL_12:.*]] = memref.alloc() : memref<64xf64>
// CHECK-HIR:           linalg.copy(%[[VAL_11]], %[[VAL_12]]) : memref<64xf64>, memref<64xf64>
// CHECK-HIR:           scf.for %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-HIR:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK-HIR:             %[[VAL_15:.*]] = addi %[[VAL_13]], %[[VAL_5]] : index
// CHECK-HIR:             %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-HIR:             %[[VAL_17:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_13]]] : memref<64xf64>
// CHECK-HIR:             %[[VAL_18:.*]] = scf.for %[[VAL_19:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_5]] iter_args(%[[VAL_20:.*]] = %[[VAL_17]]) -> (f64) {
// CHECK-HIR:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK-HIR:               %[[VAL_22:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_19]]] : memref<?xf64>
// CHECK-HIR:               %[[VAL_23:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_21]]] : memref<64xf64>
// CHECK-HIR:               %[[VAL_24:.*]] = mulf %[[VAL_22]], %[[VAL_23]] : f64
// CHECK-HIR:               %[[VAL_25:.*]] = addf %[[VAL_20]], %[[VAL_24]] : f64
// CHECK-HIR:               scf.yield %[[VAL_25]] : f64
// CHECK-HIR:             }
// CHECK-HIR:             store %[[VAL_26:.*]], %[[VAL_12]]{{\[}}%[[VAL_13]]] : memref<64xf64>
// CHECK-HIR:           }
// CHECK-HIR:           %[[VAL_27:.*]] = memref.tensor_load %[[VAL_12]] : memref<64xf64>
// CHECK-HIR:           return %[[VAL_27]] : tensor<64xf64>
// CHECK-HIR:         }

// CHECK-MIR-LABEL:   func @matvec(
// CHECK-MIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-MIR-SAME:                 %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-MIR-SAME:                 %[[VAL_2:.*]]: tensor<64xf64>) -> tensor<64xf64> {
// CHECK-MIR:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-MIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-MIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-MIR:           %[[VAL_6:.*]] = call @sparsePointers64(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_7:.*]] = call @sparseIndices64(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-MIR:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<64xf64>
// CHECK-MIR:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_2]] : memref<64xf64>
// CHECK-MIR:           %[[VAL_11:.*]] = memref.alloc() : memref<64xf64>
// CHECK-MIR:           scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-MIR:             %[[VAL_13:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_12]]] : memref<64xf64>
// CHECK-MIR:             store %[[VAL_13]], %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<64xf64>
// CHECK-MIR:           }
// CHECK-MIR:           scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-MIR:             %[[VAL_15:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK-MIR:             %[[VAL_16:.*]] = addi %[[VAL_14]], %[[VAL_5]] : index
// CHECK-MIR:             %[[VAL_17:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK-MIR:             %[[VAL_18:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_14]]] : memref<64xf64>
// CHECK-MIR:             %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_5]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (f64) {
// CHECK-MIR:               %[[VAL_22:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK-MIR:               %[[VAL_23:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xf64>
// CHECK-MIR:               %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_22]]] : memref<64xf64>
// CHECK-MIR:               %[[VAL_25:.*]] = mulf %[[VAL_23]], %[[VAL_24]] : f64
// CHECK-MIR:               %[[VAL_26:.*]] = addf %[[VAL_21]], %[[VAL_25]] : f64
// CHECK-MIR:               scf.yield %[[VAL_26]] : f64
// CHECK-MIR:             }
// CHECK-MIR:             store %[[VAL_27:.*]], %[[VAL_11]]{{\[}}%[[VAL_14]]] : memref<64xf64>
// CHECK-MIR:           }
// CHECK-MIR:           %[[VAL_28:.*]] = memref.tensor_load %[[VAL_11]] : memref<64xf64>
// CHECK-MIR:           return %[[VAL_28]] : tensor<64xf64>
// CHECK-MIR:         }

// CHECK-LIR-LABEL:   func @matvec(
// CHECK-LIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-LIR-SAME:                 %[[VAL_1:.*]]: memref<64xf64>,
// CHECK-LIR-SAME:                 %[[VAL_2:.*]]: memref<64xf64>) -> memref<64xf64> {
// CHECK-LIR:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-LIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-LIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-LIR:           %[[VAL_6:.*]] = call @sparsePointers64(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_7:.*]] = call @sparseIndices64(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-LIR:           %[[VAL_9:.*]] = memref.alloc() : memref<64xf64>
// CHECK-LIR:           scf.for %[[VAL_10:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-LIR:             %[[VAL_11:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_10]]] : memref<64xf64>
// CHECK-LIR:             store %[[VAL_11]], %[[VAL_9]]{{\[}}%[[VAL_10]]] : memref<64xf64>
// CHECK-LIR:           }
// CHECK-LIR:           scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-LIR:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK-LIR:             %[[VAL_14:.*]] = addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK-LIR:             %[[VAL_15:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK-LIR:             %[[VAL_16:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<64xf64>
// CHECK-LIR:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_5]] iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (f64) {
// CHECK-LIR:               %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-LIR:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK-LIR:               %[[VAL_22:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_20]]] : memref<64xf64>
// CHECK-LIR:               %[[VAL_23:.*]] = mulf %[[VAL_21]], %[[VAL_22]] : f64
// CHECK-LIR:               %[[VAL_24:.*]] = addf %[[VAL_19]], %[[VAL_23]] : f64
// CHECK-LIR:               scf.yield %[[VAL_24]] : f64
// CHECK-LIR:             }
// CHECK-LIR:             store %[[VAL_25:.*]], %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<64xf64>
// CHECK-LIR:           }
// CHECK-LIR:           return %[[VAL_9]] : memref<64xf64>
// CHECK-LIR:         }

// CHECK-FAST-LABEL:   func @matvec(
// CHECK-FAST-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-FAST-SAME:                 %[[VAL_1:.*]]: memref<64xf64>,
// CHECK-FAST-SAME:                 %[[VAL_2:.*]]: memref<64xf64>) -> memref<64xf64> {
// CHECK-FAST:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-FAST:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-FAST:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-FAST:           %[[VAL_6:.*]] = call @sparsePointers64(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-FAST:           %[[VAL_7:.*]] = call @sparseIndices64(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-FAST:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-FAST:           scf.for %[[VAL_9:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-FAST:             %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_9]]] : memref<?xindex>
// CHECK-FAST:             %[[VAL_11:.*]] = addi %[[VAL_9]], %[[VAL_5]] : index
// CHECK-FAST:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK-FAST:             %[[VAL_13:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]] : memref<64xf64>
// CHECK-FAST:             %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_5]] iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (f64) {
// CHECK-FAST:               %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-FAST:               %[[VAL_18:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xf64>
// CHECK-FAST:               %[[VAL_19:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_17]]] : memref<64xf64>
// CHECK-FAST:               %[[VAL_20:.*]] = mulf %[[VAL_18]], %[[VAL_19]] : f64
// CHECK-FAST:               %[[VAL_21:.*]] = addf %[[VAL_16]], %[[VAL_20]] : f64
// CHECK-FAST:               scf.yield %[[VAL_21]] : f64
// CHECK-FAST:             }
// CHECK-FAST:             store %[[VAL_22:.*]], %[[VAL_2]]{{\[}}%[[VAL_9]]] : memref<64xf64>
// CHECK-FAST:           }
// CHECK-FAST:           return %[[VAL_2]] : memref<64xf64>
// CHECK-FAST:         }

!SparseTensor = type !llvm.ptr<i8>

func @matvec(%argA: !SparseTensor, %argb: tensor<64xf64>, %argx: tensor<64xf64>) -> tensor<64xf64> {
  %arga = linalg.sparse_tensor %argA : !SparseTensor to tensor<64x64xf64>
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<64x64xf64>, tensor<64xf64>)
      outs(%argx: tensor<64xf64>) {
    ^bb(%A: f64, %b: f64, %x: f64):
      %0 = mulf %A, %b : f64
      %1 = addf %x, %0 : f64
      linalg.yield %1 : f64
  } -> tensor<64xf64>
  return %0 : tensor<64xf64>
}
