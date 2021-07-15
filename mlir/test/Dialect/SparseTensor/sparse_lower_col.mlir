// RUN: mlir-opt %s -sparsification | FileCheck %s --check-prefix=CHECK-HIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion | \
// RUN: FileCheck %s --check-prefix=CHECK-MIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion \
// RUN: --func-bufferize --tensor-constant-bufferize           \
// RUN: --tensor-bufferize --finalizing-bufferize |            \
// RUN: FileCheck %s --check-prefix=CHECK-LIR

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

#trait_matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel","reduction"],
  doc = "x(i) += A(i,j) * b(j)"
}

// CHECK-HIR-LABEL:   func @matvec(
// CHECK-HIR-SAME:                 %[[VAL_0:.*]]: tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>,
// CHECK-HIR-SAME:                 %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-HIR-SAME:                 %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-HIR:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-HIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-HIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-HIR:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK-HIR:           %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK-HIR:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
// CHECK-HIR:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<64xf64>
// CHECK-HIR:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_2]] : memref<32xf64>
// CHECK-HIR:           %[[VAL_11:.*]] = memref.alloc() : memref<32xf64>
// CHECK-HIR:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<32xf64> to memref<32xf64>
// CHECK-HIR:           scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-HIR:             %[[VAL_13:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<64xf64>
// CHECK-HIR:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK-HIR:             %[[VAL_15:.*]] = addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK-HIR:             %[[VAL_16:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-HIR:             scf.for %[[VAL_17:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_5]] {
// CHECK-HIR:               %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK-HIR:               %[[VAL_19:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_18]]] : memref<32xf64>
// CHECK-HIR:               %[[VAL_20:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK-HIR:               %[[VAL_21:.*]] = mulf %[[VAL_20]], %[[VAL_13]] : f64
// CHECK-HIR:               %[[VAL_22:.*]] = addf %[[VAL_19]], %[[VAL_21]] : f64
// CHECK-HIR:               memref.store %[[VAL_22]], %[[VAL_11]]{{\[}}%[[VAL_18]]] : memref<32xf64>
// CHECK-HIR:             }
// CHECK-HIR:           }
// CHECK-HIR:           %[[VAL_23:.*]] = memref.tensor_load %[[VAL_11]] : memref<32xf64>
// CHECK-HIR:           return %[[VAL_23]] : tensor<32xf64>
// CHECK-HIR:         }

// CHECK-MIR-LABEL:   func @matvec(
// CHECK-MIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-MIR-SAME:                 %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-MIR-SAME:                 %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-MIR:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-MIR:           %[[VAL_5:.*]] = constant 0 : index
// CHECK-MIR:           %[[VAL_6:.*]] = constant 1 : index
// CHECK-MIR:           %[[VAL_7:.*]] = call @sparsePointers(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_8:.*]] = call @sparseIndices(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_9:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-MIR:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_1]] : memref<64xf64>
// CHECK-MIR:           %[[VAL_11:.*]] = memref.buffer_cast %[[VAL_2]] : memref<32xf64>
// CHECK-MIR:           %[[VAL_12:.*]] = memref.alloc() : memref<32xf64>
// CHECK-MIR:           memref.copy %[[VAL_11]], %[[VAL_12]] : memref<32xf64> to memref<32xf64>
// CHECK-MIR:           scf.for %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK-MIR:             %[[VAL_16:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_15]]] : memref<64xf64>
// CHECK-MIR:             %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-MIR:             %[[VAL_18:.*]] = addi %[[VAL_15]], %[[VAL_6]] : index
// CHECK-MIR:             %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-MIR:             scf.for %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_6]] {
// CHECK-MIR:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK-MIR:               %[[VAL_22:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_21]]] : memref<32xf64>
// CHECK-MIR:               %[[VAL_23:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<?xf64>
// CHECK-MIR:               %[[VAL_24:.*]] = mulf %[[VAL_23]], %[[VAL_16]] : f64
// CHECK-MIR:               %[[VAL_25:.*]] = addf %[[VAL_22]], %[[VAL_24]] : f64
// CHECK-MIR:               memref.store %[[VAL_25]], %[[VAL_12]]{{\[}}%[[VAL_21]]] : memref<32xf64>
// CHECK-MIR:             }
// CHECK-MIR:           }
// CHECK-MIR:           %[[VAL_26:.*]] = memref.tensor_load %[[VAL_12]] : memref<32xf64>
// CHECK-MIR:           return %[[VAL_26]] : tensor<32xf64>
// CHECK-MIR:         }

// CHECK-LIR-LABEL:   func @matvec(
// CHECK-LIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-LIR-SAME:                 %[[VAL_1:.*]]: memref<64xf64>,
// CHECK-LIR-SAME:                 %[[VAL_2:.*]]: memref<32xf64>) -> memref<32xf64> {
// CHECK-LIR:           %[[VAL_3:.*]] = constant 64 : index
// CHECK-LIR:           %[[VAL_5:.*]] = constant 0 : index
// CHECK-LIR:           %[[VAL_6:.*]] = constant 1 : index
// CHECK-LIR:           %[[VAL_7:.*]] = call @sparsePointers(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_8:.*]] = call @sparseIndices(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_9:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-LIR:           %[[VAL_10:.*]] = memref.alloc() : memref<32xf64>
// CHECK-LIR:           memref.copy %[[VAL_2]], %[[VAL_10]] : memref<32xf64> to memref<32xf64>
// CHECK-LIR:           scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK-LIR:             %[[VAL_14:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_13]]] : memref<64xf64>
// CHECK-LIR:             %[[VAL_15:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK-LIR:             %[[VAL_16:.*]] = addi %[[VAL_13]], %[[VAL_6]] : index
// CHECK-LIR:             %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK-LIR:             scf.for %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_6]] {
// CHECK-LIR:               %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-LIR:               %[[VAL_20:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK-LIR:               %[[VAL_21:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK-LIR:               %[[VAL_22:.*]] = mulf %[[VAL_21]], %[[VAL_14]] : f64
// CHECK-LIR:               %[[VAL_23:.*]] = addf %[[VAL_20]], %[[VAL_22]] : f64
// CHECK-LIR:               memref.store %[[VAL_23]], %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK-LIR:             }
// CHECK-LIR:           }
// CHECK-LIR:           return %[[VAL_10]] : memref<32xf64>
// CHECK-LIR:         }

func @matvec(%arga: tensor<32x64xf64, #CSC>,
             %argb: tensor<64xf64>,
             %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<32x64xf64, #CSC>, tensor<64xf64>)
      outs(%argx: tensor<32xf64>) {
    ^bb(%A: f64, %b: f64, %x: f64):
      %0 = mulf %A, %b : f64
      %1 = addf %x, %0 : f64
      linalg.yield %1 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}
