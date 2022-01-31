// RUN: mlir-opt %s -sparsification | FileCheck %s --check-prefix=CHECK-HIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion | \
// RUN: FileCheck %s --check-prefix=CHECK-MIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion \
// RUN: --func-bufferize --arith-bufferize           \
// RUN: --tensor-bufferize --finalizing-bufferize |            \
// RUN: FileCheck %s --check-prefix=CHECK-LIR

#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

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
// CHECK-HIR-SAME:      %[[VAL_0:.*]]: tensor<32x64xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR-SAME:      %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-HIR-SAME:      %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-HIR-DAG:       %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-HIR-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-HIR-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-HIR:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<32x64xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<32x64xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_1]] : memref<64xf64>
// CHECK-HIR:           %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK-HIR:           %[[VAL_11:.*]] = memref.alloc() : memref<32xf64>
// CHECK-HIR:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<32xf64> to memref<32xf64>
// CHECK-HIR:           scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-HIR-DAG:         %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK-HIR-DAG:         %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK-HIR-DAG:         %[[VAL_15:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK-HIR-DAG:         %[[VAL_16:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<32xf64>
// CHECK-HIR:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_5]] iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (f64) {
// CHECK-HIR:               %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-HIR:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK-HIR:               %[[VAL_22:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<64xf64>
// CHECK-HIR:               %[[VAL_23:.*]] = arith.mulf %[[VAL_21]], %[[VAL_22]] : f64
// CHECK-HIR:               %[[VAL_24:.*]] = arith.addf %[[VAL_19]], %[[VAL_23]] : f64
// CHECK-HIR:               scf.yield %[[VAL_24]] : f64
// CHECK-HIR:             }
// CHECK-HIR:             memref.store %[[VAL_17]], %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<32xf64>
// CHECK-HIR:           }
// CHECK-HIR:           %[[VAL_26:.*]] = bufferization.to_tensor %[[VAL_11]] : memref<32xf64>
// CHECK-HIR:           return %[[VAL_26]] : tensor<32xf64>
// CHECK-HIR:         }

// CHECK-MIR-LABEL:   func @matvec(
// CHECK-MIR-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-MIR-SAME:      %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-MIR-SAME:      %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-MIR-DAG:       %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-MIR-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-MIR-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-MIR:           %[[VAL_6:.*]] = call @sparsePointers(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_7:.*]] = call @sparseIndices(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-MIR:           %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_1]] : memref<64xf64>
// CHECK-MIR:           %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK-MIR:           %[[VAL_11:.*]] = memref.alloc() : memref<32xf64>
// CHECK-MIR:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<32xf64> to memref<32xf64>
// CHECK-MIR:           scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-MIR-DAG:         %[[VAL_15:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK-MIR-DAG:         %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_5]] : index
// CHECK-MIR-DAG:         %[[VAL_17:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK-MIR-DAG:         %[[VAL_18:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_14]]] : memref<32xf64>
// CHECK-MIR:             %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_5]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (f64) {
// CHECK-MIR:               %[[VAL_22:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK-MIR:               %[[VAL_23:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xf64>
// CHECK-MIR:               %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_22]]] : memref<64xf64>
// CHECK-MIR:               %[[VAL_25:.*]] = arith.mulf %[[VAL_23]], %[[VAL_24]] : f64
// CHECK-MIR:               %[[VAL_26:.*]] = arith.addf %[[VAL_21]], %[[VAL_25]] : f64
// CHECK-MIR:               scf.yield %[[VAL_26]] : f64
// CHECK-MIR:             }
// CHECK-MIR:             memref.store %[[VAL_19]], %[[VAL_11]]{{\[}}%[[VAL_14]]] : memref<32xf64>
// CHECK-MIR:           }
// CHECK-MIR:           %[[VAL_28:.*]] = bufferization.to_tensor %[[VAL_11]] : memref<32xf64>
// CHECK-MIR:           return %[[VAL_28]] : tensor<32xf64>
// CHECK-MIR:         }

// CHECK-LIR-LABEL:   func @matvec(
// CHECK-LIR-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-LIR-SAME:      %[[VAL_1:.*]]: memref<64xf64>,
// CHECK-LIR-SAME:      %[[VAL_2:.*]]: memref<32xf64>) -> memref<32xf64> {
// CHECK-LIR-DAG:       %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-LIR-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-LIR-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-LIR:           %[[VAL_6:.*]] = call @sparsePointers(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_7:.*]] = call @sparseIndices(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-LIR:           %[[VAL_9:.*]] = memref.alloc() : memref<32xf64>
// CHECK-LIR:           memref.copy %[[VAL_2]], %[[VAL_9]] : memref<32xf64> to memref<32xf64>
// CHECK-LIR:           scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-LIR-DAG:         %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK-LIR-DAG:         %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK-LIR-DAG:         %[[VAL_15:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK-LIR-DAG:         %[[VAL_16:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<32xf64>
// CHECK-LIR:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_5]] iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (f64) {
// CHECK-LIR:               %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-LIR:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK-LIR:               %[[VAL_22:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_20]]] : memref<64xf64>
// CHECK-LIR:               %[[VAL_23:.*]] = arith.mulf %[[VAL_21]], %[[VAL_22]] : f64
// CHECK-LIR:               %[[VAL_24:.*]] = arith.addf %[[VAL_19]], %[[VAL_23]] : f64
// CHECK-LIR:               scf.yield %[[VAL_24]] : f64
// CHECK-LIR:             }
// CHECK-LIR:             memref.store %[[VAL_17]], %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<32xf64>
// CHECK-LIR:           }
// CHECK-LIR:           return %[[VAL_9]] : memref<32xf64>
// CHECK-LIR:         }

func @matvec(%arga: tensor<32x64xf64, #CSR>,
             %argb: tensor<64xf64>,
             %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<32x64xf64, #CSR>, tensor<64xf64>)
      outs(%argx: tensor<32xf64>) {
    ^bb(%A: f64, %b: f64, %x: f64):
      %0 = arith.mulf %A, %b : f64
      %1 = arith.addf %x, %0 : f64
      linalg.yield %1 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}
