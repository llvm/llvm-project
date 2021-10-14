// RUN: mlir-opt %s -sparsification --canonicalize | FileCheck %s --check-prefix=CHECK-HIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion --canonicalize | \
// RUN: FileCheck %s --check-prefix=CHECK-MIR

#X = #sparse_tensor.encoding<{
 dimLevelType = [ "dense", "dense", "dense" ],
 dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j,k) -> (k,i,j)>,  // A (in)
    affine_map<(i,j,k) -> ()>        // X (out)
  ],
  iterator_types = ["reduction", "reduction", "reduction"]
}

// CHECK-HIR-LABEL:   func @sparse_dynamic_dims(
// CHECK-HIR-SAME:                                      %[[VAL_0:.*]]: tensor<?x?x?xf32,  #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-HIR-SAME:                                      %[[VAL_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK-HIR-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-HIR-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-HIR-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-HIR:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[C2]] : tensor<?x?x?xf32,  #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?x?x?xf32,  #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?x?xf32,  #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?x?xf32,  #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-HIR:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<f32>
// CHECK-HIR:           %[[VAL_10:.*]] = memref.alloc() : memref<f32>
// CHECK-HIR:           memref.copy %[[VAL_9]], %[[VAL_10]] : memref<f32> to memref<f32>
// CHECK-HIR:           scf.for %[[VAL_11:.*]] = %[[C0]] to %[[VAL_5]] step %[[C1]] {
// CHECK-HIR:             scf.for %[[VAL_12:.*]] = %[[C0]] to %[[VAL_6]] step %[[C1]] {
// CHECK-HIR:               %[[VAL_13:.*]] = arith.muli %[[VAL_6]], %[[VAL_11]] : index
// CHECK-HIR:               %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : index
// CHECK-HIR:               %[[VAL_15:.*]] = memref.load %[[VAL_10]][] : memref<f32>
// CHECK-HIR:               %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[C0]] to %[[VAL_7]] step %[[C1]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (f32) {
// CHECK-HIR:                 %[[VAL_19:.*]] = arith.muli %[[VAL_7]], %[[VAL_14]] : index
// CHECK-HIR:                 %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_17]] : index
// CHECK-HIR:                 %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xf32>
// CHECK-HIR:                 %[[VAL_22:.*]] = arith.addf %[[VAL_18]], %[[VAL_21]] : f32
// CHECK-HIR:                 scf.yield %[[VAL_22]] : f32
// CHECK-HIR:               }
// CHECK-HIR:               memref.store %[[VAL_23:.*]], %[[VAL_10]][] : memref<f32>
// CHECK-HIR:             }
// CHECK-HIR:           }
// CHECK-HIR:           %[[VAL_24:.*]] = memref.tensor_load %[[VAL_10]] : memref<f32>
// CHECK-HIR:           return %[[VAL_24]] : tensor<f32>
// CHECK-HIR:         }
//
// CHECK-MIR-LABEL:   func @sparse_dynamic_dims(
// CHECK-MIR-SAME:                                      %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-MIR-SAME:                                      %[[VAL_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK-MIR-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-MIR-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-MIR-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-MIR:           %[[VAL_5:.*]] = call @sparseDimSize(%[[VAL_0]], %[[C0]]) : (!llvm.ptr<i8>, index) -> index
// CHECK-MIR:           %[[VAL_6:.*]] = call @sparseDimSize(%[[VAL_0]], %[[C1]]) : (!llvm.ptr<i8>, index) -> index
// CHECK-MIR:           %[[VAL_7:.*]] = call @sparseDimSize(%[[VAL_0]], %[[C2]]) : (!llvm.ptr<i8>, index) -> index
// CHECK-MIR:           %[[VAL_8:.*]] = call @sparseValuesF32(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf32>
// CHECK-MIR:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<f32>
// CHECK-MIR:           %[[VAL_10:.*]] = memref.alloc() : memref<f32>
// CHECK-MIR:           memref.copy %[[VAL_9]], %[[VAL_10]] : memref<f32> to memref<f32>
// CHECK-MIR:           scf.for %[[VAL_11:.*]] = %[[C0]] to %[[VAL_5]] step %[[C1]] {
// CHECK-MIR:             scf.for %[[VAL_12:.*]] = %[[C0]] to %[[VAL_6]] step %[[C1]] {
// CHECK-MIR:               %[[VAL_13:.*]] = arith.muli %[[VAL_6]], %[[VAL_11]] : index
// CHECK-MIR:               %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : index
// CHECK-MIR:               %[[VAL_15:.*]] = memref.load %[[VAL_10]][] : memref<f32>
// CHECK-MIR:               %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[C0]] to %[[VAL_7]] step %[[C1]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (f32) {
// CHECK-MIR:                 %[[VAL_19:.*]] = arith.muli %[[VAL_7]], %[[VAL_14]] : index
// CHECK-MIR:                 %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_17]] : index
// CHECK-MIR:                 %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xf32>
// CHECK-MIR:                 %[[VAL_22:.*]] = arith.addf %[[VAL_18]], %[[VAL_21]] : f32
// CHECK-MIR:                 scf.yield %[[VAL_22]] : f32
// CHECK-MIR:               }
// CHECK-MIR:               memref.store %[[VAL_23:.*]], %[[VAL_10]][] : memref<f32>
// CHECK-MIR:             }
// CHECK-MIR:           }
// CHECK-MIR:           %[[VAL_24:.*]] = memref.tensor_load %[[VAL_10]] : memref<f32>
// CHECK-MIR:           return %[[VAL_24]] : tensor<f32>
// CHECK-MIR:         }
func @sparse_dynamic_dims(%arga: tensor<?x?x?xf32, #X>,
                          %argx: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic #trait
    ins(%arga: tensor<?x?x?xf32, #X>)
    outs(%argx: tensor<f32>) {
      ^bb(%a : f32, %x: f32):
        %0 = arith.addf %x, %a : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
