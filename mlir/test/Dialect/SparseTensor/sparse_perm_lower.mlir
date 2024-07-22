// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification --canonicalize | FileCheck %s --check-prefix=CHECK-HIR
//
// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification --sparse-tensor-conversion --canonicalize | \
// RUN: FileCheck %s --check-prefix=CHECK-MIR

#X = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : dense, d1 : dense)
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j,k) -> (k,i,j)>,  // A (in)
    affine_map<(i,j,k) -> ()>        // X (out)
  ],
  iterator_types = ["reduction", "reduction", "reduction"]
}

// CHECK-HIR-LABEL:   func @sparse_dynamic_dims(
// CHECK-HIR-SAME:      %[[VAL_0:.*]]: tensor<?x?x?xf32, #sparse{{[0-9]*}}>,
// CHECK-HIR-SAME:      %[[VAL_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK-HIR-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-HIR-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-HIR-DAG:       %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-HIR:           %[[DEMAP:. *]] = sparse_tensor.reinterpret_map %[[VAL_0]]
// CHECK-HIR-DAG:       %[[VAL_5:.*]] = sparse_tensor.lvl %[[DEMAP]], %[[VAL_3]] : tensor<?x?x?xf32, #sparse{{[0-9]*}}>
// CHECK-HIR-DAG:       %[[VAL_6:.*]] = sparse_tensor.lvl %[[DEMAP]], %[[VAL_2]] : tensor<?x?x?xf32, #sparse{{[0-9]*}}>
// CHECK-HIR-DAG:       %[[VAL_7:.*]] = sparse_tensor.lvl %[[DEMAP]], %[[VAL_4]] : tensor<?x?x?xf32, #sparse{{[0-9]*}}>
// CHECK-HIR-DAG:       %[[VAL_8:.*]] = sparse_tensor.values %[[DEMAP]] : tensor<?x?x?xf32, #sparse{{[0-9]*}}>
// CHECK-HIR-DAG:       %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_1]] : memref<f32>
// CHECK-HIR:           %[[VAL_11:.*]] = tensor.extract %[[VAL_1]][] : tensor<f32>
// CHECK-HIR:           %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (f32) {
// CHECK-HIR:             %[[VAL_18:.*]] = arith.muli %[[VAL_13]], %[[VAL_6]] : index
// CHECK-HIR:             %[[VAL_15:.*]] = scf.for %[[VAL_16:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_2]] iter_args(%[[VAL_17:.*]] = %[[VAL_14]]) -> (f32) {
// CHECK-HIR:               %[[VAL_19:.*]] = arith.addi %[[VAL_16]], %[[VAL_18]] : index
// CHECK-HIR:               %[[VAL_23:.*]] = arith.muli %[[VAL_19]], %[[VAL_7]] : index
// CHECK-HIR:               %[[VAL_20:.*]] = scf.for %[[VAL_21:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_2]] iter_args(%[[VAL_22:.*]] = %[[VAL_17]]) -> (f32) {
// CHECK-HIR:                 %[[VAL_24:.*]] = arith.addi %[[VAL_21]], %[[VAL_23]] : index
// CHECK-HIR:                 %[[VAL_25:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// CHECK-HIR:                 %[[VAL_26:.*]] = arith.addf %[[VAL_22]], %[[VAL_25]] : f32
// CHECK-HIR:                 scf.yield %[[VAL_26]] : f32
// CHECK-HIR:               }
// CHECK-HIR:               scf.yield %[[VAL_20]] : f32
// CHECK-HIR:             }
// CHECK-HIR:             scf.yield %[[VAL_15]] : f32
// CHECK-HIR:           }
// CHECK-HIR:           memref.store %[[VAL_12]], %[[VAL_10]][] : memref<f32>
// CHECK-HIR:           %[[VAL_30:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<f32>
// CHECK-HIR:           return %[[VAL_30]] : tensor<f32>
// CHECK-HIR:         }
//
// CHECK-MIR-LABEL:   func @sparse_dynamic_dims(
// CHECK-MIR-SAME:      %[[ARGA:.*]]: !llvm.ptr,
// CHECK-MIR-SAME:      %[[ARGX:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK-MIR-DAG:       %[[I0:.*]] = arith.constant 0 : index
// CHECK-MIR-DAG:       %[[I1:.*]] = arith.constant 1 : index
// CHECK-MIR-DAG:       %[[I2:.*]] = arith.constant 2 : index
// CHECK-MIR-DAG:       %[[DimSize0:.*]] = call @sparseLvlSize(%[[ARGA]], %[[I0]])
// CHECK-MIR-DAG:       %[[DimSize1:.*]] = call @sparseLvlSize(%[[ARGA]], %[[I1]])
// CHECK-MIR-DAG:       %[[DimSize2:.*]] = call @sparseLvlSize(%[[ARGA]], %[[I2]])
// CHECK-MIR-DAG:       %[[VAL_8:.*]] = call @sparseValuesF32(%[[ARGA]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-MIR-DAG:       %[[VAL_10:.*]] = bufferization.to_memref %[[ARGX]] : memref<f32>
// CHECK-MIR:           %[[VAL_11:.*]] = tensor.extract %[[ARGX]][] : tensor<f32>
// CHECK-MIR:           %[[VAL_12:.*]] = scf.for %[[D2:.*]] = %[[I0]] to %[[DimSize0]] step %[[I1]] iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (f32) {
// CHECK-MIR:             %[[VAL_18:.*]] = arith.muli %[[D2]], %[[DimSize1]] : index
// CHECK-MIR:             %[[VAL_15:.*]] = scf.for %[[D0:.*]] = %[[I0]] to %[[DimSize1]] step %[[I1]] iter_args(%[[VAL_17:.*]] = %[[VAL_14]]) -> (f32) {
// CHECK-MIR:               %[[VAL_19:.*]] = arith.addi %[[D0]], %[[VAL_18]] : index
// CHECK-MIR:               %[[VAL_23:.*]] = arith.muli %[[VAL_19]], %[[DimSize2]] : index
// CHECK-MIR:               %[[VAL_20:.*]] = scf.for %[[D1:.*]] = %[[I0]] to %[[DimSize2]] step %[[I1]] iter_args(%[[VAL_22:.*]] = %[[VAL_17]]) -> (f32) {
// CHECK-MIR:                 %[[VAL_24:.*]] = arith.addi %[[D1]], %[[VAL_23]] : index
// CHECK-MIR:                 %[[VAL_25:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// CHECK-MIR:                 %[[VAL_26:.*]] = arith.addf %[[VAL_22]], %[[VAL_25]] : f32
// CHECK-MIR:                 scf.yield %[[VAL_26]] : f32
// CHECK-MIR:               }
// CHECK-MIR:               scf.yield %[[VAL_20]] : f32
// CHECK-MIR:             }
// CHECK-MIR:             scf.yield %[[VAL_15]] : f32
// CHECK-MIR:           }
// CHECK-MIR:           memref.store %[[VAL_12]], %[[VAL_10]][] : memref<f32>
// CHECK-MIR:           %[[VAL_30:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<f32>
// CHECK-MIR:           return %[[VAL_30]] : tensor<f32>
// CHECK-MIR:         }
func.func @sparse_dynamic_dims(%arga: tensor<?x?x?xf32, #X>,
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
