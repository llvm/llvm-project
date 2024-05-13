// RUN: mlir-opt %s --linalg-fuse-elementwise-ops \
// RUN:             --sparsification-and-bufferization | FileCheck %s

#Sparse = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed),
  explicitVal = 1.0 : f32
}>

#trait3p = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>,  // A
    affine_map<(i,j,k) -> (i,j,k)>,  // B
    affine_map<(i,j,k) -> (i,j,k)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
}

#trait3r = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>,  // A
    affine_map<(i,j,k) -> ()>        // X (out)
  ],
  iterator_types = ["reduction", "reduction", "reduction"]
}

//
// Make sure X += A * A => X += 1 in single loop.
//
//
// CHECK-LABEL:   func.func @sum_squares(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf32>,
// CHECK-SAME:      %[[VAL_3:.*3]]: !sparse_tensor.storage_specifier<#{{.*}}>) -> memref<f32> {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK:           linalg.fill ins(%[[VAL_9]] : f32) outs(%[[VAL_10]] : memref<f32>)
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_10]][] : memref<f32>
// CHECK:           %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_5]] iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (f32) {
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_13]], %[[VAL_7]] : index
// CHECK:             %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_5]] iter_args(%[[VAL_18:.*]] = %[[VAL_14]]) -> (f32) {
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_15]] : index
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:               %[[VAL_21:.*]] = arith.addi %[[VAL_19]], %[[VAL_5]] : index
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:               %[[VAL_23:.*]] = scf.for %[[VAL_24:.*]] = %[[VAL_20]] to %[[VAL_22]] step %[[VAL_5]] iter_args(%[[VAL_25:.*]] = %[[VAL_18]]) -> (f32) {
// CHECK:                 %[[VAL_26:.*]] = arith.addf %[[VAL_25]], %[[VAL_4]] : f32
// CHECK:                 scf.yield %[[VAL_26]] : f32
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               scf.yield %[[VAL_23]] : f32
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             scf.yield %[[VAL_16]] : f32
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           memref.store %[[VAL_12]], %[[VAL_10]][] : memref<f32>
// CHECK:           return %[[VAL_10]] : memref<f32>
// CHECK:         }
//
func.func @sum_squares(%a: tensor<2x3x8xf32, #Sparse>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x3x8xf32>
  %1 = linalg.generic #trait3p
      ins(%a, %a : tensor<2x3x8xf32, #Sparse>, tensor<2x3x8xf32, #Sparse>)
      outs(%0 : tensor<2x3x8xf32>) {
        ^bb0(%in1: f32, %in2: f32, %out: f32):
          %mul = arith.mulf %in1, %in2 : f32
          linalg.yield %mul : f32
      } -> tensor<2x3x8xf32>
  %2 = tensor.empty() : tensor<f32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
  %4 = linalg.generic #trait3r
      ins(%1 : tensor<2x3x8xf32>)
      outs(%3 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %add = arith.addf %in, %out : f32
          linalg.yield %add : f32
      } -> tensor<f32>

  return %4 : tensor<f32>
}

//
// Make sure X += A * B => X += B in single loop.
//
// CHECK-LABEL:   func.func @sum_products(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf32>,
// CHECK-SAME:      %[[VAL_3:.*3]]: !sparse_tensor.storage_specifier<#{{.*}}>,
// CHECK-SAME:      %[[VAL_4:.*4]]: memref<2x3x8xf32>) -> memref<f32> {
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK:           linalg.fill ins(%[[VAL_9]] : f32) outs(%[[VAL_10]] : memref<f32>)
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_10]][] : memref<f32>
// CHECK:           %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_5]] iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (f32) {
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_13]], %[[VAL_7]] : index
// CHECK:             %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_5]] iter_args(%[[VAL_18:.*]] = %[[VAL_14]]) -> (f32) {
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_15]] : index
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:               %[[VAL_21:.*]] = arith.addi %[[VAL_19]], %[[VAL_5]] : index
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:               %[[VAL_23:.*]] = scf.for %[[VAL_24:.*]] = %[[VAL_20]] to %[[VAL_22]] step %[[VAL_5]] iter_args(%[[VAL_25:.*]] = %[[VAL_18]]) -> (f32) {
// CHECK:                 %[[VAL_26:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_24]]] : memref<?xindex>
// CHECK:                 %[[VAL_27:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_13]], %[[VAL_17]], %[[VAL_26]]] : memref<2x3x8xf32>
// CHECK:                 %[[VAL_28:.*]] = arith.addf %[[VAL_27]], %[[VAL_25]] : f32
// CHECK:                 scf.yield %[[VAL_28]] : f32
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               scf.yield %[[VAL_23]] : f32
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             scf.yield %[[VAL_16]] : f32
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           memref.store %[[VAL_12]], %[[VAL_10]][] : memref<f32>
// CHECK:           return %[[VAL_10]] : memref<f32>
// CHECK:         }
//
func.func @sum_products(%a: tensor<2x3x8xf32, #Sparse>, %b: tensor<2x3x8xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x3x8xf32>
  %1 = linalg.generic #trait3p
      ins(%a, %b : tensor<2x3x8xf32, #Sparse>, tensor<2x3x8xf32>)
      outs(%0 : tensor<2x3x8xf32>) {
        ^bb0(%in1: f32, %in2: f32, %out: f32):
          %mul = arith.mulf %in1, %in2 : f32
          linalg.yield %mul : f32
      } -> tensor<2x3x8xf32>
  %2 = tensor.empty() : tensor<f32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
  %4 = linalg.generic #trait3r
      ins(%1 : tensor<2x3x8xf32>)
      outs(%3 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %add = arith.addf %in, %out : f32
          linalg.yield %add : f32
      } -> tensor<f32>

  return %4 : tensor<f32>
}
