// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -canonicalize | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#elemwise = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) OP B(i,j)"
}


// CHECK-LABEL:   func.func @padded_mul(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<4x4xf32, #sparse>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<8x8xf32>) -> tensor<8x8xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant -1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 6 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_9:.*]] = tensor.empty() : tensor<8x8xf32>
// CHECK-DAG:       %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_8]] : f32) outs(%[[VAL_9]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<4x4xf32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<4x4xf32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<4x4xf32, #sparse> to memref<?xf32>
// CHECK-DAG:       %[[VAL_14:.*]] = bufferization.to_memref %[[VAL_10]] :
// CHECK-DAG:       linalg.fill ins(%[[VAL_8]] : f32) outs(%[[VAL_14]] : memref<8x8xf32>)
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_5]] {
// CHECK:             %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %[[VAL_7]] : index
// CHECK:             %[[VAL_17:.*]] = arith.cmpi ult, %[[VAL_15]], %[[VAL_7]] : index
// CHECK:             %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_15]], %[[VAL_3]] : index
// CHECK:             %[[VAL_19:.*]] = arith.ori %[[VAL_17]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_20:.*]]:2 = scf.if %[[VAL_19]] -> (index, index) {
// CHECK:               scf.yield %[[VAL_6]], %[[VAL_6]] : index, index
// CHECK:             } else {
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_15]], %[[VAL_2]] : index
// CHECK:               %[[VAL_23:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK:               scf.yield %[[VAL_21]], %[[VAL_23]] : index, index
// CHECK:             }
// CHECK:             scf.for %[[VAL_24:.*]] = %[[VAL_20]]#0 to %[[VAL_20]]#1 step %[[VAL_5]] {
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_24]]] : memref<?xindex>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_7]] : index
// CHECK:               %[[VAL_28:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// CHECK:               %[[VAL_29:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_15]], %[[VAL_27]]] : tensor<8x8xf32>
// CHECK:               %[[VAL_30:.*]] = arith.mulf %[[VAL_28]], %[[VAL_29]] : f32
// CHECK:               memref.store %[[VAL_30]], %[[VAL_14]]{{\[}}%[[VAL_15]], %[[VAL_27]]] : memref<8x8xf32>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_31:.*]] = bufferization.to_tensor %[[VAL_14]] :
// CHECK:           return %[[VAL_31]] : tensor<8x8xf32>
// CHECK:         }
func.func @padded_mul(%arg0: tensor<4x4xf32, #CSR>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst_0 = arith.constant 0.00000e+00 : f32
  %buf = tensor.empty() : tensor<8x8xf32>
  %s = linalg.fill ins(%cst_0 : f32) outs(%buf : tensor<8x8xf32>) -> tensor<8x8xf32>

  %padded = tensor.pad %arg0 low[2, 2] high[2, 2] {
  ^bb0(%arg75: index, %arg76: index):
    tensor.yield %cst_0 : f32
  } : tensor<4x4xf32, #CSR> to tensor<8x8xf32, #CSR>

  %0 = linalg.generic #elemwise
     ins(%padded, %arg1: tensor<8x8xf32, #CSR>, tensor<8x8xf32>)
    outs(%s: tensor<8x8xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<8x8xf32>

  return %0 : tensor<8x8xf32>
}
