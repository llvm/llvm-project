// RUN: mlir-opt %s --linalg-fuse-elementwise-ops --sparse-reinterpret-map --sparsification | FileCheck %s

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // A
    affine_map<(i) -> (i)>  // B (out)
  ],
  iterator_types = ["parallel"],
  doc = "B(i) = OP A(i)"
}


// CHECK-LABEL:   func.func @sparse_fusion(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<100xf64, #sparse>) -> tensor<100xf64> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 100 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1.000000e+02 : f64
// CHECK-DAG:       %[[VAL_8:.*]] = tensor.empty() : tensor<100xf64>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<100xf64, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<100xf64, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<100xf64, #sparse> to memref<?xf64>
// CHECK-DAG:       %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_8]] :
// CHECK-DAG:        linalg.fill ins(%[[VAL_4]] : f64) outs(%[[VAL_12]] : memref<100xf64>)
// CHECK-DAG:        %[[VAL_13:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-DAG:        %[[VAL_14:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_15:.*]]:2 = scf.while (%[[VAL_16:.*]] = %[[VAL_13]], %[[VAL_17:.*]] = %[[VAL_3]]) : (index, index) -> (index, index) {
// CHECK:             %[[VAL_18:.*]] = arith.cmpi ult, %[[VAL_16]], %[[VAL_14]] : index
// CHECK:             scf.condition(%[[VAL_18]]) %[[VAL_16]], %[[VAL_17]] : index, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_19:.*]]: index, %[[VAL_20:.*]]: index):
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:             %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_20]] : index
// CHECK:             scf.if %[[VAL_22]] {
// CHECK:               %[[VAL_23:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<?xf64>
// CHECK:               %[[VAL_24:.*]] = arith.addf %[[VAL_23]], %[[VAL_6]] : f64
// CHECK:               %[[VAL_25:.*]] = math.exp %[[VAL_24]] : f64
// CHECK:               %[[VAL_26:.*]] = arith.maximumf %[[VAL_25]], %[[VAL_7]] : f64
// CHECK:               memref.store %[[VAL_26]], %[[VAL_12]]{{\[}}%[[VAL_20]]] : memref<100xf64>
// CHECK:             } else {
// CHECK:               scf.if %[[VAL_1]] {
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_20]]] : memref<100xf64>
// CHECK:               } else {
// CHECK:               }
// CHECK:             }
// CHECK:             %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_20]] : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_19]], %[[VAL_2]] : index
// CHECK:             %[[VAL_29:.*]] = arith.select %[[VAL_27]], %[[VAL_28]], %[[VAL_19]] : index
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_20]], %[[VAL_2]] : index
// CHECK:             scf.yield %[[VAL_29]], %[[VAL_30]] : index, index
// CHECK:           }
// CHECK:           scf.for %[[VAL_31:.*]] = %[[VAL_32:.*]]#1 to %[[VAL_5]] step %[[VAL_2]] {
// CHECK:             memref.store %[[VAL_7]], %[[VAL_12]]{{\[}}%[[VAL_31]]] : memref<100xf64>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = bufferization.to_tensor %[[VAL_12]] :
// CHECK:           return %[[VAL_33]] : tensor<100xf64>
// CHECK:         }
func.func @sparse_fusion(%argA: tensor<100xf64, #SV>) -> tensor<100xf64> {
  %c1 = arith.constant 1.0 : f64
  %c100 = arith.constant 100.0 : f64

  %t0 = tensor.empty() : tensor<100xf64>
  %l0 = linalg.generic #trait
      ins(%argA: tensor<100xf64, #SV>) outs(%t0: tensor<100xf64>) {
    ^bb0(%in0: f64, %out0: f64):
      %b0 = arith.addf %in0, %c1 : f64
      linalg.yield %b0 : f64
  } -> tensor<100xf64>
  %t1 = tensor.empty() : tensor<100xf64>
  %l1 = linalg.generic #trait
      ins(%l0: tensor<100xf64>) outs(%t1: tensor<100xf64>) {
    ^bb0(%in1: f64, %out1: f64):
      %b1 = math.exp %in1 : f64
      linalg.yield %b1 : f64
  } -> tensor<100xf64>
  %t2 = tensor.empty() : tensor<100xf64>
  %l2 = linalg.generic #trait
      ins(%l1: tensor<100xf64>) outs(%t2: tensor<100xf64>) {
    ^bb0(%in2: f64, %out2: f64):
      %b2 = arith.maximumf %in2, %c100 : f64
      linalg.yield %b2 : f64
  } -> tensor<100xf64>

  return %l2 : tensor<100xf64>
}
