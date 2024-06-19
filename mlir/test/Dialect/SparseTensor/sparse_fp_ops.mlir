// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait1 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = OP a(i)"
}

#trait2 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}

#traitc = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP c"
}

// CHECK-LABEL: func @abs(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK-DAG:     %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK:         %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:         %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] {
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = math.absf %[[VAL_12]] : f64
// CHECK:             memref.store %[[VAL_13]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<32xf64>
// CHECK:         return %[[VAL_14]] : tensor<32xf64>
// CHECK:       }
func.func @abs(%arga: tensor<32xf64, #SV>,
               %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait1
     ins(%arga: tensor<32xf64, #SV>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = math.absf %a : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @ceil(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK-DAG:     %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK:         %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:         %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] {
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = math.ceil %[[VAL_12]] : f64
// CHECK:           memref.store %[[VAL_13]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<32xf64>
// CHECK:         return %[[VAL_14]] : tensor<32xf64>
// CHECK:       }
func.func @ceil(%arga: tensor<32xf64, #SV>,
                %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait1
     ins(%arga: tensor<32xf64, #SV>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = math.ceil %a : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @floor(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK-DAG:     %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK:         %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:         %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] {
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = math.floor %[[VAL_12]] : f64
// CHECK:           memref.store %[[VAL_13]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<32xf64>
// CHECK:         return %[[VAL_14]] : tensor<32xf64>
// CHECK:       }
func.func @floor(%arga: tensor<32xf64, #SV>,
                 %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait1
     ins(%arga: tensor<32xf64, #SV>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = math.floor %a : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @neg(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK-DAG:     %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK:         %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:         %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] {
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = arith.negf %[[VAL_12]] : f64
// CHECK:           memref.store %[[VAL_13]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<32xf64>
// CHECK:         return %[[VAL_14]] : tensor<32xf64>
// CHECK:       }
func.func @neg(%arga: tensor<32xf64, #SV>,
               %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait1
     ins(%arga: tensor<32xf64, #SV>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = arith.negf %a : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @add(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_5:.*]] = arith.constant true
// CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK-DAG:     %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK:         %[[VAL_12:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:         %[[VAL_14:.*]]:2 = scf.while (%[[VAL_15:.*]] = %[[VAL_12]], %[[VAL_16:.*]] = %[[VAL_4]]) : (index, index) -> (index, index) {
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ult, %[[VAL_15]], %[[VAL_13]] : index
// CHECK:           scf.condition(%[[VAL_17]]) %[[VAL_15]], %[[VAL_16]] : index, index
// CHECK:         } do {
// CHECK:         ^bb0(%[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_19]] : index
// CHECK:           scf.if %[[VAL_21]] {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:             %[[VAL_24:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : f64
// CHECK:             memref.store %[[VAL_24]], %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:           } else {
// CHECK:             scf.if %[[VAL_5]] {
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:               memref.store %[[VAL_25]], %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:             } else {
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_19]] : index
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_18]], %[[VAL_6]] : index
// CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_26]], %[[VAL_27]], %[[VAL_18]] : index
// CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_19]], %[[VAL_6]] : index
// CHECK:           scf.yield %[[VAL_28]], %[[VAL_29]] : index, index
// CHECK:         }
// CHECK:         scf.for %[[VAL_30:.*]] = %[[VAL_31:.*]]#1 to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:           %[[VAL_32:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_30]]] : memref<32xf64>
// CHECK:           memref.store %[[VAL_32]], %[[VAL_11]]{{\[}}%[[VAL_30]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_33:.*]] = bufferization.to_tensor %[[VAL_11]] : memref<32xf64>
// CHECK:         return %[[VAL_33]] : tensor<32xf64>
// CHECK:       }
func.func @add(%arga: tensor<32xf64, #SV>,
               %argb: tensor<32xf64>,
               %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait2
     ins(%arga, %argb: tensor<32xf64, #SV>, tensor<32xf64>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.addf %a, %b : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @sub(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_5:.*]] = arith.constant true
// CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:     %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK-DAG:     %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK-DAG:     %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK:         %[[VAL_12:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:         %[[VAL_14:.*]]:2 = scf.while (%[[VAL_15:.*]] = %[[VAL_12]], %[[VAL_16:.*]] = %[[VAL_4]]) : (index, index) -> (index, index) {
// CHECK:         %[[VAL_17:.*]] = arith.cmpi ult, %[[VAL_15]], %[[VAL_13]] : index
// CHECK:         scf.condition(%[[VAL_17]]) %[[VAL_15]], %[[VAL_16]] : index, index
// CHECK:         } do {
// CHECK:         ^bb0(%[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_19]] : index
// CHECK:           scf.if %[[VAL_21]] {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:             %[[VAL_24:.*]] = arith.subf %[[VAL_22]], %[[VAL_23]] : f64
// CHECK:             memref.store %[[VAL_24]], %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:           } else {
// CHECK:             scf.if %[[VAL_5]] {
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:               %[[VAL_26:.*]] = arith.negf %[[VAL_25]] : f64
// CHECK:               memref.store %[[VAL_26]], %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<32xf64>
// CHECK:             } else {
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_19]] : index
// CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_18]], %[[VAL_6]] : index
// CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_27]], %[[VAL_28]], %[[VAL_18]] : index
// CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_19]], %[[VAL_6]] : index
// CHECK:           scf.yield %[[VAL_29]], %[[VAL_30]] : index, index
// CHECK:         }
// CHECK:         scf.for %[[VAL_31:.*]] = %[[VAL_32:.*]]#1 to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:           %[[VAL_33:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_31]]] : memref<32xf64>
// CHECK:           %[[VAL_34:.*]] = arith.negf %[[VAL_33]] : f64
// CHECK:           memref.store %[[VAL_34]], %[[VAL_11]]{{\[}}%[[VAL_31]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_35:.*]] = bufferization.to_tensor %[[VAL_11]] : memref<32xf64>
// CHECK:         return %[[VAL_35]] : tensor<32xf64>
// CHECK:       }
func.func @sub(%arga: tensor<32xf64, #SV>,
               %argb: tensor<32xf64>,
               %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait2
     ins(%arga, %argb: tensor<32xf64, #SV>, tensor<32xf64>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.subf %a, %b : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @mul(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK-DAG:     %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK:         %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_12:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_4]] {
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_12]]] : memref<?xf64>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_13]]] : memref<32xf64>
// CHECK:           %[[VAL_16:.*]] = arith.mulf %[[VAL_14]], %[[VAL_15]] : f64
// CHECK:           memref.store %[[VAL_16]], %[[VAL_9]]{{\[}}%[[VAL_13]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_17:.*]] = bufferization.to_tensor %[[VAL_9]] : memref<32xf64>
// CHECK:         return %[[VAL_17]] : tensor<32xf64>
// CHECK:       }
func.func @mul(%arga: tensor<32xf64, #SV>,
               %argb: tensor<32xf64>,
               %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait2
     ins(%arga, %argb: tensor<32xf64, #SV>, tensor<32xf64>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @divbyc(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:     %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32xf64>
// CHECK:         %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_4]] {
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:           %[[VAL_14:.*]] = arith.divf %[[VAL_13]], %[[VAL_2]] : f64
// CHECK:           memref.store %[[VAL_14]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<32xf64>
// CHECK:         return %[[VAL_15]] : tensor<32xf64>
// CHECK:       }
func.func @divbyc(%arga: tensor<32xf64, #SV>,
                  %argx: tensor<32xf64>) -> tensor<32xf64> {
  %c = arith.constant 2.0 : f64
  %0 = linalg.generic #traitc
     ins(%arga: tensor<32xf64, #SV>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = arith.divf %a, %c : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL:   func.func @zero_preserving_math(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32xf64, #sparse{{[0-9]*}}>) -> tensor<32xf64, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = tensor.empty() : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK:           %[[VAL_7:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[T:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_2]] {{.*}} {
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<?xindex>
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_9]]] : memref<?xf64>
// CHECK:             %[[VAL_12:.*]] = math.absf %[[VAL_11]] : f64
// CHECK:             %[[VAL_13:.*]] = math.ceil %[[VAL_12]] : f64
// CHECK:             %[[VAL_14:.*]] = math.floor %[[VAL_13]] : f64
// CHECK:             %[[VAL_15:.*]] = math.sqrt %[[VAL_14]] : f64
// CHECK:             %[[VAL_16:.*]] = math.expm1 %[[VAL_15]] : f64
// CHECK:             %[[VAL_17:.*]] = math.log1p %[[VAL_16]] : f64
// CHECK:             %[[VAL_18:.*]] = math.sin %[[VAL_17]] : f64
// CHECK:             %[[VAL_19:.*]] = math.tanh %[[VAL_18]] : f64
// CHECK:             %[[Y:.*]] = tensor.insert %[[VAL_19]] into %{{.*}}[%[[VAL_10]]] : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK:             scf.yield %[[Y]]
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.load %[[T]] hasInserts : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_20]] : tensor<32xf64, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @zero_preserving_math(%arga: tensor<32xf64, #SV>) -> tensor<32xf64, #SV> {
  %c32 = arith.constant 32 : index
  %xinp = tensor.empty() : tensor<32xf64, #SV>
  %0 = linalg.generic #trait1
     ins(%arga: tensor<32xf64, #SV>)
    outs(%xinp: tensor<32xf64, #SV>) {
      ^bb(%a: f64, %x: f64):
	%0 = math.absf %a : f64
        %1 = math.ceil %0 : f64
        %2 = math.floor %1 : f64
        %3 = math.sqrt %2 : f64
        %4 = math.expm1 %3 : f64
        %5 = math.log1p %4 : f64
        %6 = math.sin %5 : f64
        %7 = math.tanh %6 : f64
        linalg.yield %7 : f64
  } -> tensor<32xf64, #SV>
  return %0 : tensor<32xf64, #SV>
}

// CHECK-LABEL:   func.func @complex_divbyc(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32xcomplex<f64>, #sparse{{.*}}> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_4:.*]] = tensor.empty() : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}> to memref<?xcomplex<f64>>
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[T:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_2]] {{.*}} {
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<?xcomplex<f64>>
// CHECK:             %[[VAL_13:.*]] = complex.div %[[VAL_12]], %[[VAL_3]] : complex<f64>
// CHECK:             %[[Y:.*]] = tensor.insert %[[VAL_13]] into %{{.*}}[%[[VAL_11]]] : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}>
// CHECK:             scf.yield %[[Y]]
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.load %[[T]] hasInserts : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_14]] : tensor<32xcomplex<f64>, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @complex_divbyc(%arg0: tensor<32xcomplex<f64>, #SV>) -> tensor<32xcomplex<f64>, #SV> {
  %c = complex.constant [0.0, 1.0] : complex<f64>
  %init = tensor.empty() : tensor<32xcomplex<f64>, #SV>
  %0 = linalg.generic #traitc
     ins(%arg0: tensor<32xcomplex<f64>, #SV>)
    outs(%init: tensor<32xcomplex<f64>, #SV>) {
      ^bb(%a: complex<f64>, %x: complex<f64>):
        %0 = complex.div %a, %c : complex<f64>
        linalg.yield %0 : complex<f64>
  } -> tensor<32xcomplex<f64>, #SV>
  return %0 : tensor<32xcomplex<f64>, #SV>
}
