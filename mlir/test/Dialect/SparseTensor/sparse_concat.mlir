// RUN: mlir-opt %s --lower-sparse-ops-to-foreach="enable-runtime-library=false enable-convert=false" --lower-sparse-foreach-to-scf \
// RUN: | FileCheck %s
// RUN: mlir-opt %s --lower-sparse-ops-to-foreach="enable-runtime-library=true enable-convert=false" --lower-sparse-foreach-to-scf \
// RUN: | FileCheck %s


#DCSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>

// CHECK-LABEL: @concat_sparse_sparse(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse>
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse>
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_c0:.*]] = arith.constant 0 : index
//       CHECK:  %[[TMP_c1:.*]] = arith.constant 1 : index
//       CHECK:  %[[TMP_c5:.*]] = arith.constant 5 : index
//       CHECK:  %[[TMP_c2:.*]] = arith.constant 2 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor() : tensor<9x4xf64, #sparse>
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_6:.*]] = memref.load %[[TMP_1]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_7:.*]] = memref.load %[[TMP_1]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[RET_1:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_6]] to %[[TMP_7]] step %[[TMP_c1]] iter_args(%[[A0:.*]] = %[[TMP_0]])
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_2]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_3]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_3]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    %[[RET_4:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]] iter_args(%[[A1:.*]] = %[[A0]])
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_4]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_5]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[NEW_1:.*]] = tensor.insert %[[TMP_28]] into %[[A1]][%[[TMP_23]], %[[TMP_27]]] : tensor<9x4xf64, #sparse>
//       CHECK:      scf.yield %[[NEW_1]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_4]]
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_13:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_14:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[RET_2:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_13]] to %[[TMP_14]] step %[[TMP_c1]] iter_args(%[[A2:.*]] = %[[RET_1]])
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_9]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_10]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_10]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    %[[RET_5:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]] iter_args(%[[A3:.*]] = %[[A2]])
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_11]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_12]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c2]] : index
//       CHECK:      %[[NEW_2:.*]] = tensor.insert %[[TMP_28]] into %[[A3]][%[[TMP_29]], %[[TMP_27]]] : tensor<9x4xf64, #sparse>
//       CHECK:      scf.yield %[[NEW_2]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_5]]
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_20:.*]] = memref.load %[[TMP_15]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_21:.*]] = memref.load %[[TMP_15]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[RET_3:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_20]] to %[[TMP_21]] step %[[TMP_c1]] iter_args(%[[A4:.*]] = %[[RET_2]])
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_16]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_25:.*]] = memref.load %[[TMP_17]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_17]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    %[[RET_6:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]] iter_args(%[[A5:.*]] = %[[A4]])
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_18]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_19]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c5]] : index
//       CHECK:      %[[NEW_3:.*]] = tensor.insert %[[TMP_28]] into %[[A5]][%[[TMP_29]], %[[TMP_27]]] : tensor<9x4xf64, #sparse>
//       CHECK:      scf.yield %[[NEW_3]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_6]]
//       CHECK:  }
//       CHECK:  %[[TMP_23:.*]] = sparse_tensor.load %[[RET_3]] hasInserts
//       CHECK:  return %[[TMP_23]] : tensor<9x4xf64, #sparse>
func.func @concat_sparse_sparse(%arg0: tensor<2x4xf64, #DCSR>,
                                %arg1: tensor<3x4xf64, #DCSR>,
                                %arg2: tensor<4x4xf64, #DCSR>)
                                -> tensor<9x4xf64, #DCSR> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #DCSR>,
           tensor<3x4xf64, #DCSR>,
           tensor<4x4xf64, #DCSR> to tensor<9x4xf64, #DCSR>
    return %0 : tensor<9x4xf64, #DCSR>
}

// CHECK-LABEL: @concat_sparse_sparse_dynamic(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse>
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse>
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse>
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[TMP_c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[TMP_c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:  %[[TMP_c9:.*]] = arith.constant 9 : index
//   CHECK-DAG:  %[[TMP_c4:.*]] = arith.constant 4 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor(%[[TMP_c9]], %[[TMP_c4]]) : tensor<?x?xf64, #sparse>
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse>
//       CHECK:  %[[TMP_6:.*]] = memref.load %[[TMP_1]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_7:.*]] = memref.load %[[TMP_1]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[RET_1:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_6]] to %[[TMP_7]] step %[[TMP_c1]] iter_args(%[[A0:.*]] = %[[TMP_0]])
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_2]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_3]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_3]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    %[[RET_4:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]] iter_args(%[[A1:.*]] = %[[A0]])
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_4]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_5]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[NEW_1:.*]] = tensor.insert %[[TMP_28]] into %[[A1]][%[[TMP_23]], %[[TMP_27]]] : tensor<?x?xf64, #sparse>
//       CHECK:      scf.yield %[[NEW_1]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_4]]
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse>
//       CHECK:  %[[TMP_13:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_14:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[RET_2:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_13]] to %[[TMP_14]] step %[[TMP_c1]] iter_args(%[[A2:.*]] = %[[RET_1]])
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_9]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_10]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_10]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    %[[RET_5:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]] iter_args(%[[A3:.*]] = %[[A2]])
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_11]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_12]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c2]] : index
//       CHECK:      %[[NEW_2:.*]] = tensor.insert %[[TMP_28]] into %[[A3]][%[[TMP_29]], %[[TMP_27]]] : tensor<?x?xf64, #sparse>
//       CHECK:      scf.yield %[[NEW_2]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_5]]
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse>
//       CHECK:  %[[TMP_20:.*]] = memref.load %[[TMP_15]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_21:.*]] = memref.load %[[TMP_15]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[RET_3:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_20]] to %[[TMP_21]] step %[[TMP_c1]] iter_args(%[[A4:.*]] = %[[RET_2]])
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_16]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_25:.*]] = memref.load %[[TMP_17]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_17]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    %[[RET_6:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]] iter_args(%[[A5:.*]] = %[[A4]])
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_18]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_19]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c5]] : index
//       CHECK:      %[[NEW_3:.*]] = tensor.insert %[[TMP_28]] into %[[A5]][%[[TMP_29]], %[[TMP_27]]] : tensor<?x?xf64, #sparse>
//       CHECK:      scf.yield %[[NEW_3]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_6]]
//       CHECK:  }
//       CHECK:  %[[TMP_23:.*]] = sparse_tensor.load %[[RET_3]] hasInserts
//       CHECK:  return %[[TMP_23]] : tensor<?x?xf64, #sparse>
func.func @concat_sparse_sparse_dynamic(%arg0: tensor<2x4xf64, #DCSR>,
                                %arg1: tensor<3x4xf64, #DCSR>,
                                %arg2: tensor<4x4xf64, #DCSR>)
                                -> tensor<?x?xf64, #DCSR> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #DCSR>,
           tensor<3x4xf64, #DCSR>,
           tensor<4x4xf64, #DCSR> to tensor<?x?xf64, #DCSR>
    return %0 : tensor<?x?xf64, #DCSR>
}

// CHECK-LABEL:   func.func @concat_sparse_sparse_dense(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x4xf64, #sparse>
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x4xf64, #sparse>
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<4x4xf64, #sparse>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 9 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_10:.*]] = bufferization.alloc_tensor(%[[VAL_4]], %[[VAL_3]]) : tensor<?x?xf64>
// CHECK:           %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_6]] : f64) outs(%[[VAL_10]] : tensor<?x?xf64>) -> tensor<?x?xf64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<2x4xf64, #sparse>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<2x4xf64, #sparse>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<2x4xf64, #sparse>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<2x4xf64, #sparse>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<2x4xf64, #sparse>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_18]] step %[[VAL_8]] iter_args(%[[VAL_21:.*]] = %[[VAL_11]]) -> (tensor<?x?xf64>) {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_20]], %[[VAL_8]] : index
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_24]]] : memref<?xindex>
// CHECK:             %[[VAL_26:.*]] = scf.for %[[VAL_27:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_8]] iter_args(%[[VAL_28:.*]] = %[[VAL_21]]) -> (tensor<?x?xf64>) {
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_27]]] : memref<?xindex>
// CHECK:               %[[VAL_30:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_27]]] : memref<?xf64>
// CHECK:               %[[VAL_31:.*]] = tensor.insert %[[VAL_30]] into %[[VAL_28]]{{\[}}%[[VAL_22]], %[[VAL_29]]] : tensor<?x?xf64>
// CHECK:               scf.yield %[[VAL_31]] : tensor<?x?xf64>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_26]] : tensor<?x?xf64>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<3x4xf64, #sparse>
// CHECK:           %[[VAL_33:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<3x4xf64, #sparse>
// CHECK:           %[[VAL_34:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<3x4xf64, #sparse>
// CHECK:           %[[VAL_35:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<3x4xf64, #sparse>
// CHECK:           %[[VAL_36:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<3x4xf64, #sparse>
// CHECK:           %[[VAL_37:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:           %[[VAL_38:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:           %[[VAL_39:.*]] = scf.for %[[VAL_40:.*]] = %[[VAL_37]] to %[[VAL_38]] step %[[VAL_8]] iter_args(%[[VAL_41:.*]] = %[[VAL_19]]) -> (tensor<?x?xf64>) {
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:             %[[VAL_43:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_40]], %[[VAL_8]] : index
// CHECK:             %[[VAL_45:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_44]]] : memref<?xindex>
// CHECK:             %[[VAL_46:.*]] = scf.for %[[VAL_47:.*]] = %[[VAL_43]] to %[[VAL_45]] step %[[VAL_8]] iter_args(%[[VAL_48:.*]] = %[[VAL_41]]) -> (tensor<?x?xf64>) {
// CHECK:               %[[VAL_49:.*]] = memref.load %[[VAL_35]]{{\[}}%[[VAL_47]]] : memref<?xindex>
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_36]]{{\[}}%[[VAL_47]]] : memref<?xf64>
// CHECK:               %[[VAL_51:.*]] = arith.addi %[[VAL_42]], %[[VAL_9]] : index
// CHECK:               %[[VAL_52:.*]] = tensor.insert %[[VAL_50]] into %[[VAL_48]]{{\[}}%[[VAL_51]], %[[VAL_49]]] : tensor<?x?xf64>
// CHECK:               scf.yield %[[VAL_52]] : tensor<?x?xf64>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_46]] : tensor<?x?xf64>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = sparse_tensor.positions %[[VAL_2]] {level = 0 : index} : tensor<4x4xf64, #sparse>
// CHECK:           %[[VAL_54:.*]] = sparse_tensor.coordinates %[[VAL_2]] {level = 0 : index} : tensor<4x4xf64, #sparse>
// CHECK:           %[[VAL_55:.*]] = sparse_tensor.positions %[[VAL_2]] {level = 1 : index} : tensor<4x4xf64, #sparse>
// CHECK:           %[[VAL_56:.*]] = sparse_tensor.coordinates %[[VAL_2]] {level = 1 : index} : tensor<4x4xf64, #sparse>
// CHECK:           %[[VAL_57:.*]] = sparse_tensor.values %[[VAL_2]] : tensor<4x4xf64, #sparse>
// CHECK:           %[[VAL_58:.*]] = memref.load %[[VAL_53]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:           %[[VAL_59:.*]] = memref.load %[[VAL_53]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:           %[[VAL_60:.*]] = scf.for %[[VAL_61:.*]] = %[[VAL_58]] to %[[VAL_59]] step %[[VAL_8]] iter_args(%[[VAL_62:.*]] = %[[VAL_39]]) -> (tensor<?x?xf64>) {
// CHECK:             %[[VAL_63:.*]] = memref.load %[[VAL_54]]{{\[}}%[[VAL_61]]] : memref<?xindex>
// CHECK:             %[[VAL_64:.*]] = memref.load %[[VAL_55]]{{\[}}%[[VAL_61]]] : memref<?xindex>
// CHECK:             %[[VAL_65:.*]] = arith.addi %[[VAL_61]], %[[VAL_8]] : index
// CHECK:             %[[VAL_66:.*]] = memref.load %[[VAL_55]]{{\[}}%[[VAL_65]]] : memref<?xindex>
// CHECK:             %[[VAL_67:.*]] = scf.for %[[VAL_68:.*]] = %[[VAL_64]] to %[[VAL_66]] step %[[VAL_8]] iter_args(%[[VAL_69:.*]] = %[[VAL_62]]) -> (tensor<?x?xf64>) {
// CHECK:               %[[VAL_70:.*]] = memref.load %[[VAL_56]]{{\[}}%[[VAL_68]]] : memref<?xindex>
// CHECK:               %[[VAL_71:.*]] = memref.load %[[VAL_57]]{{\[}}%[[VAL_68]]] : memref<?xf64>
// CHECK:               %[[VAL_72:.*]] = arith.addi %[[VAL_63]], %[[VAL_5]] : index
// CHECK:               %[[VAL_73:.*]] = tensor.insert %[[VAL_71]] into %[[VAL_69]]{{\[}}%[[VAL_72]], %[[VAL_70]]] : tensor<?x?xf64>
// CHECK:               scf.yield %[[VAL_73]] : tensor<?x?xf64>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_67]] : tensor<?x?xf64>
// CHECK:           }
// CHECK:           return %[[VAL_60]] : tensor<?x?xf64>
// CHECK:         }
func.func @concat_sparse_sparse_dense(%arg0: tensor<2x4xf64, #DCSR>,
                                %arg1: tensor<3x4xf64, #DCSR>,
                                %arg2: tensor<4x4xf64, #DCSR>)
                                -> tensor<?x?xf64> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #DCSR>,
           tensor<3x4xf64, #DCSR>,
           tensor<4x4xf64, #DCSR> to tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
}
