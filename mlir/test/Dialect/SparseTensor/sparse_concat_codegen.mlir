// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-convert=false" \
// RUN: | FileCheck %s

#DCSR = #sparse_tensor.encoding<{lvlTypes = ["compressed", "compressed"]}>
#DENSE = #sparse_tensor.encoding<{lvlTypes = ["dense", "dense"]}>
#DENSE_P = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "dense"],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>
// CHECK-LABEL: @concat_sparse_sparse(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_c0:.*]] = arith.constant 0 : index
//       CHECK:  %[[TMP_c1:.*]] = arith.constant 1 : index
//       CHECK:  %[[TMP_c5:.*]] = arith.constant 5 : index
//       CHECK:  %[[TMP_c2:.*]] = arith.constant 2 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor() : tensor<9x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse_tensor
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
//       CHECK:      %[[NEW_1:.*]] = sparse_tensor.insert %[[TMP_28]] into %[[A1]][%[[TMP_23]], %[[TMP_27]]] : tensor<9x4xf64, #sparse_tensor
//       CHECK:      scf.yield %[[NEW_1]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_4]]
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse_tensor
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
//       CHECK:      %[[NEW_2:.*]] = sparse_tensor.insert %[[TMP_28]] into %[[A3]][%[[TMP_29]], %[[TMP_27]]] : tensor<9x4xf64, #sparse_tensor
//       CHECK:      scf.yield %[[NEW_2]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_5]]
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse_tensor
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
//       CHECK:      %[[NEW_3:.*]] = sparse_tensor.insert %[[TMP_28]] into %[[A5]][%[[TMP_29]], %[[TMP_27]]] : tensor<9x4xf64, #sparse_tensor
//       CHECK:      scf.yield %[[NEW_3]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_6]]
//       CHECK:  }
//       CHECK:  %[[TMP_23:.*]] = sparse_tensor.load %[[RET_3]] hasInserts
//       CHECK:  return %[[TMP_23]] : tensor<9x4xf64, #sparse_tensor
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
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse_tensor
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[TMP_c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[TMP_c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:  %[[TMP_c9:.*]] = arith.constant 9 : index
//   CHECK-DAG:  %[[TMP_c4:.*]] = arith.constant 4 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor(%[[TMP_c9]], %[[TMP_c4]]) : tensor<?x?xf64, #sparse_tensor
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse_tensor
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
//       CHECK:      %[[NEW_1:.*]] = sparse_tensor.insert %[[TMP_28]] into %[[A1]][%[[TMP_23]], %[[TMP_27]]] : tensor<?x?xf64, #sparse_tensor
//       CHECK:      scf.yield %[[NEW_1]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_4]]
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse_tensor
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
//       CHECK:      %[[NEW_2:.*]] = sparse_tensor.insert %[[TMP_28]] into %[[A3]][%[[TMP_29]], %[[TMP_27]]] : tensor<?x?xf64, #sparse_tensor
//       CHECK:      scf.yield %[[NEW_2]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_5]]
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse_tensor
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
//       CHECK:      %[[NEW_3:.*]] = sparse_tensor.insert %[[TMP_28]] into %[[A5]][%[[TMP_29]], %[[TMP_27]]] : tensor<?x?xf64, #sparse_tensor
//       CHECK:      scf.yield %[[NEW_3]]
//       CHECK:    }
//       CHECK:    scf.yield %[[RET_6]]
//       CHECK:  }
//       CHECK:  %[[TMP_23:.*]] = sparse_tensor.load %[[RET_3]] hasInserts
//       CHECK:  return %[[TMP_23]] : tensor<?x?xf64, #sparse_tensor
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

// CHECK-LABEL: @concat_sparse_sparse_dense(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse_tensor
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[TMP_c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[TMP_c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:  %[[TMP_c9:.*]] = arith.constant 9 : index
//   CHECK-DAG:  %[[TMP_c4:.*]] = arith.constant 4 : index
//   CHECK-DAG:  %[[TMP_d0:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:  %[[A:.*]] = memref.alloc(%[[TMP_c9]], %[[TMP_c4]]) : memref<?x?xf64>
//       CHECK:  linalg.fill ins(%[[TMP_d0]] : f64) outs(%[[A]] : memref<?x?xf64>)
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_6:.*]] = memref.load %[[TMP_1]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_7:.*]] = memref.load %[[TMP_1]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_6]] to %[[TMP_7]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_2]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_3]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_3]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_4]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_5]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      memref.store %[[TMP_28]], %[[A]]{{\[}}%[[TMP_23]], %[[TMP_27]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_13:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_14:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_13]] to %[[TMP_14]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_9]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_10]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_10]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_11]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_12]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c2]] : index
//       CHECK:      memref.store %[[TMP_28]], %[[A]]{{\[}}%[[TMP_29]], %[[TMP_27]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_20:.*]] = memref.load %[[TMP_15]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_21:.*]] = memref.load %[[TMP_15]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_20]] to %[[TMP_21]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_16]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_25:.*]] = memref.load %[[TMP_17]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_17]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_18]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_19]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c5]] : index
//       CHECK:      memref.store %[[TMP_28]], %[[A]]{{\[}}%[[TMP_29]], %[[TMP_27]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[R:.*]] = bufferization.to_tensor %[[A]] : memref<?x?xf64>
//       CHECK:  return %[[R]] : tensor<?x?xf64>
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

// CHECK-LABEL: @concat_sparse_sparse_annotated_dense(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse_tensor
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[TMP_c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[TMP_c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:  %[[TMP_c9:.*]] = arith.constant 9 : index
//   CHECK-DAG:  %[[TMP_c4:.*]] = arith.constant 4 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor(%[[TMP_c9]], %[[TMP_c4]]) : tensor<?x?xf64, #sparse_tensor
//       CHECK:  %[[VAL_0:.*]] = sparse_tensor.values %[[TMP_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>> to memref<?xf64>
//       CHECK:  %[[DIM_0:.*]] = memref.alloca() : memref<2xindex>
//       CHECK:  memref.store %[[TMP_c9]], %[[DIM_0]][%[[TMP_c0]]] : memref<2xindex>
//       CHECK:  memref.store %[[TMP_c4]], %[[DIM_0]][%[[TMP_c1]]] : memref<2xindex>
//       CHECK:  %[[VAL_1:.*]] = memref.reshape %[[VAL_0]](%[[DIM_0]]) : (memref<?xf64>, memref<2xindex>) -> memref<?x?xf64>
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_6:.*]] = memref.load %[[TMP_1]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_7:.*]] = memref.load %[[TMP_1]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_6]] to %[[TMP_7]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_2]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_3]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_3]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_4]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_5]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      memref.store %[[TMP_28]], %[[VAL_1]][%[[TMP_23]], %[[TMP_27]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_13:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_14:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_13]] to %[[TMP_14]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_9]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_10]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_10]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_11]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_12]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c2]] : index
//       CHECK:      memref.store %[[TMP_28]], %[[VAL_1]][%[[TMP_29]], %[[TMP_27]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_20:.*]] = memref.load %[[TMP_15]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_21:.*]] = memref.load %[[TMP_15]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_20]] to %[[TMP_21]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_16]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_25:.*]] = memref.load %[[TMP_17]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_17]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_18]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_19]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c5]] : index
//       CHECK:      memref.store %[[TMP_28]], %[[VAL_1]][%[[TMP_29]], %[[TMP_27]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[R:.*]] = sparse_tensor.convert %[[TMP_0]]
//       CHECK:  return %[[R]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>
func.func @concat_sparse_sparse_annotated_dense(%arg0: tensor<2x4xf64, #DCSR>,
                                %arg1: tensor<3x4xf64, #DCSR>,
                                %arg2: tensor<4x4xf64, #DCSR>)
                                -> tensor<?x?xf64, #DENSE> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #DCSR>,
           tensor<3x4xf64, #DCSR>,
           tensor<4x4xf64, #DCSR> to tensor<?x?xf64, #DENSE>
    return %0 : tensor<?x?xf64, #DENSE>
}

// CHECK-LABEL: @concat_sparse_sparse_annotated_dense_permute(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<2x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg1:.*]]: tensor<3x4xf64, #sparse_tensor
//  CHECK-SAME:  %[[TMP_arg2:.*]]: tensor<4x4xf64, #sparse_tensor
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[TMP_c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[TMP_c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:  %[[TMP_c9:.*]] = arith.constant 9 : index
//   CHECK-DAG:  %[[TMP_c4:.*]] = arith.constant 4 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor(%[[TMP_c9]], %[[TMP_c4]]) : tensor<?x?xf64, #sparse_tensor
//       CHECK:  %[[VAL_0:.*]] = sparse_tensor.values %[[TMP_0]] : tensor<?x?xf64, #sparse_tensor
//       CHECK:  %[[DIM_0:.*]] = memref.alloca() : memref<2xindex>
//       CHECK:  memref.store %[[TMP_c4]], %[[DIM_0]][%[[TMP_c0]]] : memref<2xindex>
//       CHECK:  memref.store %[[TMP_c9]], %[[DIM_0]][%[[TMP_c1]]] : memref<2xindex>
//       CHECK:  %[[VAL_1:.*]] = memref.reshape %[[VAL_0]](%[[DIM_0]]) : (memref<?xf64>, memref<2xindex>) -> memref<?x?xf64>
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index} : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]] : tensor<2x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_6:.*]] = memref.load %[[TMP_1]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_7:.*]] = memref.load %[[TMP_1]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_6]] to %[[TMP_7]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_2]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_3]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_3]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_4]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_5]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      memref.store %[[TMP_28]], %[[VAL_1]][%[[TMP_27]], %[[TMP_23]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_9:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 0 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_10:.*]] = sparse_tensor.positions %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_11:.*]] = sparse_tensor.coordinates %[[TMP_arg1]] {level = 1 : index} : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_12:.*]] = sparse_tensor.values %[[TMP_arg1]] : tensor<3x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_13:.*]] = memref.load %[[TMP_8]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_14:.*]] = memref.load %[[TMP_8]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_13]] to %[[TMP_14]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_9]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_25:.*]] = memref.load %[[TMP_10]][%[[TMP_arg3]]] : memref<?xindex>
//   CHECK-DAG:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_10]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_11]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_12]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c2]] : index
//       CHECK:      memref.store %[[TMP_28]], %[[VAL_1]][%[[TMP_27]], %[[TMP_29]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[TMP_15:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_16:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 0 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_17:.*]] = sparse_tensor.positions %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_18:.*]] = sparse_tensor.coordinates %[[TMP_arg2]] {level = 1 : index} : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_19:.*]] = sparse_tensor.values %[[TMP_arg2]] : tensor<4x4xf64, #sparse_tensor
//       CHECK:  %[[TMP_20:.*]] = memref.load %[[TMP_15]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_21:.*]] = memref.load %[[TMP_15]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  scf.for %[[TMP_arg3:.*]] = %[[TMP_20]] to %[[TMP_21]] step %[[TMP_c1]]
//       CHECK:    %[[TMP_23:.*]] = memref.load %[[TMP_16]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_25:.*]] = memref.load %[[TMP_17]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:    %[[TMP_24:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
//       CHECK:    %[[TMP_26:.*]] = memref.load %[[TMP_17]][%[[TMP_24]]] : memref<?xindex>
//       CHECK:    scf.for %[[TMP_arg4:.*]] = %[[TMP_25]] to %[[TMP_26]] step %[[TMP_c1]]
//       CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_18]][%[[TMP_arg4]]] : memref<?xindex>
//       CHECK:      %[[TMP_28:.*]] = memref.load %[[TMP_19]][%[[TMP_arg4]]] : memref<?xf64>
//       CHECK:      %[[TMP_29:.*]] = arith.addi %[[TMP_23]], %[[TMP_c5]] : index
//       CHECK:      memref.store %[[TMP_28]], %[[VAL_1]][%[[TMP_27]], %[[TMP_29]]] : memref<?x?xf64>
//       CHECK:    }
//       CHECK:  }
//       CHECK:  %[[R:.*]] = sparse_tensor.convert %[[TMP_0]]
//       CHECK:  return %[[R]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ], dimToLvl = affine_map<(d0, d1) -> (d1, d0)> }>>
func.func @concat_sparse_sparse_annotated_dense_permute(%arg0: tensor<2x4xf64, #DCSR>,
                                %arg1: tensor<3x4xf64, #DCSR>,
                                %arg2: tensor<4x4xf64, #DCSR>)
                                -> tensor<?x?xf64, #DENSE_P> {
    %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
         : tensor<2x4xf64, #DCSR>,
           tensor<3x4xf64, #DCSR>,
           tensor<4x4xf64, #DCSR> to tensor<?x?xf64, #DENSE_P>
    return %0 : tensor<?x?xf64, #DENSE_P>
}
