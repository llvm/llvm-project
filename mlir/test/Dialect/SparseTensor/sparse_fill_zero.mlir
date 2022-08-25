// RUN: mlir-opt %s --linalg-generalize-named-ops --sparsification --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>
// CHECK-LABEL:  func.func @fill_zero_after_alloc
// CHECK-SAME:     %[[TMP_arg0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:     %[[TMP_arg1:.*]]: !llvm.ptr<i8>
// CHECK:    %[[TMP_cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:    %[[TMP_c1_i32:.*]] = arith.constant 1 : i32
// CHECK:    %[[TMP_c0_i32:.*]] = arith.constant 0 : i32
// CHECK:    %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK:    %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK:    %[[TMP_false:.*]] = arith.constant false
// CHECK:    %[[TMP_true:.*]] = arith.constant true
// CHECK:    %[[TMP_c100:.*]] = arith.constant 100 : index
// CHECK:    %[[TMP_c1_i8:.*]] = arith.constant 1 : i8
// CHECK:    %[[TMP_0:.*]] = memref.alloca() : memref<2xi8>
// CHECK:    %[[TMP_1:.*]] = memref.cast %[[TMP_0]] : memref<2xi8> to memref<?xi8>
// CHECK:    memref.store %[[TMP_c1_i8]], %[[TMP_0]][%[[TMP_c0]]] : memref<2xi8>
// CHECK:    memref.store %[[TMP_c1_i8]], %[[TMP_0]][%[[TMP_c1]]] : memref<2xi8>
// CHECK:    %[[TMP_2:.*]] = memref.alloca() : memref<2xindex>
// CHECK:    %[[TMP_3:.*]] = memref.cast %[[TMP_2]] : memref<2xindex> to memref<?xindex>
// CHECK:    memref.store %[[TMP_c100]], %[[TMP_2]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:    memref.store %[[TMP_c100]], %[[TMP_2]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:    %[[TMP_4:.*]] = memref.alloca() : memref<2xindex>
// CHECK:    %[[TMP_5:.*]] = memref.cast %[[TMP_4]] : memref<2xindex> to memref<?xindex>
// CHECK:    memref.store %[[TMP_c0]], %[[TMP_4]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:    memref.store %[[TMP_c1]], %[[TMP_4]][%[[TMP_c1]]] : memref<2xindex>
// CHECK:    %[[TMP_6:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:    %[[TMP_7:.*]] = call @newSparseTensor(%[[TMP_1]], %[[TMP_3]], %[[TMP_5]], %[[TMP_c0_i32]], %[[TMP_c0_i32]], %[[TMP_c1_i32]], %[[TMP_c0_i32]], %[[TMP_6]])
// CHECK:    %[[TMP_8:.*]] = call @sparseDimSize(%[[TMP_7]], %[[TMP_c1]])
// CHECK:    %[[TMP_9:.*]] = memref.alloc(%[[TMP_8]]) : memref<?xf64>
// CHECK:    %[[TMP_10:.*]] = memref.alloc(%[[TMP_8]]) : memref<?xi1>
// CHECK:    %[[TMP_11:.*]] = memref.alloc(%[[TMP_8]]) : memref<?xindex>
// CHECK:    linalg.fill ins(%[[TMP_cst]] : f64) outs(%[[TMP_9]] : memref<?xf64>)
// CHECK:    linalg.fill ins(%[[TMP_false]] : i1) outs(%[[TMP_10]] : memref<?xi1>)
// CHECK:    %[[TMP_12:.*]] = call @sparsePointers0(%[[TMP_arg0]], %[[TMP_c0]])
// CHECK:    %[[TMP_13:.*]] = call @sparseIndices0(%[[TMP_arg0]], %[[TMP_c0]])
// CHECK:    %[[TMP_14:.*]] = call @sparsePointers0(%[[TMP_arg0]], %[[TMP_c1]])
// CHECK:    %[[TMP_15:.*]] = call @sparseIndices0(%[[TMP_arg0]], %[[TMP_c1]])
// CHECK:    %[[TMP_16:.*]] = call @sparseValuesF64(%[[TMP_arg0]])
// CHECK:    %[[TMP_17:.*]] = call @sparsePointers0(%[[TMP_arg1]], %[[TMP_c0]])
// CHECK:    %[[TMP_18:.*]] = call @sparseIndices0(%[[TMP_arg1]], %[[TMP_c0]])
// CHECK:    %[[TMP_19:.*]] = call @sparsePointers0(%[[TMP_arg1]], %[[TMP_c1]])
// CHECK:    %[[TMP_20:.*]] = call @sparseIndices0(%[[TMP_arg1]], %[[TMP_c1]])
// CHECK:    %[[TMP_21:.*]] = call @sparseValuesF64(%[[TMP_arg1]])
// CHECK:    %[[TMP_22:.*]] = memref.alloca() : memref<2xindex>
// CHECK:    %[[TMP_23:.*]] = memref.cast %[[TMP_22]] : memref<2xindex> to memref<?xindex>
// CHECK:    %[[TMP_24:.*]] = memref.load %[[TMP_12]][%[[TMP_c0]]] : memref<?xindex>
// CHECK:    %[[TMP_25:.*]] = memref.load %[[TMP_12]][%[[TMP_c1]]] : memref<?xindex>
// CHECK:    scf.for %[[TMP_arg2:.*]] = %[[TMP_24]] to %[[TMP_25]] step %[[TMP_c1]] {
// CHECK:      %[[TMP_26:.*]] = memref.load %[[TMP_13]][%[[TMP_arg2]]] : memref<?xindex>
// CHECK:      memref.store %[[TMP_26]], %[[TMP_22]][%[[TMP_c0]]] : memref<2xindex>
// CHECK:      %[[TMP_27:.*]] = memref.load %[[TMP_14]][%[[TMP_arg2]]] : memref<?xindex>
// CHECK:      %[[TMP_28:.*]] = arith.addi %[[TMP_arg2]], %[[TMP_c1]] : index
// CHECK:      %[[TMP_29:.*]] = memref.load %[[TMP_14]][%[[TMP_28]]] : memref<?xindex>
// CHECK:      %[[TMP_30:.*]] = memref.load %[[TMP_17]][%[[TMP_c0]]] : memref<?xindex>
// CHECK:      %[[TMP_31:.*]] = memref.load %[[TMP_17]][%[[TMP_c1]]] : memref<?xindex>
// CHECK:      %[[TMP_32:.*]]:3 = scf.while (%[[TMP_arg3:.*]] = %[[TMP_27]], %[[TMP_arg4:.*]] = %[[TMP_30]], %[[TMP_arg5:.*]] = %[[TMP_c0]]) : (index, index, index) -> (index, index, index) {
// CHECK:        %[[TMP_33:.*]] = arith.cmpi ult, %[[TMP_arg3]], %[[TMP_29]] : index
// CHECK:        %[[TMP_34:.*]] = arith.cmpi ult, %[[TMP_arg4]], %[[TMP_31]] : index
// CHECK:        %[[TMP_35:.*]] = arith.andi %[[TMP_33]], %[[TMP_34]] : i1
// CHECK:        scf.condition(%[[TMP_35]]) %[[TMP_arg3]], %[[TMP_arg4]], %[[TMP_arg5]] : index, index, index
// CHECK:      } do {
// CHECK:      ^bb0(%[[TMP_arg3:.*]]: index, %[[TMP_arg4:.*]]: index, %[[TMP_arg5:.*]]: index):
// CHECK:        %[[TMP_33:.*]] = memref.load %[[TMP_15]][%[[TMP_arg3]]] : memref<?xindex>
// CHECK:        %[[TMP_34:.*]] = memref.load %[[TMP_18]][%[[TMP_arg4]]] : memref<?xindex>
// CHECK:        %[[TMP_35:.*]] = arith.cmpi ult, %[[TMP_34]], %[[TMP_33]] : index
// CHECK:        %[[TMP_36:.*]] = arith.select %[[TMP_35]], %[[TMP_34]], %[[TMP_33]] : index
// CHECK:        %[[TMP_37:.*]] = arith.cmpi eq, %[[TMP_33]], %[[TMP_36]] : index
// CHECK:        %[[TMP_38:.*]] = arith.cmpi eq, %[[TMP_34]], %[[TMP_36]] : index
// CHECK:        %[[TMP_39:.*]] = arith.andi %[[TMP_37]], %[[TMP_38]] : i1
// CHECK:        %[[TMP_40:.*]] = scf.if %[[TMP_39]] -> (index) {
// CHECK:          %[[TMP_45:.*]] = memref.load %[[TMP_16]][%[[TMP_arg3]]] : memref<?xf64>
// CHECK:          %[[TMP_46:.*]] = memref.load %[[TMP_19]][%[[TMP_arg4]]] : memref<?xindex>
// CHECK:          %[[TMP_47:.*]] = arith.addi %[[TMP_arg4]], %[[TMP_c1]] : index
// CHECK:          %[[TMP_48:.*]] = memref.load %[[TMP_19]][%[[TMP_47]]] : memref<?xindex>
// CHECK:          %[[TMP_49:.*]] = scf.for %[[TMP_arg6:.*]] = %[[TMP_46]] to %[[TMP_48]] step %[[TMP_c1]] iter_args(%[[TMP_arg7:.*]] = %[[TMP_arg5]]) -> (index) {
// CHECK:            %[[TMP_50:.*]] = memref.load %[[TMP_20]][%[[TMP_arg6]]] : memref<?xindex>
// CHECK:            %[[TMP_51:.*]] = memref.load %[[TMP_9]][%[[TMP_50]]] : memref<?xf64>
// CHECK:            %[[TMP_52:.*]] = memref.load %[[TMP_21]][%[[TMP_arg6]]] : memref<?xf64>
// CHECK:            %[[TMP_53:.*]] = arith.mulf %[[TMP_45]], %[[TMP_52]] : f64
// CHECK:            %[[TMP_54:.*]] = arith.addf %[[TMP_51]], %[[TMP_53]] : f64
// CHECK:            %[[TMP_55:.*]] = memref.load %[[TMP_10]][%[[TMP_50]]] : memref<?xi1>
// CHECK:            %[[TMP_56:.*]] = arith.cmpi eq, %[[TMP_55]], %[[TMP_false]] : i1
// CHECK:            %[[TMP_57:.*]] = scf.if %[[TMP_56]] -> (index) {
// CHECK:              memref.store %[[TMP_true]], %[[TMP_10]][%[[TMP_50]]] : memref<?xi1>
// CHECK:              memref.store %[[TMP_50]], %[[TMP_11]][%[[TMP_arg7]]] : memref<?xindex>
// CHECK:              %[[TMP_58:.*]] = arith.addi %[[TMP_arg7]], %[[TMP_c1]] : index
// CHECK:              scf.yield %[[TMP_58]] : index
// CHECK:            } else {
// CHECK:              scf.yield %[[TMP_arg7]] : index
// CHECK:            }
// CHECK:            memref.store %[[TMP_54]], %[[TMP_9]][%[[TMP_50]]] : memref<?xf64>
// CHECK:            scf.yield %[[TMP_57]] : index
// CHECK:          }
// CHECK:          scf.yield %[[TMP_49]] : index
// CHECK:        } else {
// CHECK:          scf.yield %[[TMP_arg5]] : index
// CHECK:        }
// CHECK:        %[[TMP_41:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
// CHECK:        %[[TMP_42:.*]] = arith.select %[[TMP_37]], %[[TMP_41]], %[[TMP_arg3]] : index
// CHECK:        %[[TMP_43:.*]] = arith.addi %[[TMP_arg4]], %[[TMP_c1]] : index
// CHECK:        %[[TMP_44:.*]] = arith.select %[[TMP_38]], %[[TMP_43]], %[[TMP_arg4]] : index
// CHECK:        scf.yield %[[TMP_42]], %[[TMP_44]], %[[TMP_40]] : index, index, index
// CHECK:      }
// CHECK:      func.call @expInsertF64(%[[TMP_7]], %[[TMP_23]], %[[TMP_9]], %[[TMP_10]], %[[TMP_11]], %[[TMP_32]]#2)
// CHECK:    }
// CHECK:    memref.dealloc %[[TMP_9]] : memref<?xf64>
// CHECK:    memref.dealloc %[[TMP_10]] : memref<?xi1>
// CHECK:    memref.dealloc %[[TMP_11]] : memref<?xindex>
// CHECK:    call @endInsert(%[[TMP_7]]) : (!llvm.ptr<i8>) -> ()
// CHECK:    return %[[TMP_7]] : !llvm.ptr<i8>
func.func @fill_zero_after_alloc(%arg0: tensor<100x100xf64, #DCSR>,
                                 %arg1: tensor<100x100xf64, #DCSR>) -> tensor<100x100xf64, #DCSR> {
  %0 = bufferization.alloc_tensor() : tensor<100x100xf64, #DCSR>
  %cst = arith.constant 0.000000e+00 : f64
  %1 = linalg.fill ins(%cst : f64)
                   outs(%0 : tensor<100x100xf64, #DCSR>) -> tensor<100x100xf64, #DCSR>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<100x100xf64, #DCSR>, tensor<100x100xf64, #DCSR>)
                     outs(%1 : tensor<100x100xf64, #DCSR>) -> tensor<100x100xf64, #DCSR>
  return %2 : tensor<100x100xf64, #DCSR>
}
