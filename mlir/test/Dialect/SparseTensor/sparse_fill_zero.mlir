// RUN: mlir-opt %s --linalg-generalize-named-ops --sparsification --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

// CHECK-LABEL:   func.func @fill_zero_after_alloc(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:      %[[VAL_1:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant false
// CHECK:           %[[VAL_8:.*]] = arith.constant true
// CHECK:           %[[VAL_9:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 300 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i8
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<2xi8>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<2xi8> to memref<?xi8>
// CHECK:           memref.store %[[VAL_11]], %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<2xi8>
// CHECK:           memref.store %[[VAL_11]], %[[VAL_12]]{{\[}}%[[VAL_6]]] : memref<2xi8>
// CHECK:           %[[VAL_14:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           %[[VAL_16:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_16]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[VAL_5]], %[[VAL_16]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_16]]{{\[}}%[[VAL_6]]] : memref<2xindex>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           %[[VAL_19:.*]] = call @newSparseTensor(%[[VAL_13]], %[[VAL_15]], %[[VAL_17]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_4]], %[[VAL_18]]) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_20:.*]] = memref.alloc() : memref<300xf64>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_20]] : memref<300xf64> to memref<?xf64>
// CHECK:           %[[VAL_22:.*]] = memref.alloc() : memref<300xi1>
// CHECK:           %[[VAL_23:.*]] = memref.cast %[[VAL_22]] : memref<300xi1> to memref<?xi1>
// CHECK:           %[[VAL_24:.*]] = memref.alloc() : memref<300xindex>
// CHECK:           %[[VAL_25:.*]] = memref.cast %[[VAL_24]] : memref<300xindex> to memref<?xindex>
// CHECK:           linalg.fill ins(%[[VAL_2]] : f64) outs(%[[VAL_20]] : memref<300xf64>)
// CHECK:           linalg.fill ins(%[[VAL_7]] : i1) outs(%[[VAL_22]] : memref<300xi1>)
// CHECK:           %[[VAL_26:.*]] = call @sparsePointers0(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_27:.*]] = call @sparseIndices0(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_28:.*]] = call @sparsePointers0(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_29:.*]] = call @sparseIndices0(%[[VAL_0]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_30:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK:           %[[VAL_31:.*]] = call @sparsePointers0(%[[VAL_1]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_32:.*]] = call @sparseIndices0(%[[VAL_1]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_33:.*]] = call @sparsePointers0(%[[VAL_1]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_34:.*]] = call @sparseIndices0(%[[VAL_1]], %[[VAL_6]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_35:.*]] = call @sparseValuesF64(%[[VAL_1]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK:           %[[VAL_36:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[VAL_37:.*]] = memref.cast %[[VAL_36]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_38:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           %[[VAL_39:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_40:.*]] = %[[VAL_38]] to %[[VAL_39]] step %[[VAL_6]] {
// CHECK:             %[[VAL_41:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:             memref.store %[[VAL_41]], %[[VAL_36]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_40]], %[[VAL_6]] : index
// CHECK:             %[[VAL_44:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_43]]] : memref<?xindex>
// CHECK:             %[[VAL_45:.*]] = memref.load %[[VAL_31]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:             %[[VAL_46:.*]] = memref.load %[[VAL_31]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:             %[[VAL_47:.*]]:3 = scf.while (%[[VAL_48:.*]] = %[[VAL_42]], %[[VAL_49:.*]] = %[[VAL_45]], %[[VAL_50:.*]] = %[[VAL_5]]) : (index, index, index) -> (index, index, index) {
// CHECK:               %[[VAL_51:.*]] = arith.cmpi ult, %[[VAL_48]], %[[VAL_44]] : index
// CHECK:               %[[VAL_52:.*]] = arith.cmpi ult, %[[VAL_49]], %[[VAL_46]] : index
// CHECK:               %[[VAL_53:.*]] = arith.andi %[[VAL_51]], %[[VAL_52]] : i1
// CHECK:               scf.condition(%[[VAL_53]]) %[[VAL_48]], %[[VAL_49]], %[[VAL_50]] : index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_54:.*]]: index, %[[VAL_55:.*]]: index, %[[VAL_56:.*]]: index):
// CHECK:               %[[VAL_57:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_54]]] : memref<?xindex>
// CHECK:               %[[VAL_58:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:               %[[VAL_59:.*]] = arith.cmpi ult, %[[VAL_58]], %[[VAL_57]] : index
// CHECK:               %[[VAL_60:.*]] = arith.select %[[VAL_59]], %[[VAL_58]], %[[VAL_57]] : index
// CHECK:               %[[VAL_61:.*]] = arith.cmpi eq, %[[VAL_57]], %[[VAL_60]] : index
// CHECK:               %[[VAL_62:.*]] = arith.cmpi eq, %[[VAL_58]], %[[VAL_60]] : index
// CHECK:               %[[VAL_63:.*]] = arith.andi %[[VAL_61]], %[[VAL_62]] : i1
// CHECK:               %[[VAL_64:.*]] = scf.if %[[VAL_63]] -> (index) {
// CHECK:                 %[[VAL_65:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_54]]] : memref<?xf64>
// CHECK:                 %[[VAL_66:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:                 %[[VAL_67:.*]] = arith.addi %[[VAL_55]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_68:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_67]]] : memref<?xindex>
// CHECK:                 %[[VAL_69:.*]] = scf.for %[[VAL_70:.*]] = %[[VAL_66]] to %[[VAL_68]] step %[[VAL_6]] iter_args(%[[VAL_71:.*]] = %[[VAL_56]]) -> (index) {
// CHECK:                   %[[VAL_72:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_70]]] : memref<?xindex>
// CHECK:                   %[[VAL_73:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_72]]] : memref<300xf64>
// CHECK:                   %[[VAL_74:.*]] = memref.load %[[VAL_35]]{{\[}}%[[VAL_70]]] : memref<?xf64>
// CHECK:                   %[[VAL_75:.*]] = arith.mulf %[[VAL_65]], %[[VAL_74]] : f64
// CHECK:                   %[[VAL_76:.*]] = arith.addf %[[VAL_73]], %[[VAL_75]] : f64
// CHECK:                   %[[VAL_77:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_72]]] : memref<300xi1>
// CHECK:                   %[[VAL_78:.*]] = arith.cmpi eq, %[[VAL_77]], %[[VAL_7]] : i1
// CHECK:                   %[[VAL_79:.*]] = scf.if %[[VAL_78]] -> (index) {
// CHECK:                     memref.store %[[VAL_8]], %[[VAL_22]]{{\[}}%[[VAL_72]]] : memref<300xi1>
// CHECK:                     memref.store %[[VAL_72]], %[[VAL_24]]{{\[}}%[[VAL_71]]] : memref<300xindex>
// CHECK:                     %[[VAL_80:.*]] = arith.addi %[[VAL_71]], %[[VAL_6]] : index
// CHECK:                     scf.yield %[[VAL_80]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_71]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[VAL_76]], %[[VAL_20]]{{\[}}%[[VAL_72]]] : memref<300xf64>
// CHECK:                   scf.yield %[[VAL_81:.*]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_82:.*]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_56]] : index
// CHECK:               }
// CHECK:               %[[VAL_83:.*]] = arith.addi %[[VAL_54]], %[[VAL_6]] : index
// CHECK:               %[[VAL_84:.*]] = arith.select %[[VAL_61]], %[[VAL_83]], %[[VAL_54]] : index
// CHECK:               %[[VAL_85:.*]] = arith.addi %[[VAL_55]], %[[VAL_6]] : index
// CHECK:               %[[VAL_86:.*]] = arith.select %[[VAL_62]], %[[VAL_85]], %[[VAL_55]] : index
// CHECK:               scf.yield %[[VAL_84]], %[[VAL_86]], %[[VAL_87:.*]] : index, index, index
// CHECK:             }
// CHECK:             func.call @expInsertF64(%[[VAL_19]], %[[VAL_37]], %[[VAL_21]], %[[VAL_23]], %[[VAL_25]], %[[VAL_88:.*]]#2) : (!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index) -> ()
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_20]] : memref<300xf64>
// CHECK:           memref.dealloc %[[VAL_22]] : memref<300xi1>
// CHECK:           memref.dealloc %[[VAL_24]] : memref<300xindex>
// CHECK:           call @endInsert(%[[VAL_19]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           return %[[VAL_19]] : !llvm.ptr<i8>
// CHECK:         }
func.func @fill_zero_after_alloc(%arg0: tensor<100x200xf64, #DCSR>,
                                 %arg1: tensor<200x300xf64, #DCSR>) -> tensor<100x300xf64, #DCSR> {
  %0 = bufferization.alloc_tensor() : tensor<100x300xf64, #DCSR>
  %cst = arith.constant 0.000000e+00 : f64
  %1 = linalg.fill ins(%cst : f64)
                   outs(%0 : tensor<100x300xf64, #DCSR>) -> tensor<100x300xf64, #DCSR>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<100x200xf64, #DCSR>, tensor<200x300xf64, #DCSR>)
                     outs(%1 : tensor<100x300xf64, #DCSR>) -> tensor<100x300xf64, #DCSR>
  return %2 : tensor<100x300xf64, #DCSR>
}
