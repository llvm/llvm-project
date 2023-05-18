// RUN: mlir-opt %s --linalg-generalize-named-ops --pre-sparsification-rewrite --sparsification --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#DCSR = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>

// CHECK-LABEL:   func.func @fill_zero_after_alloc(
// CHECK-SAME:      %[[Arg0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:      %[[Arg1:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[F0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[I0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[I1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[False:.*]] = arith.constant false
// CHECK-DAG:       %[[True:.*]] = arith.constant true
// CHECK-DAG:       %[[I100:.*]] = arith.constant 100 : index
// CHECK-DAG:       %[[I300:.*]] = arith.constant 300 : index
// CHECK-DAG:       %[[CompressedDLT:.*]] = arith.constant 8 : i8
// CHECK-DAG:       %[[LvlTypes:.*]] = memref.alloca() : memref<2xi8>
// CHECK-DAG:       %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<2xi8> to memref<?xi8>
// CHECK-DAG:       memref.store %[[CompressedDLT]], %[[LvlTypes]]{{\[}}%[[I0]]] : memref<2xi8>
// CHECK-DAG:       memref.store %[[CompressedDLT]], %[[LvlTypes]]{{\[}}%[[I1]]] : memref<2xi8>
// CHECK-DAG:       %[[DimSizes:.*]] = memref.alloca() : memref<2xindex>
// CHECK-DAG:       %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<2xindex> to memref<?xindex>
// CHECK-DAG:       memref.store %[[I100]], %[[DimSizes]]{{\[}}%[[I0]]] : memref<2xindex>
// CHECK-DAG:       memref.store %[[I300]], %[[DimSizes]]{{\[}}%[[I1]]] : memref<2xindex>
// CHECK-DAG:       %[[LvlSizes:.*]] = memref.alloca() : memref<2xindex>
// CHECK-DAG:       %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<2xindex> to memref<?xindex>
// CHECK-DAG:       memref.store %[[I100]], %[[LvlSizes]]{{\[}}%[[I0]]] : memref<2xindex>
// CHECK-DAG:       memref.store %[[I300]], %[[LvlSizes]]{{\[}}%[[I1]]] : memref<2xindex>
// CHECK-DAG:       %[[Iota:.*]] = memref.alloca() : memref<2xindex>
// CHECK-DAG:       %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<2xindex> to memref<?xindex>
// CHECK-DAG:       memref.store %[[I0]], %[[Iota]]{{\[}}%[[I0]]] : memref<2xindex>
// CHECK-DAG:       memref.store %[[I1]], %[[Iota]]{{\[}}%[[I1]]] : memref<2xindex>
// CHECK-DAG:       %[[NullPtr:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           %[[VAL_19:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %[[C0]], %[[C0]], %[[C1]], %[[C0]], %[[NullPtr]]) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_20:.*]] = memref.alloc() : memref<300xf64>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_20]] : memref<300xf64> to memref<?xf64>
// CHECK:           %[[VAL_22:.*]] = memref.alloc() : memref<300xi1>
// CHECK:           %[[VAL_23:.*]] = memref.cast %[[VAL_22]] : memref<300xi1> to memref<?xi1>
// CHECK:           %[[VAL_24:.*]] = memref.alloc() : memref<300xindex>
// CHECK:           %[[VAL_25:.*]] = memref.cast %[[VAL_24]] : memref<300xindex> to memref<?xindex>
// CHECK:           linalg.fill ins(%[[F0]] : f64) outs(%[[VAL_20]] : memref<300xf64>)
// CHECK:           linalg.fill ins(%[[False]] : i1) outs(%[[VAL_22]] : memref<300xi1>)
// CHECK:           %[[VAL_26:.*]] = call @sparsePositions0(%[[Arg0]], %[[I0]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_27:.*]] = call @sparseCoordinates0(%[[Arg0]], %[[I0]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_28:.*]] = call @sparsePositions0(%[[Arg0]], %[[I1]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_29:.*]] = call @sparseCoordinates0(%[[Arg0]], %[[I1]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_30:.*]] = call @sparseValuesF64(%[[Arg0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK:           %[[VAL_31:.*]] = call @sparsePositions0(%[[Arg1]], %[[I0]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_32:.*]] = call @sparseCoordinates0(%[[Arg1]], %[[I0]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_33:.*]] = call @sparsePositions0(%[[Arg1]], %[[I1]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_34:.*]] = call @sparseCoordinates0(%[[Arg1]], %[[I1]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK:           %[[VAL_35:.*]] = call @sparseValuesF64(%[[Arg1]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK:           %[[VAL_36:.*]] = memref.load %[[VAL_26]]{{\[}}%[[I0]]] : memref<?xindex>
// CHECK:           %[[VAL_37:.*]] = memref.load %[[VAL_26]]{{\[}}%[[I1]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_38:.*]] = %[[VAL_36]] to %[[VAL_37]] step %[[I1]] {
// CHECK:             %[[VAL_39:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_38]]] : memref<?xindex>
// CHECK:             %[[VAL_40:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_38]]] : memref<?xindex>
// CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_38]], %[[I1]] : index
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_41]]] : memref<?xindex>
// CHECK:             %[[VAL_43:.*]] = memref.load %[[VAL_31]]{{\[}}%[[I0]]] : memref<?xindex>
// CHECK:             %[[VAL_44:.*]] = memref.load %[[VAL_31]]{{\[}}%[[I1]]] : memref<?xindex>
// CHECK:             %[[VAL_45:.*]]:3 = scf.while (%[[VAL_46:.*]] = %[[VAL_40]], %[[VAL_47:.*]] = %[[VAL_43]], %[[VAL_48:.*]] = %[[I0]]) : (index, index, index) -> (index, index, index) {
// CHECK:               %[[VAL_49:.*]] = arith.cmpi ult, %[[VAL_46]], %[[VAL_42]] : index
// CHECK:               %[[VAL_50:.*]] = arith.cmpi ult, %[[VAL_47]], %[[VAL_44]] : index
// CHECK:               %[[VAL_51:.*]] = arith.andi %[[VAL_49]], %[[VAL_50]] : i1
// CHECK:               scf.condition(%[[VAL_51]]) %[[VAL_46]], %[[VAL_47]], %[[VAL_48]] : index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_52:.*]]: index, %[[VAL_53:.*]]: index, %[[VAL_54:.*]]: index):
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_52]]] : memref<?xindex>
// CHECK:               %[[VAL_56:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_53]]] : memref<?xindex>
// CHECK:               %[[VAL_57:.*]] = arith.cmpi ult, %[[VAL_56]], %[[VAL_55]] : index
// CHECK:               %[[VAL_58:.*]] = arith.select %[[VAL_57]], %[[VAL_56]], %[[VAL_55]] : index
// CHECK:               %[[VAL_59:.*]] = arith.cmpi eq, %[[VAL_55]], %[[VAL_58]] : index
// CHECK:               %[[VAL_60:.*]] = arith.cmpi eq, %[[VAL_56]], %[[VAL_58]] : index
// CHECK:               %[[VAL_61:.*]] = arith.andi %[[VAL_59]], %[[VAL_60]] : i1
// CHECK:               %[[VAL_62:.*]] = scf.if %[[VAL_61]] -> (index) {
// CHECK:                 %[[VAL_63:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_52]]] : memref<?xf64>
// CHECK:                 %[[VAL_64:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_53]]] : memref<?xindex>
// CHECK:                 %[[VAL_65:.*]] = arith.addi %[[VAL_53]], %[[I1]] : index
// CHECK:                 %[[VAL_66:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_65]]] : memref<?xindex>
// CHECK:                 %[[VAL_67:.*]] = scf.for %[[VAL_68:.*]] = %[[VAL_64]] to %[[VAL_66]] step %[[I1]] iter_args(%[[VAL_69:.*]] = %[[VAL_54]]) -> (index) {
// CHECK:                   %[[VAL_70:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_68]]] : memref<?xindex>
// CHECK:                   %[[VAL_71:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_70]]] : memref<300xf64>
// CHECK:                   %[[VAL_72:.*]] = memref.load %[[VAL_35]]{{\[}}%[[VAL_68]]] : memref<?xf64>
// CHECK:                   %[[VAL_73:.*]] = arith.mulf %[[VAL_63]], %[[VAL_72]] : f64
// CHECK:                   %[[VAL_74:.*]] = arith.addf %[[VAL_71]], %[[VAL_73]] : f64
// CHECK:                   %[[VAL_75:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_70]]] : memref<300xi1>
// CHECK:                   %[[VAL_76:.*]] = arith.cmpi eq, %[[VAL_75]], %[[False]] : i1
// CHECK:                   %[[VAL_77:.*]] = scf.if %[[VAL_76]] -> (index) {
// CHECK:                     memref.store %[[True]], %[[VAL_22]]{{\[}}%[[VAL_70]]] : memref<300xi1>
// CHECK:                     memref.store %[[VAL_70]], %[[VAL_24]]{{\[}}%[[VAL_69]]] : memref<300xindex>
// CHECK:                     %[[VAL_78:.*]] = arith.addi %[[VAL_69]], %[[I1]] : index
// CHECK:                     scf.yield %[[VAL_78]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_69]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[VAL_74]], %[[VAL_20]]{{\[}}%[[VAL_70]]] : memref<300xf64>
// CHECK:                   scf.yield %[[VAL_79:.*]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_80:.*]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_54]] : index
// CHECK:               }
// CHECK:               %[[VAL_81:.*]] = arith.addi %[[VAL_52]], %[[I1]] : index
// CHECK:               %[[VAL_82:.*]] = arith.select %[[VAL_59]], %[[VAL_81]], %[[VAL_52]] : index
// CHECK:               %[[VAL_83:.*]] = arith.addi %[[VAL_53]], %[[I1]] : index
// CHECK:               %[[VAL_84:.*]] = arith.select %[[VAL_60]], %[[VAL_83]], %[[VAL_53]] : index
// CHECK:               scf.yield %[[VAL_82]], %[[VAL_84]], %[[VAL_85:.*]] : index, index, index
// CHECK:             }
// CHECK:             %[[VAL_86:.*]] = memref.alloca() : memref<2xindex>
// CHECK:             %[[VAL_87:.*]] = memref.cast %[[VAL_86]] : memref<2xindex> to memref<?xindex>
// CHECK:             memref.store %[[VAL_39]], %[[VAL_86]]{{\[}}%[[I0]]] : memref<2xindex>
// CHECK:             func.call @expInsertF64(%[[VAL_19]], %[[VAL_87]], %[[VAL_21]], %[[VAL_23]], %[[VAL_25]], %[[VAL_88:.*]]#2) : (!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index) -> ()
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
