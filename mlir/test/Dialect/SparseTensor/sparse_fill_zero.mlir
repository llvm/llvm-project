// RUN: mlir-opt %s --linalg-generalize-named-ops --pre-sparsification-rewrite --sparse-reinterpret-map --sparsification --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

// CHECK-LABEL:   func.func @fill_zero_after_alloc(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 100 : index
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 300 : index
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 262144 : i64
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<2xi64>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ALLOCA_0]] : memref<2xi64> to memref<?xi64>
// CHECK:           memref.store %[[CONSTANT_9]], %[[ALLOCA_0]]{{\[}}%[[CONSTANT_6]]] : memref<2xi64>
// CHECK:           memref.store %[[CONSTANT_9]], %[[ALLOCA_0]]{{\[}}%[[CONSTANT_5]]] : memref<2xi64>
// CHECK:           %[[ALLOCA_1:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[ALLOCA_1]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[CONSTANT_7]], %[[ALLOCA_1]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK:           memref.store %[[CONSTANT_8]], %[[ALLOCA_1]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK:           %[[ALLOCA_2:.*]] = memref.alloca() : memref<2xindex>
// CHECK:           %[[CAST_2:.*]] = memref.cast %[[ALLOCA_2]] : memref<2xindex> to memref<?xindex>
// CHECK:           memref.store %[[CONSTANT_6]], %[[ALLOCA_2]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK:           memref.store %[[CONSTANT_5]], %[[ALLOCA_2]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK:           %[[VAL_0:.*]] = call @newSparseTensor(%[[CAST_1]], %[[CAST_1]], %[[CAST_0]], %[[CAST_2]], %[[CAST_2]], %[[CONSTANT_2]], %[[CONSTANT_2]], %[[CONSTANT_1]], %[[CONSTANT_2]], %[[MLIR_0]]) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<300xf64>
// CHECK:           %[[CAST_3:.*]] = memref.cast %[[ALLOC_0]] : memref<300xf64> to memref<?xf64>
// CHECK:           %[[ALLOC_1:.*]] = memref.alloc() : memref<300xi1>
// CHECK:           %[[CAST_4:.*]] = memref.cast %[[ALLOC_1]] : memref<300xi1> to memref<?xi1>
// CHECK:           %[[ALLOC_2:.*]] = memref.alloc() : memref<300xindex>
// CHECK:           %[[CAST_5:.*]] = memref.cast %[[ALLOC_2]] : memref<300xindex> to memref<?xindex>
// CHECK:           linalg.fill ins(%[[CONSTANT_0]] : f64) outs(%[[ALLOC_0]] : memref<300xf64>)
// CHECK:           linalg.fill ins(%[[CONSTANT_4]] : i1) outs(%[[ALLOC_1]] : memref<300xi1>)
// CHECK:           %[[VAL_1:.*]] = call @sparseValuesF64(%[[ARG0]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK:           %[[VAL_2:.*]] = call @sparseValuesF64(%[[ARG1]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK:           %[[VAL_3:.*]] = call @sparsePositions0(%[[ARG0]], %[[CONSTANT_6]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_4:.*]] = call @sparseCoordinates0(%[[ARG0]], %[[CONSTANT_6]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_5:.*]] = call @sparsePositions0(%[[ARG0]], %[[CONSTANT_5]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = call @sparseCoordinates0(%[[ARG0]], %[[CONSTANT_5]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = call @sparsePositions0(%[[ARG1]], %[[CONSTANT_6]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = call @sparseCoordinates0(%[[ARG1]], %[[CONSTANT_6]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = call @sparsePositions0(%[[ARG1]], %[[CONSTANT_5]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = call @sparseCoordinates0(%[[ARG1]], %[[CONSTANT_5]]) : (!llvm.ptr, index) -> memref<?xindex>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[VAL_3]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[VAL_3]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_11:.*]] = %[[LOAD_0]] to %[[LOAD_1]] step %[[CONSTANT_5]] {
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             %[[LOAD_3:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_11]], %[[CONSTANT_5]] : index
// CHECK:             %[[LOAD_4:.*]] = memref.load %[[VAL_5]]{{\[}}%[[ADDI_0]]] : memref<?xindex>
// CHECK:             %[[LOAD_5:.*]] = memref.load %[[VAL_7]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK:             %[[LOAD_6:.*]] = memref.load %[[VAL_7]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK:             %[[WHILE_0:.*]]:3 = scf.while (%[[VAL_12:.*]] = %[[LOAD_3]], %[[VAL_13:.*]] = %[[LOAD_5]], %[[VAL_14:.*]] = %[[CONSTANT_6]]) : (index, index, index) -> (index, index, index) {
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_12]], %[[LOAD_4]] : index
// CHECK:               %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_13]], %[[LOAD_6]] : index
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:               scf.condition(%[[ANDI_0]]) %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index, %[[VAL_17:.*]]: index):
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_16]], %[[CONSTANT_5]] : index
// CHECK:               %[[LOAD_7:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:               %[[LOAD_8:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[CMPI_2:.*]] = arith.cmpi ult, %[[LOAD_8]], %[[LOAD_7]] : index
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[LOAD_8]], %[[LOAD_7]] : index
// CHECK:               %[[CMPI_3:.*]] = arith.cmpi eq, %[[LOAD_7]], %[[SELECT_0]] : index
// CHECK:               %[[CMPI_4:.*]] = arith.cmpi eq, %[[LOAD_8]], %[[SELECT_0]] : index
// CHECK:               %[[ANDI_1:.*]] = arith.andi %[[CMPI_3]], %[[CMPI_4]] : i1
// CHECK:               %[[IF_0:.*]] = scf.if %[[ANDI_1]] -> (index) {
// CHECK:                 %[[LOAD_9:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_15]]] : memref<?xf64>
// CHECK:                 %[[LOAD_10:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:                 %[[LOAD_11:.*]] = memref.load %[[VAL_9]]{{\[}}%[[ADDI_1]]] : memref<?xindex>
// CHECK:                 %[[FOR_0:.*]] = scf.for %[[VAL_18:.*]] = %[[LOAD_10]] to %[[LOAD_11]] step %[[CONSTANT_5]] iter_args(%[[VAL_19:.*]] = %[[VAL_17]]) -> (index) {
// CHECK:                   %[[LOAD_12:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:                   %[[LOAD_13:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[LOAD_12]]] : memref<300xf64>
// CHECK:                   %[[LOAD_14:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:                   %[[MULF_0:.*]] = arith.mulf %[[LOAD_9]], %[[LOAD_14]] : f64
// CHECK:                   %[[ADDF_0:.*]] = arith.addf %[[LOAD_13]], %[[MULF_0]] : f64
// CHECK:                   %[[LOAD_15:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[LOAD_12]]] : memref<300xi1>
// CHECK:                   %[[CMPI_5:.*]] = arith.cmpi eq, %[[LOAD_15]], %[[CONSTANT_4]] : i1
// CHECK:                   %[[IF_1:.*]] = scf.if %[[CMPI_5]] -> (index) {
// CHECK:                     memref.store %[[CONSTANT_3]], %[[ALLOC_1]]{{\[}}%[[LOAD_12]]] : memref<300xi1>
// CHECK:                     memref.store %[[LOAD_12]], %[[ALLOC_2]]{{\[}}%[[VAL_19]]] : memref<300xindex>
// CHECK:                     %[[ADDI_2:.*]] = arith.addi %[[VAL_19]], %[[CONSTANT_5]] : index
// CHECK:                     scf.yield %[[ADDI_2]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_19]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[ADDF_0]], %[[ALLOC_0]]{{\[}}%[[LOAD_12]]] : memref<300xf64>
// CHECK:                   scf.yield %[[IF_1]] : index
// CHECK:                 } {"Emitted from" = "linalg.generic"}
// CHECK:                 scf.yield %[[FOR_0]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_17]] : index
// CHECK:               }
// CHECK:               %[[ADDI_3:.*]] = arith.addi %[[VAL_15]], %[[CONSTANT_5]] : index
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[CMPI_3]], %[[ADDI_3]], %[[VAL_15]] : index
// CHECK:               %[[SELECT_2:.*]] = arith.select %[[CMPI_4]], %[[ADDI_1]], %[[VAL_16]] : index
// CHECK:               scf.yield %[[SELECT_1]], %[[SELECT_2]], %[[IF_0]] : index, index, index
// CHECK:             }
// CHECK:             %[[ALLOCA_3:.*]] = memref.alloca() : memref<2xindex>
// CHECK:             %[[CAST_6:.*]] = memref.cast %[[ALLOCA_3]] : memref<2xindex> to memref<?xindex>
// CHECK:             memref.store %[[LOAD_2]], %[[ALLOCA_3]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK:             func.call @expInsertF64(%[[VAL_0]], %[[CAST_6]], %[[CAST_3]], %[[CAST_4]], %[[CAST_5]], %[[VAL_20:.*]]#2) : (!llvm.ptr, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index) -> ()
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<300xf64>
// CHECK:           memref.dealloc %[[ALLOC_1]] : memref<300xi1>
// CHECK:           memref.dealloc %[[ALLOC_2]] : memref<300xindex>
// CHECK:           call @endLexInsert(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           return %[[VAL_0]] : !llvm.ptr
// CHECK:         }
func.func @fill_zero_after_alloc(%arg0: tensor<100x200xf64, #DCSR>,
                                       %arg1: tensor<200x300xf64, #DCSR>) -> tensor<100x300xf64, #DCSR> {
  %0 = tensor.empty() : tensor<100x300xf64, #DCSR>
  %cst = arith.constant 0.000000e+00 : f64
  %1 = linalg.fill ins(%cst : f64)
                   outs(%0 : tensor<100x300xf64, #DCSR>) -> tensor<100x300xf64, #DCSR>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<100x200xf64, #DCSR>, tensor<200x300xf64, #DCSR>)
                     outs(%1 : tensor<100x300xf64, #DCSR>) -> tensor<100x300xf64, #DCSR>
  return %2 : tensor<100x300xf64, #DCSR>
}
