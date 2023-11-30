// RUN: mlir-opt %s \
// RUN: --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN: --sparse-reinterpret-map --sparsification | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

// CHECK-LABEL:   func.func @matmul1(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10x20xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<20x30xf32>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<10x30xf32>) -> tensor<10x30xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 30 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_1]] : memref<20x30xf32>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_2]] : memref<10x30xf32>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_5]] {
// CHECK:             %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_5]] : index
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_5]] {
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_20]]] : memref<?xf32>
// CHECK:               scf.for %[[VAL_23:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_24:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_16]], %[[VAL_23]]] : memref<10x30xf32>
// CHECK:                 %[[VAL_25:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_21]], %[[VAL_23]]] : memref<20x30xf32>
// CHECK:                 %[[VAL_26:.*]] = arith.mulf %[[VAL_22]], %[[VAL_25]] : f32
// CHECK:                 %[[VAL_27:.*]] = arith.addf %[[VAL_24]], %[[VAL_26]] : f32
// CHECK:                 memref.store %[[VAL_27]], %[[VAL_12]]{{\[}}%[[VAL_16]], %[[VAL_23]]] : memref<10x30xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<10x30xf32>
// CHECK:           return %[[VAL_28]] : tensor<10x30xf32>
// CHECK:         }
func.func @matmul1(%a: tensor<10x20xf32, #DCSR>,
              %b: tensor<20x30xf32>,
              %c: tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul
    ins(%a, %b: tensor<10x20xf32, #DCSR>, tensor<20x30xf32>)
    outs(%c: tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}

// CHECK-LABEL:   func.func @matmul_sparse_rhs(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10x20xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<20x30xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<10x30xf32>) -> tensor<10x30xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<10x20xf32>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index}
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index}
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index}
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index}
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_1]]
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_2]] : memref<10x30xf32>
// CHECK:           scf.for %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_5]] {
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]], %[[VAL_17]]] : memref<10x20xf32>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_16]], %[[VAL_5]] : index
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:               scf.for %[[VAL_22:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_23:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK:                 %[[VAL_24:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_23]]] : memref<10x30xf32>
// CHECK:                 %[[VAL_25:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_22]]] : memref<?xf32>
// CHECK:                 %[[VAL_26:.*]] = arith.mulf %[[VAL_18]], %[[VAL_25]] : f32
// CHECK:                 %[[VAL_27:.*]] = arith.addf %[[VAL_24]], %[[VAL_26]] : f32
// CHECK:                 memref.store %[[VAL_27]], %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_23]]] : memref<10x30xf32>
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_28:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<10x30xf32>
// CHECK:           return %[[VAL_28]] : tensor<10x30xf32>
// CHECK:         }
// IMPORTANT! Ensures that dense input are visit in row-major order.
func.func @matmul_sparse_rhs(%a: tensor<10x20xf32>,
                             %b: tensor<20x30xf32, #DCSR>,
                             %c: tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul
    ins(%a, %b: tensor<10x20xf32>, tensor<20x30xf32,#DCSR>)
    outs(%c: tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}


//
// Computes C = A x B with all matrices sparse (SpMSpM) in DCSR.
//
// CHECK-LABEL:   func.func @matmul2(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<4x8xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<8x4xf64, #sparse{{[0-9]*}}>) -> tensor<4x4xf64, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.empty() : tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<4x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<4x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<4x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<4x8xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<4x8xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<8x4xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<8x4xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<8x4xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<8x4xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<8x4xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_18]] step %[[VAL_3]] iter_args(%[[VAL_21:.*]] = %[[VAL_6]]) -> (tensor<4x4xf64, #sparse{{[0-9]*}}>) {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]] = sparse_tensor.expand %[[VAL_6]] : tensor<4x4xf64, #sparse{{[0-9]*}}> to memref<?xf64>, memref<?xi1>, memref<?xindex>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_20]], %[[VAL_3]] : index
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_28]]] : memref<?xindex>
// CHECK:             %[[VAL_30:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:             %[[VAL_32:.*]]:4 = scf.while (%[[VAL_33:.*]] = %[[VAL_27]], %[[VAL_34:.*]] = %[[VAL_30]], %[[VAL_35:.*]] = %[[VAL_26]], %[[VAL_36:.*]] = %[[VAL_21]]) : (index, index, index, tensor<4x4xf64, #sparse{{[0-9]*}}>) -> (index, index, index, tensor<4x4xf64, #sparse{{[0-9]*}}>) {
// CHECK:               %[[VAL_37:.*]] = arith.cmpi ult, %[[VAL_33]], %[[VAL_29]] : index
// CHECK:               %[[VAL_38:.*]] = arith.cmpi ult, %[[VAL_34]], %[[VAL_31]] : index
// CHECK:               %[[VAL_39:.*]] = arith.andi %[[VAL_37]], %[[VAL_38]] : i1
// CHECK:               scf.condition(%[[VAL_39]]) %[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]] : index, index, index, tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_40:.*]]: index, %[[VAL_41:.*]]: index, %[[VAL_42:.*]]: index, %[[VAL_43:.*]]: tensor<4x4xf64, #sparse{{[0-9]*}}>):
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:               %[[VAL_45:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_41]]] : memref<?xindex>
// CHECK:               %[[VAL_46:.*]] = arith.cmpi ult, %[[VAL_45]], %[[VAL_44]] : index
// CHECK:               %[[VAL_47:.*]] = arith.select %[[VAL_46]], %[[VAL_45]], %[[VAL_44]] : index
// CHECK:               %[[VAL_48:.*]] = arith.cmpi eq, %[[VAL_44]], %[[VAL_47]] : index
// CHECK:               %[[VAL_49:.*]] = arith.cmpi eq, %[[VAL_45]], %[[VAL_47]] : index
// CHECK:               %[[VAL_50:.*]] = arith.andi %[[VAL_48]], %[[VAL_49]] : i1
// CHECK:               %[[VAL_51:.*]]:2 = scf.if %[[VAL_50]] -> (index, tensor<4x4xf64, #sparse{{[0-9]*}}>) {
// CHECK:                 %[[VAL_52:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_40]]] : memref<?xf64>
// CHECK:                 %[[VAL_53:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_41]]] : memref<?xindex>
// CHECK:                 %[[VAL_54:.*]] = arith.addi %[[VAL_41]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_55:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_54]]] : memref<?xindex>
// CHECK:                 %[[VAL_56:.*]] = scf.for %[[VAL_57:.*]] = %[[VAL_53]] to %[[VAL_55]] step %[[VAL_3]] iter_args(%[[VAL_58:.*]] = %[[VAL_42]]) -> (index) {
// CHECK:                   %[[VAL_59:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_57]]] : memref<?xindex>
// CHECK:                   %[[VAL_60:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_59]]] : memref<?xf64>
// CHECK:                   %[[VAL_61:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_57]]] : memref<?xf64>
// CHECK:                   %[[VAL_62:.*]] = arith.mulf %[[VAL_52]], %[[VAL_61]] : f64
// CHECK:                   %[[VAL_63:.*]] = arith.addf %[[VAL_60]], %[[VAL_62]] : f64
// CHECK:                   %[[VAL_64:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_59]]] : memref<?xi1>
// CHECK:                   %[[VAL_65:.*]] = arith.cmpi eq, %[[VAL_64]], %[[VAL_4]] : i1
// CHECK:                   %[[VAL_66:.*]] = scf.if %[[VAL_65]] -> (index) {
// CHECK:                     memref.store %[[VAL_5]], %[[VAL_24]]{{\[}}%[[VAL_59]]] : memref<?xi1>
// CHECK:                     memref.store %[[VAL_59]], %[[VAL_25]]{{\[}}%[[VAL_58]]] : memref<?xindex>
// CHECK:                     %[[VAL_67:.*]] = arith.addi %[[VAL_58]], %[[VAL_3]] : index
// CHECK:                     scf.yield %[[VAL_67]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_58]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[VAL_63]], %[[VAL_23]]{{\[}}%[[VAL_59]]] : memref<?xf64>
// CHECK:                   scf.yield %[[VAL_68:.*]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_69:.*]], %[[VAL_43]] : index, tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_42]], %[[VAL_43]] : index, tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:               }
// CHECK:               %[[VAL_70:.*]] = arith.cmpi eq, %[[VAL_44]], %[[VAL_47]] : index
// CHECK:               %[[VAL_71:.*]] = arith.addi %[[VAL_40]], %[[VAL_3]] : index
// CHECK:               %[[VAL_72:.*]] = arith.select %[[VAL_70]], %[[VAL_71]], %[[VAL_40]] : index
// CHECK:               %[[VAL_73:.*]] = arith.cmpi eq, %[[VAL_45]], %[[VAL_47]] : index
// CHECK:               %[[VAL_74:.*]] = arith.addi %[[VAL_41]], %[[VAL_3]] : index
// CHECK:               %[[VAL_75:.*]] = arith.select %[[VAL_73]], %[[VAL_74]], %[[VAL_41]] : index
// CHECK:               scf.yield %[[VAL_72]], %[[VAL_75]], %[[VAL_76:.*]]#0, %[[VAL_76]]#1 : index, index, index, tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:             }
// CHECK:             %[[VAL_77:.*]] = sparse_tensor.compress %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_78:.*]]#2 into %[[VAL_78]]#3{{\[}}%[[VAL_22]]] : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:             scf.yield %[[VAL_77]] : tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:           }
// CHECK:           %[[VAL_79:.*]] = sparse_tensor.load %[[VAL_80:.*]] hasInserts : tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_79]] : tensor<4x4xf64, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @matmul2(%A: tensor<4x8xf64, #DCSR>,
              %B: tensor<8x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
  %c4 = arith.constant 4 : index
  %C = tensor.empty() : tensor<4x4xf64, #DCSR>
  %D = linalg.matmul
    ins(%A, %B: tensor<4x8xf64, #DCSR>, tensor<8x4xf64, #DCSR>)
       outs(%C: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR>
  return %D: tensor<4x4xf64, #DCSR>
}


// CHECK-LABEL:   func.func @conv2d(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xi32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x3xi32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<6x6xi32>) -> tensor<6x6xi32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 6 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<8x8xi32>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<3x3xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<3x3xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<3x3xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<3x3xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<3x3xi32, #sparse{{[0-9]*}}> to memref<?xi32>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_2]] : memref<6x6xi32>
// CHECK:           scf.for %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:               %[[VAL_15:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<6x6xi32>
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:               %[[VAL_18:.*]] = scf.for %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_5]] iter_args(%[[VAL_20:.*]] = %[[VAL_15]]) -> (i32) {
// CHECK:                 %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:                 %[[VAL_22:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:                 %[[VAL_23:.*]] = arith.addi %[[VAL_19]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_23]]] : memref<?xindex>
// CHECK:                 %[[VAL_25:.*]] = scf.for %[[VAL_26:.*]] = %[[VAL_22]] to %[[VAL_24]] step %[[VAL_5]] iter_args(%[[VAL_27:.*]] = %[[VAL_20]]) -> (i32) {
// CHECK:                   %[[VAL_28:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_26]]] : memref<?xindex>
// CHECK:                   %[[VAL_29:.*]] = arith.addi %[[VAL_13]], %[[VAL_21]] : index
// CHECK:                   %[[VAL_30:.*]] = arith.addi %[[VAL_14]], %[[VAL_28]] : index
// CHECK:                   %[[VAL_31:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_29]], %[[VAL_30]]] : memref<8x8xi32>
// CHECK:                   %[[VAL_32:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_26]]] : memref<?xi32>
// CHECK:                   %[[VAL_33:.*]] = arith.muli %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:                   %[[VAL_34:.*]] = arith.addi %[[VAL_27]], %[[VAL_33]] : i32
// CHECK:                   scf.yield %[[VAL_34]] : i32
// CHECK:                 } {"Emitted from" = "linalg.generic"}
// CHECK:                 scf.yield %[[VAL_25]] : i32
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               memref.store %[[VAL_18]], %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<6x6xi32>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_35:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<6x6xi32>
// CHECK:           return %[[VAL_35]] : tensor<6x6xi32>
// CHECK:         }
func.func @conv2d(%input:  tensor<8x8xi32>,
             %filter: tensor<3x3xi32, #DCSR>,
             %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = linalg.conv_2d
    ins  (%input, %filter: tensor<8x8xi32>, tensor<3x3xi32, #DCSR>)
    outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
  return %0 : tensor<6x6xi32>
}

// CHECK-LABEL:   func.func @quantized_matmul(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<5x3xi8>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x6xi8, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<5x6xi64>) -> tensor<5x6xi64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_0]] : memref<5x3xi8>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<3x6xi8, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<3x6xi8, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<3x6xi8, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<3x6xi8, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<3x6xi8, #sparse{{[0-9]*}}> to memref<?xi8>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_memref %[[VAL_2]] : memref<5x6xi64>
// CHECK:           scf.for %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:             %[[VAL_16:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_17:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_5]] {
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_14]], %[[VAL_18]]] : memref<5x3xi8>
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:               %[[VAL_21:.*]] = arith.addi %[[VAL_17]], %[[VAL_5]] : index
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:               scf.for %[[VAL_23:.*]] = %[[VAL_20]] to %[[VAL_22]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_24:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_23]]] : memref<?xindex>
// CHECK:                 %[[VAL_25:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_14]], %[[VAL_24]]] : memref<5x6xi64>
// CHECK:                 %[[VAL_26:.*]] = arith.extsi %[[VAL_19]] : i8 to i64
// CHECK:                 %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_6]] : i64
// CHECK:                 %[[VAL_28:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_23]]] : memref<?xi8>
// CHECK:                 %[[VAL_29:.*]] = arith.extsi %[[VAL_28]] : i8 to i64
// CHECK:                 %[[VAL_30:.*]] = arith.muli %[[VAL_27]], %[[VAL_29]] : i64
// CHECK:                 %[[VAL_31:.*]] = arith.addi %[[VAL_25]], %[[VAL_30]] : i64
// CHECK:                 memref.store %[[VAL_31]], %[[VAL_13]]{{\[}}%[[VAL_14]], %[[VAL_24]]] : memref<5x6xi64>
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_32:.*]] = bufferization.to_tensor %[[VAL_13]] : memref<5x6xi64>
// CHECK:           return %[[VAL_32]] : tensor<5x6xi64>
// CHECK:         }
func.func @quantized_matmul(%input1: tensor<5x3xi8>,
                       %input2: tensor<3x6xi8, #DCSR>,
                       %output: tensor<5x6xi64>) -> tensor<5x6xi64> {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %0 = linalg.quantized_matmul
    ins(%input1, %input2, %c2, %c0 : tensor<5x3xi8>, tensor<3x6xi8, #DCSR>, i32, i32)
    outs(%output : tensor<5x6xi64>) -> tensor<5x6xi64>
  return %0: tensor<5x6xi64>
}

// CHECK-LABEL:   func.func @sparse_dot(
// CHECK-SAME:      %[[VAL_0:.*0]]: tensor<1024xf32, #sparse{{[0-9]*}}>, %[[VAL_1:.*1]]: tensor<1024xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_2:.*2]]: tensor<f32>) -> tensor<f32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<1024xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<1024xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<1024xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<1024xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<1024xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<1024xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_2]] : memref<f32>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_11]][] : memref<f32>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_17:.*]]:3 = scf.while (%[[VAL_18:.*]] = %[[VAL_13]], %[[VAL_19:.*]] = %[[VAL_15]], %[[VAL_20:.*]] = %[[VAL_12]]) : (index, index, f32) -> (index, index, f32) {
// CHECK:             %[[VAL_21:.*]] = arith.cmpi ult, %[[VAL_18]], %[[VAL_14]] : index
// CHECK:             %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_19]], %[[VAL_16]] : index
// CHECK:             %[[VAL_23:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// CHECK:             scf.condition(%[[VAL_23]]) %[[VAL_18]], %[[VAL_19]], %[[VAL_20]] : index, index, f32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_24:.*]]: index, %[[VAL_25:.*]]: index, %[[VAL_26:.*]]: f32):
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_24]]] : memref<?xindex>
// CHECK:             %[[VAL_28:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_25]]] : memref<?xindex>
// CHECK:             %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_27]] : index
// CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_28]], %[[VAL_27]] : index
// CHECK:             %[[VAL_31:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_30]] : index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:             %[[VAL_33:.*]] = arith.andi %[[VAL_31]], %[[VAL_32]] : i1
// CHECK:             %[[VAL_34:.*]] = scf.if %[[VAL_33]] -> (f32) {
// CHECK:               %[[VAL_35:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// CHECK:               %[[VAL_36:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_25]]] : memref<?xf32>
// CHECK:               %[[VAL_37:.*]] = arith.mulf %[[VAL_35]], %[[VAL_36]] : f32
// CHECK:               %[[VAL_38:.*]] = arith.addf %[[VAL_26]], %[[VAL_37]] : f32
// CHECK:               scf.yield %[[VAL_38]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_26]] : f32
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_30]] : index
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_24]], %[[VAL_4]] : index
// CHECK:             %[[VAL_41:.*]] = arith.select %[[VAL_39]], %[[VAL_40]], %[[VAL_24]] : index
// CHECK:             %[[VAL_42:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_25]], %[[VAL_4]] : index
// CHECK:             %[[VAL_44:.*]] = arith.select %[[VAL_42]], %[[VAL_43]], %[[VAL_25]] : index
// CHECK:             scf.yield %[[VAL_41]], %[[VAL_44]], %[[VAL_45:.*]] : index, index, f32
// CHECK:           }
// CHECK:           memref.store %[[VAL_46:.*]]#2, %[[VAL_11]][] : memref<f32>
// CHECK:           %[[VAL_47:.*]] = bufferization.to_tensor %[[VAL_11]] : memref<f32>
// CHECK:           return %[[VAL_47]] : tensor<f32>
// CHECK:         }
func.func @sparse_dot(%a: tensor<1024xf32, #SparseVector>,
                 %b: tensor<1024xf32, #SparseVector>,
		 %x: tensor<f32>) -> tensor<f32> {
  %dot = linalg.dot ins(%a, %b: tensor<1024xf32, #SparseVector>,
                                tensor<1024xf32, #SparseVector>)
                   outs(%x: tensor<f32>) -> tensor<f32>
  return %dot : tensor<f32>
}
