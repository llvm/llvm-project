// RUN: mlir-opt %s -sparsification | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

#SparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ]
}>

#trait_scale_inpl = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) *= 2 or X(i,j) += X(i,j)"
}

// CHECK-LABEL:   func.func @sparse_simply_dynamic1(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> to memref<?xf32>
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_3]] {
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_3]] : index
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_3]] {
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xf32>
// CHECK:               %[[VAL_17:.*]] = arith.mulf %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:               memref.store %[[VAL_17]], %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           return %[[VAL_18]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:         }
func.func @sparse_simply_dynamic1(%argx: tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR> {
  %c = arith.constant 2.0 : f32
  %0 = linalg.generic #trait_scale_inpl
    outs(%argx: tensor<32x16xf32, #DCSR>) {
      ^bb(%x: f32):
        %1 = arith.mulf %x, %c : f32
        linalg.yield %1 : f32
  } -> tensor<32x16xf32, #DCSR>
  return %0 : tensor<32x16xf32, #DCSR>
}

// CHECK-LABEL:   func.func @sparse_simply_dynamic2(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           %[[VAL_6:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_2]] {
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_2]] : index
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_11]] step %[[VAL_2]] {
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:               %[[VAL_14:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:               %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:               memref.store %[[VAL_15]], %[[VAL_5]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           return %[[VAL_16]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:         }
func.func @sparse_simply_dynamic2(%argx: tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR> {
  %0 = linalg.generic #trait_scale_inpl
    outs(%argx: tensor<32x16xf32, #DCSR>) {
      ^bb(%x: f32):
        %1 = arith.addf %x, %x : f32
        linalg.yield %1 : f32
  } -> tensor<32x16xf32, #DCSR>
  return %0 : tensor<32x16xf32, #DCSR>
}

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * 2.0"
}

// CHECK-LABEL:   func.func @sparse_truly_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = bufferization.alloc_tensor() : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xf32>
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_3]] {
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_9]]] : memref<?xindex>
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_3]] : index
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_3]] {
// CHECK:               %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:               %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_13]]] : memref<?xf32>
// CHECK:               %[[VAL_16:.*]] = arith.mulf %[[VAL_15]], %[[VAL_4]] : f32
// CHECK:               sparse_tensor.insert %[[VAL_16]] into %[[VAL_5]]{{\[}}%[[VAL_9]], %[[VAL_14]]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.load %[[VAL_5]] hasInserts : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           return %[[VAL_17]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:         }
func.func @sparse_truly_dynamic(%arga: tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #DCSR> {
  %s = arith.constant 2.0 : f32
  %xm = bufferization.alloc_tensor() : tensor<10x20xf32, #DCSR>
  %0 = linalg.generic #trait_scale
     ins(%arga: tensor<10x20xf32, #CSR>)
      outs(%xm: tensor<10x20xf32, #DCSR>) {
      ^bb(%a: f32, %x: f32):
        %1 = arith.mulf %a, %s : f32
        linalg.yield %1 : f32
  } -> tensor<10x20xf32, #DCSR>
  return %0 : tensor<10x20xf32, #DCSR>
}

#trait_sumred = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>, // A
    affine_map<(i,j,k) -> (i,j,k)>, // B
    affine_map<(i,j,k) -> (i,j)>    // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) = SUM_k A(i,j,k) * B(i,j,k)"
}

// CHECK-LABEL:   func.func @sumred(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>>) -> tensor<?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_7:.*]] = bufferization.alloc_tensor(%[[VAL_5]], %[[VAL_6]]) : tensor<?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 0 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 2 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 2 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xi32>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 0 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 1 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 2 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 2 : index} : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ] }>> to memref<?xi32>
// CHECK:           %[[VAL_22:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_25:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_26:.*]]:2 = scf.while (%[[VAL_27:.*]] = %[[VAL_22]], %[[VAL_28:.*]] = %[[VAL_24]]) : (index, index) -> (index, index) {
// CHECK:             %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_27]], %[[VAL_23]] : index
// CHECK:             %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_25]] : index
// CHECK:             %[[VAL_31:.*]] = arith.andi %[[VAL_29]], %[[VAL_30]] : i1
// CHECK:             scf.condition(%[[VAL_31]]) %[[VAL_27]], %[[VAL_28]] : index, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index):
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_32]]] : memref<?xindex>
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_33]]] : memref<?xindex>
// CHECK:             %[[VAL_36:.*]] = arith.cmpi ult, %[[VAL_35]], %[[VAL_34]] : index
// CHECK:             %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_35]], %[[VAL_34]] : index
// CHECK:             %[[VAL_38:.*]] = arith.cmpi eq, %[[VAL_34]], %[[VAL_37]] : index
// CHECK:             %[[VAL_39:.*]] = arith.cmpi eq, %[[VAL_35]], %[[VAL_37]] : index
// CHECK:             %[[VAL_40:.*]] = arith.andi %[[VAL_38]], %[[VAL_39]] : i1
// CHECK:             scf.if %[[VAL_40]] {
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_32]]] : memref<?xindex>
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_32]], %[[VAL_3]] : index
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_42]]] : memref<?xindex>
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_33]]] : memref<?xindex>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : index
// CHECK:               %[[VAL_46:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_45]]] : memref<?xindex>
// CHECK:               %[[VAL_47:.*]]:2 = scf.while (%[[VAL_48:.*]] = %[[VAL_41]], %[[VAL_49:.*]] = %[[VAL_44]]) : (index, index) -> (index, index) {
// CHECK:                 %[[VAL_50:.*]] = arith.cmpi ult, %[[VAL_48]], %[[VAL_43]] : index
// CHECK:                 %[[VAL_51:.*]] = arith.cmpi ult, %[[VAL_49]], %[[VAL_46]] : index
// CHECK:                 %[[VAL_52:.*]] = arith.andi %[[VAL_50]], %[[VAL_51]] : i1
// CHECK:                 scf.condition(%[[VAL_52]]) %[[VAL_48]], %[[VAL_49]] : index, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_53:.*]]: index, %[[VAL_54:.*]]: index):
// CHECK:                 %[[VAL_55:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_53]]] : memref<?xindex>
// CHECK:                 %[[VAL_56:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_54]]] : memref<?xindex>
// CHECK:                 %[[VAL_57:.*]] = arith.cmpi ult, %[[VAL_56]], %[[VAL_55]] : index
// CHECK:                 %[[VAL_58:.*]] = arith.select %[[VAL_57]], %[[VAL_56]], %[[VAL_55]] : index
// CHECK:                 %[[VAL_59:.*]] = arith.cmpi eq, %[[VAL_55]], %[[VAL_58]] : index
// CHECK:                 %[[VAL_60:.*]] = arith.cmpi eq, %[[VAL_56]], %[[VAL_58]] : index
// CHECK:                 %[[VAL_61:.*]] = arith.andi %[[VAL_59]], %[[VAL_60]] : i1
// CHECK:                 scf.if %[[VAL_61]] {
// CHECK:                   %[[VAL_62:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_53]]] : memref<?xindex>
// CHECK:                   %[[VAL_63:.*]] = arith.addi %[[VAL_53]], %[[VAL_3]] : index
// CHECK:                   %[[VAL_64:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_63]]] : memref<?xindex>
// CHECK:                   %[[VAL_65:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_54]]] : memref<?xindex>
// CHECK:                   %[[VAL_66:.*]] = arith.addi %[[VAL_54]], %[[VAL_3]] : index
// CHECK:                   %[[VAL_67:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_66]]] : memref<?xindex>
// CHECK:                   %[[VAL_68:.*]]:3 = scf.while (%[[VAL_69:.*]] = %[[VAL_62]], %[[VAL_70:.*]] = %[[VAL_65]], %[[VAL_71:.*]] = %[[VAL_4]]) : (index, index, i32) -> (index, index, i32) {
// CHECK:                     %[[VAL_72:.*]] = arith.cmpi ult, %[[VAL_69]], %[[VAL_64]] : index
// CHECK:                     %[[VAL_73:.*]] = arith.cmpi ult, %[[VAL_70]], %[[VAL_67]] : index
// CHECK:                     %[[VAL_74:.*]] = arith.andi %[[VAL_72]], %[[VAL_73]] : i1
// CHECK:                     scf.condition(%[[VAL_74]]) %[[VAL_69]], %[[VAL_70]], %[[VAL_71]] : index, index, i32
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_75:.*]]: index, %[[VAL_76:.*]]: index, %[[VAL_77:.*]]: i32):
// CHECK:                     %[[VAL_78:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_75]]] : memref<?xindex>
// CHECK:                     %[[VAL_79:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_76]]] : memref<?xindex>
// CHECK:                     %[[VAL_80:.*]] = arith.cmpi ult, %[[VAL_79]], %[[VAL_78]] : index
// CHECK:                     %[[VAL_81:.*]] = arith.select %[[VAL_80]], %[[VAL_79]], %[[VAL_78]] : index
// CHECK:                     %[[VAL_82:.*]] = arith.cmpi eq, %[[VAL_78]], %[[VAL_81]] : index
// CHECK:                     %[[VAL_83:.*]] = arith.cmpi eq, %[[VAL_79]], %[[VAL_81]] : index
// CHECK:                     %[[VAL_84:.*]] = arith.andi %[[VAL_82]], %[[VAL_83]] : i1
// CHECK:                     %[[VAL_85:.*]] = scf.if %[[VAL_84]] -> (i32) {
// CHECK:                       %[[VAL_86:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_75]]] : memref<?xi32>
// CHECK:                       %[[VAL_87:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_76]]] : memref<?xi32>
// CHECK:                       %[[VAL_88:.*]] = arith.muli %[[VAL_86]], %[[VAL_87]] : i32
// CHECK:                       %[[VAL_89:.*]] = arith.addi %[[VAL_77]], %[[VAL_88]] : i32
// CHECK:                       scf.yield %[[VAL_89]] : i32
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_77]] : i32
// CHECK:                     }
// CHECK:                     %[[VAL_90:.*]] = arith.cmpi eq, %[[VAL_78]], %[[VAL_81]] : index
// CHECK:                     %[[VAL_91:.*]] = arith.addi %[[VAL_75]], %[[VAL_3]] : index
// CHECK:                     %[[VAL_92:.*]] = arith.select %[[VAL_90]], %[[VAL_91]], %[[VAL_75]] : index
// CHECK:                     %[[VAL_93:.*]] = arith.cmpi eq, %[[VAL_79]], %[[VAL_81]] : index
// CHECK:                     %[[VAL_94:.*]] = arith.addi %[[VAL_76]], %[[VAL_3]] : index
// CHECK:                     %[[VAL_95:.*]] = arith.select %[[VAL_93]], %[[VAL_94]], %[[VAL_76]] : index
// CHECK:                     scf.yield %[[VAL_92]], %[[VAL_95]], %[[VAL_96:.*]] : index, index, i32
// CHECK:                   }
// CHECK:                   sparse_tensor.insert %[[VAL_97:.*]]#2 into %[[VAL_7]]{{\[}}%[[VAL_37]], %[[VAL_58]]] : tensor<?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:                 } else {
// CHECK:                 }
// CHECK:                 %[[VAL_98:.*]] = arith.cmpi eq, %[[VAL_55]], %[[VAL_58]] : index
// CHECK:                 %[[VAL_99:.*]] = arith.addi %[[VAL_53]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_100:.*]] = arith.select %[[VAL_98]], %[[VAL_99]], %[[VAL_53]] : index
// CHECK:                 %[[VAL_101:.*]] = arith.cmpi eq, %[[VAL_56]], %[[VAL_58]] : index
// CHECK:                 %[[VAL_102:.*]] = arith.addi %[[VAL_54]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_103:.*]] = arith.select %[[VAL_101]], %[[VAL_102]], %[[VAL_54]] : index
// CHECK:                 scf.yield %[[VAL_100]], %[[VAL_103]] : index, index
// CHECK:               }
// CHECK:             } else {
// CHECK:             }
// CHECK:             %[[VAL_104:.*]] = arith.cmpi eq, %[[VAL_34]], %[[VAL_37]] : index
// CHECK:             %[[VAL_105:.*]] = arith.addi %[[VAL_32]], %[[VAL_3]] : index
// CHECK:             %[[VAL_106:.*]] = arith.select %[[VAL_104]], %[[VAL_105]], %[[VAL_32]] : index
// CHECK:             %[[VAL_107:.*]] = arith.cmpi eq, %[[VAL_35]], %[[VAL_37]] : index
// CHECK:             %[[VAL_108:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : index
// CHECK:             %[[VAL_109:.*]] = arith.select %[[VAL_107]], %[[VAL_108]], %[[VAL_33]] : index
// CHECK:             scf.yield %[[VAL_106]], %[[VAL_109]] : index, index
// CHECK:           }
// CHECK:           %[[VAL_110:.*]] = sparse_tensor.load %[[VAL_7]] hasInserts : tensor<?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           return %[[VAL_110]] : tensor<?x?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:         }
func.func @sumred(%arga: tensor<?x?x?xi32, #SparseTensor>,
             %argb: tensor<?x?x?xi32, #SparseTensor>) -> tensor<?x?xi32, #DCSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xi32, #SparseTensor>
  %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xi32, #SparseTensor>
  %xinit = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xi32, #DCSR>
  %0 = linalg.generic #trait_sumred
    ins(%arga, %argb: tensor<?x?x?xi32, #SparseTensor>,
                      tensor<?x?x?xi32, #SparseTensor>)
    outs(%xinit: tensor<?x?xi32, #DCSR>) {
      ^bb(%a: i32, %b: i32, %x: i32):
        %0 = arith.muli %a, %b : i32
        %1 = arith.addi %x, %0 : i32
        linalg.yield %1 : i32
  } -> tensor<?x?xi32, #DCSR>
  return %0 : tensor<?x?xi32, #DCSR>
}

#trait_matmat = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // C (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "C(i,j) = SUM_k A(i,k) * B(k,j)"
}

// CHECK-LABEL:   func.func @matmat(
// CHECK-SAME:                      %[[VAL_0:.*]]: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_8:.*]] = bufferization.alloc_tensor(%[[VAL_6]], %[[VAL_7]]) : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf32>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 1 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf32>
// CHECK:           %[[VAL_19:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_21:.*]] = %[[VAL_19]] to %[[VAL_20]] step %[[VAL_3]] {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:             %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]] = sparse_tensor.expand %[[VAL_8]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf32>, memref<?xi1>, memref<?xindex>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_21]], %[[VAL_3]] : index
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_28]]] : memref<?xindex>
// CHECK:             %[[VAL_30:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:             %[[VAL_32:.*]]:3 = scf.while (%[[VAL_33:.*]] = %[[VAL_27]], %[[VAL_34:.*]] = %[[VAL_30]], %[[VAL_35:.*]] = %[[VAL_26]]) : (index, index, index) -> (index, index, index) {
// CHECK:               %[[VAL_36:.*]] = arith.cmpi ult, %[[VAL_33]], %[[VAL_29]] : index
// CHECK:               %[[VAL_37:.*]] = arith.cmpi ult, %[[VAL_34]], %[[VAL_31]] : index
// CHECK:               %[[VAL_38:.*]] = arith.andi %[[VAL_36]], %[[VAL_37]] : i1
// CHECK:               scf.condition(%[[VAL_38]]) %[[VAL_33]], %[[VAL_34]], %[[VAL_35]] : index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_39:.*]]: index, %[[VAL_40:.*]]: index, %[[VAL_41:.*]]: index):
// CHECK:               %[[VAL_42:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_39]]] : memref<?xindex>
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:               %[[VAL_44:.*]] = arith.cmpi ult, %[[VAL_43]], %[[VAL_42]] : index
// CHECK:               %[[VAL_45:.*]] = arith.select %[[VAL_44]], %[[VAL_43]], %[[VAL_42]] : index
// CHECK:               %[[VAL_46:.*]] = arith.cmpi eq, %[[VAL_42]], %[[VAL_45]] : index
// CHECK:               %[[VAL_47:.*]] = arith.cmpi eq, %[[VAL_43]], %[[VAL_45]] : index
// CHECK:               %[[VAL_48:.*]] = arith.andi %[[VAL_46]], %[[VAL_47]] : i1
// CHECK:               %[[VAL_49:.*]] = scf.if %[[VAL_48]] -> (index) {
// CHECK:                 %[[VAL_50:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_39]]] : memref<?xf32>
// CHECK:                 %[[VAL_51:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:                 %[[VAL_52:.*]] = arith.addi %[[VAL_40]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_53:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_52]]] : memref<?xindex>
// CHECK:                 %[[VAL_54:.*]] = scf.for %[[VAL_55:.*]] = %[[VAL_51]] to %[[VAL_53]] step %[[VAL_3]] iter_args(%[[VAL_56:.*]] = %[[VAL_41]]) -> (index) {
// CHECK:                   %[[VAL_57:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:                   %[[VAL_58:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_57]]] : memref<?xf32>
// CHECK:                   %[[VAL_59:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_55]]] : memref<?xf32>
// CHECK:                   %[[VAL_60:.*]] = arith.mulf %[[VAL_50]], %[[VAL_59]] : f32
// CHECK:                   %[[VAL_61:.*]] = arith.addf %[[VAL_58]], %[[VAL_60]] : f32
// CHECK:                   %[[VAL_62:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_57]]] : memref<?xi1>
// CHECK:                   %[[VAL_63:.*]] = arith.cmpi eq, %[[VAL_62]], %[[VAL_4]] : i1
// CHECK:                   %[[VAL_64:.*]] = scf.if %[[VAL_63]] -> (index) {
// CHECK:                     memref.store %[[VAL_5]], %[[VAL_24]]{{\[}}%[[VAL_57]]] : memref<?xi1>
// CHECK:                     memref.store %[[VAL_57]], %[[VAL_25]]{{\[}}%[[VAL_56]]] : memref<?xindex>
// CHECK:                     %[[VAL_65:.*]] = arith.addi %[[VAL_56]], %[[VAL_3]] : index
// CHECK:                     scf.yield %[[VAL_65]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_56]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[VAL_61]], %[[VAL_23]]{{\[}}%[[VAL_57]]] : memref<?xf32>
// CHECK:                   scf.yield %[[VAL_66:.*]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_67:.*]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_41]] : index
// CHECK:               }
// CHECK:               %[[VAL_68:.*]] = arith.cmpi eq, %[[VAL_42]], %[[VAL_45]] : index
// CHECK:               %[[VAL_69:.*]] = arith.addi %[[VAL_39]], %[[VAL_3]] : index
// CHECK:               %[[VAL_70:.*]] = arith.select %[[VAL_68]], %[[VAL_69]], %[[VAL_39]] : index
// CHECK:               %[[VAL_71:.*]] = arith.cmpi eq, %[[VAL_43]], %[[VAL_45]] : index
// CHECK:               %[[VAL_72:.*]] = arith.addi %[[VAL_40]], %[[VAL_3]] : index
// CHECK:               %[[VAL_73:.*]] = arith.select %[[VAL_71]], %[[VAL_72]], %[[VAL_40]] : index
// CHECK:               scf.yield %[[VAL_70]], %[[VAL_73]], %[[VAL_74:.*]] : index, index, index
// CHECK:             }
// CHECK:             sparse_tensor.compress %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_75:.*]]#2 into %[[VAL_8]]{{\[}}%[[VAL_22]]] : memref<?xf32>, memref<?xi1>, memref<?xindex>, tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           }
// CHECK:           %[[VAL_76:.*]] = sparse_tensor.load %[[VAL_8]] hasInserts : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           return %[[VAL_76]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:         }
func.func @matmat(%arga: tensor<?x?xf32, #DCSR>,
             %argb: tensor<?x?xf32, #DCSR>) -> tensor<?x?xf32, #DCSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arga, %c0 : tensor<?x?xf32, #DCSR>
  %d1 = tensor.dim %argb, %c1 : tensor<?x?xf32, #DCSR>
  %cinit = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf32, #DCSR>
  %0 = linalg.generic #trait_matmat
       ins(%arga, %argb: tensor<?x?xf32, #DCSR>,
                         tensor<?x?xf32, #DCSR>)
      outs(%cinit: tensor<?x?xf32, #DCSR>) {
    ^bb(%a: f32, %b: f32, %c: f32):
      %1 = arith.mulf %a, %b : f32
      %2 = arith.addf %c, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<?x?xf32, #DCSR>
  return %0 : tensor<?x?xf32, #DCSR>
}
