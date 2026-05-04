// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#trait_scale_inpl = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) *= 2 or X(i,j) += X(i,j)"
}

// CHECK-LABEL:   func.func @sparse_simply_dynamic1(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x16xf32, #sparse{{[0-9]*}}>) -> tensor<32x16xf32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32x16xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<32x16xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x16xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK:           %[[VAL_7:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<?xindex>
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_2]] {
// CHECK:               %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xf32>
// CHECK:               %[[VAL_15:.*]] = arith.mulf %[[VAL_14]], %[[VAL_3]] : f32
// CHECK:               memref.store %[[VAL_15]], %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<32x16xf32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_16]] : tensor<32x16xf32, #sparse{{[0-9]*}}>
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
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x16xf32, #sparse{{[0-9]*}}>) -> tensor<32x16xf32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32x16xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<32x16xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x16xf32, #sparse{{[0-9]*}}> to memref<?xf32>
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
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<32x16xf32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_16]] : tensor<32x16xf32, #sparse{{[0-9]*}}>
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
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10x20xf32, #sparse{{[0-9]*}}>) -> tensor<10x20xf32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_5:.*]] = tensor.empty() : tensor<10x20xf32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10x20xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK:           %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_3]] iter_args(%[[VAL_11:.*]] = %[[VAL_5]]) -> (tensor<10x20xf32, #sparse{{[0-9]*}}>) {
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_10]], %[[VAL_3]] : index
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:             %[[VAL_15:.*]] = scf.for %[[VAL_16:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_3]] iter_args(%[[VAL_17:.*]] = %[[VAL_11]]) -> (tensor<10x20xf32, #sparse{{[0-9]*}}>) {
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xf32>
// CHECK:               %[[VAL_20:.*]] = arith.mulf %[[VAL_19]], %[[VAL_4]] : f32
// CHECK:               %[[VAL_21:.*]] = tensor.insert %[[VAL_20]] into %[[VAL_17]]{{\[}}%[[VAL_10]], %[[VAL_18]]] : tensor<10x20xf32, #sparse{{[0-9]*}}>
// CHECK:               scf.yield %[[VAL_21]] : tensor<10x20xf32, #sparse{{[0-9]*}}>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_22:.*]] : tensor<10x20xf32, #sparse{{[0-9]*}}>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.load %[[VAL_24:.*]] hasInserts : tensor<10x20xf32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_23]] : tensor<10x20xf32, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @sparse_truly_dynamic(%arga: tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #DCSR> {
  %s = arith.constant 2.0 : f32
  %xm = tensor.empty() : tensor<10x20xf32, #DCSR>
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
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?x?xi32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?x?xi32, #sparse{{[0-9]*}}>) -> tensor<?x?xi32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_FALSE:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_TRUE:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?x?xi32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?x?xi32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = tensor.empty(%[[VAL_5]], %[[VAL_6]]) : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 2 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 2 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_18:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_19:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 2 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_20:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 2 : index} : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_21:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?x?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
// CHECK:           %[[VAL_22:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_25:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_26:.*]]:3 = scf.while (%[[VAL_27:.*]] = %[[VAL_22]], %[[VAL_28:.*]] = %[[VAL_24]], %[[VAL_29:.*]] = %[[VAL_7]]) : (index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>) -> (index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:             %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_27]], %[[VAL_23]] : index
// CHECK:             %[[VAL_31:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_25]] : index
// CHECK:             %[[VAL_32:.*]] = arith.andi %[[VAL_30]], %[[VAL_31]] : i1
// CHECK:             scf.condition(%[[VAL_32]]) %[[VAL_27]], %[[VAL_28]], %[[VAL_29]] : index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_33:.*]]: index, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: tensor<?x?xi32, #sparse{{[0-9]*}}>):
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_33]]] : memref<?xindex>
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_34]]] : memref<?xindex>
// CHECK:             %[[VAL_38:.*]] = arith.cmpi ult, %[[VAL_37]], %[[VAL_36]] : index
// CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_38]], %[[VAL_37]], %[[VAL_36]] : index
// CHECK:             %[[VAL_40:.*]] = arith.cmpi eq, %[[VAL_36]], %[[VAL_39]] : index
// CHECK:             %[[VAL_41:.*]] = arith.cmpi eq, %[[VAL_37]], %[[VAL_39]] : index
// CHECK:             %[[VAL_42:.*]] = arith.andi %[[VAL_40]], %[[VAL_41]] : i1
// CHECK:             %[[VAL_43:.*]] = scf.if %[[VAL_42]] -> (tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_33]]] : memref<?xindex>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : index
// CHECK:               %[[VAL_46:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_45]]] : memref<?xindex>
// CHECK:               %[[VAL_47:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_34]]] : memref<?xindex>
// CHECK:               %[[VAL_48:.*]] = arith.addi %[[VAL_34]], %[[VAL_3]] : index
// CHECK:               %[[VAL_49:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_48]]] : memref<?xindex>
// CHECK:               %[[VAL_50:.*]]:3 = scf.while (%[[VAL_51:.*]] = %[[VAL_44]], %[[VAL_52:.*]] = %[[VAL_47]], %[[VAL_53:.*]] = %[[VAL_35]]) : (index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>) -> (index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:                 %[[VAL_54:.*]] = arith.cmpi ult, %[[VAL_51]], %[[VAL_46]] : index
// CHECK:                 %[[VAL_55:.*]] = arith.cmpi ult, %[[VAL_52]], %[[VAL_49]] : index
// CHECK:                 %[[VAL_56:.*]] = arith.andi %[[VAL_54]], %[[VAL_55]] : i1
// CHECK:                 scf.condition(%[[VAL_56]]) %[[VAL_51]], %[[VAL_52]], %[[VAL_53]] : index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_57:.*]]: index, %[[VAL_58:.*]]: index, %[[VAL_59:.*]]: tensor<?x?xi32, #sparse{{[0-9]*}}>):
// CHECK:                 %[[VAL_60:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_57]]] : memref<?xindex>
// CHECK:                 %[[VAL_61:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_58]]] : memref<?xindex>
// CHECK:                 %[[VAL_62:.*]] = arith.cmpi ult, %[[VAL_61]], %[[VAL_60]] : index
// CHECK:                 %[[VAL_63:.*]] = arith.select %[[VAL_62]], %[[VAL_61]], %[[VAL_60]] : index
// CHECK:                 %[[VAL_64:.*]] = arith.cmpi eq, %[[VAL_60]], %[[VAL_63]] : index
// CHECK:                 %[[VAL_65:.*]] = arith.cmpi eq, %[[VAL_61]], %[[VAL_63]] : index
// CHECK:                 %[[VAL_66:.*]] = arith.andi %[[VAL_64]], %[[VAL_65]] : i1
// CHECK:                 %[[VAL_67:.*]] = scf.if %[[VAL_66]] -> (tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:                   %[[VAL_68:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_57]]] : memref<?xindex>
// CHECK:                   %[[VAL_69:.*]] = arith.addi %[[VAL_57]], %[[VAL_3]] : index
// CHECK:                   %[[VAL_70:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_69]]] : memref<?xindex>
// CHECK:                   %[[VAL_71:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_58]]] : memref<?xindex>
// CHECK:                   %[[VAL_72:.*]] = arith.addi %[[VAL_58]], %[[VAL_3]] : index
// CHECK:                   %[[VAL_73:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_72]]] : memref<?xindex>
// CHECK:                   %[[VAL_74:.*]]:5 = scf.while (%[[VAL_75:.*]] = %[[VAL_68]], %[[VAL_76:.*]] = %[[VAL_71]], %[[VAL_77:.*]] = %[[VAL_4]], %[[VAL_200:.*]] = %[[VAL_FALSE]], %[[VAL_78:.*]] = %[[VAL_59]]) : (index, index, i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>) -> (index, index, i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:                     %[[VAL_79:.*]] = arith.cmpi ult, %[[VAL_75]], %[[VAL_70]] : index
// CHECK:                     %[[VAL_80:.*]] = arith.cmpi ult, %[[VAL_76]], %[[VAL_73]] : index
// CHECK:                     %[[VAL_81:.*]] = arith.andi %[[VAL_79]], %[[VAL_80]] : i1
// CHECK:                     scf.condition(%[[VAL_81]]) %[[VAL_75]], %[[VAL_76]], %[[VAL_77]], %[[VAL_200]], %[[VAL_78]] : index, index, i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_82:.*]]: index, %[[VAL_83:.*]]: index, %[[VAL_84:.*]]: i32, %[[VAL_201:.*]]: i1, %[[VAL_85:.*]]: tensor<?x?xi32, #sparse{{[0-9]*}}>):
// CHECK:                     %[[VAL_86:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_82]]] : memref<?xindex>
// CHECK:                     %[[VAL_87:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_83]]] : memref<?xindex>
// CHECK:                     %[[VAL_88:.*]] = arith.cmpi ult, %[[VAL_87]], %[[VAL_86]] : index
// CHECK:                     %[[VAL_89:.*]] = arith.select %[[VAL_88]], %[[VAL_87]], %[[VAL_86]] : index
// CHECK:                     %[[VAL_90:.*]] = arith.cmpi eq, %[[VAL_86]], %[[VAL_89]] : index
// CHECK:                     %[[VAL_91:.*]] = arith.cmpi eq, %[[VAL_87]], %[[VAL_89]] : index
// CHECK:                     %[[VAL_92:.*]] = arith.andi %[[VAL_90]], %[[VAL_91]] : i1
// CHECK:                     %[[VAL_93:.*]]:3 = scf.if %[[VAL_92]] -> (i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:                       %[[VAL_94:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_82]]] : memref<?xi32>
// CHECK:                       %[[VAL_95:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_83]]] : memref<?xi32>
// CHECK:                       %[[VAL_96:.*]] = arith.muli %[[VAL_94]], %[[VAL_95]] : i32
// CHECK:                       %[[VAL_97:.*]] = arith.addi %[[VAL_84]], %[[VAL_96]] : i32
// CHECK:                       scf.yield %[[VAL_97]], %[[VAL_TRUE]], %[[VAL_85]] : i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_84]], %[[VAL_201]], %[[VAL_85]] : i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                     }
// CHECK:                     %[[VAL_98:.*]] = arith.cmpi eq, %[[VAL_86]], %[[VAL_89]] : index
// CHECK:                     %[[VAL_99:.*]] = arith.addi %[[VAL_82]], %[[VAL_3]] : index
// CHECK:                     %[[VAL_100:.*]] = arith.select %[[VAL_98]], %[[VAL_99]], %[[VAL_82]] : index
// CHECK:                     %[[VAL_101:.*]] = arith.cmpi eq, %[[VAL_87]], %[[VAL_89]] : index
// CHECK:                     %[[VAL_102:.*]] = arith.addi %[[VAL_83]], %[[VAL_3]] : index
// CHECK:                     %[[VAL_103:.*]] = arith.select %[[VAL_101]], %[[VAL_102]], %[[VAL_83]] : index
// CHECK:                     scf.yield %[[VAL_100]], %[[VAL_103]], %[[VAL_104:.*]]#0, %[[VAL_104]]#1, %[[VAL_104]]#2 : index, index, i32, i1, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                   }
// CHECK:                   %[[VAL_202:.*]] = scf.if %[[VAL_74]]#3 -> (tensor<?x?xi32, #sparse{{[0-9]*}}>) {
// CHECK:                     %[[VAL_105:.*]] = tensor.insert %[[VAL_74]]#2 into %[[VAL_74]]#4{{\[}}%[[VAL_39]], %[[VAL_63]]] : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                     scf.yield %[[VAL_105]] : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_74]]#4 : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_202]] : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_59]] : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:                 }
// CHECK:                 %[[VAL_107:.*]] = arith.cmpi eq, %[[VAL_60]], %[[VAL_63]] : index
// CHECK:                 %[[VAL_108:.*]] = arith.addi %[[VAL_57]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_109:.*]] = arith.select %[[VAL_107]], %[[VAL_108]], %[[VAL_57]] : index
// CHECK:                 %[[VAL_110:.*]] = arith.cmpi eq, %[[VAL_61]], %[[VAL_63]] : index
// CHECK:                 %[[VAL_111:.*]] = arith.addi %[[VAL_58]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_112:.*]] = arith.select %[[VAL_110]], %[[VAL_111]], %[[VAL_58]] : index
// CHECK:                 scf.yield %[[VAL_109]], %[[VAL_112]], %[[VAL_113:.*]] : index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:               }
// CHECK:               scf.yield %[[VAL_114:.*]]#2 : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_35]] : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:             }
// CHECK:             %[[VAL_115:.*]] = arith.cmpi eq, %[[VAL_36]], %[[VAL_39]] : index
// CHECK:             %[[VAL_116:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : index
// CHECK:             %[[VAL_117:.*]] = arith.select %[[VAL_115]], %[[VAL_116]], %[[VAL_33]] : index
// CHECK:             %[[VAL_118:.*]] = arith.cmpi eq, %[[VAL_37]], %[[VAL_39]] : index
// CHECK:             %[[VAL_119:.*]] = arith.addi %[[VAL_34]], %[[VAL_3]] : index
// CHECK:             %[[VAL_120:.*]] = arith.select %[[VAL_118]], %[[VAL_119]], %[[VAL_34]] : index
// CHECK:             scf.yield %[[VAL_117]], %[[VAL_120]], %[[VAL_121:.*]] : index, index, tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:           }
// CHECK:           %[[VAL_122:.*]] = sparse_tensor.load %[[VAL_123:.*]]#2 hasInserts : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_122]] : tensor<?x?xi32, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @sumred(%arga: tensor<?x?x?xi32, #SparseTensor>,
             %argb: tensor<?x?x?xi32, #SparseTensor>) -> tensor<?x?xi32, #DCSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xi32, #SparseTensor>
  %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xi32, #SparseTensor>
  %xinit = tensor.empty(%d0, %d1) : tensor<?x?xi32, #DCSR>
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
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?xf32, #sparse{{[0-9]*}}>) -> tensor<?x?xf32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = tensor.empty(%[[VAL_6]], %[[VAL_7]]) : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// CHECK:           %[[VAL_19:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = scf.for %[[VAL_22:.*]] = %[[VAL_19]] to %[[VAL_20]] step %[[VAL_3]] iter_args(%[[VAL_23:.*]] = %[[VAL_8]]) -> (tensor<?x?xf32, #sparse{{[0-9]*}}>) {
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK:             %[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]], %[[VAL_28:.*]] = sparse_tensor.expand %[[VAL_8]] : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xf32>, memref<?xi1>, memref<?xindex>
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : index
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_30]]] : memref<?xindex>
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:             %[[VAL_33:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:             %[[VAL_34:.*]]:4 = scf.while (%[[VAL_35:.*]] = %[[VAL_29]], %[[VAL_36:.*]] = %[[VAL_32]], %[[VAL_37:.*]] = %[[VAL_28]], %[[VAL_38:.*]] = %[[VAL_23]]) : (index, index, index, tensor<?x?xf32, #sparse{{[0-9]*}}>) -> (index, index, index, tensor<?x?xf32, #sparse{{[0-9]*}}>) {
// CHECK:               %[[VAL_39:.*]] = arith.cmpi ult, %[[VAL_35]], %[[VAL_31]] : index
// CHECK:               %[[VAL_40:.*]] = arith.cmpi ult, %[[VAL_36]], %[[VAL_33]] : index
// CHECK:               %[[VAL_41:.*]] = arith.andi %[[VAL_39]], %[[VAL_40]] : i1
// CHECK:               scf.condition(%[[VAL_41]]) %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]] : index, index, index, tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_42:.*]]: index, %[[VAL_43:.*]]: index, %[[VAL_44:.*]]: index, %[[VAL_45:.*]]: tensor<?x?xf32, #sparse{{[0-9]*}}>):
// CHECK:               %[[VAL_46:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_42]]] : memref<?xindex>
// CHECK:               %[[VAL_47:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_43]]] : memref<?xindex>
// CHECK:               %[[VAL_48:.*]] = arith.cmpi ult, %[[VAL_47]], %[[VAL_46]] : index
// CHECK:               %[[VAL_49:.*]] = arith.select %[[VAL_48]], %[[VAL_47]], %[[VAL_46]] : index
// CHECK:               %[[VAL_50:.*]] = arith.cmpi eq, %[[VAL_46]], %[[VAL_49]] : index
// CHECK:               %[[VAL_51:.*]] = arith.cmpi eq, %[[VAL_47]], %[[VAL_49]] : index
// CHECK:               %[[VAL_52:.*]] = arith.andi %[[VAL_50]], %[[VAL_51]] : i1
// CHECK:               %[[VAL_53:.*]]:2 = scf.if %[[VAL_52]] -> (index, tensor<?x?xf32, #sparse{{[0-9]*}}>) {
// CHECK:                 %[[VAL_54:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_42]]] : memref<?xf32>
// CHECK:                 %[[VAL_55:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_43]]] : memref<?xindex>
// CHECK:                 %[[VAL_56:.*]] = arith.addi %[[VAL_43]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_57:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_56]]] : memref<?xindex>
// CHECK:                 %[[VAL_58:.*]] = scf.for %[[VAL_59:.*]] = %[[VAL_55]] to %[[VAL_57]] step %[[VAL_3]] iter_args(%[[VAL_60:.*]] = %[[VAL_44]]) -> (index) {
// CHECK:                   %[[VAL_61:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_59]]] : memref<?xindex>
// CHECK:                   %[[VAL_62:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_61]]] : memref<?xf32>
// CHECK:                   %[[VAL_63:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_59]]] : memref<?xf32>
// CHECK:                   %[[VAL_64:.*]] = arith.mulf %[[VAL_54]], %[[VAL_63]] : f32
// CHECK:                   %[[VAL_65:.*]] = arith.addf %[[VAL_62]], %[[VAL_64]] : f32
// CHECK:                   %[[VAL_66:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_61]]] : memref<?xi1>
// CHECK:                   %[[VAL_67:.*]] = arith.cmpi eq, %[[VAL_66]], %[[VAL_4]] : i1
// CHECK:                   %[[VAL_68:.*]] = scf.if %[[VAL_67]] -> (index) {
// CHECK:                     memref.store %[[VAL_5]], %[[VAL_26]]{{\[}}%[[VAL_61]]] : memref<?xi1>
// CHECK:                     memref.store %[[VAL_61]], %[[VAL_27]]{{\[}}%[[VAL_60]]] : memref<?xindex>
// CHECK:                     %[[VAL_69:.*]] = arith.addi %[[VAL_60]], %[[VAL_3]] : index
// CHECK:                     scf.yield %[[VAL_69]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_60]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[VAL_65]], %[[VAL_25]]{{\[}}%[[VAL_61]]] : memref<?xf32>
// CHECK:                   scf.yield %[[VAL_70:.*]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_71:.*]], %[[VAL_45]] : index, tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_44]], %[[VAL_45]] : index, tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:               }
// CHECK:               %[[VAL_72:.*]] = arith.cmpi eq, %[[VAL_46]], %[[VAL_49]] : index
// CHECK:               %[[VAL_73:.*]] = arith.addi %[[VAL_42]], %[[VAL_3]] : index
// CHECK:               %[[VAL_74:.*]] = arith.select %[[VAL_72]], %[[VAL_73]], %[[VAL_42]] : index
// CHECK:               %[[VAL_75:.*]] = arith.cmpi eq, %[[VAL_47]], %[[VAL_49]] : index
// CHECK:               %[[VAL_76:.*]] = arith.addi %[[VAL_43]], %[[VAL_3]] : index
// CHECK:               %[[VAL_77:.*]] = arith.select %[[VAL_75]], %[[VAL_76]], %[[VAL_43]] : index
// CHECK:               scf.yield %[[VAL_74]], %[[VAL_77]], %[[VAL_78:.*]]#0, %[[VAL_78]]#1 : index, index, index, tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:             }
// CHECK:             %[[VAL_79:.*]] = sparse_tensor.compress %[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_80:.*]]#2 into %[[VAL_80]]#3{{\[}}%[[VAL_24]]] : memref<?xf32>, memref<?xi1>, memref<?xindex>, tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:             scf.yield %[[VAL_79]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:           }
// CHECK:           %[[VAL_81:.*]] = sparse_tensor.load %[[VAL_82:.*]] hasInserts : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_81]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @matmat(%arga: tensor<?x?xf32, #DCSR>,
             %argb: tensor<?x?xf32, #DCSR>) -> tensor<?x?xf32, #DCSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arga, %c0 : tensor<?x?xf32, #DCSR>
  %d1 = tensor.dim %argb, %c1 : tensor<?x?xf32, #DCSR>
  %cinit = tensor.empty(%d0, %d1) : tensor<?x?xf32, #DCSR>
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
