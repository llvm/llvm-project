// RUN: mlir-opt %s --canonicalize --post-sparsification-rewrite="enable-runtime-library=false" --sparse-tensor-codegen -cse --canonicalize | FileCheck %s

#COO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
  crdWidth=32
}>

// CHECK-LABEL:   func.func @sparse_pack(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<6xf64>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<2xindex>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<6x2xi32>)
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_1]] : memref<2xindex>
// CHECK-DAG:       %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<2xindex> to memref<?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_2]] : memref<6x2xi32>
// CHECK-DAG:       %[[VAL_9:.*]] = memref.collapse_shape %[[VAL_8]] {{\[\[}}0, 1]] : memref<6x2xi32> into memref<12xi32>
// CHECK-DAG:       %[[VAL_10:.*]] = memref.cast %[[VAL_9]] : memref<12xi32> to memref<?xi32>
// CHECK-DAG:       %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_0]] : memref<6xf64>
// CHECK-DAG:       %[[VAL_12:.*]] = memref.cast %[[VAL_11]] : memref<6xf64> to memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.storage_specifier.init
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.storage_specifier.set %[[VAL_13]]  lvl_sz at 0 with %[[VAL_4]]
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.storage_specifier.set %[[VAL_14]]  pos_mem_sz at 0 with %[[VAL_3]]
// CHECK:           %[[VAL_16:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_5]]] : tensor<2xindex>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[VAL_3]] : index
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.storage_specifier.set %[[VAL_15]]  crd_mem_sz at 0 with %[[VAL_17]]
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.storage_specifier.set %[[VAL_18]]  lvl_sz at 1 with %[[VAL_4]]
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.storage_specifier.set %[[VAL_19]]  val_mem_sz with %[[VAL_16]]
// CHECK:           return %[[VAL_7]], %[[VAL_10]], %[[VAL_12]], %[[VAL_20]]
// CHECK:         }
func.func @sparse_pack(%values: tensor<6xf64>, %pos:tensor<2xindex>, %coordinates: tensor<6x2xi32>)
                    -> tensor<100x100xf64, #COO> {
  %0 = sparse_tensor.pack %values, %pos, %coordinates
     : tensor<6xf64>, tensor<2xindex>, tensor<6x2xi32> to tensor<100x100xf64, #COO>
  return %0 : tensor<100x100xf64, #COO>
}

// CHECK-LABEL:   func.func @sparse_unpack(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<?xi32>,
// CHECK-SAME:      %[[VAL_2:.*]]: memref<?xf64>,
// CHECK-SAME:      %[[VAL_3:.*]]: !sparse_tensor.storage_specifier<#sparse_tensor.encoding<{ lvlTypes = [ "compressed", "singleton" ] }>>,
// CHECK-SAME:      %[[VAL_4:.*]]: tensor<6xf64>,
// CHECK-SAME:      %[[VAL_5:.*]]: tensor<2xindex>,
// CHECK-SAME:      %[[VAL_6:.*]]: tensor<6x2xi32>) -> (tensor<6xf64>, tensor<2xindex>, tensor<6x2xi32>) {
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.storage_specifier.get %[[VAL_3]]  pos_mem_sz at 0
// CHECK:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_5]] : memref<2xindex>
// CHECK:           %[[VAL_9:.*]] = memref.subview %[[VAL_8]][0] {{\[}}%[[VAL_7]]] [1] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = memref.subview %[[VAL_0]][0] {{\[}}%[[VAL_7]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK:           memref.copy %[[VAL_10]], %[[VAL_9]] : memref<?xindex> to memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.storage_specifier.get %[[VAL_3]]  crd_mem_sz at 0
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_6]] : memref<6x2xi32>
// CHECK:           %[[VAL_13:.*]] = memref.collapse_shape %[[VAL_12]] {{\[\[}}0, 1]] : memref<6x2xi32> into memref<12xi32>
// CHECK:           %[[VAL_14:.*]] = memref.subview %[[VAL_13]][0] {{\[}}%[[VAL_11]]] [1] : memref<12xi32> to memref<?xi32>
// CHECK:           %[[VAL_15:.*]] = memref.subview %[[VAL_1]][0] {{\[}}%[[VAL_11]]] [1] : memref<?xi32> to memref<?xi32>
// CHECK:           memref.copy %[[VAL_15]], %[[VAL_14]] : memref<?xi32> to memref<?xi32>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.storage_specifier.get %[[VAL_3]]  val_mem_sz
// CHECK:           %[[VAL_17:.*]] = bufferization.to_memref %[[VAL_4]] : memref<6xf64>
// CHECK:           %[[VAL_18:.*]] = memref.subview %[[VAL_17]][0] {{\[}}%[[VAL_16]]] [1] : memref<6xf64> to memref<?xf64>
// CHECK:           %[[VAL_19:.*]] = memref.subview %[[VAL_2]][0] {{\[}}%[[VAL_16]]] [1] : memref<?xf64> to memref<?xf64>
// CHECK:           memref.copy %[[VAL_19]], %[[VAL_18]] : memref<?xf64> to memref<?xf64>
// CHECK:           %[[VAL_20:.*]] = bufferization.to_tensor %[[VAL_17]] : memref<6xf64>
// CHECK:           %[[VAL_21:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<2xindex>
// CHECK:           %[[VAL_22:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<6x2xi32>
// CHECK:           return %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] : tensor<6xf64>, tensor<2xindex>, tensor<6x2xi32>
// CHECK:         }
func.func @sparse_unpack(%sp : tensor<100x100xf64, #COO>,
                         %od : tensor<6xf64>,
                         %op : tensor<2xindex>,
                         %oi : tensor<6x2xi32>)
                       -> (tensor<6xf64>, tensor<2xindex>, tensor<6x2xi32>) {
  %rd, %rp, %ri, %dl, %pl, %il = sparse_tensor.unpack %sp : tensor<100x100xf64, #COO>
                                 outs(%od, %op, %oi : tensor<6xf64>, tensor<2xindex>, tensor<6x2xi32>)
                                 -> tensor<6xf64>, (tensor<2xindex>, tensor<6x2xi32>), index, (index, index)
  return %rd, %rp, %ri : tensor<6xf64>, tensor<2xindex>, tensor<6x2xi32>
}
