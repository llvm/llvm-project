// RUN: mlir-opt %s --canonicalize --post-sparsification-rewrite="enable-runtime-library=false" --sparse-tensor-codegen -cse | FileCheck %s

#COO = #sparse_tensor.encoding<{
  dimLevelType = ["compressed-nu", "singleton"],
  indexBitWidth=32
}>

// CHECK-LABEL:   func.func @sparse_pack(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<6xf64>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<6x2xi32>) -> (memref<?xindex>, memref<?xi32>, memref<?xf64>,
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<[0, 6]> : tensor<2xindex>
// CHECK:           %[[VAL_3:.*]] = bufferization.to_memref %[[VAL_2]] : memref<2xindex>
// CHECK:           %[[VAL_4:.*]] = memref.cast %[[VAL_3]] : memref<2xindex> to memref<?xindex>
// CHECK:           %[[VAL_5:.*]] = bufferization.to_memref %[[VAL_1]] : memref<6x2xi32>
// CHECK:           %[[VAL_6:.*]] = memref.collapse_shape %[[VAL_5]] {{\[\[}}0, 1]] : memref<6x2xi32> into memref<12xi32>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<12xi32> to memref<?xi32>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<6xf64>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<6xf64> to memref<?xf64>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.storage_specifier.init :
// CHECK:           %[[VAL_11:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i32
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.storage_specifier.set %[[VAL_10]]  dim_sz at 0 with %[[VAL_13]] : i32,
// CHECK:           %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_16:.*]] = arith.index_cast %[[VAL_15]] : index to i32
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.storage_specifier.set %[[VAL_14]]  ptr_mem_sz at 0 with %[[VAL_16]] : i32,
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_11]] : index to i32
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.storage_specifier.set %[[VAL_17]]  idx_mem_sz at 0 with %[[VAL_18]] : i32,
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.storage_specifier.set %[[VAL_19]]  dim_sz at 1 with %[[VAL_13]] : i32,
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.storage_specifier.set %[[VAL_20]]  idx_mem_sz at 1 with %[[VAL_18]] : i32,
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.storage_specifier.set %[[VAL_21]]  val_mem_sz with %[[VAL_18]] : i32,
// CHECK:           return %[[VAL_4]], %[[VAL_7]], %[[VAL_9]], %[[VAL_22]] : memref<?xindex>, memref<?xi32>, memref<?xf64>,
// CHECK:         }
func.func @sparse_pack(%data: tensor<6xf64>, %index: tensor<6x2xi32>)
                    -> tensor<100x100xf64, #COO> {
  %0 = sparse_tensor.pack %data, %index : tensor<6xf64>, tensor<6x2xi32>
                                       to tensor<100x100xf64, #COO>
  return %0 : tensor<100x100xf64, #COO>
}

// CHECK-LABEL:   func.func @sparse_unpack(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<?xi32>,
// CHECK-SAME:      %[[VAL_2:.*]]: memref<?xf64>,
// CHECK-SAME:      %[[VAL_3:.*]]: !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_4:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_2]], %[[VAL_5]] : memref<?xf64>
// CHECK:           %[[VAL_7:.*]] = arith.cmpi ult, %[[VAL_4]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = scf.if %[[VAL_7]] -> (memref<6xf64>) {
// CHECK:             %[[VAL_9:.*]] = memref.realloc %[[VAL_2]] : memref<?xf64> to memref<6xf64>
// CHECK:             scf.yield %[[VAL_9]] : memref<6xf64>
// CHECK:           } else {
// CHECK:             %[[VAL_10:.*]] = memref.subview %[[VAL_2]][0] [6] [1] : memref<?xf64> to memref<6xf64>
// CHECK:             scf.yield %[[VAL_10]] : memref<6xf64>
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = arith.constant 12 : index
// CHECK:           %[[VAL_12:.*]] = memref.dim %[[VAL_1]], %[[VAL_5]] : memref<?xi32>
// CHECK:           %[[VAL_13:.*]] = arith.cmpi ult, %[[VAL_11]], %[[VAL_12]] : index
// CHECK:           %[[VAL_14:.*]] = scf.if %[[VAL_13]] -> (memref<12xi32>) {
// CHECK:             %[[VAL_15:.*]] = memref.realloc %[[VAL_1]] : memref<?xi32> to memref<12xi32>
// CHECK:             scf.yield %[[VAL_15]] : memref<12xi32>
// CHECK:           } else {
// CHECK:             %[[VAL_16:.*]] = memref.subview %[[VAL_1]][0] [12] [1] : memref<?xi32> to memref<12xi32>
// CHECK:             scf.yield %[[VAL_16]] : memref<12xi32>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = memref.expand_shape %[[VAL_18:.*]] {{\[\[}}0, 1]] : memref<12xi32> into memref<6x2xi32>
// CHECK:           %[[VAL_19:.*]] = bufferization.to_tensor %[[VAL_20:.*]] : memref<6xf64>
// CHECK:           %[[VAL_21:.*]] = bufferization.to_tensor %[[VAL_17]] : memref<6x2xi32>
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.storage_specifier
// CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_22]] : i32 to index
// CHECK:           return %[[VAL_19]], %[[VAL_21]], %[[VAL_23]] : tensor<6xf64>, tensor<6x2xi32>, index
// CHECK:         }
func.func @sparse_unpack(%sp: tensor<100x100xf64, #COO>) -> (tensor<6xf64>, tensor<6x2xi32>, index) {
  %d, %i, %nnz = sparse_tensor.unpack %sp : tensor<100x100xf64, #COO>
                                         to tensor<6xf64>, tensor<6x2xi32>, index
  return %d, %i, %nnz : tensor<6xf64>, tensor<6x2xi32>, index
}
