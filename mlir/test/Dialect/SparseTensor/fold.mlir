// RUN: mlir-opt %s  --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

// CHECK-LABEL: func @sparse_nop_dense2dense_convert(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//   CHECK-NOT: sparse_tensor.convert
//       CHECK: return %[[A]] : tensor<64xf32>
func.func @sparse_nop_dense2dense_convert(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func @sparse_dce_convert(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//   CHECK-NOT: sparse_tensor.convert
//       CHECK: return
func.func @sparse_dce_convert(%arg0: tensor<64xf32>) {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32, #SparseVector>
  return
}

// CHECK-LABEL: func @sparse_dce_getters(
//  CHECK-SAME: %[[A:.*]]: tensor<64xf32, #sparse_tensor.encoding<{{{.*}}}>>)
//   CHECK-NOT: sparse_tensor.positions
//   CHECK-NOT: sparse_tensor.coordinates
//   CHECK-NOT: sparse_tensor.values
//       CHECK: return
func.func @sparse_dce_getters(%arg0: tensor<64xf32, #SparseVector>) {
  %0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<64xf32, #SparseVector> to memref<?xindex>
  %1 = sparse_tensor.coordinates %arg0 { level = 0 : index } : tensor<64xf32, #SparseVector> to memref<?xindex>
  %2 = sparse_tensor.values %arg0 : tensor<64xf32, #SparseVector> to memref<?xf32>
  return
}

// CHECK-LABEL: func @sparse_concat_dce(
//   CHECK-NOT: sparse_tensor.concatenate
//       CHECK: return
func.func @sparse_concat_dce(%arg0: tensor<2xf64, #SparseVector>,
                             %arg1: tensor<3xf64, #SparseVector>,
                             %arg2: tensor<4xf64, #SparseVector>) {
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<2xf64, #SparseVector>,
         tensor<3xf64, #SparseVector>,
         tensor<4xf64, #SparseVector> to tensor<9xf64, #SparseVector>
  return
}

// CHECK-LABEL: func @sparse_get_specifier_dce_fold(
//  CHECK-SAME:  %[[A0:.*]]: !sparse_tensor.storage_specifier
//  CHECK-SAME:  %[[A1:.*]]: index,
//  CHECK-SAME:  %[[A2:.*]]: index)
//   CHECK-NOT:  sparse_tensor.storage_specifier.set
//   CHECK-NOT:  sparse_tensor.storage_specifier.get
//       CHECK:  return %[[A1]]
func.func @sparse_get_specifier_dce_fold(%arg0: !sparse_tensor.storage_specifier<#SparseVector>, %arg1: index, %arg2: index) -> index {
  %0 = sparse_tensor.storage_specifier.set %arg0 lvl_sz at 0 with %arg1
       : !sparse_tensor.storage_specifier<#SparseVector>
  %1 = sparse_tensor.storage_specifier.set %0 pos_mem_sz at 0 with %arg2
       : !sparse_tensor.storage_specifier<#SparseVector>
  %2 = sparse_tensor.storage_specifier.get %1 lvl_sz at 0
       : !sparse_tensor.storage_specifier<#SparseVector>
  return %2 : index
}
