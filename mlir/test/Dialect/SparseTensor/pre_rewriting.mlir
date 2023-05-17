// RUN: mlir-opt %s -pre-sparsification-rewrite | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"]
}>

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

#Slice = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ],
  slice = [ (?, 1, 1), (?, 3, 1) ]
}>

// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A:.*]]: tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>>)
//       CHECK: return %[[A]] : tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_nop_cast(%a : tensor<?xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %a : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  %1 = tensor.cast %0 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  %2 = tensor.cast %1 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %2 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_repair_cast(
//  CHECK-SAME: %[[A:.*]]: tensor<?xf32>)
//       CHECK: %[[C:.*]] = sparse_tensor.convert %[[A]] : tensor<?xf32> to tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>
//       CHECK: return %[[C]] : tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_repair_cast(%a : tensor<?xf32>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %a : tensor<?xf32> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_fuse_slice(
//  CHECK-SAME: %[[A:.*]]: tensor<2x3xi64, #sparse_tensor.encoding<{{{.*}}}>>)
//       CHECK: %[[E:.*]] = tensor.extract_slice %[[A]][1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #sparse_tensor.encoding<{{{.*}}}>> to tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>>
//       CHECK: %[[C:.*]] = sparse_tensor.convert %[[E]] : tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>> to tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>>
//       CHECK: return %[[C]] : tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_fuse_slice(%a : tensor<2x3xi64, #SortedCOO>) -> tensor<1x3xi64, #SortedCOO> {
  %extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64>
  %cast = tensor.cast %extracted_slice : tensor<1x3xi64> to tensor<1x3xi64, #Slice>
  %0 = sparse_tensor.convert %cast : tensor<1x3xi64, #Slice> to tensor<1x3xi64, #SortedCOO>
  return %0 : tensor<1x3xi64, #SortedCOO>
}
