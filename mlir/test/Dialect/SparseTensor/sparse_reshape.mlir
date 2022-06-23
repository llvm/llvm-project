// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// TODO: check lowering to an actual implementation

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
#SparseMatrix = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

// CHECK-LABEL: func.func @sparse_expand(
// CHECK-SAME:  %[[A:.*]]: tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK:  %[[E:.*]] = tensor.expand_shape %[[A]] {{\[\[}}0, 1]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK:  return %[[E]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_expand(%arg0: tensor<100xf64, #SparseVector>) -> tensor<10x10xf64, #SparseMatrix> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] :
    tensor<100xf64, #SparseVector> into tensor<10x10xf64, #SparseMatrix>
  return %0 : tensor<10x10xf64, #SparseMatrix>
}

// CHECK-LABEL: func.func @sparse_collapse(
// CHECK-SAME:  %[[A:.*]]: tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK:  %[[C:.*]] = tensor.collapse_shape %[[A]] {{\[\[}}0, 1]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK:  return %[[C]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_collapse(%arg0: tensor<10x10xf64, #SparseMatrix>) -> tensor<100xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] :
    tensor<10x10xf64, #SparseMatrix> into tensor<100xf64, #SparseVector>
  return %0 : tensor<100xf64, #SparseVector>
}
