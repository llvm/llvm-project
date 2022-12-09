// RUN: mlir-opt %s -post-sparsification-rewrite | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

// CHECK-LABEL: func.func @expand_dense(
// CHECK-SAME:    %[[A:.*]]: tensor<12xf64>) -> tensor<3x4xf64> {
// CHECK:         %[[E:.*]] = tensor.expand_shape %[[A]] {{.*}} : tensor<12xf64> into tensor<3x4xf64>
// CHECK:         return %[[E]] : tensor<3x4xf64>
// CHECK:       }
func.func @expand_dense(%arg0: tensor<12xf64>) -> tensor<3x4xf64> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64> into tensor<3x4xf64>
  return %0 : tensor<3x4xf64>
}

// CHECK-LABEL: func.func @expand_from_sparse(
// CHECK-SAME:    %[[A:.*]]: tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<3x4xf64> {
// CHECK:         %[[C:.*]] = sparse_tensor.convert %[[A]] : tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>> to tensor<12xf64>
// CHECK:         %[[E:.*]] = tensor.expand_shape %[[C]] {{.*}} : tensor<12xf64> into tensor<3x4xf64>
// CHECK:         return %[[E]] : tensor<3x4xf64>
// CHECK:       }
func.func @expand_from_sparse(%arg0: tensor<12xf64, #SparseVector>) -> tensor<3x4xf64> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64, #SparseVector> into tensor<3x4xf64>
  return %0 : tensor<3x4xf64>
}

// CHECK-LABEL: func.func @expand_to_sparse(
// CHECK-SAME:    %[[A:.*]]: tensor<12xf64>) -> tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[E:.*]] = tensor.expand_shape %[[A]] {{.*}} : tensor<12xf64> into tensor<3x4xf64>
// CHECK:         %[[C:.*]] = sparse_tensor.convert %[[E]] : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         return %[[C]] : tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:       }
func.func @expand_to_sparse(%arg0: tensor<12xf64>) -> tensor<3x4xf64, #SparseMatrix> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64> into tensor<3x4xf64, #SparseMatrix>
  return %0 : tensor<3x4xf64, #SparseMatrix>
}

//
// Not rewritten, needs conversion.
//
// CHECK-LABEL:   func.func @expand_sparse2sparse(
// CHECK-SAME:    %[[A:.*]]: tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[E:.*]] = tensor.expand_shape %[[A]] {{.*}} : tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         return %[[E]] : tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:       }
func.func @expand_sparse2sparse(%arg0: tensor<12xf64, #SparseVector>) -> tensor<3x4xf64, #SparseMatrix> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64, #SparseVector> into tensor<3x4xf64, #SparseMatrix>
  return %0 : tensor<3x4xf64, #SparseMatrix>
}

// CHECK-LABEL: func.func @collapse_dense(
// CHECK-SAME:    %[[A:.*]]: tensor<3x4xf64>) -> tensor<12xf64> {
// CHECK:         %[[R:.*]] = tensor.collapse_shape %[[A]] {{.*}} : tensor<3x4xf64> into tensor<12xf64>
// CHECK:         return %[[R]] : tensor<12xf64>
// CHECK:       }
func.func @collapse_dense(%arg0: tensor<3x4xf64>) -> tensor<12xf64> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64> into tensor<12xf64>
  return %0 : tensor<12xf64>
}

// CHECK-LABEL: func.func @collapse_from_sparse(
// CHECK-SAME:    %[[A:.*]]: tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<12xf64> {
// CHECK:         %[[C:.*]] = sparse_tensor.convert %[[A]] : tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>> to tensor<3x4xf64>
// CHECK:         %[[R:.*]] = tensor.collapse_shape %[[C]] {{.*}} : tensor<3x4xf64> into tensor<12xf64>
// CHECK:         return %[[R]] : tensor<12xf64>
// CHECK:       }
func.func @collapse_from_sparse(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64, #SparseMatrix> into tensor<12xf64>
  return %0 : tensor<12xf64>
}

// CHECK-LABEL: func.func @collapse_to_sparse(
// CHECK-SAME:    %[[A:.*]]: tensor<3x4xf64>) -> tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[R:.*]] = tensor.collapse_shape %[[A]] {{.*}} : tensor<3x4xf64> into tensor<12xf64>
// CHECK:         %[[C:.*]] = sparse_tensor.convert %[[R]] : tensor<12xf64> to tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         return %[[C]] : tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:       }
func.func @collapse_to_sparse(%arg0: tensor<3x4xf64>) -> tensor<12xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64> into tensor<12xf64, #SparseVector>
  return %0 : tensor<12xf64, #SparseVector>
}

//
// Not rewritten, needs conversion.
//
// CHECK-LABEL:   func.func @collapse_sparse2sparse(
// CHECK-SAME:    %[[A:.*]]: tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[C:.*]] = tensor.collapse_shape %[[A]] {{.*}} : tensor<3x4xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         return %[[C]] : tensor<12xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:       }
func.func @collapse_sparse2sparse(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64, #SparseMatrix> into tensor<12xf64, #SparseVector>
  return %0 : tensor<12xf64, #SparseVector>
}
