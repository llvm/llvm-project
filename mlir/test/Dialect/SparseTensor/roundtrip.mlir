// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_new(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = sparse_tensor.new %[[A]] : !llvm.ptr<i8> to tensor<128xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<128xf64, #{{.*}}>
func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<128xf64, #SparseVector>
  return %0 : tensor<128xf64, #SparseVector>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_dealloc(
// CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>
//       CHECK: bufferization.dealloc_tensor %[[A]] : tensor<128xf64, #{{.*}}>
//       CHECK: return
func.func @sparse_dealloc(%arg0: tensor<128xf64, #SparseVector>) {
  bufferization.dealloc_tensor %arg0 : tensor<128xf64, #SparseVector>
  return
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_convert_1d_to_sparse(
// CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//       CHECK: %[[T:.*]] = sparse_tensor.convert %[[A]] : tensor<64xf32> to tensor<64xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<64xf32, #{{.*}}>
func.func @sparse_convert_1d_to_sparse(%arg0: tensor<64xf32>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// -----

#SparseTensor = #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense", "compressed" ] }>

// CHECK-LABEL: func @sparse_convert_3d_from_sparse(
// CHECK-SAME: %[[A:.*]]: tensor<8x8x8xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.convert %[[A]] : tensor<8x8x8xf64, #{{.*}}> to tensor<8x8x8xf64>
//       CHECK: return %[[T]] : tensor<8x8x8xf64>
func.func @sparse_convert_3d_from_sparse(%arg0: tensor<8x8x8xf64, #SparseTensor>) -> tensor<8x8x8xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<8x8x8xf64, #SparseTensor> to tensor<8x8x8xf64>
  return %0 : tensor<8x8x8xf64>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_pointers(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.pointers %[[A]] {dimension = 0 : index} : tensor<128xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_pointers(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %0 = sparse_tensor.pointers %arg0 {dimension = 0 : index} : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.indices %[[A]] {dimension = 0 : index} : tensor<128xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func.func @sparse_indices(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %0 = sparse_tensor.indices %arg0 {dimension = 0 : index} : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_values(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.values %[[A]] : tensor<128xf64, #{{.*}}> to memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func.func @sparse_values(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<128xf64, #SparseVector> to memref<?xf64>
  return %0 : memref<?xf64>
}

// -----

#DenseMatrix = #sparse_tensor.encoding<{dimLevelType = ["dense","dense"]}>

// CHECK-LABEL: func @sparse_load(
//  CHECK-SAME: %[[A:.*]]: tensor<16x32xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.load %[[A]] : tensor<16x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<16x32xf64, #{{.*}}>
func.func @sparse_load(%arg0: tensor<16x32xf64, #DenseMatrix>) -> tensor<16x32xf64, #DenseMatrix> {
  %0 = sparse_tensor.load %arg0 : tensor<16x32xf64, #DenseMatrix>
  return %0 : tensor<16x32xf64, #DenseMatrix>
}

// -----

#DenseMatrix = #sparse_tensor.encoding<{dimLevelType = ["dense","dense"]}>

// CHECK-LABEL: func @sparse_load_ins(
//  CHECK-SAME: %[[A:.*]]: tensor<16x32xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.load %[[A]] hasInserts : tensor<16x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<16x32xf64, #{{.*}}>
func.func @sparse_load_ins(%arg0: tensor<16x32xf64, #DenseMatrix>) -> tensor<16x32xf64, #DenseMatrix> {
  %0 = sparse_tensor.load %arg0 hasInserts : tensor<16x32xf64, #DenseMatrix>
  return %0 : tensor<16x32xf64, #DenseMatrix>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[B:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[C:.*]]: f64) {
//       CHECK: sparse_tensor.insert %[[A]], %[[B]], %[[C]] : tensor<128xf64, #{{.*}}>, memref<?xindex>, f64
//       CHECK: return
func.func @sparse_insert(%arg0: tensor<128xf64, #SparseVector>, %arg1: memref<?xindex>, %arg2: f64) {
  sparse_tensor.insert %arg0, %arg1, %arg2 : tensor<128xf64, #SparseVector>, memref<?xindex>, f64
  return
}

// -----

// CHECK-LABEL: func @sparse_push_back(
//  CHECK-SAME: %[[A:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> memref<?xf64> {
//       CHECK: %[[D:.*]] = sparse_tensor.push_back %[[A]], %[[B]], %[[C]] {idx = 2 : index} : memref<?xindex>, memref<?xf64>, f64 to memref<?xf64>
//       CHECK: return %[[D]]
func.func @sparse_push_back(%arg0: memref<?xindex>, %arg1: memref<?xf64>, %arg2: f64) -> memref<?xf64> {
  %0 = sparse_tensor.push_back %arg0, %arg1, %arg2 {idx = 2 : index} : memref<?xindex>, memref<?xf64>, f64 to memref<?xf64>
  return %0 : memref<?xf64>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_expansion(
//  CHECK-SAME: %[[A:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>)
//       CHECK: sparse_tensor.expand %[[A]]
//       CHECK: return
func.func @sparse_expansion(%arg0: tensor<8x8xf64, #SparseMatrix>) {
  %values, %filled, %added, %count = sparse_tensor.expand %arg0
    : tensor<8x8xf64, #SparseMatrix> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>,
//       CHECK: sparse_tensor.compress %[[A]]
//       CHECK: return
func.func @sparse_compression(%arg0: tensor<8x8xf64, #SparseMatrix>,
                         %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xi1>,
                         %arg4: memref<?xindex>, %arg5: index) {
  sparse_tensor.compress %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    : tensor<8x8xf64, #SparseMatrix>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_out(
//  CHECK-SAME: %[[A:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[B:.*]]: !llvm.ptr<i8>)
//       CHECK: sparse_tensor.out %[[A]], %[[B]] : tensor<?x?xf64, #sparse_tensor.encoding<{{.*}}>>, !llvm.ptr<i8>
//       CHECK: return
func.func @sparse_out(%arg0: tensor<?x?xf64, #SparseMatrix>, %arg1: !llvm.ptr<i8>) {
  sparse_tensor.out %arg0, %arg1 : tensor<?x?xf64, #SparseMatrix>, !llvm.ptr<i8>
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_binary(
//  CHECK-SAME:   %[[A:.*]]: f64, %[[B:.*]]: i64) -> f64 {
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:   %[[C1:.*]] = sparse_tensor.binary %[[A]], %[[B]] : f64, i64 to f64
//       CHECK:     overlap = {
//       CHECK:       ^bb0(%[[A1:.*]]: f64, %[[B1:.*]]: i64):
//       CHECK:         sparse_tensor.yield %[[A1]] : f64
//       CHECK:     }
//       CHECK:     left = identity
//       CHECK:     right = {
//       CHECK:       ^bb0(%[[A2:.*]]: i64):
//       CHECK:         sparse_tensor.yield %[[Z]] : f64
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_binary(%arg0: f64, %arg1: i64) -> f64 {
  %cf0 = arith.constant 0.0 : f64
  %r = sparse_tensor.binary %arg0, %arg1 : f64, i64 to f64
    overlap={
      ^bb0(%x: f64, %y: i64):
        sparse_tensor.yield %x : f64
    }
    left=identity
    right={
      ^bb0(%y: i64):
        sparse_tensor.yield %cf0 : f64
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_unary(
//  CHECK-SAME:   %[[A:.*]]: f64) -> f64 {
//       CHECK:   %[[C1:.*]] = sparse_tensor.unary %[[A]] : f64 to f64
//       CHECK:     present = {
//       CHECK:       ^bb0(%[[A1:.*]]: f64):
//       CHECK:         sparse_tensor.yield %[[A1]] : f64
//       CHECK:     }
//       CHECK:     absent = {
//       CHECK:       %[[R:.*]] = arith.constant -1.000000e+00 : f64
//       CHECK:       sparse_tensor.yield %[[R]] : f64
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_unary(%arg0: f64) -> f64 {
  %r = sparse_tensor.unary %arg0 : f64 to f64
    present={
      ^bb0(%x: f64):
        sparse_tensor.yield %x : f64
    } absent={
      ^bb0:
        %cf1 = arith.constant -1.0 : f64
        sparse_tensor.yield %cf1 : f64
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_unary(
//  CHECK-SAME:   %[[A:.*]]: f64) -> i64 {
//       CHECK:   %[[C1:.*]] = sparse_tensor.unary %[[A]] : f64 to i64
//       CHECK:     present = {
//       CHECK:       ^bb0(%[[A1:.*]]: f64):
//       CHECK:         %[[R:.*]] = arith.fptosi %[[A1]] : f64 to i64
//       CHECK:         sparse_tensor.yield %[[R]] : i64
//       CHECK:     }
//       CHECK:     absent = {
//       CHECK:     }
//       CHECK:   return %[[C1]] : i64
//       CHECK: }
func.func @sparse_unary(%arg0: f64) -> i64 {
  %r = sparse_tensor.unary %arg0 : f64 to i64
    present={
      ^bb0(%x: f64):
        %ret = arith.fptosi %x : f64 to i64
        sparse_tensor.yield %ret : i64
    }
    absent={}
  return %r : i64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_reduce_2d_to_1d(
//  CHECK-SAME:   %[[A:.*]]: f64, %[[B:.*]]: f64) -> f64 {
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:   %[[C1:.*]] = sparse_tensor.reduce %[[A]], %[[B]], %[[Z]] : f64 {
//       CHECK:       ^bb0(%[[A1:.*]]: f64, %[[B1:.*]]: f64):
//       CHECK:         sparse_tensor.yield %[[A1]] : f64
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_reduce_2d_to_1d(%arg0: f64, %arg1: f64) -> f64 {
  %cf0 = arith.constant 0.0 : f64
  %r = sparse_tensor.reduce %arg0, %arg1, %cf0 : f64 {
      ^bb0(%x: f64, %y: f64):
        sparse_tensor.yield %x : f64
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_select(
//  CHECK-SAME:   %[[A:.*]]: f64) -> f64 {
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:   %[[C1:.*]] = sparse_tensor.select %[[A]] : f64 {
//       CHECK:       ^bb0(%[[A1:.*]]: f64):
//       CHECK:         %[[B1:.*]] = arith.cmpf ogt, %[[A1]], %[[Z]] : f64
//       CHECK:         sparse_tensor.yield %[[B1]] : i1
//       CHECK:     }
//       CHECK:   return %[[C1]] : f64
//       CHECK: }
func.func @sparse_select(%arg0: f64) -> f64 {
  %cf0 = arith.constant 0.0 : f64
  %r = sparse_tensor.select %arg0 : f64 {
      ^bb0(%x: f64):
        %cmp = arith.cmpf "ogt", %x, %cf0 : f64
        sparse_tensor.yield %cmp : i1
    }
  return %r : f64
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @concat_sparse_sparse(
//  CHECK-SAME:   %[[A0:.*]]: tensor<2x4xf64
//  CHECK-SAME:   %[[A1:.*]]: tensor<3x4xf64
//  CHECK-SAME:   %[[A2:.*]]: tensor<4x4xf64
//       CHECK:   %[[TMP0:.*]] = sparse_tensor.concatenate %[[A0]], %[[A1]], %[[A2]] {dimension = 0 : index} :
//  CHECK-SAME:   tensor<2x4xf64
//  CHECK-SAME:   tensor<3x4xf64
//  CHECK-SAME:   tensor<4x4xf64
//  CHECK-SAME:   tensor<9x4xf64
//       CHECK:   return %[[TMP0]] : tensor<9x4xf64
func.func @concat_sparse_sparse(%arg0: tensor<2x4xf64, #SparseMatrix>,
                                %arg1: tensor<3x4xf64, #SparseMatrix>,
                                %arg2: tensor<4x4xf64, #SparseMatrix>) -> tensor<9x4xf64, #SparseMatrix> {
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<2x4xf64, #SparseMatrix>,
         tensor<3x4xf64, #SparseMatrix>,
         tensor<4x4xf64, #SparseMatrix> to tensor<9x4xf64, #SparseMatrix>
  return %0 : tensor<9x4xf64, #SparseMatrix>
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_tensor_foreach(
//  CHECK-SAME: %[[A0:.*]]: tensor<2x4xf64
//       CHECK: sparse_tensor.foreach in %[[A0]] : 
//       CHECK:  ^bb0(%arg1: index, %arg2: index, %arg3: f64):
func.func @sparse_tensor_foreach(%arg0: tensor<2x4xf64, #DCSR>) -> () {
  sparse_tensor.foreach in %arg0 : tensor<2x4xf64, #DCSR> do {
    ^bb0(%1: index, %2: index, %v: f64) :
  }
  return
}

// ----

// CHECK-LABEL: func @sparse_sort_1d0v(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xindex>)
//       CHECK: sparse_tensor.sort %[[A]], %[[B]] : memref<?xindex>
//       CHECK: return %[[B]]
func.func @sparse_sort_1d0v(%arg0: index, %arg1: memref<?xindex>) -> (memref<?xindex>) {
  sparse_tensor.sort %arg0, %arg1 : memref<?xindex>
  return %arg1 : memref<?xindex>
}

// -----

// CHECK-LABEL: func @sparse_sort_1d2v(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<20xindex>,
//  CHECK-SAME: %[[C:.*]]: memref<10xindex>,
//  CHECK-SAME: %[[D:.*]]: memref<?xf32>)
//       CHECK: sparse_tensor.sort %[[A]], %[[B]] jointly %[[C]], %[[D]] : memref<20xindex> jointly memref<10xindex>, memref<?xf32>
//       CHECK: return %[[B]], %[[C]], %[[D]]
func.func @sparse_sort_1d2v(%arg0: index, %arg1: memref<20xindex>, %arg2: memref<10xindex>, %arg3: memref<?xf32>) -> (memref<20xindex>, memref<10xindex>, memref<?xf32>) {
  sparse_tensor.sort %arg0, %arg1 jointly %arg2, %arg3 : memref<20xindex> jointly memref<10xindex>, memref<?xf32>
  return %arg1, %arg2, %arg3 : memref<20xindex>, memref<10xindex>, memref<?xf32>
}

// -----

// CHECK-LABEL: func @sparse_sort_2d1v(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<10xi8>,
//  CHECK-SAME: %[[C:.*]]: memref<20xi8>,
//  CHECK-SAME: %[[D:.*]]: memref<10xf64>)
//       CHECK: sparse_tensor.sort %[[A]], %[[B]], %[[C]] jointly %[[D]] : memref<10xi8>, memref<20xi8> jointly memref<10xf64>
//       CHECK: return %[[B]], %[[C]], %[[D]]
func.func @sparse_sort_2d1v(%arg0: index, %arg1: memref<10xi8>, %arg2: memref<20xi8>, %arg3: memref<10xf64>) -> (memref<10xi8>, memref<20xi8>, memref<10xf64>) {
  sparse_tensor.sort %arg0, %arg1, %arg2 jointly %arg3 : memref<10xi8>, memref<20xi8> jointly memref<10xf64>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<20xi8>, memref<10xf64>
}
