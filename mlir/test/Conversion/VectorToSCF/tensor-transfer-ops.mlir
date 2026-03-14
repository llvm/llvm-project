// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf{lower-tensors=true lower-scalable=true}))" -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @transfer_read_2d(
//       CHECK: %[[ALLOC:.*]] = memref.alloca() : memref<vector<4x9xf32>>
//       CHECK: %[[CASTED:.*]] = vector.type_cast %[[ALLOC]] : memref<vector<4x9xf32>> to memref<4xvector<9xf32>>
//       CHECK: scf.for {{.*}} {
//       CHECK:   %[[READ:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %cst {in_bounds = [true]} : tensor<?x?xf32>, vector<9xf32>
//       CHECK:   memref.store %[[READ]], %[[CASTED]][%{{.*}}] : memref<4xvector<9xf32>>
//       CHECK: }
//       CHECK: %[[LOADED:.*]] = memref.load %[[ALLOC]][] : memref<vector<4x9xf32>>
//       CHECK: return %[[LOADED]] : vector<4x9xf32>
func.func @transfer_read_2d(%A : tensor<?x?xf32>, %base1 : index, %base2 : index)
    -> (vector<4x9xf32>){
  %p = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %p {in_bounds = [true, true]}
      : tensor<?x?xf32>, vector<4x9xf32>
  return %f : vector<4x9xf32>
}

// -----

// CHECK-LABEL: func @transfer_write_2d(
//       CHECK: %[[ALLOC:.*]] = memref.alloca() : memref<vector<2x3xf32>>
//       CHECK: memref.store {{.*}}, %[[ALLOC]][] : memref<vector<2x3xf32>>
//       CHECK: %[[CASTED:.*]] = vector.type_cast %[[ALLOC]] : memref<vector<2x3xf32>> to memref<2xvector<3xf32>>
//       CHECK: %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[STATE:.*]] = %{{.*}}) -> (tensor<?x?xf32>) {
//       CHECK:   %[[LOADED:.*]] = memref.load %[[CASTED]][%{{.*}}] : memref<2xvector<3xf32>>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[LOADED]], %[[STATE]][{{.*}}] {in_bounds = [true]} : vector<3xf32>, tensor<?x?xf32>
//       CHECK:   scf.yield %[[WRITE]] : tensor<?x?xf32>
//       CHECK: }
//       CHECK: return %[[RESULT]] : tensor<?x?xf32>
func.func @transfer_write_2d(%A : tensor<?x?xf32>, %vec : vector<2x3xf32>,
                        %base1 : index, %base2 : index) -> (tensor<?x?xf32>) {
  %t = vector.transfer_write %vec, %A[%base1, %base2] {in_bounds = [true, true]}
      : vector<2x3xf32>, tensor<?x?xf32>
  return %t : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @scalable_transpose_store
//  CHECK-SAME: %[[TENSOR:[a-z0-9]+]]: tensor<?x?xf32>
//       CHECK: %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[ITER_ARG:.*]] = %[[TENSOR]]) -> (tensor<?x?xf32>)
//       CHECK:   %[[WRITE_SLICE:.*]] = vector.transfer_write %{{.*}} %[[ITER_ARG]]
//       CHECK:   scf.yield %[[WRITE_SLICE]]
//       CHECK: return %[[RESULT]]
func.func @scalable_transpose_store(%vec: vector<4x[4]xf32>, %A: tensor<?x?xf32>, %base1: index, %base2: index) -> tensor<?x?xf32> {
  %transpose = vector.transpose %vec, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
  %result = vector.transfer_write %transpose, %A[%base1, %base2] {in_bounds = [true, true]} : vector<[4]x4xf32>,  tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
