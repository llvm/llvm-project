// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf{lower-permutation-maps=true}))" -split-input-file | FileCheck %s

// Ensure that the permutation map is lowered (by inserting a transpose op)
// before lowering the vector.transfer_read.

// CHECK-LABEL: func @transfer_read_2d_mask_transposed(
//   CHECK-DAG:   %[[PADDING:.*]] = arith.constant dense<-4.200000e+01> : vector<9xf32>
//   CHECK-DAG:   %[[MASK:.*]] = arith.constant dense<{{.*}}> : vector<4x9xi1>
//       CHECK:   %[[MASK_MEM:.*]] = memref.alloca() : memref<vector<4x9xi1>>
//       CHECK:   memref.store %[[MASK]], %[[MASK_MEM]][] : memref<vector<4x9xi1>>
//       CHECK:   %[[MASK_CASTED:.*]] = vector.type_cast %[[MASK_MEM]] : memref<vector<4x9xi1>> to memref<4xvector<9xi1>>
//       CHECK:   scf.for {{.*}} {
//       CHECK:     scf.if {{.*}} {
//       CHECK:       %[[MASK_LOADED:.*]] = memref.load %[[MASK_CASTED]][%{{.*}}] : memref<4xvector<9xi1>>
//       CHECK:       %[[READ:.*]] = vector.transfer_read %{{.*}}, %{{.*}}, %[[MASK_LOADED]] : memref<?x?xf32>, vector<9xf32>
//       CHECK:       memref.store %[[READ]], %{{.*}} : memref<4xvector<9xf32>>
//       CHECK:     }
//       CHECK:   }
//       CHECK:   %[[RESULT:.*]] = memref.load %{{.*}} : memref<vector<4x9xf32>>
//       CHECK:   %[[RESULT_T:.*]] = vector.transpose %[[RESULT]], [1, 0] : vector<4x9xf32> to vector<9x4xf32>
//       CHECK:   return %[[RESULT_T]] : vector<9x4xf32>

// Vector load with mask + transpose.
func.func @transfer_read_2d_mask_transposed(
    %A : memref<?x?xf32>, %base1: index, %base2: index) -> (vector<9x4xf32>) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[[1, 0, 1, 0, 1, 1, 1, 0, 1],
                                [0, 0, 1, 1, 1, 1, 1, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 0, 1],
                                [0, 0, 1, 0, 1, 1, 1, 0, 1]]> : vector<4x9xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} :
    memref<?x?xf32>, vector<9x4xf32>
  return %f : vector<9x4xf32>
}
