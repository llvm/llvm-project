// RUN: mlir-opt %s --sparse-tensor-codegen --sparse-storage-specifier-to-llvm | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A0:.*0]]: memref<?xindex>,
//  CHECK-SAME: %[[A1:.*1]]: memref<?xindex>,
//  CHECK-SAME: %[[A2:.*2]]: memref<?xf64>,
//  CHECK-SAME: %[[A3:.*3]]: !llvm.struct<(array<1 x i64>, array<3 x i64>)>)
//       CHECK: return %[[A0]], %[[A1]], %[[A2]], %[[A3]] :
//  CHECK-SAME: memref<?xindex>, memref<?xindex>, memref<?xf64>, !llvm.struct<(array<1 x i64>, array<3 x i64>)>
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}
