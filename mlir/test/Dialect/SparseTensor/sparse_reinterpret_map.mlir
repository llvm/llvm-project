// RUN: mlir-opt %s --sparse-reinterpret-map | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>)
//       CHECK: return %[[A0]]
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}
