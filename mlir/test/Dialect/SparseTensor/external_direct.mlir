// RUN: mlir-opt %s --sparse-assembler="direct-out=True" -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @sparse_in(
// CHECK-SAME:    %[[B:.*0]]: tensor<?xindex>,
// CHECK-SAME:    %[[C:.*1]]: tensor<?xindex>,
// CHECK-SAME:    %[[A:.*]]: tensor<?xf32>) -> tensor<64x64xf32> {
// CHECK:         %[[I:.*]] = sparse_tensor.assemble (%[[B]], %[[C]]), %[[A]]
// CHECK:         %[[F:.*]] = call @_internal_sparse_in(%[[I]])
// CHECK:         return %[[F]] : tensor<64x64xf32>
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_in
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_in(%arg0: tensor<64x64xf32, #sparse>) -> tensor<64x64xf32> {
  %0 = sparse_tensor.convert %arg0 : tensor<64x64xf32, #sparse> to tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @sparse_out(
// CHECK-SAME:    %[[X:.*0]]: tensor<64x64xf32>)
// CHECK:         %[[F:.*]] = call @_internal_sparse_out(%[[X]])
// CHECK:         %[[P:.*]] = sparse_tensor.positions %[[F]]
// CHECK:         %[[C:.*]] = sparse_tensor.coordinates %[[F]]
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[F]]
// CHECK:         return %[[P]], %[[C]], %[[V]]
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_out
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_out(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32, #sparse> {
  %0 = sparse_tensor.convert %arg0 : tensor<64x64xf32> to tensor<64x64xf32, #sparse>
  return %0 : tensor<64x64xf32, #sparse>
}

// -----

// CHECK-LABEL: func.func @sparse_out2(
// CHECK-SAME:    %[[X:.*0]]: tensor<64x64xf32>)
// CHECK:         %[[F:.*]]:2 = call @_internal_sparse_out2(%[[X]])
// CHECK:         %[[P:.*]] = sparse_tensor.positions %[[F]]#1
// CHECK:         %[[C:.*]] = sparse_tensor.coordinates %[[F]]#1
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[F]]#1
// CHECK:         return %[[F]]#0, %[[P]], %[[C]], %[[V]]
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_out2
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_out2(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32, #sparse>) {
  %0 = sparse_tensor.convert %arg0 : tensor<64x64xf32> to tensor<64x64xf32, #sparse>
  return %arg0, %0 : tensor<64x64xf32>, tensor<64x64xf32, #sparse>
}
