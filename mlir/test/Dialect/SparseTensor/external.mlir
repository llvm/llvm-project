// RUN: mlir-opt %s --sparse-assembler -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @nop(
// CHECK-SAME:    %[[A:.*]]: tensor<100xf32>) -> tensor<100xf32> {
// CHECK:         return %[[A]] : tensor<100xf32>
// CHECK:       }
func.func @nop(%arg0: tensor<100xf32>) -> tensor<100xf32> {
  return %arg0 : tensor<100xf32>
}

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

// CHECK-LABEL: func.func @sparse_in2(
// CHECK-SAME:    %[[X:.*0]]: tensor<100xf32>,
// CHECK-SAME:    %[[B:.*1]]: tensor<?xindex>,
// CHECK-SAME:    %[[C:.*2]]: tensor<?xindex>,
// CHECK-SAME:    %[[A:.*3]]: tensor<?xf32>) -> tensor<64x64xf32> {
// CHECK:         %[[I:.*]] = sparse_tensor.assemble (%[[B]], %[[C]]), %[[A]]
// CHECK:         %[[F:.*]] = call @_internal_sparse_in2(%[[X]], %[[I]])
// CHECK:         return %[[F]] : tensor<64x64xf32>
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_in2
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_in2(%arg0: tensor<100xf32>, %arg1: tensor<64x64xf32, #sparse>) -> tensor<64x64xf32> {
  %0 = sparse_tensor.convert %arg1 : tensor<64x64xf32, #sparse> to tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @sparse_out(
// CHECK-SAME:    %[[X:.*0]]: tensor<64x64xf32>,
// CHECK-SAME:    %[[B:.*1]]: tensor<?xindex>,
// CHECK-SAME:    %[[C:.*2]]: tensor<?xindex>,
// CHECK-SAME:    %[[A:.*3]]: tensor<?xf32>)
// CHECK:         %[[F:.*]] = call @_internal_sparse_out(%[[X]])
// CHECK:         sparse_tensor.disassemble %[[F]]
// CHECK:         return
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_out
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_out(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32, #sparse> {
  %0 = sparse_tensor.convert %arg0 : tensor<64x64xf32> to tensor<64x64xf32, #sparse>
  return %0 : tensor<64x64xf32, #sparse>
}

// -----

// CHECK-LABEL: func.func @sparse_out2(
// CHECK-SAME:    %[[X:.*0]]: tensor<64x64xf32>,
// CHECK-SAME:    %[[B:.*1]]: tensor<?xindex>,
// CHECK-SAME:    %[[C:.*2]]: tensor<?xindex>,
// CHECK-SAME:    %[[A:.*3]]: tensor<?xf32>)
// CHECK:         %[[F:.*]]:2 = call @_internal_sparse_out2(%[[X]])
// CHECK:         sparse_tensor.disassemble %[[F]]#1
// CHECK:         return %[[F]]#0
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_out2
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_out2(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32, #sparse>) {
  %0 = sparse_tensor.convert %arg0 : tensor<64x64xf32> to tensor<64x64xf32, #sparse>
  return %arg0, %0 : tensor<64x64xf32>, tensor<64x64xf32, #sparse>
}

// -----

// CHECK-LABEL: func.func @sparse_inout(
// CHECK-SAME:    %[[B:.*0]]: tensor<?xindex>,
// CHECK-SAME:    %[[C:.*1]]: tensor<?xindex>,
// CHECK-SAME:    %[[A:.*2]]: tensor<?xf32>,
// CHECK-SAME:    %[[E:.*3]]: tensor<?xindex>,
// CHECK-SAME:    %[[F:.*4]]: tensor<?xindex>,
// CHECK-SAME:    %[[D:.*5]]: tensor<?xf32>)
// CHECK:         %[[I:.*]] = sparse_tensor.assemble (%[[B]], %[[C]]), %[[A]]
// CHECK:         %[[F:.*]] = call @_internal_sparse_inout(%[[I]])
// CHECK:         sparse_tensor.disassemble %[[F]]
// CHECK:         return
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_inout
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @sparse_inout(%arg0: tensor<64x64xf32, #sparse>) -> tensor<64x64xf32, #sparse> {
  return %arg0 : tensor<64x64xf32, #sparse>
}

// -----

// CHECK-LABEL: func.func @sparse_inout_coo_soa(
// CHECK-SAME:    %[[B:.*0]]: tensor<?xindex>,
// CHECK-SAME:    %[[C:.*1]]: tensor<?xindex>,
// CHECK-SAME:    %[[D:.*2]]: tensor<?xindex>,
// CHECK-SAME:    %[[A:.*3]]: tensor<?xf32>,
// CHECK-SAME:    %[[F:.*4]]: tensor<?xindex>,
// CHECK-SAME:    %[[G:.*5]]: tensor<?xindex>,
// CHECK-SAME:    %[[H:.*6]]: tensor<?xindex>,
// CHECK-SAME:    %[[E:.*7]]: tensor<?xf32>)
// CHECK:         %[[I:.*]] = sparse_tensor.assemble (%[[B]], %[[C]], %[[D]]), %[[A]]
// CHECK:         %[[F:.*]] = call @_internal_sparse_inout_coo_soa(%[[I]])
// CHECK:         sparse_tensor.disassemble %[[F]]
// CHECK:         return
// CHECK:       }
// CHECK:       func.func private @_internal_sparse_inout
#sparse = #sparse_tensor.encoding<{
   map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa))
}>
func.func @sparse_inout_coo_soa(%arg0: tensor<64x64xf32, #sparse>) -> tensor<64x64xf32, #sparse> {
  return %arg0 : tensor<64x64xf32, #sparse>
}
