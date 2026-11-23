// RUN: mlir-opt %s --sparse-tensor-conversion -verify-diagnostics -split-input-file

// Regression test for https://github.com/llvm/llvm-project/issues/180310:
// sparse_tensor.new with an unsupported element type (e.g. index) must not
// crash with llvm_unreachable in primaryTypeEncoding; the conversion should
// fail gracefully.

#sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

func.func @new_index_elem_type(%arg0: index) {
  // expected-error@+1 {{invalid primary type}}
  %0 = sparse_tensor.new %arg0 : index to tensor<?xindex, #sparse>
  return
}

// -----

#sparse1d = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error@+1 {{invalid primary type}}
func.func @alloc_tensor_i1_elem_type(%sz: index) -> tensor<?xi1, #sparse1d> {
  %0 = bufferization.alloc_tensor(%sz) : tensor<?xi1, #sparse1d>
  return %0 : tensor<?xi1, #sparse1d>
}

// -----

#unordered_coo = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed(nonunique, nonordered), d1 : singleton(nonordered))}>
#ordered_coo = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)}>

// expected-error@+1 {{invalid primary type}}
func.func @reorder_coo_i1_elem_type(%arg0: tensor<?x?xi1, #unordered_coo>)
    -> tensor<?x?xi1, #ordered_coo> {
  %0 = sparse_tensor.reorder_coo quick_sort %arg0
    : tensor<?x?xi1, #unordered_coo> to tensor<?x?xi1, #ordered_coo>
  return %0 : tensor<?x?xi1, #ordered_coo>
}

// -----

#sparse1d = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error@+2 {{invalid primary type}}
func.func @assemble_i1_elem_type(%pos: tensor<2xindex>, %coords: tensor<4x1xindex>,
                                 %vals: tensor<4xi1>) -> tensor<8xi1, #sparse1d> {
  %0 = sparse_tensor.assemble (%pos, %coords), %vals
    : (tensor<2xindex>, tensor<4x1xindex>), tensor<4xi1> to tensor<8xi1, #sparse1d>
  return %0 : tensor<8xi1, #sparse1d>
}

// -----

#sparse1d = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error@+1 {{invalid primary type}}
func.func @values_i1_elem_type(%arg0: tensor<8xi1, #sparse1d>) -> memref<?xi1> {
  %0 = sparse_tensor.values %arg0 : tensor<8xi1, #sparse1d> to memref<?xi1>
  return %0 : memref<?xi1>
}

// -----

#sparse1d = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error@+1 {{invalid primary type}}
func.func @number_of_entries_i1_elem_type(%arg0: tensor<8xi1, #sparse1d>) -> index {
  %0 = sparse_tensor.number_of_entries %arg0 : tensor<8xi1, #sparse1d>
  return %0 : index
}

// -----

#sparse1d = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error@+1 {{invalid primary type}}
func.func @disassemble_i1_elem_type(%sp: tensor<8xi1, #sparse1d>,
                                      %op: tensor<2xindex>, %oi: tensor<4x1xindex>,
                                      %od: tensor<4xi1>)
    -> (tensor<2xindex>, tensor<4x1xindex>, tensor<4xi1>) {
  %rp, %ri, %d, %rpl, %ril, %dl = sparse_tensor.disassemble %sp
    : tensor<8xi1, #sparse1d>
    out_lvls(%op, %oi : tensor<2xindex>, tensor<4x1xindex>)
    out_vals(%od : tensor<4xi1>)
    -> (tensor<2xindex>, tensor<4x1xindex>), tensor<4xi1>, (index, index), index
  return %rp, %ri, %d : tensor<2xindex>, tensor<4x1xindex>, tensor<4xi1>
}

// -----

#sparse1d = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error@+1 {{invalid primary type}}
func.func @insert_i1_elem_type(%arg0: tensor<8xi1, #sparse1d>, %idx: index, %val: i1)
    -> tensor<8xi1, #sparse1d> {
  %0 = tensor.insert %val into %arg0[%idx] : tensor<8xi1, #sparse1d>
  return %0 : tensor<8xi1, #sparse1d>
}

// -----

#csr = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>

// expected-error@+1 {{invalid primary type}}
func.func @compress_i1_elem_type(%tensor: tensor<8x8xi1, #csr>,
                                 %values: memref<?xi1>, %filled: memref<?xi1>,
                                 %added: memref<?xindex>, %count: index, %i: index)
    -> tensor<8x8xi1, #csr> {
  %0 = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%i]
    : memref<?xi1>, memref<?xi1>, memref<?xindex>, tensor<8x8xi1, #csr>
  return %0 : tensor<8x8xi1, #csr>
}
