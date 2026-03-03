// RUN: mlir-opt %s --sparse-tensor-conversion -verify-diagnostics -split-input-file

// Regression test for https://github.com/llvm/llvm-project/issues/180310:
// sparse_tensor.new with an unsupported element type (e.g. index) must not
// crash with llvm_unreachable in primaryTypeEncoding; the conversion should
// fail gracefully.

#sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

func.func @new_index_elem_type(%arg0: index) {
  // expected-error@+1 {{failed to legalize operation 'sparse_tensor.new'}}
  %0 = sparse_tensor.new %arg0 : index to tensor<?xindex, #sparse>
  return
}
