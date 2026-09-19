// RUN: mlir-opt %s --sparse-tensor-codegen --verify-diagnostics
// RUN: mlir-opt %s --sparse-tensor-conversion --verify-diagnostics

// Without lower-sparse-ops-to-foreach and lower-sparse-foreach-to-scf,
// storage conversion must not silently return the values buffer length.
#Loose = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : loose_compressed)
}>

func.func @loose(%t: tensor<?x?xf64, #Loose>) -> index {
  // expected-error@+1 {{failed to legalize operation 'sparse_tensor.number_of_entries'}}
  %n = sparse_tensor.number_of_entries %t : tensor<?x?xf64, #Loose>
  return %n : index
}
