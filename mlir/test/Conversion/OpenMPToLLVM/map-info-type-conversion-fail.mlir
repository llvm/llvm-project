// RUN: mlir-opt -convert-openmp-to-llvm -split-input-file -verify-diagnostics %s

// Indicates that the TypeConversion has failed for the MPMapInfoOp.
// In this specific case, the `tensor` type (used in a TypeAttr) cannot be converted
// to an LLVM type. This test ensures that the conversion fails gracefully with a
// legalization error instead of crashing.
func.func @fail_map_info_tensor_type(%arg0: memref<?xf32>) {
  // expected-error@+1 {{failed to legalize operation 'omp.map.info' that was explicitly marked illegal}}
  %map_info = omp.map.info var_ptr(%arg0: memref<?xf32>, tensor<?xf32>) map_clauses(to) capture(ByRef) -> memref<?xf32>
  omp.target_update map_entries(%map_info: memref<?xf32>) {
    omp.terminator
  }
  return
}
