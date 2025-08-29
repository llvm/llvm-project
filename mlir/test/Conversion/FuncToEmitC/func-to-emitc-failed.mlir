// RUN: mlir-opt -convert-func-to-emitc %s -split-input-file -verify-diagnostics

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @unsuppoted_emitc_type(%arg0: i4) -> i4 {
  return %arg0 : i4
}
