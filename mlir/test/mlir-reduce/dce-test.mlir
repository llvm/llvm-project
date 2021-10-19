// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -opt-reduction-pass='opt-pass=symbol-dce test=%S/failure-test.sh' | FileCheck %s
// This input should be reduced by the pass pipeline so that only
// the @simple1 function remains as the other functions should be
// removed by the dead code elimination pass.

// CHECK-NOT: func private @dead_private_function
func private @dead_private_function()

// CHECK-NOT: func nested @dead_nested_function
func nested @dead_nested_function()

// CHECK-LABEL: func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  "test.op_crash" () : () -> ()
  return
}
