// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -opt-reduction-pass='opt-pass=symbol-dce test=%S/../failure-test.sh' | FileCheck %s
// RUN: mlir-reduce %s -opt-reduction-pass='opt-pass-file=%S/dce-pipeline test=%S/../failure-test.sh' | FileCheck %s  --check-prefix=CHECK-OPT-FILE
// This input should be reduced by the pass pipeline so that only
// the @simple1 function remains as the other functions should be
// removed by the dead code elimination pass.

// CHECK-NOT: func private @dead_private_function
// CHECK-OPT-FILE-NOT: func private @dead_private_function
func.func private @dead_private_function()

// CHECK-NOT: func nested @dead_nested_function
// CHECK-OPT-FILE-NOT: funcnested @dead_nested_function
func.func nested @dead_nested_function()

// CHECK-LABEL: func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
// CHECK-OPT-FILE-LABEL: func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
func.func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  "test.op_crash" () : () -> ()
  return
}
