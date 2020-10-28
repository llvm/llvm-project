// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -test %S/failure-test.sh -pass-test DCE | FileCheck %s
// This input should be reduced by the pass pipeline so that only
// the @simple1 function remains as the other functions should be
// removed by the dead code elimination pass.
// CHECK-LABEL: func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {

// CHECK-NOT: func @dead_nested_function
func @dead_private_function() attributes { sym_visibility = "private" }

// CHECK-NOT: func @dead_nested_function
func @dead_nested_function() attributes { sym_visibility = "nested" }

func @simple1(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  "test.crashOp" () : () -> ()
  return
}
