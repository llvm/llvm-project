// RUN: not mlir-runner %s -e entry_point -entry-point-result=void 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-POINT
// RUN: not mlir-runner %s -e entry_inputs -entry-point-result=void 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-INPUTS
// RUN: not mlir-runner %s -e entry_result -entry-point-result=void 2>&1 | FileCheck %s --check-prefix=CHECK-ENTRY-RESULT

// CHECK-ENTRY-POINT: Error: entry point not found
llvm.func @entry_point() -> ()

// CHECK-ENTRY-INPUTS: Error: function inputs not supported
llvm.func @entry_inputs(%arg0: i32) {
  llvm.return
}

// CHECK-ENTRY-RESULT: Error: expected void function
llvm.func @entry_result() -> (i32) {
  %0 = llvm.mlir.constant(0 : index) : i32
  llvm.return %0 : i32
}
