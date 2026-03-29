// RUN: not mlir-runner %s -e entry -entry-point-result=void 2>&1 | FileCheck %s

// CHECK: Error: expected void function
llvm.func @entry() -> (i32) {
  %0 = llvm.mlir.constant(0 : index) : i32
  llvm.return %0 : i32
}
