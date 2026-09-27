// RUN: mlir-opt %s --inline | FileCheck %s

llvm.func @threadidx() -> i32 {
  %tid = rocdl.workitem.id.x : i32
  llvm.return %tid : i32
}

// CHECK-LABEL: func @caller
llvm.func @caller() -> i32 {
  // CHECK-NOT: llvm.call @threadidx
  // CHECK: rocdl.workitem.id.x
  %z = llvm.call @threadidx() : () -> (i32)
  llvm.return %z : i32
}
