// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

// UNSUPPORTED: system-windows

llvm.func @threadidx() -> i32 {
  %tid = nvvm.read.ptx.sreg.tid.x : i32
  llvm.return %tid : i32
}

// CHECK-LABEL: func @caller
llvm.func @caller() -> i32 {
  // CHECK-NOT: llvm.call @threadidx
  // CHECK: nvvm.read.ptx.sreg.tid.x
  %z = llvm.call @threadidx() : () -> (i32)
  llvm.return %z : i32
}
