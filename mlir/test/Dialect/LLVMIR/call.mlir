// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s

module {
  // CHECK: llvm.func @f()
  llvm.func @f()

  // CHECK-LABEL: call_convergent
  llvm.func @call_convergent() {
    // CHECK: llvm.call @f() {convergent} : () -> ()
    llvm.call @f() {convergent} : () -> ()
    llvm.return
  }

  // CHECK-LABEL: call_no_unwind
  llvm.func @call_no_unwind() {
    // CHECK: llvm.call @f() {no_unwind} : () -> ()
    llvm.call @f() {no_unwind} : () -> ()
    llvm.return
  }

  // CHECK-LABEL: call_will_return
  llvm.func @call_will_return() {
    // CHECK: llvm.call @f() {will_return} : () -> ()
    llvm.call @f() {will_return} : () -> ()
    llvm.return
  }

  // CHECK-LABEL: call_mem_effects
  llvm.func @call_mem_effects() {
    // CHECK: llvm.call @f() {memory = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = write>} : () -> ()
    llvm.call @f() {memory = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = write>} : () -> ()
    llvm.return
  }
}
