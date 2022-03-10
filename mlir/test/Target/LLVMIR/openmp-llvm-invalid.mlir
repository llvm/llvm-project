// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file -verify-diagnostics

// Checking translation when the update is carried out by using more than one op
// in the region.
llvm.func @omp_atomic_update_multiple_step_update(%x: !llvm.ptr<i32>, %expr: i32) {
  // expected-error @+2 {{exactly two operations are allowed inside an atomic update region while lowering to LLVM IR}}
  // expected-error @+1 {{LLVM Translation failed for operation: omp.atomic.update}}
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %t1 = llvm.mul %xval, %expr : i32
    %t2 = llvm.sdiv %t1, %expr : i32
    %newval = llvm.add %xval, %t2 : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking translation when the captured variable is not used in the inner
// update operation
llvm.func @omp_atomic_update_multiple_step_update(%x: !llvm.ptr<i32>, %expr: i32) {
  // expected-error @+2 {{the update operation inside the region must be a binary operation and that update operation must have the region argument as an operand}}
  // expected-error @+1 {{LLVM Translation failed for operation: omp.atomic.update}}
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.mul %expr, %expr : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}
