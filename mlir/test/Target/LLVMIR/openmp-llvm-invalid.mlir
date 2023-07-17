// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file -verify-diagnostics

// Checking translation when the update is carried out by using more than one op
// in the region.
llvm.func @omp_atomic_update_multiple_step_update(%x: !llvm.ptr<i32>, %expr: i32) {
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
  // expected-error @+2 {{no atomic update operation with region argument as operand found inside atomic.update region}}
  // expected-error @+1 {{LLVM Translation failed for operation: omp.atomic.update}}
  omp.atomic.update %x : !llvm.ptr<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.mul %expr, %expr : i32
    omp.yield(%newval : i32)
  }
  llvm.return
}

// -----

// Checking translation when the update is carried out by using more than one
// operations in the atomic capture region.
llvm.func @omp_atomic_update_multiple_step_update(%x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>, %expr: i32) {
  // expected-error @+1 {{LLVM Translation failed for operation: omp.atomic.capture}}
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>, i32
    // expected-error @+1 {{no atomic update operation with region argument as operand found inside atomic.update region}}
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.mul %expr, %expr : i32
      omp.yield(%newval : i32)
    }
  }
  llvm.return
}

// -----

// Checking translation when the captured variable is not used in the inner
// update operation
llvm.func @omp_atomic_update_multiple_step_update(%x: !llvm.ptr<i32>, %v: !llvm.ptr<i32>, %expr: i32) {
  omp.atomic.capture memory_order(seq_cst) {
    omp.atomic.read %v = %x : !llvm.ptr<i32>, i32 
    omp.atomic.update %x : !llvm.ptr<i32> {
    ^bb0(%xval: i32):
      %t1 = llvm.mul %xval, %expr : i32
      %t2 = llvm.sdiv %t1, %expr : i32
      %newval = llvm.add %xval, %t2 : i32
      omp.yield(%newval : i32)
    }
  }
  llvm.return
}

// -----

llvm.func @omp_threadprivate() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.constant(3 : i32) : i32

  %4 = llvm.alloca %0 x i32 {in_type = i32, name = "a"} : (i64) -> !llvm.ptr<i32>

  // expected-error @below {{Addressing symbol not found}}
  // expected-error @below {{LLVM Translation failed for operation: omp.threadprivate}}
  %5 = omp.threadprivate %4 : !llvm.ptr<i32> -> !llvm.ptr<i32>

  llvm.store %1, %5 : !llvm.ptr<i32>

  omp.parallel  {
    %6 = omp.threadprivate %4 : !llvm.ptr<i32> -> !llvm.ptr<i32>
    llvm.store %2, %6 : !llvm.ptr<i32>
    omp.terminator
  }

  llvm.store %3, %5 : !llvm.ptr<i32>
  llvm.return
}
