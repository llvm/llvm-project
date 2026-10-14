// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

func.func @update_no_op(%x : memref<i32>) {
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    omp.yield(%xval : i32)
  }
  return
}

// CHECK-LABEL: func.func @update_no_op
// CHECK-NOT: omp.atomic.update

// -----

func.func @update_write_op(%x : memref<i32>, %value: i32) {
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    omp.yield(%value : i32)
  }
  return
}

// CHECK-LABEL: func.func @update_write_op
// CHECK-SAME:            (%[[X:.+]]: memref<i32>, %[[VALUE:.+]]: i32)
// CHECK: omp.atomic.write %[[X]] = %[[VALUE]] : memref<i32>, i32
// CHECK-NOT: omp.atomic.update

// -----

func.func @update_normal(%x : memref<i32>, %value: i32) {
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    %newval = arith.addi %xval, %value : i32
    omp.yield(%newval : i32)
  }
  return
}

// CHECK-LABEL: func.func @update_normal
// CHECK: omp.atomic.update
// CHECK: arith.addi
// CHECK: omp.yield

// -----

func.func @update_unnecessary_computations(%x: memref<i32>) {
  %c0 = arith.constant 0 : i32
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.addi %xval, %c0 : i32
    omp.yield(%newval: i32)
  }
  return
}

// CHECK-LABEL: func.func @update_unnecessary_computations
// CHECK-NOT: omp.atomic.update

// -----

func.func @update_unnecessary_computations(%x: memref<i32>) {
  %c0 = arith.constant 0 : i32
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.muli %xval, %c0 : i32
    omp.yield(%newval: i32)
  }
  return
}

// CHECK-LABEL: func.func @update_unnecessary_computations
// CHECK-NOT: omp.atomic.update
// CHECK: omp.atomic.write

// -----

// CHECK-LABEL: func.func @parallel_empty
func.func @parallel_empty() {
  omp.parallel {}
  return
}

// CHECK-NOT: omp.parallel

// -----

// CHECK-LABEL: func.func @parallel_only_terminator
func.func @parallel_only_terminator() {
  omp.parallel {
    omp.terminator
  }
  return
}

// CHECK-NOT: omp.parallel
// CHECK-NOT: omp.terminator

// -----

// CHECK-LABEL: func.func @parallel_no_side_effects
func.func @parallel_no_side_effects(%a: i32, %b: i32) {
  omp.parallel {
    %x = arith.addi %a, %b : i32
    omp.terminator
  }
  return
}

// CHECK-NOT: omp.parallel
// CHECK-NOT: omp.terminator

// -----

// CHECK-LABEL: func.func @parallel_maybe_side_effects
func.func @parallel_maybe_side_effects(%a: i32, %b: i32) {
  omp.parallel {
    func.call @foo() : () -> ()
    omp.terminator
  }
  return
}

func.func private @foo() -> ()

// CHECK: omp.parallel
// CHECK: func.call @foo() : () -> ()
// CHECK: omp.terminator

// -----

func.func @constant_hoisting_target(%x : !llvm.ptr) {
  omp.target kernel_type(generic) {
    ^bb0(%arg0: !llvm.ptr):
    %c1 = arith.constant 10 : i32
    llvm.store %c1, %arg0 : i32, !llvm.ptr
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @constant_hoisting_target
// CHECK-NOT: arith.constant
// CHECK: omp.target
// CHECK: arith.constant

// -----

// Pure sections and their enclosing nowait construct can both be removed.
// CHECK-LABEL: func.func @sections_nowait_no_side_effects
// CHECK-NEXT: return
func.func @sections_nowait_no_side_effects(%a: i32, %b: i32) {
  omp.sections nowait {
    omp.section {
      %unused = arith.addi %a, %b : i32
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

// Sections also retain their implicit barrier when their body is empty.
// CHECK-LABEL: func.func @sections_only_terminator
// CHECK-NEXT: omp.sections {
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @sections_only_terminator() {
  omp.sections {
    omp.terminator
  }
  return
}

// -----

// Effects must be considered recursively through both sections and section.
// CHECK-LABEL: func.func @sections_nowait_store
// CHECK-SAME: (%[[X:.*]]: memref<i32>, %[[VALUE:.*]]: i32)
// CHECK-NEXT: omp.sections nowait {
// CHECK-NEXT: omp.section {
// CHECK-NEXT: memref.store %[[VALUE]], %[[X]][] : memref<i32>
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @sections_nowait_store(%x: memref<i32>, %value: i32) {
  omp.sections nowait {
    omp.section {
      memref.store %value, %x[] : memref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

// Unknown effects in a nested operation also prevent removal.
// CHECK-LABEL: func.func @sections_nowait_call
// CHECK-NEXT: omp.sections nowait {
// CHECK-NEXT: omp.section {
// CHECK-NEXT: func.call @opaque() : () -> ()
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @sections_nowait_call() {
  omp.sections nowait {
    omp.section {
      func.call @opaque() : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

func.func private @opaque()
