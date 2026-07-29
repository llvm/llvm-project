// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Variant selection (e.g. from Fortran `declare variant`) is resolved in the
// frontend, so at the MLIR level the dispatch region simply wraps a call to the
// selected variant procedure.

// CHECK-LABEL: func.func @omp_dispatch
// CHECK-SAME: (%[[X:.*]]: memref<i32>)
func.func @omp_dispatch(%x : memref<i32>) -> () {
  // CHECK: omp.dispatch {
  // CHECK-NEXT: func.call @variant(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  omp.dispatch {
    func.call @variant(%x) : (memref<i32>) -> ()
    omp.terminator
  }
  return
}

// Test that the generic form of omp.dispatch roundtrips to pretty-printed form.
// CHECK-LABEL: func.func @omp_dispatch_generic_to_pretty
// CHECK-SAME: (%[[X:.*]]: memref<i32>)
func.func @omp_dispatch_generic_to_pretty(%x : memref<i32>) -> () {
  // A plain call (outside any dispatch region) is left untouched.
  // CHECK: call @omp_dispatch(%[[X]]) : (memref<i32>) -> ()
  func.call @omp_dispatch(%x) : (memref<i32>) -> ()
  // CHECK: omp.dispatch {
  // CHECK-NEXT: func.call @variant(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  "omp.dispatch" () ({
    func.call @variant(%x) : (memref<i32>) -> ()
    "omp.terminator" () : () -> ()
  }) : () -> ()
  return
}

// Test the nowait clause on omp.dispatch.
// CHECK-LABEL: func.func @omp_dispatch_nowait
// CHECK-SAME: (%[[X:.*]]: memref<i32>)
func.func @omp_dispatch_nowait(%x : memref<i32>) -> () {
  // CHECK: omp.dispatch nowait {
  // CHECK-NEXT: func.call @variant(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  omp.dispatch nowait {
    func.call @variant(%x) : (memref<i32>) -> ()
    omp.terminator
  }
  return
}

// novariants clause round-trip; the frontend materializes the runtime
// base/variant selection inside the region.
// CHECK-LABEL: func.func @omp_dispatch_novariants
// CHECK-SAME: (%[[COND:.*]]: i1, %[[X:.*]]: memref<i32>)
func.func @omp_dispatch_novariants(%cond : i1, %x : memref<i32>) -> () {
  // CHECK: omp.dispatch novariants(%[[COND]]) {
  // CHECK-NEXT: func.call @variant(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  omp.dispatch novariants(%cond) {
    func.call @variant(%x) : (memref<i32>) -> ()
    omp.terminator
  }
  return
}

// novariants and nowait together.
// CHECK-LABEL: func.func @omp_dispatch_novariants_nowait
// CHECK-SAME: (%[[COND:.*]]: i1, %[[X:.*]]: memref<i32>)
func.func @omp_dispatch_novariants_nowait(%cond : i1, %x : memref<i32>) -> () {
  // CHECK: omp.dispatch novariants(%[[COND]]) nowait {
  // CHECK-NEXT: func.call @variant(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  omp.dispatch novariants(%cond) nowait {
    func.call @variant(%x) : (memref<i32>) -> ()
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func private @variant(memref<i32>)
func.func private @variant(memref<i32>) -> ()
