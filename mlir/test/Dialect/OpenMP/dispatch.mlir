// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @foo_dispatch
// CHECK-SAME: (%[[X:.*]]: memref<i32>)
func.func @foo_dispatch(%x : memref<i32>) -> () {
  // CHECK: %[[V:.*]] = memref.load %[[X]][] : memref<i32>
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[V]], %[[C1]] : i32
  // CHECK: cf.cond_br %[[CMP]], ^[[BB1:.*]], ^[[BB2:.*]]
  %v = memref.load %x[] : memref<i32>
  %c1 = arith.constant 1 : i32
  %cmp = arith.cmpi eq, %v, %c1 : i32
  cf.cond_br %cmp, ^bb1, ^bb2
// CHECK: ^[[BB1]]:
// CHECK: call @variant1() : () -> ()
^bb1:
  func.call @variant1() : () -> ()
  cf.br ^bb3
// CHECK: ^[[BB2]]:
// CHECK: call @variant2() : () -> ()
^bb2:
  func.call @variant2() : () -> ()
  cf.br ^bb3
^bb3:
  return
}

// Test that the generic form of omp.dispatch roundtrips to pretty-printed form.
// CHECK-LABEL: func @omp_dispatch_generic_to_pretty
// CHECK-SAME: (%[[X:.*]]: memref<i32>)
func.func @omp_dispatch_generic_to_pretty(%x : memref<i32>) -> () {
  // CHECK: omp.dispatch {
  // CHECK-NEXT: func.call @foo_dispatch(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  "omp.dispatch" () ({
    func.call @foo_dispatch(%x) : (memref<i32>) -> ()
    "omp.terminator" () : () -> ()
  }) : () -> ()
  return
}

// Test multiple dispatch regions with stores selecting different variants.
// CHECK-LABEL: func @omp_dispatch_multiple
// CHECK-SAME: (%[[X:.*]]: memref<i32>)
func.func @omp_dispatch_multiple(%x : memref<i32>) -> () {
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: memref.store %[[C1]], %[[X]][] : memref<i32>
  %c1 = arith.constant 1 : i32
  memref.store %c1, %x[] : memref<i32>
  // CHECK: omp.dispatch {
  // CHECK-NEXT: func.call @foo_dispatch(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  "omp.dispatch" () ({
    "func.call" (%x) {callee = @foo_dispatch} : (memref<i32>) -> ()
    "omp.terminator" () : () -> ()
  }) : () -> ()
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: memref.store %[[C2]], %[[X]][] : memref<i32>
  %c2 = arith.constant 2 : i32
  memref.store %c2, %x[] : memref<i32>
  // CHECK: omp.dispatch {
  // CHECK-NEXT: func.call @foo_dispatch(%[[X]]) : (memref<i32>) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  "omp.dispatch" () ({
    "func.call" (%x) {callee = @foo_dispatch} : (memref<i32>) -> ()
    "omp.terminator" () : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func private @variant1()
// CHECK-LABEL: func private @variant2()
func.func private @variant1() -> ()
func.func private @variant2() -> ()
