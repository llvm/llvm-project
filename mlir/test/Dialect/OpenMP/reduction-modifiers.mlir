// RUN: mlir-opt %s | mlir-opt | FileCheck %s

omp.declare_reduction @add : i32 init {
^bb0(%v: i32):
  %zero = arith.constant 0 : i32
  omp.yield(%zero : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %sum = arith.addi %a, %b : i32
  omp.yield(%sum : i32)
}

// CHECK-LABEL: func.func @mixed
func.func @mixed(%x: !llvm.ptr, %y: !llvm.ptr) {
  // CHECK: omp.parallel reduction(task @add %{{.*}} -> %{{.*}}, @add %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr)
  omp.parallel reduction(task @add %x -> %px, @add %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @uniform_items
func.func @uniform_items(%x: !llvm.ptr, %y: !llvm.ptr) {
  // CHECK: omp.parallel reduction(task @add %{{.*}} -> %{{.*}}, task @add %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr)
  omp.parallel reduction(task @add %x -> %px, task @add %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @defaults
func.func @defaults(%x: !llvm.ptr, %y: !llvm.ptr) {
  // CHECK: omp.parallel reduction(@add
  omp.parallel reduction(@add %x -> %px, @add %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @different_types
func.func @different_types(%x: !llvm.ptr, %y: memref<i32>) {
  // CHECK: omp.parallel reduction(task @add %{{.*}} -> %{{.*}}, @add %{{.*}} -> %{{.*}} : !llvm.ptr, memref<i32>)
  omp.parallel reduction(task @add %x -> %px, defaultmod @add %y -> %py : !llvm.ptr, memref<i32>) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @byref_item
func.func @byref_item(%x: !llvm.ptr, %y: !llvm.ptr) {
  // CHECK: omp.parallel reduction(task byref @add %{{.*}} -> %{{.*}}, byref @add %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr)
  omp.parallel reduction(task byref @add %x -> %px, byref @add %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  return
}

// Lowering may add ordinary reductions for compiler-generated state alongside
// an INSCAN reduction, for example for conditional lastprivate.
// CHECK-LABEL: func.func @inscan_with_default
func.func @inscan_with_default(%x: !llvm.ptr, %y: !llvm.ptr, %lb: i32, %ub: i32, %step: i32) {
  // CHECK: omp.wsloop reduction(inscan @add %{{.*}} -> %{{.*}}, @add %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr)
  omp.wsloop reduction(inscan @add %x -> %px, @add %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      // CHECK: omp.scan inclusive
      omp.scan inclusive(%px : !llvm.ptr)
      omp.yield
    }
  }
  return
}
