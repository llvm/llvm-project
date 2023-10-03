// RUN: mlir-opt --test-transform-dialect-interpreter=debug-transform-root-tag=all %s | FileCheck %s --check-prefixes=CHECK,ALL
// RUN: mlir-opt --test-transform-dialect-interpreter=debug-transform-root-tag=small %s | FileCheck %s --check-prefixes=CHECK,SMALL

func.func private @callee(memref<*xf32>)

// CHECK-LABEL: @large_alloc
func.func @large_alloc() {
  // SMALL: memref.alloc()
  // ALL:   memref.alloca
  %0 = memref.alloc() : memref<100x100xf32>
  %1 = memref.cast %0 : memref<100x100xf32> to memref<*xf32>
  call @callee(%1) : (memref<*xf32>) -> ()
  // SMALL: memref.dealloc
  // ALL-NOT: memref.dealloc
  memref.dealloc %0 : memref<100x100xf32>
  return
}

// CHECK-LABEL: @small_alloc
func.func @small_alloc() {
  // CHECK: memref.alloca
  %0 = memref.alloc() : memref<2x2xf32>
  %1 = memref.cast %0 : memref<2x2xf32> to memref<*xf32>
  call @callee(%1) : (memref<*xf32>) -> ()
  // CHECK-NOT: memref.dealloc
  memref.dealloc %0 : memref<2x2xf32>
  return
}

// CHECK-LABEL: @no_dealloc
func.func @no_dealloc() {
  // CHECK: memref.alloc()
  %0 = memref.alloc() : memref<2x2xf32>
  %1 = memref.cast %0 : memref<2x2xf32> to memref<*xf32>
  call @callee(%1) : (memref<*xf32>) -> ()
  return
}

// CHECK-LABEL: @mismatching_scope
func.func @mismatching_scope() {
  // CHECK: memref.alloc()
  %0 = memref.alloc() : memref<2x2xf32>
  %1 = memref.cast %0 : memref<2x2xf32> to memref<*xf32>
  call @callee(%1) : (memref<*xf32>) -> ()
  scf.execute_region {
    memref.dealloc %0 : memref<2x2xf32>
    scf.yield
  }
  return
}

transform.sequence failures(propagate) attributes {transform.target_tag = "all"} {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %0 {
    transform.apply_patterns.memref.alloc_to_alloca
  } : !transform.any_op
  transform.yield
}

transform.sequence failures(propagate) attributes {transform.target_tag = "small"} {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %0 {
    transform.apply_patterns.memref.alloc_to_alloca size_limit(32)
  } : !transform.any_op
  transform.yield
}
