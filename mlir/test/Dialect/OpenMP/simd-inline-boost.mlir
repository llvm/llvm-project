// RUN: mlir-opt -omp-simd-inline-boost %s | FileCheck %s

func.func @callee(%arg0: f32) -> f32 {
  omp.declare_simd
  return %arg0 : f32
}

func.func @no_simd_callee(%arg0: f32) -> f32 {
  return %arg0 : f32
}

// CHECK-LABEL: func.func @simd_with_call
func.func @simd_with_call(%lb: index, %ub: index, %step: index, %a: memref<?xf32>) {
  omp.simd {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %val = memref.load %a[%iv] : memref<?xf32>
      // CHECK: func.call @callee(%{{.*}}) {omp.simd_inline_boost} : (f32) -> f32
      %res = func.call @callee(%val) : (f32) -> f32
      memref.store %res, %a[%iv] : memref<?xf32>
      omp.yield
    }
  }
  return
}

// Calls to functions without declare simd should NOT be boosted.
// CHECK-LABEL: func.func @simd_without_declare_simd
func.func @simd_without_declare_simd(%lb: index, %ub: index, %step: index, %a: memref<?xf32>) {
  omp.simd {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %val = memref.load %a[%iv] : memref<?xf32>
      // CHECK: func.call @no_simd_callee(%{{.*}}) : (f32) -> f32
      // CHECK-NOT: omp.simd_inline_boost
      %res = func.call @no_simd_callee(%val) : (f32) -> f32
      memref.store %res, %a[%iv] : memref<?xf32>
      omp.yield
    }
  }
  return
}

// Calls outside omp.simd should NOT be modified.
// CHECK-LABEL: func.func @no_simd
func.func @no_simd(%v: f32) -> f32 {
  // CHECK: call @callee(%{{.*}}) : (f32) -> f32
  // CHECK-NOT: omp.simd_inline_boost
  %res = func.call @callee(%v) : (f32) -> f32
  return %res : f32
}

// Calls to declare simd functions should be boosted.
// CHECK-LABEL: func.func @wsloop_simd_with_call
func.func @wsloop_simd_with_call(%lb: index, %ub: index, %step: index, %a: memref<?xf32>) {
  omp.wsloop {
    omp.simd {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        %val = memref.load %a[%iv] : memref<?xf32>
        // CHECK: func.call @callee(%{{.*}}) {omp.simd_inline_boost} : (f32) -> f32
        %res = func.call @callee(%val) : (f32) -> f32
        memref.store %res, %a[%iv] : memref<?xf32>
        omp.yield
      }
    } {omp.composite}
  } {omp.composite}
  return
}

// Calls already marked should not be re-marked.
// CHECK-LABEL: func.func @already_marked
func.func @already_marked(%lb: index, %ub: index, %step: index, %v: f32) {
  omp.simd {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      // CHECK: func.call @callee(%{{.*}}) {omp.simd_inline_boost} : (f32) -> f32
      %res = func.call @callee(%v) {omp.simd_inline_boost} : (f32) -> f32
      omp.yield
    }
  }
  return
}
