// RUN: mlir-opt -convert-scf-to-openmp='num-threads=4' %s | FileCheck %s

// CHECK-LABEL: @parallel
func.func @parallel(%arg0: index, %arg1: index, %arg2: index,
          %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) {
  // CHECK: omp.wsloop for (%[[LVAR1:.*]], %[[LVAR2:.*]]) : index = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
  // CHECK: memref.alloca_scope
  scf.parallel (%i, %j) = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
    // CHECK: "test.payload"(%[[LVAR1]], %[[LVAR2]]) : (index, index) -> ()
    "test.payload"(%i, %j) : (index, index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}

// CHECK-LABEL: @nested_loops
func.func @nested_loops(%arg0: index, %arg1: index, %arg2: index,
                   %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) {
  // CHECK: omp.wsloop for (%[[LVAR_OUT1:.*]]) : index = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: memref.alloca_scope
  scf.parallel (%i) = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: omp.parallel
    // CHECK: omp.wsloop for (%[[LVAR_IN1:.*]]) : index = (%arg1) to (%arg3) step (%arg5) {
    // CHECK: memref.alloca_scope
    scf.parallel (%j) = (%arg1) to (%arg3) step (%arg5) {
      // CHECK: "test.payload"(%[[LVAR_OUT1]], %[[LVAR_IN1]]) : (index, index) -> ()
      "test.payload"(%i, %j) : (index, index) -> ()
      // CHECK: }
    }
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}

// CHECK-LABEL: @adjacent_loops
func.func @adjacent_loops(%arg0: index, %arg1: index, %arg2: index,
                     %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) {
  // CHECK: omp.wsloop for (%[[LVAR_AL1:.*]]) : index = (%arg0) to (%arg2) step (%arg4) {
  // CHECK: memref.alloca_scope
  scf.parallel (%i) = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: "test.payload1"(%[[LVAR_AL1]]) : (index) -> ()
    "test.payload1"(%i) : (index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   omp.terminator
  // CHECK: }

  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) {
  // CHECK: omp.wsloop for (%[[LVAR_AL2:.*]]) : index = (%arg1) to (%arg3) step (%arg5) {
  // CHECK: memref.alloca_scope
  scf.parallel (%j) = (%arg1) to (%arg3) step (%arg5) {
    // CHECK: "test.payload2"(%[[LVAR_AL2]]) : (index) -> ()
    "test.payload2"(%j) : (index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}
