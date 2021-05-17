// RUN: mlir-opt -convert-scf-to-openmp %s | FileCheck %s

// CHECK-LABEL: @parallel
func @parallel(%arg0: index, %arg1: index, %arg2: index,
          %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: omp.parallel {
  // CHECK: omp.wsloop (%[[LVAR1:.*]], %[[LVAR2:.*]]) : index = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
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
func @nested_loops(%arg0: index, %arg1: index, %arg2: index,
                   %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: omp.parallel {
  // CHECK: omp.wsloop (%[[LVAR_OUT1:.*]]) : index = (%arg0) to (%arg2) step (%arg4) {
  // CHECK-NOT: omp.parallel
  scf.parallel (%i) = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: omp.wsloop (%[[LVAR_IN1:.*]]) : index = (%arg1) to (%arg3) step (%arg5) {
    scf.parallel (%j) = (%arg1) to (%arg3) step (%arg5) {
      // CHECK: "test.payload"(%[[LVAR_OUT1]], %[[LVAR_IN1]]) : (index, index) -> ()
      "test.payload"(%i, %j) : (index, index) -> ()
      // CHECK:   omp.yield
      // CHECK: }
    }
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}

func @adjacent_loops(%arg0: index, %arg1: index, %arg2: index,
                     %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: omp.parallel {
  // CHECK: omp.wsloop (%[[LVAR_AL1:.*]]) : index = (%arg0) to (%arg2) step (%arg4) {
  scf.parallel (%i) = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: "test.payload1"(%[[LVAR_AL1]]) : (index) -> ()
    "test.payload1"(%i) : (index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   omp.terminator
  // CHECK: }

  // CHECK: omp.parallel {
  // CHECK: omp.wsloop (%[[LVAR_AL2:.*]]) : index = (%arg1) to (%arg3) step (%arg5) {
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
