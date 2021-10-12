// RUN: mlir-opt %s                                                            \
// RUN:     -test-print-number-of-operation-executions                         \
// RUN:     -split-input-file 2>&1                                             \
// RUN:   | FileCheck %s

// CHECK-LABEL: Number of executions: empty
func @empty() {
  // CHECK: Operation: std.return
  // CHECK-NEXT: Number of executions: 1
  return
}

// -----

// CHECK-LABEL: Number of executions: propagate_parent_num_executions
func @propagate_parent_num_executions() {
  // CHECK: Operation: arith.constant
  // CHECK-NEXT: Number of executions: 1
  %c0 = arith.constant 0 : index
  // CHECK: Operation: arith.constant
  // CHECK-NEXT: Number of executions: 1
  %c1 = arith.constant 1 : index
  // CHECK: Operation: arith.constant
  // CHECK-NEXT: Number of executions: 1
  %c2 = arith.constant 2 : index

  // CHECK-DAG: Operation: scf.for
  // CHECK-NEXT: Number of executions: 1
  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK-DAG: Operation: async.execute
    // CHECK-NEXT: Number of executions: 2
    async.execute {
      // CHECK-DAG: Operation: async.yield
      // CHECK-NEXT: Number of executions: 2
      async.yield
    }
  }

  return
}

// -----

// CHECK-LABEL: Number of executions: clear_num_executions
func @clear_num_executions(%step : index) {
  // CHECK: Operation: arith.constant
  // CHECK-NEXT: Number of executions: 1
  %c0 = arith.constant 0 : index
  // CHECK: Operation: arith.constant
  // CHECK-NEXT: Number of executions: 1
  %c2 = arith.constant 2 : index

  // CHECK: Operation: scf.for
  // CHECK-NEXT: Number of executions: 1
  scf.for %i = %c0 to %c2 step %step {
    // CHECK: Operation: async.execute
    // CHECK-NEXT: Number of executions: <unknown>
    async.execute {
      // CHECK: Operation: async.yield
      // CHECK-NEXT: Number of executions: <unknown>
      async.yield
    }
  }

  return
}
