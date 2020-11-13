// RUN: mlir-opt %s                                                            \
// RUN:     -test-print-number-of-block-executions                             \
// RUN:     -split-input-file 2>&1                                             \
// RUN:   | FileCheck %s --dump-input=always

// CHECK-LABEL: Number of executions: empty
func @empty() {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  return
}

// -----

// CHECK-LABEL: Number of executions: sequential
func @sequential() {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  br ^bb1
^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: Number of executions: 1
  br ^bb2
^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: Number of executions: 1
  return
}

// -----

// CHECK-LABEL: Number of executions: conditional
func @conditional(%cond : i1) {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  br ^bb1
^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: Number of executions: 1
  cond_br %cond, ^bb2, ^bb3
^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: Number of executions: <unknown>
  br ^bb4
^bb3:
  // CHECK: Block: 3
  // CHECK-NEXT: Number of executions: <unknown>
  br ^bb4
^bb4:
  // CHECK: Block: 4
  // CHECK-NEXT: Number of executions: <unknown>
  return
}

// -----

// CHECK-LABEL: Number of executions: loop
func @loop(%cond : i1) {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  br ^bb1
^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: Number of executions: <unknown>
  br ^bb2
^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: Number of executions: <unknown>
  br ^bb3
^bb3:
  // CHECK: Block: 3
  // CHECK-NEXT: Number of executions: <unknown>
  cond_br %cond, ^bb1, ^bb4
^bb4:
  // CHECK: Block: 4
  // CHECK-NEXT: Number of executions: <unknown>
  return
}

// -----

// CHECK-LABEL: Number of executions: scf_if_dynamic_branch
func @scf_if_dynamic_branch(%cond : i1) {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  scf.if %cond {
    // CHECK: Block: 1
    // CHECK-NEXT: Number of executions: <unknown>
  } else {
    // CHECK: Block: 2
    // CHECK-NEXT: Number of executions: <unknown>
  }
  return
}

// -----

// CHECK-LABEL: Number of executions: async_execute
func @async_execute() {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  async.execute {
   // CHECK: Block: 1
   // CHECK-NEXT: Number of executions: 1
    async.yield
  }
  return
}

// -----

// CHECK-LABEL: Number of executions: async_execute_with_scf_if
func @async_execute_with_scf_if(%cond : i1) {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  async.execute {
    // CHECK: Block: 1
    // CHECK-NEXT: Number of executions: 1
    scf.if %cond {
    // CHECK: Block: 2
    // CHECK-NEXT: Number of executions: <unknown>
    } else {
    // CHECK: Block: 3
    // CHECK-NEXT: Number of executions: <unknown>
    }
    async.yield
  }
  return
}

// -----

// CHECK-LABEL: Number of executions: scf_for_constant_bounds
func @scf_for_constant_bounds() {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: Block: 1
    // CHECK-NEXT: Number of executions: 2
  }

  return
}

// -----

// CHECK-LABEL: Number of executions: propagate_parent_num_executions
func @propagate_parent_num_executions() {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: Block: 1
    // CHECK-NEXT: Number of executions: 2
    async.execute {
      // CHECK: Block: 2
      // CHECK-NEXT: Number of executions: 2
      async.yield
    }
  }

  return
}

// -----

// CHECK-LABEL: Number of executions: clear_num_executions
func @clear_num_executions(%step : index) {
  // CHECK: Block: 0
  // CHECK-NEXT: Number of executions: 1
  %c0 = constant 0 : index
  %c2 = constant 2 : index

  scf.for %i = %c0 to %c2 step %step {
    // CHECK: Block: 1
    // CHECK-NEXT: Number of executions: <unknown>
    async.execute {
      // CHECK: Block: 2
      // CHECK-NEXT: Number of executions: <unknown>
      async.yield
    }
  }

  return
}
