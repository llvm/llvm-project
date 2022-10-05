// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @get_parent_for_op
func.func @get_parent_for_op(%arg0: index, %arg1: index, %arg2: index) {
  // expected-remark @below {{first loop}}
  scf.for %i = %arg0 to %arg1 step %arg2 {
    // expected-remark @below {{second loop}}
    scf.for %j = %arg0 to %arg1 step %arg2 {
      // expected-remark @below {{third loop}}
      scf.for %k = %arg0 to %arg1 step %arg2 {
        arith.addi %i, %j : index
      }
    }
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1
    // CHECK: = transform.loop.get_parent_for
    %1 = transform.loop.get_parent_for %0
    %2 = transform.loop.get_parent_for %0 { num_loops = 2 }
    %3 = transform.loop.get_parent_for %0 { num_loops = 3 }
    transform.test_print_remark_at_operand %1, "third loop"
    transform.test_print_remark_at_operand %2, "second loop"
    transform.test_print_remark_at_operand %3, "first loop"
  }
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  arith.addi %arg0, %arg1 : index
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1
    // expected-error @below {{could not find an 'scf.for' parent}}
    %1 = transform.loop.get_parent_for %0
  }
}

// -----

// Outlined functions:
//
// CHECK: func @foo(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
// CHECK:   scf.for
// CHECK:     arith.addi
//
// CHECK: func @foo[[SUFFIX:.+]](%{{.+}}, %{{.+}}, %{{.+}})
// CHECK:   scf.for
// CHECK:     arith.addi
//
// CHECK-LABEL @loop_outline_op
func.func @loop_outline_op(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK: scf.for
  // CHECK-NOT: scf.for
  // CHECK:   scf.execute_region
  // CHECK:     func.call @foo
  scf.for %i = %arg0 to %arg1 step %arg2 {
    scf.for %j = %arg0 to %arg1 step %arg2 {
      arith.addi %i, %j : index
    }
  }
  // CHECK: scf.execute_region
  // CHECK-NOT: scf.for
  // CHECK:   func.call @foo[[SUFFIX]]
  scf.for %j = %arg0 to %arg1 step %arg2 {
    arith.addi %j, %j : index
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1
    %1 = transform.loop.get_parent_for %0
    // CHECK: = transform.loop.outline %{{.*}}
    transform.loop.outline %1 {func_name = "foo"}
  }
}

// -----

func.func private @cond() -> i1
func.func private @body()

func.func @loop_outline_op_multi_region() {
  // expected-note @below {{target op}}
  scf.while : () -> () {
    %0 = func.call @cond() : () -> i1
    scf.condition(%0)
  } do {
  ^bb0:
    func.call @body() : () -> ()
    scf.yield
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["scf.while"]} in %arg1
    // expected-error @below {{failed to outline}}
    transform.loop.outline %0 {func_name = "foo"}
  }
}

// -----

// CHECK-LABEL: @loop_peel_op
func.func @loop_peel_op() {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[C42:.+]] = arith.constant 42
  // CHECK: %[[C5:.+]] = arith.constant 5
  // CHECK: %[[C40:.+]] = arith.constant 40
  // CHECK: scf.for %{{.+}} = %[[C0]] to %[[C40]] step %[[C5]]
  // CHECK:   arith.addi
  // CHECK: scf.for %{{.+}} = %[[C40]] to %[[C42]] step %[[C5]]
  // CHECK:   arith.addi
  %0 = arith.constant 0 : index
  %1 = arith.constant 42 : index
  %2 = arith.constant 5 : index
  scf.for %i = %0 to %1 step %2 {
    arith.addi %i, %i : index
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1
    %1 = transform.loop.get_parent_for %0
    transform.loop.peel %1
  }
}

// -----

func.func @loop_pipeline_op(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  // CHECK: memref.load %[[MEMREF:.+]][%{{.+}}]
  // CHECK: memref.load %[[MEMREF]]
  // CHECK: arith.addf
  // CHECK: scf.for
  // CHECK:   memref.load
  // CHECK:   arith.addf
  // CHECK:   memref.store
  // CHECK: arith.addf
  // CHECK: memref.store
  // CHECK: memref.store
  // expected-remark @below {{transformed}}
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_elem = memref.load %A[%i0] : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf : f32
    memref.store %A1_elem, %result[%i0] : memref<?xf32>
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["arith.addf"]} in %arg1
    %1 = transform.loop.get_parent_for %0
    %2 = transform.loop.pipeline %1
    // Verify that the returned handle is usable.
    transform.test_print_remark_at_operand %2, "transformed"
  }
}

// -----

// CHECK-LABEL: @loop_unroll_op
func.func @loop_unroll_op() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  // CHECK: scf.for %[[I:.+]] =
  scf.for %i = %c0 to %c42 step %c5 {
    // CHECK-COUNT-4: arith.addi %[[I]]
    arith.addi %i, %i : index
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1
    %1 = transform.loop.get_parent_for %0
    transform.loop.unroll %1 { factor = 4 }
  }
}

