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

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // CHECK: = transform.loop.get_parent_for
  %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !transform.op<"scf.for">
  %2 = transform.loop.get_parent_for %0 { num_loops = 2 } : (!transform.any_op) -> !transform.op<"scf.for">
  %3 = transform.loop.get_parent_for %0 { num_loops = 3 } : (!transform.any_op) -> !transform.op<"scf.for">
  transform.test_print_remark_at_operand %1, "third loop" : !transform.op<"scf.for">
  transform.test_print_remark_at_operand %2, "second loop" : !transform.op<"scf.for">
  transform.test_print_remark_at_operand %3, "first loop" : !transform.op<"scf.for">
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  arith.addi %arg0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{could not find an 'scf.for' parent}}
  %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !transform.op<"scf.for">
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

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0  : (!transform.any_op) -> !transform.op<"scf.for">
  // CHECK: = transform.loop.outline %{{.*}}
  transform.loop.outline %1 {func_name = "foo"} : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
}

// -----

// CHECK-LABEL: @loop_peel_op
func.func @loop_peel_op() {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[C41:.+]] = arith.constant 41
  // CHECK: %[[C5:.+]] = arith.constant 5
  // CHECK: %[[C40:.+]] = arith.constant 40
  // CHECK: scf.for %{{.+}} = %[[C0]] to %[[C40]] step %[[C5]]
  // CHECK:   arith.addi
  // CHECK: scf.for %{{.+}} = %[[C40]] to %[[C41]] step %[[C5]]
  // CHECK:   arith.addi
  %0 = arith.constant 0 : index
  %1 = arith.constant 41 : index
  %2 = arith.constant 5 : index
  // expected-remark @below {{main loop}}
  // expected-remark @below {{remainder loop}}
  scf.for %i = %0 to %1 step %2 {
    arith.addi %i, %i : index
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !transform.op<"scf.for">
  %main_loop, %remainder = transform.loop.peel %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
  // Verify that both of the generated loop handles are valid
  transform.test_print_remark_at_operand %main_loop, "main loop" : !transform.op<"scf.for">
  transform.test_print_remark_at_operand %remainder, "remainder loop" : !transform.op<"scf.for">
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

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !transform.op<"scf.for">
  %2 = transform.loop.pipeline %1 : (!transform.op<"scf.for">) -> !transform.any_op
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %2, "transformed" : !transform.any_op
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

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0 : (!transform.any_op) -> !transform.op<"scf.for">
  transform.loop.unroll %1 { factor = 4 } : !transform.op<"scf.for">
}

// -----

// CHECK-LABEL: @get_parent_for_op
func.func @get_parent_for_op(%arg0: index, %arg1: index, %arg2: index) {
  // expected-remark @below {{first loop}}
  affine.for %i = %arg0 to %arg1 {
    // expected-remark @below {{second loop}}
    affine.for %j = %arg0 to %arg1 {
      // expected-remark @below {{third loop}}
      affine.for %k = %arg0 to %arg1 {
        arith.addi %i, %j : index
      }
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // CHECK: = transform.loop.get_parent_for
  %1 = transform.loop.get_parent_for %0 { affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
  %2 = transform.loop.get_parent_for %0 { num_loops = 2, affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
  %3 = transform.loop.get_parent_for %0 { num_loops = 3, affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
  transform.test_print_remark_at_operand %1, "third loop" : !transform.op<"affine.for">
  transform.test_print_remark_at_operand %2, "second loop" : !transform.op<"affine.for">
  transform.test_print_remark_at_operand %3, "first loop" : !transform.op<"affine.for">
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  arith.addi %arg0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{could not find an 'affine.for' parent}}
  %1 = transform.loop.get_parent_for %0 { affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
}

// -----

func.func @loop_unroll_op() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  // CHECK: affine.for %[[I:.+]] =
  // expected-remark @below {{affine for loop}}
  affine.for %i = %c0 to %c42 {
    // CHECK-COUNT-4: arith.addi
    arith.addi %i, %i : index
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0 { affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
  transform.test_print_remark_at_operand %1, "affine for loop" : !transform.op<"affine.for">
  transform.loop.unroll %1 { factor = 4, affine = true } : !transform.op<"affine.for">
}

// -----

func.func @test_mixed_loops() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  scf.for %j = %c0 to %c42 step %c5 {
    // CHECK: affine.for %[[I:.+]] =
    // expected-remark @below {{affine for loop}}
    affine.for %i = %c0 to %c42 {
      // CHECK-COUNT-4: arith.addi
      arith.addi %i, %i : index
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0 { num_loops = 1, affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
  transform.test_print_remark_at_operand %1, "affine for loop" : !transform.op<"affine.for">
  transform.loop.unroll %1 { factor = 4 } : !transform.op<"affine.for">
}

// -----

// CHECK-LABEL: func @test_promote_if_one_iteration(
//   CHECK-NOT:   scf.for
//       CHECK:   %[[r:.*]] = "test.foo"
//       CHECK:   return %[[r]]
func.func @test_promote_if_one_iteration(%a: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %j = %c0 to %c1 step %c1 iter_args(%arg0 = %a) -> index {
    %1 = "test.foo"(%a) : (index) -> (index)
    scf.yield %1 : index
  }
  return %0 : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.loop.promote_if_one_iteration %0 : !transform.any_op
}


// -----

// CHECK-LABEL: func @test_structural_conversion_patterns(
// CHECK: scf.for {{.*}} -> (memref<f32>) {

func.func @test_structural_conversion_patterns(%a: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.for %j = %c0 to %c10 step %c1 iter_args(%arg0 = %a) -> tensor<f32> {
    %1 = "test.foo"(%arg0) : (tensor<f32>) -> (tensor<f32>)
    scf.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    transform.apply_conversion_patterns.scf.structural_conversions
  } with type_converter {
    transform.apply_conversion_patterns.transform.test_type_converter
  } {  partial_conversion  } : !transform.any_op
}
