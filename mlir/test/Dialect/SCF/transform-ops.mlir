// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// Outlined functions:
//
// CHECK: func @foo(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
// CHECK:   scf.for
// CHECK:     arith.addi
//
// CHECK: func @foo[[$SUFFIX:.+]](%{{.+}}, %{{.+}}, %{{.+}})
// CHECK:   scf.for
// CHECK:     arith.addi
//
// CHECK-LABEL: @loop_outline_op
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
  // CHECK:   func.call @foo[[$SUFFIX]]
  scf.for %j = %arg0 to %arg1 step %arg2 {
    arith.addi %j, %j : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // CHECK: = transform.loop.outline %{{.*}}
    transform.loop.outline %1 {func_name = "foo"} : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %main_loop, %remainder = transform.loop.peel %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
    // Make sure 
    transform.debug.emit_remark_at %main_loop, "main loop" : !transform.op<"scf.for">
    transform.debug.emit_remark_at %remainder, "remainder loop" : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_peel_first_iter_op
func.func @loop_peel_first_iter_op() {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[C41:.+]] = arith.constant 41
  // CHECK: %[[C5:.+]] = arith.constant 5
  // CHECK: %[[C5_0:.+]] = arith.constant 5
  // CHECK: scf.for %{{.+}} = %[[C0]] to %[[C5_0]] step %[[C5]]
  // CHECK:   arith.addi
  // CHECK: scf.for %{{.+}} = %[[C5_0]] to %[[C41]] step %[[C5]]
  // CHECK:   arith.addi
  %0 = arith.constant 0 : index
  %1 = arith.constant 41 : index
  %2 = arith.constant 5 : index
  scf.for %i = %0 to %1 step %2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %main_loop, %remainder = transform.loop.peel %1 {peel_front = true} : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %2 = transform.loop.pipeline %1 : (!transform.op<"scf.for">) -> !transform.any_op
    // Verify that the returned handle is usable.
    transform.debug.emit_remark_at %2, "transformed" : !transform.any_op
    transform.yield
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    transform.debug.emit_remark_at %1, "affine for loop" : !transform.op<"affine.for">
    transform.loop.unroll %1 { factor = 4, affine = true } : !transform.op<"affine.for">
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    transform.debug.emit_remark_at %1, "affine for loop" : !transform.op<"affine.for">
    transform.loop.unroll %1 { factor = 4 } : !transform.op<"affine.for">
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.loop.promote_if_one_iteration %0 : !transform.any_op
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.scf.structural_conversions
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {  partial_conversion  } : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @coalesce_i32_loops(

// This test checks for loop coalescing success for non-index loop boundaries and step type
func.func @coalesce_i32_loops() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 128 : i32
  // CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
  // CHECK:           %[[VAL_3:.*]] = arith.constant 64 : i32
  %0 = arith.constant 0 : i32
  %1 = arith.constant 128 : i32
  %2 = arith.constant 2 : i32
  %3 = arith.constant 64 : i32
  // CHECK:           %[[VAL_4:.*]] = arith.constant 64 : i32
  // CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_7:.*]] = arith.constant 32 : i32
  // CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i32
  // CHECK:           %[[UB:.*]] = arith.muli %[[VAL_4]], %[[VAL_7]] : i32
  // CHECK:           scf.for %[[VAL_11:.*]] = %[[ZERO]] to %[[UB]] step %[[ONE]]  : i32 {
  scf.for %i = %0 to %1 step %2 : i32 {
    scf.for %j = %0 to %3 step %2 : i32 {
      arith.addi %i, %j : i32
    }
  } {coalesce}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1: (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}


// -----
 
// CHECK-LABEL: func.func @loop_pipeline
func.func @loop_pipeline(%arg0: memref<4x16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
   %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c3 = arith.constant 3 : index
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_read
  // CHECK: arith.addf
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = scf.for %arg2 = %c0 to %c3 step %c1 iter_args(%arg3 = %arg1) -> (vector<16xf32>) {
    %1 = vector.transfer_read %arg0[%arg2, %c0], %cst {in_bounds = [true]} : memref<4x16xf32>, vector<16xf32>
    %2 = arith.addf %1, %arg3 : vector<16xf32>
    scf.yield %2 : vector<16xf32>
  }
  return %0 : vector<16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.for">
    %1 = transform.loop.pipeline %0 {iteration_interval = 1 : i64, read_latency = 5 : i64,  scheduling_type = "full-loops"} : (!transform.op<"scf.for">) -> !transform.any_op
     transform.yield
 }
}
 
 
// -----
 
// CHECK-LABEL: func.func @loop_pipeline_lb_gt_0
func.func @loop_pipeline_lb_gt_0(%arg0: memref<4x16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c3 = arith.constant 3 : index
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_read
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = scf.for %arg2 = %c1 to %c3 step %c1 iter_args(%arg3 = %arg1) -> (vector<16xf32>) {
    %1 = vector.transfer_read %arg0[%arg2, %c1], %cst {in_bounds = [true]} : memref<4x16xf32>, vector<16xf32>
    %2 = arith.addf %1, %arg3 : vector<16xf32>
    scf.yield %2 : vector<16xf32>
  }
  return %0 : vector<16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.for">
    %1 = transform.loop.pipeline %0 {iteration_interval = 1 : i64, read_latency = 5 : i64,  scheduling_type = "full-loops"} : (!transform.op<"scf.for">) -> !transform.any_op
     transform.yield
 }
}
