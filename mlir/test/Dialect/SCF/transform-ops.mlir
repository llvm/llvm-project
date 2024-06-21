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

// CHECK-LABEL: @loop_unroll_and_jam_op
func.func @loop_unroll_and_jam_op() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_1:.*]] = arith.constant 40 : index
  // CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
  // CHECK:           %[[FACTOR:.*]] = arith.constant 4 : index
  // CHECK:           %[[STEP:.*]] = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %c2 = arith.constant 2 : index
  // CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[STEP]] {
  scf.for %i = %c0 to %c40 step %c2 {
  // CHECK:             %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_5]] : index
  // CHECK:             %[[VAL_7:.*]] = arith.constant 2 : index
  // CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_5]], %[[VAL_7]] : index
  // CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_8]] : index
  // CHECK:             %[[VAL_10:.*]] = arith.constant 4 : index
  // CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_5]], %[[VAL_10]] : index
  // CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_11]] : index
  // CHECK:             %[[VAL_13:.*]] = arith.constant 6 : index
  // CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_5]], %[[VAL_13]] : index
  // CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_14]] : index
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
// CHECK:       %[[VAL_0:.*]]: memref<96x128xi8, 3>, %[[VAL_1:.*]]: memref<128xi8, 3>) {
func.func private @loop_unroll_and_jam_op(%arg0: memref<96x128xi8, 3>, %arg1: memref<128xi8, 3>) {
  // CHECK:           %[[UB_INNER:.*]] = arith.constant 96
  // CHECK:           %[[STEP_INNER:.*]] = arith.constant 1
  // CHECK:           %[[UB_OUTER:.*]] = arith.constant 128
  // CHECK:           %[[LB:.*]] = arith.constant 0
  // CHECK:           %[[UNUSED:.*]] = arith.constant 4
  // CHECK:           %[[UNROLL_FACTOR:.*]] = arith.constant 4
  %c96 = arith.constant 96 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  // CHECK:           scf.for %[[OUTER_I:.*]] = %[[LB]] to %[[UB_OUTER]] step %[[UNROLL_FACTOR]] {
	scf.for %arg2 = %c0 to %c128 step %c1 {
    // CHECK:             %[[LOAD0:.*]] = memref.load %[[VAL_1]]{{\[}}%[[OUTER_I]]]
    // CHECK:             %[[ONE_0:.*]] = arith.constant 1
    // CHECK:             %[[INC_LOAD1:.*]] = arith.addi %[[OUTER_I]], %[[ONE_0]]
    // CHECK:             %[[LOAD1:.*]] = memref.load %[[VAL_1]]{{\[}}%[[INC_LOAD1]]]
    // CHECK:             %[[TWO_0:.*]] = arith.constant 2
    // CHECK:             %[[INC_LOAD2:.*]] = arith.addi %[[OUTER_I]], %[[TWO_0]]
    // CHECK:             %[[LOAD2:.*]] = memref.load %[[VAL_1]]{{\[}}%[[INC_LOAD2]]]
    // CHECK:             %[[THREE_0:.*]] = arith.constant 3
    // CHECK:             %[[INC_LOAD3:.*]] = arith.addi %[[OUTER_I]], %[[THREE_0]]
    // CHECK:             %[[LOAD3:.*]] = memref.load %[[VAL_1]]{{\[}}%[[INC_LOAD3]]]
	  %3 = memref.load %arg1[%arg2] : memref<128xi8, 3>
    // CHECK:             %[[VAL_19:.*]]:4 = scf.for %[[VAL_20:.*]] = %[[LB]] to %[[UB_INNER]] step %[[STEP_INNER]] iter_args(%[[VAL_21:.*]] = %[[LOAD0]], %[[VAL_22:.*]] = %[[LOAD1]], %[[VAL_23:.*]] = %[[LOAD2]], %[[VAL_24:.*]] = %[[LOAD3]])
	  %sum = scf.for %arg3 = %c0 to %c96 step %c1 iter_args(%does_not_alias_aggregated = %3) -> (i8) {
    // CHECK:               %[[LOAD0_INNER:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[OUTER_I]]]
    // CHECK:               %[[SUM_0:.*]] = arith.addi %[[LOAD0_INNER]], %[[LOAD0]]
    // CHECK:               %[[ONE_1:.*]] = arith.constant 1
    // CHECK:               %[[INC1_INNER:.*]] = arith.addi %[[OUTER_I]], %[[ONE_1]]
    // CHECK:               %[[LOAD1_INNER:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[INC1_INNER]]]
    // CHECK:               %[[SUM_1:.*]] = arith.addi %[[LOAD1_INNER]], %[[LOAD1]]
    // CHECK:               %[[TWO_1:.*]] = arith.constant 2
    // CHECK:               %[[INC2_INNER:.*]] = arith.addi %[[OUTER_I]], %[[TWO_1]]
    // CHECK:               %[[LOAD2_INNER:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[INC2_INNER]]]
    // CHECK:               %[[SUM_2:.*]] = arith.addi %[[LOAD2_INNER]], %[[LOAD2]]
    // CHECK:               %[[THREE_1:.*]] = arith.constant 3
    // CHECK:               %[[INC3_INNER:.*]] = arith.addi %[[OUTER_I]], %[[THREE_1]]
    // CHECK:               %[[LOAD3_INNER:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[INC3_INNER]]]
    // CHECK:               %[[SUM_3:.*]] = arith.addi %[[LOAD3_INNER]], %[[LOAD3]]
		%2 = memref.load %arg0[%arg3, %arg2] : memref<96x128xi8, 3>
		%4 = arith.addi %2, %3 : i8
    // CHECK:               scf.yield %[[SUM_0]], %[[SUM_1]], %[[SUM_2]], %[[SUM_3]]
		scf.yield %4 : i8
	  }
	  memref.store %sum, %arg1[%arg2] : memref<128xi8, 3>
    // CHECK:             memref.store %[[VAL_39:.*]]#0, %[[VAL_1]]{{\[}}%[[OUTER_I]]]
    // CHECK:             %[[ONE_2:.*]] = arith.constant 1
    // CHECK:             %[[INC_STORE1:.*]] = arith.addi %[[OUTER_I]], %[[ONE_2]]
    // CHECK:             memref.store %[[VAL_39]]#1, %[[VAL_1]]{{\[}}%[[INC_STORE1]]]
    // CHECK:             %[[TWO_2:.*]] = arith.constant 2
    // CHECK:             %[[INC_STORE2:.*]] = arith.addi %[[OUTER_I]], %[[TWO_2]]
    // CHECK:             memref.store %[[VAL_39]]#2, %[[VAL_1]]{{\[}}%[[INC_STORE2]]]
    // CHECK:             %[[THREE_2:.*]] = arith.constant 3
    // CHECK:             %[[INC_STORE3:.*]] = arith.addi %[[OUTER_I]], %[[THREE_2]]
    // CHECK:             memref.store %[[VAL_39]]#3, %[[VAL_1]]{{\[}}%[[INC_STORE3]]]
	}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.store"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
func.func @loop_unroll_and_jam_op() {
  // CHECK:           %[[ZERO:.*]] = arith.constant 0
  // CHECK:           %[[UNUSED_FACTOR:.*]] = arith.constant 4
  // CHECK:           %[[UNUSED_STEP:.*]] = arith.constant 2
  // CHECK:           %[[UNUSED_STEP2:.*]] = arith.constant 2
  // CHECK:           %[[UNUSED_UB:.*]] = arith.constant 4
  // CHECK:           %[[ITER_0_RES:.*]] = arith.addi %[[ZERO]], %[[ZERO]]
  // CHECK:           %[[TWO:.*]] = arith.constant 2
  // CHECK:           %[[STEP_1_I:.*]] = arith.addi %[[ZERO]], %[[TWO]]
  // CHECK:           %[[ITER_1_RES:.*]] = arith.addi %[[STEP_1_I]], %[[STEP_1_I]]
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c4 step %c2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
func.func @loop_unroll_and_jam_op() {
  // CHECK:           %[[LB:.*]] = arith.constant 0
  // CHECK:           %[[UB:.*]] = arith.constant 4
  // CHECK:           %[[STEP:.*]] = arith.constant 2
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  // CHECK:           scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
  scf.for %i = %c0 to %c4 step %c2 {
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 2 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
// CHECK:  %[[VAL_0:.*]]: memref<21x30xf32, 1>, %[[INIT0:.*]]: f32, %[[INIT1:.*]]: f32) {
func.func @loop_unroll_and_jam_op(%arg0: memref<21x30xf32, 1>, %init : f32, %init1 : f32) {
  // CHECK:           %[[LAST_OUT_ITER:.*]] = arith.constant 20
  // CHECK:           %[[VAL_4:.*]]:2 = affine.for %[[OUTER_I:.*]] = 0 to 20 step 2 iter_args(%[[VAL_6:.*]] = %[[INIT0]], %[[VAL_7:.*]] = %[[INIT0]])
  %0 = affine.for %arg3 = 0 to 21 iter_args(%arg4 = %init) -> (f32) {
    // CHECK:             %[[VAL_8:.*]]:2 = affine.for %[[INNER_I:.*]] = 0 to 30 iter_args(%[[SUM0:.*]] = %[[INIT1]], %[[SUM1:.*]] = %[[INIT1]])
    %1 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %init1) -> (f32) {
      // CHECK:               %[[LOAD0:.*]] = affine.load %[[VAL_0]]{{\[}}%[[OUTER_I]], %[[INNER_I]]]
      // CHECK:               %[[ITER_SUM0:.*]] = arith.addf %[[SUM0]], %[[LOAD0]]
      // CHECK:               %[[APPLY_OUTER_I:.*]] = affine.apply #map(%[[OUTER_I]])
      // CHECK:               %[[LOAD1:.*]] = affine.load %[[VAL_0]]{{\[}}%[[APPLY_OUTER_I]], %[[INNER_I]]]
      // CHECK:               %[[ITER_SUM1:.*]] = arith.addf %[[SUM1]], %[[LOAD1]]
      // CHECK:               affine.yield %[[ITER_SUM0]], %[[ITER_SUM1]]
      %3 = affine.load %arg0[%arg3, %arg5] : memref<21x30xf32, 1>
      %4 = arith.addf %arg6, %3 : f32
      affine.yield %4 : f32
    }
    // CHECK:             %[[MUL0:.*]] = arith.mulf %[[VAL_6]], %[[VAL_18:.*]]#0
    // CHECK:             %[[VAL_19:.*]] = affine.apply #map(%[[OUTER_I]])
    // CHECK:             %[[MUL1:.*]] = arith.mulf %[[VAL_7]], %[[VAL_18]]#1
    // CHECK:             affine.yield %[[MUL0]], %[[MUL1]]
    // CHECK:           }
    // CHECK:           %[[VAL_21:.*]] = arith.mulf %[[VAL_22:.*]]#0, %[[VAL_22]]#1
    // CHECK:           %[[VAL_23:.*]] = affine.for %[[SUFFIX_I:.*]] = 0 to 30 iter_args(%[[ITER_I:.*]] = %[[INIT1]])
    // CHECK:             %[[LOAD_SUFFIX:.*]] = affine.load %[[VAL_0]]{{\[}}%[[LAST_OUT_ITER]], %[[SUFFIX_I]]]
    // CHECK:             %[[RES:.*]] = arith.addf %[[ITER_I]], %[[LOAD_SUFFIX]]
    // CHECK:             affine.yield %[[RES]]
    %2 = arith.mulf %arg4, %1 : f32
    affine.yield %2 : f32
  }
  // CHECK:           %[[VAL_28:.*]] = arith.mulf %[[VAL_21]], %[[VAL_29:.*]]
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.get_parent_op %1 {op_name = "affine.for"} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.loop.unroll_and_jam %2 { factor = 2 } : !transform.op<"affine.for">
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
