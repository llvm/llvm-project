// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-std %s | FileCheck %s

// CHECK-LABEL: func @simple_std_for_loop(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  br ^bb1(%{{.*}} : index)
//  CHECK-NEXT:  ^bb1(%{{.*}}: index):    // 2 preds: ^bb0, ^bb2
//  CHECK-NEXT:    %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    cond_br %{{.*}}, ^bb2, ^bb3
//  CHECK-NEXT:  ^bb2:   // pred: ^bb1
//  CHECK-NEXT:    %{{.*}} = constant 1 : index
//  CHECK-NEXT:    %[[iv:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    br ^bb1(%[[iv]] : index)
//  CHECK-NEXT:  ^bb3:   // pred: ^bb1
//  CHECK-NEXT:    return
func @simple_std_for_loop(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
  }
  return
}

// CHECK-LABEL: func @simple_std_2_for_loops(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:    br ^bb1(%{{.*}} : index)
//  CHECK-NEXT:  ^bb1(%[[ub0:.*]]: index):    // 2 preds: ^bb0, ^bb5
//  CHECK-NEXT:    %[[cond0:.*]] = cmpi "slt", %[[ub0]], %{{.*}} : index
//  CHECK-NEXT:    cond_br %[[cond0]], ^bb2, ^bb6
//  CHECK-NEXT:  ^bb2:   // pred: ^bb1
//  CHECK-NEXT:    %{{.*}} = constant 1 : index
//  CHECK-NEXT:    br ^bb3(%{{.*}} : index)
//  CHECK-NEXT:  ^bb3(%[[ub1:.*]]: index):    // 2 preds: ^bb2, ^bb4
//  CHECK-NEXT:    %[[cond1:.*]] = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    cond_br %[[cond1]], ^bb4, ^bb5
//  CHECK-NEXT:  ^bb4:   // pred: ^bb3
//  CHECK-NEXT:    %{{.*}} = constant 1 : index
//  CHECK-NEXT:    %[[iv1:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    br ^bb3(%[[iv1]] : index)
//  CHECK-NEXT:  ^bb5:   // pred: ^bb3
//  CHECK-NEXT:    %[[iv0:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    br ^bb1(%[[iv0]] : index)
//  CHECK-NEXT:  ^bb6:   // pred: ^bb1
//  CHECK-NEXT:    return
func @simple_std_2_for_loops(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %c1_0 = constant 1 : index
    }
  }
  return
}

// CHECK-LABEL: func @simple_std_if(%{{.*}}: i1) {
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb2
//  CHECK-NEXT:   ^bb1:   // pred: ^bb0
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb2
//  CHECK-NEXT:   ^bb2:   // 2 preds: ^bb0, ^bb1
//  CHECK-NEXT:     return
func @simple_std_if(%arg0: i1) {
  scf.if %arg0 {
    %c1 = constant 1 : index
  }
  return
}

// CHECK-LABEL: func @simple_std_if_else(%{{.*}}: i1) {
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb2
//  CHECK-NEXT:   ^bb1:   // pred: ^bb0
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb3
//  CHECK-NEXT:   ^bb2:   // pred: ^bb0
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb3
//  CHECK-NEXT:   ^bb3:   // 2 preds: ^bb1, ^bb2
//  CHECK-NEXT:     return
func @simple_std_if_else(%arg0: i1) {
  scf.if %arg0 {
    %c1 = constant 1 : index
  } else {
    %c1_0 = constant 1 : index
  }
  return
}

// CHECK-LABEL: func @simple_std_2_ifs(%{{.*}}: i1) {
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb5
//  CHECK-NEXT: ^bb1:   // pred: ^bb0
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb3
//  CHECK-NEXT: ^bb2:   // pred: ^bb1
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb3:   // pred: ^bb1
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb4:   // 2 preds: ^bb2, ^bb3
//  CHECK-NEXT:   br ^bb5
//  CHECK-NEXT: ^bb5:   // 2 preds: ^bb0, ^bb4
//  CHECK-NEXT:   return
func @simple_std_2_ifs(%arg0: i1) {
  scf.if %arg0 {
    %c1 = constant 1 : index
    scf.if %arg0 {
      %c1_0 = constant 1 : index
    } else {
      %c1_1 = constant 1 : index
    }
  }
  return
}

// CHECK-LABEL: func @simple_std_for_loop_with_2_ifs(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: i1) {
//  CHECK-NEXT:   br ^bb1(%{{.*}} : index)
//  CHECK-NEXT:   ^bb1(%{{.*}}: index):    // 2 preds: ^bb0, ^bb7
//  CHECK-NEXT:     %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:     cond_br %{{.*}}, ^bb2, ^bb8
//  CHECK-NEXT:   ^bb2:   // pred: ^bb1
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     cond_br %{{.*}}, ^bb3, ^bb7
//  CHECK-NEXT:   ^bb3:   // pred: ^bb2
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     cond_br %{{.*}}, ^bb4, ^bb5
//  CHECK-NEXT:   ^bb4:   // pred: ^bb3
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb6
//  CHECK-NEXT:   ^bb5:   // pred: ^bb3
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb6
//  CHECK-NEXT:   ^bb6:   // 2 preds: ^bb4, ^bb5
//  CHECK-NEXT:     br ^bb7
//  CHECK-NEXT:   ^bb7:   // 2 preds: ^bb2, ^bb6
//  CHECK-NEXT:     %[[iv0:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:     br ^bb1(%[[iv0]] : index)
//  CHECK-NEXT:   ^bb8:   // pred: ^bb1
//  CHECK-NEXT:     return
//  CHECK-NEXT: }
func @simple_std_for_loop_with_2_ifs(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : i1) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
    scf.if %arg3 {
      %c1_0 = constant 1 : index
      scf.if %arg3 {
        %c1_1 = constant 1 : index
      } else {
        %c1_2 = constant 1 : index
      }
    }
  }
  return
}

// CHECK-LABEL: func @simple_if_yield
func @simple_if_yield(%arg0: i1) -> (i1, i1) {
// CHECK:   cond_br %{{.*}}, ^[[then:.*]], ^[[else:.*]]
  %0:2 = scf.if %arg0 -> (i1, i1) {
// CHECK: ^[[then]]:
// CHECK:   %[[v0:.*]] = constant false
// CHECK:   %[[v1:.*]] = constant true
// CHECK:   br ^[[dom:.*]](%[[v0]], %[[v1]] : i1, i1)
    %c0 = constant false
    %c1 = constant true
    scf.yield %c0, %c1 : i1, i1
  } else {
// CHECK: ^[[else]]:
// CHECK:   %[[v2:.*]] = constant false
// CHECK:   %[[v3:.*]] = constant true
// CHECK:   br ^[[dom]](%[[v3]], %[[v2]] : i1, i1)
    %c0 = constant false
    %c1 = constant true
    scf.yield %c1, %c0 : i1, i1
  }
// CHECK: ^[[dom]](%[[arg1:.*]]: i1, %[[arg2:.*]]: i1):
// CHECK:   br ^[[cont:.*]]
// CHECK: ^[[cont]]:
// CHECK:   return %[[arg1]], %[[arg2]]
  return %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: func @nested_if_yield
func @nested_if_yield(%arg0: i1) -> (index) {
// CHECK:   cond_br %{{.*}}, ^[[first_then:.*]], ^[[first_else:.*]]
  %0 = scf.if %arg0 -> i1 {
// CHECK: ^[[first_then]]:
    %1 = constant true
// CHECK:   br ^[[first_dom:.*]]({{.*}})
    scf.yield %1 : i1
  } else {
// CHECK: ^[[first_else]]:
    %2 = constant false
// CHECK:   br ^[[first_dom]]({{.*}})
    scf.yield %2 : i1
  }
// CHECK: ^[[first_dom]](%[[arg1:.*]]: i1):
// CHECK:   br ^[[first_cont:.*]]
// CHECK: ^[[first_cont]]:
// CHECK:   cond_br %[[arg1]], ^[[second_outer_then:.*]], ^[[second_outer_else:.*]]
  %1 = scf.if %0 -> index {
// CHECK: ^[[second_outer_then]]:
// CHECK:   cond_br %arg0, ^[[second_inner_then:.*]], ^[[second_inner_else:.*]]
    %3 = scf.if %arg0 -> index {
// CHECK: ^[[second_inner_then]]:
      %4 = constant 40 : index
// CHECK:   br ^[[second_inner_dom:.*]]({{.*}})
      scf.yield %4 : index
    } else {
// CHECK: ^[[second_inner_else]]:
      %5 = constant 41 : index
// CHECK:   br ^[[second_inner_dom]]({{.*}})
      scf.yield %5 : index
    }
// CHECK: ^[[second_inner_dom]](%[[arg2:.*]]: index):
// CHECK:   br ^[[second_inner_cont:.*]]
// CHECK: ^[[second_inner_cont]]:
// CHECK:   br ^[[second_outer_dom:.*]]({{.*}})
    scf.yield %3 : index
  } else {
// CHECK: ^[[second_outer_else]]:
    %6 = constant 42 : index
// CHECK:   br ^[[second_outer_dom]]({{.*}}
    scf.yield %6 : index
  }
// CHECK: ^[[second_outer_dom]](%[[arg3:.*]]: index):
// CHECK:   br ^[[second_outer_cont:.*]]
// CHECK: ^[[second_outer_cont]]:
// CHECK:   return %[[arg3]] : index
  return %1 : index
}

// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:                        [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: index, [[VAL_3:%.*]]: index, [[VAL_4:%.*]]: index) {
// CHECK:           [[VAL_5:%.*]] = constant 1 : index
// CHECK:           br ^bb1([[VAL_0]] : index)
// CHECK:         ^bb1([[VAL_6:%.*]]: index):
// CHECK:           [[VAL_7:%.*]] = cmpi "slt", [[VAL_6]], [[VAL_2]] : index
// CHECK:           cond_br [[VAL_7]], ^bb2, ^bb6
// CHECK:         ^bb2:
// CHECK:           br ^bb3([[VAL_1]] : index)
// CHECK:         ^bb3([[VAL_8:%.*]]: index):
// CHECK:           [[VAL_9:%.*]] = cmpi "slt", [[VAL_8]], [[VAL_3]] : index
// CHECK:           cond_br [[VAL_9]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           [[VAL_10:%.*]] = constant 1 : index
// CHECK:           [[VAL_11:%.*]] = addi [[VAL_8]], [[VAL_5]] : index
// CHECK:           br ^bb3([[VAL_11]] : index)
// CHECK:         ^bb5:
// CHECK:           [[VAL_12:%.*]] = addi [[VAL_6]], [[VAL_4]] : index
// CHECK:           br ^bb1([[VAL_12]] : index)
// CHECK:         ^bb6:
// CHECK:           return
// CHECK:         }

func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index, %arg4 : index) {
  %step = constant 1 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step) {
    %c1 = constant 1 : index
  }
  return
}

// CHECK-LABEL: @for_yield
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK:        %[[INIT0:.*]] = constant 0
// CHECK:        %[[INIT1:.*]] = constant 1
// CHECK:        br ^[[COND:.*]](%[[LB]], %[[INIT0]], %[[INIT1]] : index, f32, f32)
//
// CHECK:      ^[[COND]](%[[ITER:.*]]: index, %[[ITER_ARG0:.*]]: f32, %[[ITER_ARG1:.*]]: f32):
// CHECK:        %[[CMP:.*]] = cmpi "slt", %[[ITER]], %[[UB]] : index
// CHECK:        cond_br %[[CMP]], ^[[BODY:.*]], ^[[CONTINUE:.*]]
//
// CHECK:      ^[[BODY]]:
// CHECK:        %[[SUM:.*]] = addf %[[ITER_ARG0]], %[[ITER_ARG1]] : f32
// CHECK:        %[[STEPPED:.*]] = addi %[[ITER]], %[[STEP]] : index
// CHECK:        br ^[[COND]](%[[STEPPED]], %[[SUM]], %[[SUM]] : index, f32, f32)
//
// CHECK:      ^[[CONTINUE]]:
// CHECK:        return %[[ITER_ARG0]], %[[ITER_ARG1]] : f32, f32
func @for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> (f32, f32) {
  %s0 = constant 0.0 : f32
  %s1 = constant 1.0 : f32
  %result:2 = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = addf %si, %sj : f32
    scf.yield %sn, %sn : f32, f32
  }
  return %result#0, %result#1 : f32, f32
}

// CHECK-LABEL: @nested_for_yield
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK:         %[[INIT:.*]] = constant
// CHECK:         br ^[[COND_OUT:.*]](%[[LB]], %[[INIT]] : index, f32)
// CHECK:       ^[[COND_OUT]](%[[ITER_OUT:.*]]: index, %[[ARG_OUT:.*]]: f32):
// CHECK:         cond_br %{{.*}}, ^[[BODY_OUT:.*]], ^[[CONT_OUT:.*]]
// CHECK:       ^[[BODY_OUT]]:
// CHECK:         br ^[[COND_IN:.*]](%[[LB]], %[[ARG_OUT]] : index, f32)
// CHECK:       ^[[COND_IN]](%[[ITER_IN:.*]]: index, %[[ARG_IN:.*]]: f32):
// CHECK:         cond_br %{{.*}}, ^[[BODY_IN:.*]], ^[[CONT_IN:.*]]
// CHECK:       ^[[BODY_IN]]
// CHECK:         %[[RES:.*]] = addf
// CHECK:         br ^[[COND_IN]](%{{.*}}, %[[RES]] : index, f32)
// CHECK:       ^[[CONT_IN]]:
// CHECK:         br ^[[COND_OUT]](%{{.*}}, %[[ARG_IN]] : index, f32)
// CHECK:       ^[[CONT_OUT]]:
// CHECK:         return %[[ARG_OUT]] : f32
func @nested_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> f32 {
  %s0 = constant 1.0 : f32
  %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%iter = %s0) -> (f32) {
    %result = scf.for %i1 = %arg0 to %arg1 step %arg2 iter_args(%si = %iter) -> (f32) {
      %sn = addf %si, %si : f32
      scf.yield %sn : f32
    }
    scf.yield %result : f32
  }
  return %r : f32
}

func private @generate() -> i64

// CHECK-LABEL: @simple_parallel_reduce_loop
// CHECK-SAME: %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[INIT:.*]]: f32
func @simple_parallel_reduce_loop(%arg0: index, %arg1: index,
                                  %arg2: index, %arg3: f32) -> f32 {
  // A parallel loop with reduction is converted through sequential loops with
  // reductions into a CFG of blocks where the partially reduced value is
  // passed across as a block argument.

  // Branch to the condition block passing in the initial reduction value.
  // CHECK:   br ^[[COND:.*]](%[[LB]], %[[INIT]]

  // Condition branch takes as arguments the current value of the iteration
  // variable and the current partially reduced value.
  // CHECK: ^[[COND]](%[[ITER:.*]]: index, %[[ITER_ARG:.*]]: f32
  // CHECK:   %[[COMP:.*]] = cmpi "slt", %[[ITER]], %[[UB]]
  // CHECK:   cond_br %[[COMP]], ^[[BODY:.*]], ^[[CONTINUE:.*]]

  // Bodies of scf.reduce operations are folded into the main loop body. The
  // result of this partial reduction is passed as argument to the condition
  // block.
  // CHECK: ^[[BODY]]:
  // CHECK:   %[[CST:.*]] = constant 4.2
  // CHECK:   %[[PROD:.*]] = mulf %[[ITER_ARG]], %[[CST]]
  // CHECK:   %[[INCR:.*]] = addi %[[ITER]], %[[STEP]]
  // CHECK:   br ^[[COND]](%[[INCR]], %[[PROD]]

  // The continuation block has access to the (last value of) reduction.
  // CHECK: ^[[CONTINUE]]:
  // CHECK:   return %[[ITER_ARG]]
  %0 = scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) init(%arg3) -> f32 {
    %cst = constant 42.0 : f32
    scf.reduce(%cst) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  return %0 : f32
}

// CHECK-LABEL: parallel_reduce_loop
// CHECK-SAME: %[[INIT1:[0-9A-Za-z_]*]]: f32)
func @parallel_reduce_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                           %arg3 : index, %arg4 : index, %arg5 : f32) -> (f32, i64) {
  // Multiple reduction blocks should be folded in the same body, and the
  // reduction value must be forwarded through block structures.
  // CHECK:   %[[INIT2:.*]] = constant 42
  // CHECK:   br ^[[COND_OUT:.*]](%{{.*}}, %[[INIT1]], %[[INIT2]]
  // CHECK: ^[[COND_OUT]](%{{.*}}: index, %[[ITER_ARG1_OUT:.*]]: f32, %[[ITER_ARG2_OUT:.*]]: i64
  // CHECK:   cond_br %{{.*}}, ^[[BODY_OUT:.*]], ^[[CONT_OUT:.*]]
  // CHECK: ^[[BODY_OUT]]:
  // CHECK:   br ^[[COND_IN:.*]](%{{.*}}, %[[ITER_ARG1_OUT]], %[[ITER_ARG2_OUT]]
  // CHECK: ^[[COND_IN]](%{{.*}}: index, %[[ITER_ARG1_IN:.*]]: f32, %[[ITER_ARG2_IN:.*]]: i64
  // CHECK:   cond_br %{{.*}}, ^[[BODY_IN:.*]], ^[[CONT_IN:.*]]
  // CHECK: ^[[BODY_IN]]:
  // CHECK:   %[[REDUCE1:.*]] = addf %[[ITER_ARG1_IN]], %{{.*}}
  // CHECK:   %[[REDUCE2:.*]] = or %[[ITER_ARG2_IN]], %{{.*}}
  // CHECK:   br ^[[COND_IN]](%{{.*}}, %[[REDUCE1]], %[[REDUCE2]]
  // CHECK: ^[[CONT_IN]]:
  // CHECK:   br ^[[COND_OUT]](%{{.*}}, %[[ITER_ARG1_IN]], %[[ITER_ARG2_IN]]
  // CHECK: ^[[CONT_OUT]]:
  // CHECK:   return %[[ITER_ARG1_OUT]], %[[ITER_ARG2_OUT]]
  %step = constant 1 : index
  %init = constant 42 : i64
  %0:2 = scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                       step (%arg4, %step) init(%arg5, %init) -> (f32, i64) {
    %cf = constant 42.0 : f32
    scf.reduce(%cf) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }

    %2 = call @generate() : () -> i64
    scf.reduce(%2) : i64 {
    ^bb0(%lhs: i64, %rhs: i64):
      %3 = or %lhs, %rhs : i64
      scf.reduce.return %3 : i64
    }
  }
  return %0#0, %0#1 : f32, i64
}

// Check that the conversion is not overly conservative wrt unknown ops, i.e.
// that the presence of unknown ops does not prevent the conversion from being
// applied.
// CHECK-LABEL: @unknown_op_inside_loop
func @unknown_op_inside_loop(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK-NOT: scf.for
  scf.for %i = %arg0 to %arg1 step %arg2 {
    // CHECK: unknown.op
    "unknown.op"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: @minimal_while
func @minimal_while() {
  // CHECK:   %[[COND:.*]] = "test.make_condition"() : () -> i1
  // CHECK:   br ^[[BEFORE:.*]]
  %0 = "test.make_condition"() : () -> i1
  scf.while : () -> () {
  // CHECK: ^[[BEFORE]]:
  // CHECK:   cond_br %[[COND]], ^[[AFTER:.*]], ^[[CONT:.*]]
    scf.condition(%0)
  } do {
  // CHECK: ^[[AFTER]]:
  // CHECK:   "test.some_payload"() : () -> ()
    "test.some_payload"() : () -> ()
  // CHECK:   br ^[[BEFORE]]
    scf.yield
  }
  // CHECK: ^[[CONT]]:
  // CHECK:   return
  return
}

// CHECK-LABEL: @do_while
func @do_while(%arg0: f32) {
  // CHECK:   br ^[[BEFORE:.*]]({{.*}}: f32)
  scf.while (%arg1 = %arg0) : (f32) -> (f32) {
  // CHECK: ^[[BEFORE]](%[[VAL:.*]]: f32):
    // CHECK:   %[[COND:.*]] = "test.make_condition"() : () -> i1
    %0 = "test.make_condition"() : () -> i1
    // CHECK:   cond_br %[[COND]], ^[[BEFORE]](%[[VAL]] : f32), ^[[CONT:.*]]
    scf.condition(%0) %arg1 : f32
  } do {
  ^bb0(%arg2: f32):
    // CHECK-NOT: br ^[[BEFORE]]
    scf.yield %arg2 : f32
  }
  // CHECK: ^[[CONT]]:
  // CHECK:   return
  return
}

// CHECK-LABEL: @while_values
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: f32)
func @while_values(%arg0: i32, %arg1: f32) {
  // CHECK:     %[[COND:.*]] = "test.make_condition"() : () -> i1
  %0 = "test.make_condition"() : () -> i1
  %c0_i32 = constant 0 : i32
  %cst = constant 0.000000e+00 : f32
  // CHECK:     br ^[[BEFORE:.*]](%[[ARG0]], %[[ARG1]] : i32, f32)
  %1:2 = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (i32, f32) -> (i64, f64) {
  // CHECK:   ^bb1(%[[ARG2:.*]]: i32, %[[ARG3:.]]: f32):
    // CHECK:   %[[VAL1:.*]] = zexti %[[ARG0]] : i32 to i64
    %2 = zexti %arg0 : i32 to i64
    // CHECK:   %[[VAL2:.*]] = fpext %[[ARG3]] : f32 to f64
    %3 = fpext %arg3 : f32 to f64
    // CHECK:   cond_br %[[COND]],
    // CHECK:           ^[[AFTER:.*]](%[[VAL1]], %[[VAL2]] : i64, f64),
    // CHECK:           ^[[CONT:.*]]
    scf.condition(%0) %2, %3 : i64, f64
  } do {
  // CHECK:   ^[[AFTER]](%[[ARG4:.*]]: i64, %[[ARG5:.*]]: f64):
  ^bb0(%arg2: i64, %arg3: f64):  // no predecessors
    // CHECK:   br ^[[BEFORE]](%{{.*}}, %{{.*}} : i32, f32)
    scf.yield %c0_i32, %cst : i32, f32
  }
  // CHECK:   ^bb3:
  // CHECK:     return
  return
}

// CHECK-LABEL: @nested_while_ops
func @nested_while_ops(%arg0: f32) -> i64 {
  // CHECK:       br ^[[OUTER_BEFORE:.*]](%{{.*}} : f32)
  %0 = scf.while(%outer = %arg0) : (f32) -> i64 {
    // CHECK:   ^[[OUTER_BEFORE]](%{{.*}}: f32):
    // CHECK:     %[[OUTER_COND:.*]] = "test.outer_before_pre"() : () -> i1
    %cond = "test.outer_before_pre"() : () -> i1
    // CHECK:     br ^[[INNER_BEFORE_BEFORE:.*]](%{{.*}} : f32)
    %1 = scf.while(%inner = %outer) : (f32) -> i64 {
      // CHECK: ^[[INNER_BEFORE_BEFORE]](%{{.*}}: f32):
      // CHECK:   %[[INNER1:.*]]:2 = "test.inner_before"(%{{.*}}) : (f32) -> (i1, i64)
      %2:2 = "test.inner_before"(%inner) : (f32) -> (i1, i64)
      // CHECK:   cond_br %[[INNER1]]#0,
      // CHECK:           ^[[INNER_BEFORE_AFTER:.*]](%[[INNER1]]#1 : i64),
      // CHECK:           ^[[OUTER_BEFORE_LAST:.*]]
      scf.condition(%2#0) %2#1 : i64
    } do {
      // CHECK: ^[[INNER_BEFORE_AFTER]](%{{.*}}: i64):
    ^bb0(%arg1: i64):
      // CHECK:   %[[INNER2:.*]] = "test.inner_after"(%{{.*}}) : (i64) -> f32
      %3 = "test.inner_after"(%arg1) : (i64) -> f32
      // CHECK:   br ^[[INNER_BEFORE_BEFORE]](%[[INNER2]] : f32)
      scf.yield %3 : f32
    }
    // CHECK:   ^[[OUTER_BEFORE_LAST]]:
    // CHECK:     "test.outer_before_post"() : () -> ()
    "test.outer_before_post"() : () -> ()
    // CHECK:     cond_br %[[OUTER_COND]],
    // CHECK:             ^[[OUTER_AFTER:.*]](%[[INNER1]]#1 : i64),
    // CHECK:             ^[[CONTINUATION:.*]]
    scf.condition(%cond) %1 : i64
  } do {
    // CHECK:   ^[[OUTER_AFTER]](%{{.*}}: i64):
  ^bb2(%arg2: i64):
    // CHECK:     "test.outer_after_pre"(%{{.*}}) : (i64) -> ()
    "test.outer_after_pre"(%arg2) : (i64) -> ()
    // CHECK:     br ^[[INNER_AFTER_BEFORE:.*]](%{{.*}} : i64)
    %4 = scf.while(%inner = %arg2) : (i64) -> f32 {
      // CHECK: ^[[INNER_AFTER_BEFORE]](%{{.*}}: i64):
      // CHECK:   %[[INNER3:.*]]:2 = "test.inner2_before"(%{{.*}}) : (i64) -> (i1, f32)
      %5:2 = "test.inner2_before"(%inner) : (i64) -> (i1, f32)
      // CHECK:   cond_br %[[INNER3]]#0,
      // CHECK:           ^[[INNER_AFTER_AFTER:.*]](%[[INNER3]]#1 : f32),
      // CHECK:           ^[[OUTER_AFTER_LAST:.*]]
      scf.condition(%5#0) %5#1 : f32
    } do {
      // CHECK: ^[[INNER_AFTER_AFTER]](%{{.*}}: f32):
    ^bb3(%arg3: f32):
      // CHECK:   %{{.*}} = "test.inner2_after"(%{{.*}}) : (f32) -> i64
      %6 = "test.inner2_after"(%arg3) : (f32) -> i64
      // CHECK:   br ^[[INNER_AFTER_BEFORE]](%{{.*}} : i64)
      scf.yield %6 : i64
    }
    // CHECK:   ^[[OUTER_AFTER_LAST]]:
    // CHECK:     "test.outer_after_post"() : () -> ()
    "test.outer_after_post"() : () -> ()
    // CHECK:     br ^[[OUTER_BEFORE]](%[[INNER3]]#1 : f32)
    scf.yield %4 : f32
  }
  // CHECK:     ^[[CONTINUATION]]:
  // CHECK:       return %{{.*}} : i64
  return %0 : i64
}

