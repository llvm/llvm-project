// RUN: mlir-opt -convert-loop-to-std %s | FileCheck %s

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
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
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
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
    loop.for %i1 = %arg0 to %arg1 step %arg2 {
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
  loop.if %arg0 {
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
  loop.if %arg0 {
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
  loop.if %arg0 {
    %c1 = constant 1 : index
    loop.if %arg0 {
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
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
    loop.if %arg3 {
      %c1_0 = constant 1 : index
      loop.if %arg3 {
        %c1_1 = constant 1 : index
      } else {
        %c1_2 = constant 1 : index
      }
    }
  }
  return
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
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
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
  %result:2 = loop.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = addf %si, %sj : f32
    loop.yield %sn, %sn : f32, f32
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
  %r = loop.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%iter = %s0) -> (f32) {
    %result = loop.for %i1 = %arg0 to %arg1 step %arg2 iter_args(%si = %iter) -> (f32) {
      %sn = addf %si, %si : f32
      loop.yield %sn : f32
    }
    loop.yield %result : f32
  }
  return %r : f32
}

func @generate() -> i64

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

  // Bodies of loop.reduce operations are folded into the main loop body. The
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
  %0 = loop.parallel (%i) = (%arg0) to (%arg1) step (%arg2) init(%arg3) {
    %cst = constant 42.0 : f32
    loop.reduce(%cst) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = mulf %lhs, %rhs : f32
      loop.reduce.return %1 : f32
    } : f32
  } : f32
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
  %0:2 = loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                       step (%arg4, %step) init(%arg5, %init) {
    %cf = constant 42.0 : f32
    loop.reduce(%cf) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = addf %lhs, %rhs : f32
      loop.reduce.return %1 : f32
    } : f32

    %2 = call @generate() : () -> i64
    loop.reduce(%2) {
    ^bb0(%lhs: i64, %rhs: i64):
      %3 = or %lhs, %rhs : i64
      loop.reduce.return %3 : i64
    } : i64
  } : f32, i64
  return %0#0, %0#1 : f32, i64
}
