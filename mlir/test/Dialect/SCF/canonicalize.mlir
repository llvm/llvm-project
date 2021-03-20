// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s


// -----

func @single_iteration(%A: memref<?x?x?xi32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c10 = constant 10 : index
  scf.parallel (%i0, %i1, %i2) = (%c0, %c3, %c7) to (%c1, %c6, %c10) step (%c1, %c2, %c3) {
    %c42 = constant 42 : i32
    memref.store %c42, %A[%i0, %i1, %i2] : memref<?x?x?xi32>
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @single_iteration(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?x?xi32>) {
// CHECK-DAG:           [[C42:%.*]] = constant 42 : i32
// CHECK-DAG:           [[C7:%.*]] = constant 7 : index
// CHECK-DAG:           [[C6:%.*]] = constant 6 : index
// CHECK-DAG:           [[C3:%.*]] = constant 3 : index
// CHECK-DAG:           [[C2:%.*]] = constant 2 : index
// CHECK-DAG:           [[C0:%.*]] = constant 0 : index
// CHECK:           scf.parallel ([[V0:%.*]]) = ([[C3]]) to ([[C6]]) step ([[C2]]) {
// CHECK:             memref.store [[C42]], [[ARG0]]{{\[}}[[C0]], [[V0]], [[C7]]] : memref<?x?x?xi32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return

// -----

func @one_unused(%cond: i1) -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0, %1 = scf.if %cond -> (index, index) {
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c0, %c1 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @one_unused
// CHECK:           [[C0:%.*]] = constant 1 : index
// CHECK:           [[V0:%.*]] = scf.if %{{.*}} -> (index) {
// CHECK:             scf.yield [[C0]] : index
// CHECK:           } else
// CHECK:             scf.yield [[C0]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func @nested_unused(%cond1: i1, %cond2: i1) -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0, %1 = scf.if %cond1 -> (index, index) {
    %2, %3 = scf.if %cond2 -> (index, index) {
      scf.yield %c0, %c1 : index, index
    } else {
      scf.yield %c0, %c1 : index, index
    }
    scf.yield %2, %3 : index, index
  } else {
    scf.yield %c0, %c1 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @nested_unused
// CHECK:           [[C0:%.*]] = constant 1 : index
// CHECK:           [[V0:%.*]] = scf.if {{.*}} -> (index) {
// CHECK:             [[V1:%.*]] = scf.if {{.*}} -> (index) {
// CHECK:               scf.yield [[C0]] : index
// CHECK:             } else
// CHECK:               scf.yield [[C0]] : index
// CHECK:             }
// CHECK:             scf.yield [[V1]] : index
// CHECK:           } else
// CHECK:             scf.yield [[C0]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func private @side_effect()
func @all_unused(%cond: i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0, %1 = scf.if %cond -> (index, index) {
    call @side_effect() : () -> ()
    scf.yield %c0, %c1 : index, index
  } else {
    call @side_effect() : () -> ()
    scf.yield %c0, %c1 : index, index
  }
  return
}

// CHECK-LABEL:   func @all_unused
// CHECK:           scf.if %{{.*}} {
// CHECK:             call @side_effect() : () -> ()
// CHECK:           } else
// CHECK:             call @side_effect() : () -> ()
// CHECK:           }
// CHECK:           return

// -----

func private @make_i32() -> i32

func @for_yields_2(%lb : index, %ub : index, %step : index) -> i32 {
  %a = call @make_i32() : () -> (i32)
  %b = scf.for %i = %lb to %ub step %step iter_args(%0 = %a) -> i32 {
    scf.yield %0 : i32
  }
  return %b : i32
}

// CHECK-LABEL:   func @for_yields_2
//  CHECK-NEXT:     %[[R:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     return %[[R]] : i32

func @for_yields_3(%lb : index, %ub : index, %step : index) -> (i32, i32, i32) {
  %a = call @make_i32() : () -> (i32)
  %b = call @make_i32() : () -> (i32)
  %r:3 = scf.for %i = %lb to %ub step %step iter_args(%0 = %a, %1 = %a, %2 = %b) -> (i32, i32, i32) {
    %c = call @make_i32() : () -> (i32)
    scf.yield %0, %c, %2 : i32, i32, i32
  }
  return %r#0, %r#1, %r#2 : i32, i32, i32
}

// CHECK-LABEL:   func @for_yields_3
//  CHECK-NEXT:     %[[a:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     %[[b:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     %[[r1:.*]] = scf.for {{.*}} iter_args(%arg4 = %[[a]]) -> (i32) {
//  CHECK-NEXT:       %[[c:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:       scf.yield %[[c]] : i32
//  CHECK-NEXT:     }
//  CHECK-NEXT:     return %[[a]], %[[r1]], %[[b]] : i32, i32, i32

// -----

// CHECK-LABEL: @replace_true_if
func @replace_true_if() {
  %true = constant true
  // CHECK-NOT: scf.if
  // CHECK: "test.op"
  scf.if %true {
    "test.op"() : () -> ()
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: @remove_false_if
func @remove_false_if() {
  %false = constant false
  // CHECK-NOT: scf.if
  // CHECK-NOT: "test.op"
  scf.if %false {
    "test.op"() : () -> ()
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: @replace_true_if_with_values
func @replace_true_if_with_values() {
  %true = constant true
  // CHECK-NOT: scf.if
  // CHECK: %[[VAL:.*]] = "test.op"
  %0 = scf.if %true -> (i32) {
    %1 = "test.op"() : () -> i32
    scf.yield %1 : i32
  } else {
    %2 = "test.other_op"() : () -> i32
    scf.yield %2 : i32
  }
  // CHECK: "test.consume"(%[[VAL]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @replace_false_if_with_values
func @replace_false_if_with_values() {
  %false = constant false
  // CHECK-NOT: scf.if
  // CHECK: %[[VAL:.*]] = "test.other_op"
  %0 = scf.if %false -> (i32) {
    %1 = "test.op"() : () -> i32
    scf.yield %1 : i32
  } else {
    %2 = "test.other_op"() : () -> i32
    scf.yield %2 : i32
  }
  // CHECK: "test.consume"(%[[VAL]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @remove_zero_iteration_loop
func @remove_zero_iteration_loop() {
  %c42 = constant 42 : index
  %c1 = constant 1 : index
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> i32
  // CHECK-NOT: scf.for
  %0 = scf.for %i = %c42 to %c1 step %c1 iter_args(%arg = %init) -> (i32) {
    %1 = "test.op"(%i, %arg) : (index, i32) -> i32
    scf.yield %1 : i32
  }
  // CHECK: "test.consume"(%[[INIT]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @remove_zero_iteration_loop_vals
func @remove_zero_iteration_loop_vals(%arg0: index) {
  %c2 = constant 2 : index
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> i32
  // CHECK-NOT: scf.for
  // CHECK-NOT: test.op
  %0 = scf.for %i = %arg0 to %arg0 step %c2 iter_args(%arg = %init) -> (i32) {
    %1 = "test.op"(%i, %arg) : (index, i32) -> i32
    scf.yield %1 : i32
  }
  // CHECK: "test.consume"(%[[INIT]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @replace_single_iteration_loop_1
func @replace_single_iteration_loop_1() {
  // CHECK: %[[LB:.*]] = constant 42
  %c42 = constant 42 : index
  %c43 = constant 43 : index
  %c1 = constant 1 : index
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> i32
  // CHECK-NOT: scf.for
  // CHECK: %[[VAL:.*]] = "test.op"(%[[LB]], %[[INIT]])
  %0 = scf.for %i = %c42 to %c43 step %c1 iter_args(%arg = %init) -> (i32) {
    %1 = "test.op"(%i, %arg) : (index, i32) -> i32
    scf.yield %1 : i32
  }
  // CHECK: "test.consume"(%[[VAL]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @replace_single_iteration_loop_2
func @replace_single_iteration_loop_2() {
  // CHECK: %[[LB:.*]] = constant 5
	%c5 = constant 5 : index
	%c6 = constant 6 : index
	%c11 = constant 11 : index
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> i32
  // CHECK-NOT: scf.for
  // CHECK: %[[VAL:.*]] = "test.op"(%[[LB]], %[[INIT]])
  %0 = scf.for %i = %c5 to %c11 step %c6 iter_args(%arg = %init) -> (i32) {
    %1 = "test.op"(%i, %arg) : (index, i32) -> i32
    scf.yield %1 : i32
  }
  // CHECK: "test.consume"(%[[VAL]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @replace_single_iteration_loop_non_unit_step
func @replace_single_iteration_loop_non_unit_step() {
  // CHECK: %[[LB:.*]] = constant 42
  %c42 = constant 42 : index
  %c47 = constant 47 : index
  %c5 = constant 5 : index
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> i32
  // CHECK-NOT: scf.for
  // CHECK: %[[VAL:.*]] = "test.op"(%[[LB]], %[[INIT]])
  %0 = scf.for %i = %c42 to %c47 step %c5 iter_args(%arg = %init) -> (i32) {
    %1 = "test.op"(%i, %arg) : (index, i32) -> i32
    scf.yield %1 : i32
  }
  // CHECK: "test.consume"(%[[VAL]])
  "test.consume"(%0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: @remove_empty_parallel_loop
func @remove_empty_parallel_loop(%lb: index, %ub: index, %s: index) {
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> f32
  // CHECK-NOT: scf.parallel
  // CHECK-NOT: test.produce
  // CHECK-NOT: test.transform
  %0 = scf.parallel (%i, %j, %k) = (%lb, %ub, %lb) to (%ub, %ub, %ub) step (%s, %s, %s) init(%init) -> f32 {
    %1 = "test.produce"() : () -> f32
    scf.reduce(%1) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = "test.transform"(%lhs, %rhs) : (f32, f32) -> f32
      scf.reduce.return %2 : f32
    }
    scf.yield
  }
  // CHECK: "test.consume"(%[[INIT]])
  "test.consume"(%0) : (f32) -> ()
  return
}

// -----

func private @process(%0 : memref<128x128xf32>)
func private @process_tensor(%0 : tensor<128x128xf32>) -> memref<128x128xf32>

// CHECK-LABEL: last_value
//  CHECK-SAME:   %[[T0:[0-9a-z]*]]: tensor<128x128xf32>
//  CHECK-SAME:   %[[T1:[0-9a-z]*]]: tensor<128x128xf32>
//  CHECK-SAME:   %[[T2:[0-9a-z]*]]: tensor<128x128xf32>
//  CHECK-SAME:   %[[M0:[0-9a-z]*]]: memref<128x128xf32>
func @last_value(%t0: tensor<128x128xf32>, %t1: tensor<128x128xf32>,
                 %t2: tensor<128x128xf32>, %m0: memref<128x128xf32>,
                 %lb : index, %ub : index, %step : index)
  -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>)
{
  // CHECK-NEXT: %[[M1:.*]] = memref.buffer_cast %[[T1]] : memref<128x128xf32>
  // CHECK-NEXT: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args(%[[BBARG_T2:.*]] = %[[T2]]) -> (tensor<128x128xf32>) {
  %0:3 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %t0, %arg2 = %t1, %arg3 = %t2)
    -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>)
  {
    %m1 = memref.buffer_cast %arg2 : memref<128x128xf32>

    // CHECK-NEXT:   call @process(%[[M0]]) : (memref<128x128xf32>) -> ()
    call @process(%m0) : (memref<128x128xf32>) -> ()

    // CHECK-NEXT:   call @process(%[[M1]]) : (memref<128x128xf32>) -> ()
    call @process(%m1) : (memref<128x128xf32>) -> ()

    // This does not hoist (fails the bbArg has at most a single check).
    // CHECK-NEXT:   %[[T:.*]] = call @process_tensor(%[[BBARG_T2]]) : (tensor<128x128xf32>) -> memref<128x128xf32>
    // CHECK-NEXT:   %[[YIELD_T:.*]] = memref.tensor_load %[[T:.*]]
    %m2 = call @process_tensor(%arg3): (tensor<128x128xf32>) -> memref<128x128xf32>
    %3 = memref.tensor_load %m2 : memref<128x128xf32>

    // All this stuff goes away, incrementally
    %1 = memref.tensor_load %m0 : memref<128x128xf32>
    %2 = memref.tensor_load %m1 : memref<128x128xf32>

    // CHECK-NEXT:   scf.yield %[[YIELD_T]] : tensor<128x128xf32>
    scf.yield %1, %2, %3 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>

  // CHECK-NEXT: }
  }

  // CHECK-NEXT: %[[R0:.*]] = memref.tensor_load %[[M0]] : memref<128x128xf32>
  // CHECK-NEXT: %[[R1:.*]] = memref.tensor_load %[[M1]] : memref<128x128xf32>
  // CHECK-NEXT: return %[[R0]], %[[R1]], %[[FOR_RES]] : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  return %0#0, %0#1, %0#2 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
}

// -----

// CHECK-LABEL: fold_away_iter_with_no_use_and_yielded_input
//  CHECK-SAME:   %[[A0:[0-9a-z]*]]: i32
func @fold_away_iter_with_no_use_and_yielded_input(%arg0 : i32,
                    %ub : index, %lb : index, %step : index) -> (i32, i32) {
  // CHECK-NEXT: %[[C32:.*]] = constant 32 : i32
  %cst = constant 32 : i32
  // CHECK-NEXT: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[A0]]) -> (i32) { 
  %0:2 = scf.for %arg1 = %lb to %ub step %step iter_args(%arg2 = %arg0, %arg3 = %cst)
    -> (i32, i32) {
    %1 = addi %arg2, %cst : i32
    scf.yield %1, %cst : i32, i32
  }

  // CHECK: return %[[FOR_RES]], %[[C32]] : i32, i32
  return %0#0, %0#1 : i32, i32
}

// -----

// CHECK-LABEL: fold_away_iter_and_result_with_no_use
//  CHECK-SAME:   %[[A0:[0-9a-z]*]]: i32
func @fold_away_iter_and_result_with_no_use(%arg0 : i32,
                    %ub : index, %lb : index, %step : index) -> (i32) {
  %cst = constant 32 : i32
  // CHECK: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[A0]]) -> (i32) {
  %0:2 = scf.for %arg1 = %lb to %ub step %step iter_args(%arg2 = %arg0, %arg3 = %cst)
    -> (i32, i32) {
    %1 = addi %arg2, %cst : i32
    scf.yield %1, %1 : i32, i32
  }
  
  // CHECK: return %[[FOR_RES]] : i32
  return %0#0 : i32
}
