// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

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
    store %c42, %A[%i0, %i1, %i2] : memref<?x?x?xi32>
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @single_iteration(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?x?xi32>) {
// CHECK:           [[C0:%.*]] = constant 0 : index
// CHECK:           [[C2:%.*]] = constant 2 : index
// CHECK:           [[C3:%.*]] = constant 3 : index
// CHECK:           [[C6:%.*]] = constant 6 : index
// CHECK:           [[C7:%.*]] = constant 7 : index
// CHECK:           [[C42:%.*]] = constant 42 : i32
// CHECK:           scf.parallel ([[V0:%.*]]) = ([[C3]]) to ([[C6]]) step ([[C2]]) {
// CHECK:             store [[C42]], [[ARG0]]{{\[}}[[C0]], [[V0]], [[C7]]] : memref<?x?x?xi32>
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

// CHECK-LABEL: @replace_single_iteration_loop
func @replace_single_iteration_loop() {
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
