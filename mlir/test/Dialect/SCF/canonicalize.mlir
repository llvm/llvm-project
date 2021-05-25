// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s


// -----

func @single_iteration_some(%A: memref<?x?x?xi32>) {
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

// CHECK-LABEL:   func @single_iteration_some(
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

func @single_iteration_all(%A: memref<?x?x?xi32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c10 = constant 10 : index
  scf.parallel (%i0, %i1, %i2) = (%c0, %c3, %c7) to (%c1, %c6, %c10) step (%c1, %c3, %c3) {
    %c42 = constant 42 : i32
    memref.store %c42, %A[%i0, %i1, %i2] : memref<?x?x?xi32>
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @single_iteration_all(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?x?xi32>) {
// CHECK-DAG:           [[C42:%.*]] = constant 42 : i32
// CHECK-DAG:           [[C7:%.*]] = constant 7 : index
// CHECK-DAG:           [[C3:%.*]] = constant 3 : index
// CHECK-DAG:           [[C0:%.*]] = constant 0 : index
// CHECK-NOT:           scf.parallel
// CHECK:               memref.store [[C42]], [[ARG0]]{{\[}}[[C0]], [[C3]], [[C7]]] : memref<?x?x?xi32>
// CHECK-NOT:           scf.yield
// CHECK:               return

// -----

func @single_iteration_reduce(%A: index, %B: index) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %0:2 = scf.parallel (%i0, %i1) = (%c1, %c3) to (%c2, %c6) step (%c1, %c3) init(%A, %B) -> (index, index) {
    scf.reduce(%i0) : index {
    ^bb0(%lhs: index, %rhs: index):
      %1 = addi %lhs, %rhs : index
      scf.reduce.return %1 : index
    }
    scf.reduce(%i1) : index {
    ^bb0(%lhs: index, %rhs: index):
      %2 = muli %lhs, %rhs : index
      scf.reduce.return %2 : index
    }
    scf.yield
  }
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL:   func @single_iteration_reduce(
// CHECK-SAME:                        [[ARG0:%.*]]: index, [[ARG1:%.*]]: index)
// CHECK-DAG:           [[C3:%.*]] = constant 3 : index
// CHECK-DAG:           [[C1:%.*]] = constant 1 : index
// CHECK-NOT:           scf.parallel
// CHECK-NOT:           scf.reduce
// CHECK-NOT:           scf.reduce.return
// CHECK-NOT:           scf.yield
// CHECK:               [[V0:%.*]] = addi [[ARG0]], [[C1]]
// CHECK:               [[V1:%.*]] = muli [[ARG1]], [[C3]]
// CHECK:               return [[V0]], [[V1]]

// -----

func @nested_parallel(%0: memref<?x?x?xf64>) -> memref<?x?x?xf64> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %1 = memref.dim %0, %c0 : memref<?x?x?xf64>
  %2 = memref.dim %0, %c1 : memref<?x?x?xf64>
  %3 = memref.dim %0, %c2 : memref<?x?x?xf64>
  %4 = memref.alloc(%1, %2, %3) : memref<?x?x?xf64>
  scf.parallel (%arg1) = (%c0) to (%1) step (%c1) {
    scf.parallel (%arg2) = (%c0) to (%2) step (%c1) {
      scf.parallel (%arg3) = (%c0) to (%3) step (%c1) {
        %5 = memref.load %0[%arg1, %arg2, %arg3] : memref<?x?x?xf64>
        memref.store %5, %4[%arg1, %arg2, %arg3] : memref<?x?x?xf64>
        scf.yield
      }
      scf.yield
    }
    scf.yield
  }
  return %4 : memref<?x?x?xf64>
}

// CHECK-LABEL:   func @nested_parallel(
// CHECK-DAG:       [[C0:%.*]] = constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = constant 1 : index
// CHECK-DAG:       [[C2:%.*]] = constant 2 : index
// CHECK:           [[B0:%.*]] = memref.dim {{.*}}, [[C0]]
// CHECK:           [[B1:%.*]] = memref.dim {{.*}}, [[C1]]
// CHECK:           [[B2:%.*]] = memref.dim {{.*}}, [[C2]]
// CHECK:           scf.parallel ([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[B0]], [[B1]], [[B2]]) step ([[C1]], [[C1]], [[C1]])
// CHECK:           memref.load {{.*}}{{\[}}[[V0]], [[V1]], [[V2]]]
// CHECK:           memref.store {{.*}}{{\[}}[[V0]], [[V1]], [[V2]]]

// -----

func private @side_effect()
func @one_unused(%cond: i1) -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %0, %1 = scf.if %cond -> (index, index) {
    call @side_effect() : () -> ()
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c2, %c3 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @one_unused
// CHECK-DAG:       [[C0:%.*]] = constant 1 : index
// CHECK-DAG:       [[C3:%.*]] = constant 3 : index
// CHECK:           [[V0:%.*]] = scf.if %{{.*}} -> (index) {
// CHECK:             call @side_effect() : () -> ()
// CHECK:             scf.yield [[C0]] : index
// CHECK:           } else
// CHECK:             scf.yield [[C3]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func private @side_effect()
func @nested_unused(%cond1: i1, %cond2: i1) -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %0, %1 = scf.if %cond1 -> (index, index) {
    %2, %3 = scf.if %cond2 -> (index, index) {
      call @side_effect() : () -> ()
      scf.yield %c0, %c1 : index, index
    } else {
      scf.yield %c2, %c3 : index, index
    }
    scf.yield %2, %3 : index, index
  } else {
    scf.yield %c0, %c1 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @nested_unused
// CHECK-DAG:       [[C0:%.*]] = constant 1 : index
// CHECK-DAG:       [[C3:%.*]] = constant 3 : index
// CHECK:           [[V0:%.*]] = scf.if {{.*}} -> (index) {
// CHECK:             [[V1:%.*]] = scf.if {{.*}} -> (index) {
// CHECK:               call @side_effect() : () -> ()
// CHECK:               scf.yield [[C0]] : index
// CHECK:             } else
// CHECK:               scf.yield [[C3]] : index
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

func @empty_if1(%cond: i1) {
  scf.if %cond {
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @empty_if1
// CHECK-NOT:       scf.if
// CHECK:           return

// -----

func @empty_if2(%cond: i1) {
  scf.if %cond {
    scf.yield
  } else {
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @empty_if2
// CHECK-NOT:       scf.if
// CHECK:           return

// -----

func @to_select1(%cond: i1) -> index {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = scf.if %cond -> index {
    scf.yield %c0 : index
  } else {
    scf.yield %c1 : index
  }
  return %0 : index
}

// CHECK-LABEL:   func @to_select1
// CHECK-DAG:       [[C0:%.*]] = constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = constant 1 : index
// CHECK:           [[V0:%.*]] = select {{.*}}, [[C0]], [[C1]]
// CHECK:           return [[V0]] : index

// -----

func @to_select_same_val(%cond: i1) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0, %1 = scf.if %cond -> (index, index) {
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c1, %c1 : index, index
  }
  return %0, %1 : index, index
}

// CHECK-LABEL:   func @to_select_same_val
// CHECK-DAG:       [[C0:%.*]] = constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = constant 1 : index
// CHECK:           [[V0:%.*]] = select {{.*}}, [[C0]], [[C1]]
// CHECK:           return [[V0]], [[C1]] : index, index

// -----

func @to_select2(%cond: i1) -> (index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %0, %1 = scf.if %cond -> (index, index) {
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c2, %c3 : index, index
  }
  return %0, %1 : index, index
}

// CHECK-LABEL:   func @to_select2
// CHECK-DAG:       [[C0:%.*]] = constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = constant 1 : index
// CHECK-DAG:       [[C2:%.*]] = constant 2 : index
// CHECK-DAG:       [[C3:%.*]] = constant 3 : index
// CHECK:           [[V0:%.*]] = select {{.*}}, [[C0]], [[C2]]
// CHECK:           [[V1:%.*]] = select {{.*}}, [[C1]], [[C3]]
// CHECK:           return [[V0]], [[V1]] : index

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

// -----

func private @do(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>

// CHECK-LABEL: matmul_on_tensors
//  CHECK-SAME:   %[[T0:[0-9a-z]*]]: tensor<32x1024xf32>
//  CHECK-SAME:   %[[T1:[0-9a-z]*]]: tensor<1024x1024xf32>
func @matmul_on_tensors(%t0: tensor<32x1024xf32>, %t1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %c0 = constant 0 : index
  %c32 = constant 32 : index
  %c1024 = constant 1024 : index
//   CHECK-NOT: tensor.cast
//       CHECK: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args(%[[ITER_T0:.*]] = %[[T0]]) -> (tensor<32x1024xf32>) {
//       CHECK:   %[[CAST:.*]] = tensor.cast %[[ITER_T0]] : tensor<32x1024xf32> to tensor<?x?xf32>
//       CHECK:   %[[DONE:.*]] = call @do(%[[CAST]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK:   %[[UNCAST:.*]] = tensor.cast %[[DONE]] : tensor<?x?xf32> to tensor<32x1024xf32>
//       CHECK:   scf.yield %[[UNCAST]] : tensor<32x1024xf32>
  %0 = tensor.cast %t0 : tensor<32x1024xf32> to tensor<?x?xf32>
  %1 = scf.for %i = %c0 to %c1024 step %c32 iter_args(%iter_t0 = %0) -> (tensor<?x?xf32>) {
    %2 = call @do(%iter_t0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.yield %2 : tensor<?x?xf32>
  }
//   CHECK-NOT: tensor.cast
//       CHECK: %[[RES:.*]] = subtensor_insert %[[FOR_RES]] into %[[T1]][0, 0] [32, 1024] [1, 1] : tensor<32x1024xf32> into tensor<1024x1024xf32>
//       CHECK: return %[[RES]] : tensor<1024x1024xf32>
  %2 = tensor.cast %1 : tensor<?x?xf32> to tensor<32x1024xf32>
  %res = subtensor_insert %2 into %t1[0, 0] [32, 1024] [1, 1] : tensor<32x1024xf32> into tensor<1024x1024xf32>
  return %res : tensor<1024x1024xf32>
}



// CHECK-LABEL: @cond_prop
func @cond_prop(%arg0 : i1) -> index {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %res = scf.if %arg0 -> index {
    %res1 = scf.if %arg0 -> index {
      %v1 = "test.get_some_value"() : () -> i32
      scf.yield %c1 : index
    } else {
      %v2 = "test.get_some_value"() : () -> i32
      scf.yield %c2 : index
    }
    scf.yield %res1 : index
  } else {
    %res2 = scf.if %arg0 -> index {
      %v3 = "test.get_some_value"() : () -> i32
      scf.yield %c3 : index
    } else {
      %v4 = "test.get_some_value"() : () -> i32
      scf.yield %c4 : index
    }
    scf.yield %res2 : index
  }
  return %res : index
}
// CHECK-DAG:  %[[c1:.+]] = constant 1 : index
// CHECK-DAG:  %[[c4:.+]] = constant 4 : index
// CHECK-NEXT:  %[[if:.+]] = scf.if %arg0 -> (index) {
// CHECK-NEXT:    %{{.+}} = "test.get_some_value"() : () -> i32
// CHECK-NEXT:    scf.yield %[[c1]] : index
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %{{.+}} = "test.get_some_value"() : () -> i32
// CHECK-NEXT:    scf.yield %[[c4]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[if]] : index
// CHECK-NEXT:}

// CHECK-LABEL: @replace_if_with_cond1
func @replace_if_with_cond1(%arg0 : i1) -> (i32, i1) {
  %true = constant true
  %false = constant false
  %res:2 = scf.if %arg0 -> (i32, i1) {
    %v = "test.get_some_value"() : () -> i32
    scf.yield %v, %true : i32, i1
  } else {
    %v2 = "test.get_some_value"() : () -> i32
    scf.yield %v2, %false : i32, i1
  }
  return %res#0, %res#1 : i32, i1
}
// CHECK-NEXT:    %[[if:.+]] = scf.if %arg0 -> (i32) {
// CHECK-NEXT:      %[[sv1:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:      scf.yield %[[sv1]] : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[sv2:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:      scf.yield %[[sv2]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[if]], %arg0 : i32, i1

// CHECK-LABEL: @replace_if_with_cond2
func @replace_if_with_cond2(%arg0 : i1) -> (i32, i1) {
  %true = constant true
  %false = constant false
  %res:2 = scf.if %arg0 -> (i32, i1) {
    %v = "test.get_some_value"() : () -> i32
    scf.yield %v, %false : i32, i1
  } else {
    %v2 = "test.get_some_value"() : () -> i32
    scf.yield %v2, %true : i32, i1
  }
  return %res#0, %res#1 : i32, i1
}
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %[[toret:.+]] = xor %arg0, %true : i1
// CHECK-NEXT:     %[[if:.+]] = scf.if %arg0 -> (i32) {
// CHECK-NEXT:       %[[sv1:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:       scf.yield %[[sv1]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[sv2:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:       scf.yield %[[sv2]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[if]], %[[toret]] : i32, i1


// CHECK-LABEL: @replace_if_with_cond3
func @replace_if_with_cond3(%arg0 : i1, %arg2: i64) -> (i32, i64) {
  %res:2 = scf.if %arg0 -> (i32, i64) {
    %v = "test.get_some_value"() : () -> i32
    scf.yield %v, %arg2 : i32, i64
  } else {
    %v2 = "test.get_some_value"() : () -> i32
    scf.yield %v2, %arg2 : i32, i64
  }
  return %res#0, %res#1 : i32, i64
}
// CHECK-NEXT:     %[[if:.+]] = scf.if %arg0 -> (i32) {
// CHECK-NEXT:       %[[sv1:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:       scf.yield %[[sv1]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[sv2:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:       scf.yield %[[sv2]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[if]], %arg1 : i32, i64


// CHECK-LABEL: @while_cond_true
func @while_cond_true() {
  %0 = scf.while () : () -> i1 {
    %condition = "test.condition"() : () -> i1
    scf.condition(%condition) %condition : i1
  } do {
  ^bb0(%arg0: i1):
    "test.use"(%arg0) : (i1) -> ()
    scf.yield
  }
  return
}
// CHECK-NEXT:         %[[true:.+]] = constant true
// CHECK-NEXT:         %{{.+}} = scf.while : () -> i1 {
// CHECK-NEXT:           %[[cmp:.+]] = "test.condition"() : () -> i1
// CHECK-NEXT:           scf.condition(%[[cmp]]) %[[cmp]] : i1
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%arg0: i1):  // no predecessors
// CHECK-NEXT:           "test.use"(%[[true]]) : (i1) -> ()
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }

// -----

// CHECK-LABEL: @combineIfs
func @combineIfs(%arg0 : i1, %arg2: i64) -> (i32, i32) {
  %res = scf.if %arg0 -> i32 {
    %v = "test.firstCodeTrue"() : () -> i32
    scf.yield %v : i32
  } else {
    %v2 = "test.firstCodeFalse"() : () -> i32
    scf.yield %v2 : i32
  }
  %res2 = scf.if %arg0 -> i32 {
    %v = "test.secondCodeTrue"() : () -> i32
    scf.yield %v : i32
  } else {
    %v2 = "test.secondCodeFalse"() : () -> i32
    scf.yield %v2 : i32
  }
  return %res, %res2 : i32, i32
}
// CHECK-NEXT:     %[[res:.+]]:2 = scf.if %arg0 -> (i32, i32) {
// CHECK-NEXT:       %[[tval0:.+]] = "test.firstCodeTrue"() : () -> i32
// CHECK-NEXT:       %[[tval:.+]] = "test.secondCodeTrue"() : () -> i32
// CHECK-NEXT:       scf.yield %[[tval0]], %[[tval]] : i32, i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[fval0:.+]] = "test.firstCodeFalse"() : () -> i32
// CHECK-NEXT:       %[[fval:.+]] = "test.secondCodeFalse"() : () -> i32
// CHECK-NEXT:       scf.yield %[[fval0]], %[[fval]] : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[res]]#0, %[[res]]#1 : i32, i32


// CHECK-LABEL: @combineIfs2
func @combineIfs2(%arg0 : i1, %arg2: i64) -> i32 {
  scf.if %arg0 {
    "test.firstCodeTrue"() : () -> ()
    scf.yield
  }
  %res = scf.if %arg0 -> i32 {
    %v = "test.secondCodeTrue"() : () -> i32
    scf.yield %v : i32
  } else {
    %v2 = "test.secondCodeFalse"() : () -> i32
    scf.yield %v2 : i32
  }
  return %res : i32
}
// CHECK-NEXT:     %[[res:.+]] = scf.if %arg0 -> (i32) {
// CHECK-NEXT:       "test.firstCodeTrue"() : () -> ()
// CHECK-NEXT:       %[[tval:.+]] = "test.secondCodeTrue"() : () -> i32
// CHECK-NEXT:       scf.yield %[[tval]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[fval:.+]] = "test.secondCodeFalse"() : () -> i32
// CHECK-NEXT:       scf.yield %[[fval]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[res]] : i32


// CHECK-LABEL: @combineIfs3
func @combineIfs3(%arg0 : i1, %arg2: i64) -> i32 {
  %res = scf.if %arg0 -> i32 {
    %v = "test.firstCodeTrue"() : () -> i32
    scf.yield %v : i32
  } else {
    %v2 = "test.firstCodeFalse"() : () -> i32
    scf.yield %v2 : i32
  }
  scf.if %arg0 {
    "test.secondCodeTrue"() : () -> ()
    scf.yield
  }
  return %res : i32
}
// CHECK-NEXT:     %[[res:.+]] = scf.if %arg0 -> (i32) {
// CHECK-NEXT:       %[[tval:.+]] = "test.firstCodeTrue"() : () -> i32
// CHECK-NEXT:       "test.secondCodeTrue"() : () -> ()
// CHECK-NEXT:       scf.yield %[[tval]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[fval:.+]] = "test.firstCodeFalse"() : () -> i32
// CHECK-NEXT:       scf.yield %[[fval]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[res]] : i32

// CHECK-LABEL: @combineIfs4
func @combineIfs4(%arg0 : i1, %arg2: i64) {
  scf.if %arg0 {
    "test.firstCodeTrue"() : () -> ()
    scf.yield
  }
  scf.if %arg0 {
    "test.secondCodeTrue"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-NEXT:     scf.if %arg0 {
// CHECK-NEXT:       "test.firstCodeTrue"() : () -> ()
// CHECK-NEXT:       "test.secondCodeTrue"() : () -> ()
// CHECK-NEXT:     }
