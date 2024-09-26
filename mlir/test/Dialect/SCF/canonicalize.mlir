// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

func.func @single_iteration_some(%A: memref<?x?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c10 = arith.constant 10 : index
  scf.parallel (%i0, %i1, %i2) = (%c0, %c3, %c7) to (%c1, %c6, %c10) step (%c1, %c2, %c3) {
    %c42 = arith.constant 42 : i32
    memref.store %c42, %A[%i0, %i1, %i2] : memref<?x?x?xi32>
    scf.reduce
  }
  return
}

// CHECK-LABEL:   func @single_iteration_some(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?x?xi32>) {
// CHECK-DAG:           [[C42:%.*]] = arith.constant 42 : i32
// CHECK-DAG:           [[C7:%.*]] = arith.constant 7 : index
// CHECK-DAG:           [[C6:%.*]] = arith.constant 6 : index
// CHECK-DAG:           [[C3:%.*]] = arith.constant 3 : index
// CHECK-DAG:           [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:           [[C0:%.*]] = arith.constant 0 : index
// CHECK:           scf.parallel ([[V0:%.*]]) = ([[C3]]) to ([[C6]]) step ([[C2]]) {
// CHECK:             memref.store [[C42]], [[ARG0]]{{\[}}[[C0]], [[V0]], [[C7]]] : memref<?x?x?xi32>
// CHECK:             scf.reduce
// CHECK:           }
// CHECK:           return

// -----

func.func @single_iteration_all(%A: memref<?x?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c10 = arith.constant 10 : index
  scf.parallel (%i0, %i1, %i2) = (%c0, %c3, %c7) to (%c1, %c6, %c10) step (%c1, %c3, %c3) {
    %c42 = arith.constant 42 : i32
    memref.store %c42, %A[%i0, %i1, %i2] : memref<?x?x?xi32>
    scf.reduce
  }
  return
}

// CHECK-LABEL:   func @single_iteration_all(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?x?xi32>) {
// CHECK-DAG:           [[C42:%.*]] = arith.constant 42 : i32
// CHECK-DAG:           [[C7:%.*]] = arith.constant 7 : index
// CHECK-DAG:           [[C3:%.*]] = arith.constant 3 : index
// CHECK-DAG:           [[C0:%.*]] = arith.constant 0 : index
// CHECK-NOT:           scf.parallel
// CHECK:               memref.store [[C42]], [[ARG0]]{{\[}}[[C0]], [[C3]], [[C7]]] : memref<?x?x?xi32>
// CHECK-NOT:           scf.reduce
// CHECK:               return

// -----

func.func @single_iteration_reduce(%A: index, %B: index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %0:2 = scf.parallel (%i0, %i1) = (%c1, %c3) to (%c2, %c6) step (%c1, %c3) init(%A, %B) -> (index, index) {
    scf.reduce(%i0, %i1 : index, index)  {
    ^bb0(%lhs: index, %rhs: index):
      %1 = arith.addi %lhs, %rhs : index
      scf.reduce.return %1 : index
    }, {
    ^bb0(%lhs: index, %rhs: index):
      %2 = arith.muli %lhs, %rhs : index
      scf.reduce.return %2 : index
    }
  }
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL:   func @single_iteration_reduce(
// CHECK-SAME:                        [[ARG0:%.*]]: index, [[ARG1:%.*]]: index)
// CHECK-DAG:           [[C3:%.*]] = arith.constant 3 : index
// CHECK-DAG:           [[C1:%.*]] = arith.constant 1 : index
// CHECK-NOT:           scf.parallel
// CHECK-NOT:           scf.reduce
// CHECK-NOT:           scf.reduce.return
// CHECK-NOT:           scf.yield
// CHECK:               [[V0:%.*]] = arith.addi [[ARG0]], [[C1]]
// CHECK:               [[V1:%.*]] = arith.muli [[ARG1]], [[C3]]
// CHECK:               return [[V0]], [[V1]]

// -----

func.func @nested_parallel(%0: memref<?x?x?xf64>) -> memref<?x?x?xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %1 = memref.dim %0, %c0 : memref<?x?x?xf64>
  %2 = memref.dim %0, %c1 : memref<?x?x?xf64>
  %3 = memref.dim %0, %c2 : memref<?x?x?xf64>
  %4 = memref.alloc(%1, %2, %3) : memref<?x?x?xf64>
  scf.parallel (%arg1) = (%c0) to (%1) step (%c1) {
    scf.parallel (%arg2) = (%c0) to (%2) step (%c1) {
      scf.parallel (%arg3) = (%c0) to (%3) step (%c1) {
        %5 = memref.load %0[%arg1, %arg2, %arg3] : memref<?x?x?xf64>
        memref.store %5, %4[%arg1, %arg2, %arg3] : memref<?x?x?xf64>
        scf.reduce
      }
      scf.reduce
    }
    scf.reduce
  }
  return %4 : memref<?x?x?xf64>
}

// CHECK-LABEL:   func @nested_parallel(
// CHECK-DAG:       [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:       [[C2:%.*]] = arith.constant 2 : index
// CHECK:           [[B0:%.*]] = memref.dim {{.*}}, [[C0]]
// CHECK:           [[B1:%.*]] = memref.dim {{.*}}, [[C1]]
// CHECK:           [[B2:%.*]] = memref.dim {{.*}}, [[C2]]
// CHECK:           scf.parallel ([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[B0]], [[B1]], [[B2]]) step ([[C1]], [[C1]], [[C1]])
// CHECK:           memref.load {{.*}}{{\[}}[[V0]], [[V1]], [[V2]]]
// CHECK:           memref.store {{.*}}{{\[}}[[V0]], [[V1]], [[V2]]]

// -----

func.func private @side_effect()
func.func @one_unused(%cond: i1) -> (index) {
  %0, %1 = scf.if %cond -> (index, index) {
    func.call @side_effect() : () -> ()
    %c0 = "test.value0"() : () -> (index)
    %c1 = "test.value1"() : () -> (index)
    scf.yield %c0, %c1 : index, index
  } else {
    %c2 = "test.value2"() : () -> (index)
    %c3 = "test.value3"() : () -> (index)
    scf.yield %c2, %c3 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @one_unused
// CHECK:           [[V0:%.*]] = scf.if %{{.*}} -> (index) {
// CHECK:             call @side_effect() : () -> ()
// CHECK:             [[C1:%.*]] = "test.value1"
// CHECK:             scf.yield [[C1]] : index
// CHECK:           } else
// CHECK:             [[C3:%.*]] = "test.value3"
// CHECK:             scf.yield [[C3]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func.func private @side_effect()
func.func @nested_unused(%cond1: i1, %cond2: i1) -> (index) {
  %0, %1 = scf.if %cond1 -> (index, index) {
    %2, %3 = scf.if %cond2 -> (index, index) {
      func.call @side_effect() : () -> ()
      %c0 = "test.value0"() : () -> (index)
      %c1 = "test.value1"() : () -> (index)
      scf.yield %c0, %c1 : index, index
    } else {
      %c2 = "test.value2"() : () -> (index)
      %c3 = "test.value3"() : () -> (index)
      scf.yield %c2, %c3 : index, index
    }
    scf.yield %2, %3 : index, index
  } else {
    %c0 = "test.value0_2"() : () -> (index)
    %c1 = "test.value1_2"() : () -> (index)
    scf.yield %c0, %c1 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @nested_unused
// CHECK:           [[V0:%.*]] = scf.if {{.*}} -> (index) {
// CHECK:             [[V1:%.*]] = scf.if {{.*}} -> (index) {
// CHECK:               call @side_effect() : () -> ()
// CHECK:               [[C1:%.*]] = "test.value1"
// CHECK:               scf.yield [[C1]] : index
// CHECK:             } else
// CHECK:               [[C3:%.*]] = "test.value3"
// CHECK:               scf.yield [[C3]] : index
// CHECK:             }
// CHECK:             scf.yield [[V1]] : index
// CHECK:           } else
// CHECK:             [[C1_2:%.*]] = "test.value1_2"
// CHECK:             scf.yield [[C1_2]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func.func private @side_effect()
func.func @all_unused(%cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0, %1 = scf.if %cond -> (index, index) {
    func.call @side_effect() : () -> ()
    scf.yield %c0, %c1 : index, index
  } else {
    func.call @side_effect() : () -> ()
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

func.func @empty_if1(%cond: i1) {
  scf.if %cond {
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @empty_if1
// CHECK-NOT:       scf.if
// CHECK:           return

// -----

func.func @empty_if2(%cond: i1) {
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

func.func @empty_else(%cond: i1, %v : memref<i1>) {
  scf.if %cond {
    memref.store %cond, %v[] : memref<i1>
  } else {
  }
  return
}

// CHECK-LABEL: func @empty_else
// CHECK:         scf.if
// CHECK-NOT:     else

// -----

func.func @to_select1(%cond: i1) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.if %cond -> index {
    scf.yield %c0 : index
  } else {
    scf.yield %c1 : index
  }
  return %0 : index
}

// CHECK-LABEL:   func @to_select1
// CHECK-DAG:       [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = arith.constant 1 : index
// CHECK:           [[V0:%.*]] = arith.select {{.*}}, [[C0]], [[C1]]
// CHECK:           return [[V0]] : index

// -----

func.func @to_select_same_val(%cond: i1) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0, %1 = scf.if %cond -> (index, index) {
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c1, %c1 : index, index
  }
  return %0, %1 : index, index
}

// CHECK-LABEL:   func @to_select_same_val
// CHECK-DAG:       [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = arith.constant 1 : index
// CHECK:           [[V0:%.*]] = arith.select {{.*}}, [[C0]], [[C1]]
// CHECK:           return [[V0]], [[C1]] : index, index

// -----

func.func @to_select_with_body(%cond: i1) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.if %cond -> index {
    "test.op"() : () -> ()
    scf.yield %c0 : index
  } else {
    scf.yield %c1 : index
  }
  return %0 : index
}

// CHECK-LABEL:   func @to_select_with_body
// CHECK-DAG:       [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = arith.constant 1 : index
// CHECK:           [[V0:%.*]] = arith.select {{.*}}, [[C0]], [[C1]]
// CHECK:           scf.if {{.*}} {
// CHECK:             "test.op"() : () -> ()
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func.func @to_select2(%cond: i1) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0, %1 = scf.if %cond -> (index, index) {
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c2, %c3 : index, index
  }
  return %0, %1 : index, index
}

// CHECK-LABEL:   func @to_select2
// CHECK-DAG:       [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:       [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:       [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:       [[C3:%.*]] = arith.constant 3 : index
// CHECK:           [[V0:%.*]] = arith.select {{.*}}, [[C0]], [[C2]]
// CHECK:           [[V1:%.*]] = arith.select {{.*}}, [[C1]], [[C3]]
// CHECK:           return [[V0]], [[V1]] : index

// -----

func.func private @make_i32() -> i32

func.func @for_yields_2(%lb : index, %ub : index, %step : index) -> i32 {
  %a = call @make_i32() : () -> (i32)
  %b = scf.for %i = %lb to %ub step %step iter_args(%0 = %a) -> i32 {
    scf.yield %0 : i32
  }
  return %b : i32
}

// CHECK-LABEL:   func @for_yields_2
//  CHECK-NEXT:     %[[R:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     return %[[R]] : i32

// -----

func.func private @make_i32() -> i32

func.func @for_yields_3(%lb : index, %ub : index, %step : index) -> (i32, i32, i32) {
  %a = call @make_i32() : () -> (i32)
  %b = call @make_i32() : () -> (i32)
  %r:3 = scf.for %i = %lb to %ub step %step iter_args(%0 = %a, %1 = %a, %2 = %b) -> (i32, i32, i32) {
    %c = func.call @make_i32() : () -> (i32)
    scf.yield %0, %c, %2 : i32, i32, i32
  } {some_attr}
  return %r#0, %r#1, %r#2 : i32, i32, i32
}

// CHECK-LABEL:   func @for_yields_3
//  CHECK-NEXT:     %[[a:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     %[[b:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     %[[r1:.*]] = scf.for {{.*}} iter_args(%arg4 = %[[a]]) -> (i32) {
//  CHECK-NEXT:       %[[c:.*]] = func.call @make_i32() : () -> i32
//  CHECK-NEXT:       scf.yield %[[c]] : i32
//  CHECK-NEXT:     } {some_attr}
//  CHECK-NEXT:     return %[[a]], %[[r1]], %[[b]] : i32, i32, i32

// -----

// Test that an empty loop which iterates at least once and only returns
// values defined outside of the loop is folded away.
func.func @for_yields_4() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %a = arith.constant 3 : i32
  %b = arith.constant 4 : i32
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r : i32
}

// CHECK-LABEL:   func @for_yields_4
//  CHECK-NEXT:     %[[b:.*]] = arith.constant 4 : i32
//  CHECK-NEXT:     return %[[b]] : i32

// -----

// CHECK-LABEL: @replace_true_if
func.func @replace_true_if() {
  %true = arith.constant true
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
func.func @remove_false_if() {
  %false = arith.constant false
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
func.func @replace_true_if_with_values() {
  %true = arith.constant true
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
func.func @replace_false_if_with_values() {
  %false = arith.constant false
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

// CHECK-LABEL: @merge_nested_if
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1)
func.func @merge_nested_if(%arg0: i1, %arg1: i1) {
// CHECK: %[[COND:.*]] = arith.andi %[[ARG0]], %[[ARG1]]
// CHECK: scf.if %[[COND]] {
// CHECK-NEXT: "test.op"()
  scf.if %arg0 {
    scf.if %arg1 {
      "test.op"() : () -> ()
      scf.yield
    }
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: @merge_yielding_nested_if
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1)
func.func @merge_yielding_nested_if(%arg0: i1, %arg1: i1) -> (i32, f32, i32, i8) {
// CHECK: %[[PRE0:.*]] = "test.op"() : () -> i32
// CHECK: %[[PRE1:.*]] = "test.op1"() : () -> f32
// CHECK: %[[PRE2:.*]] = "test.op2"() : () -> i32
// CHECK: %[[PRE3:.*]] = "test.op3"() : () -> i8
// CHECK: %[[COND:.*]] = arith.andi %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]]:2 = scf.if %[[COND]] -> (f32, i32)
// CHECK:   %[[IN0:.*]] = "test.inop"() : () -> i32
// CHECK:   %[[IN1:.*]] = "test.inop1"() : () -> f32
// CHECK:   scf.yield %[[IN1]], %[[IN0]] : f32, i32
// CHECK: } else {
// CHECK:   scf.yield %[[PRE1]], %[[PRE2]] : f32, i32
// CHECK: }
// CHECK: return %[[PRE0]], %[[RES]]#0, %[[RES]]#1, %[[PRE3]] : i32, f32, i32, i8
  %0 = "test.op"() : () -> (i32)
  %1 = "test.op1"() : () -> (f32)
  %2 = "test.op2"() : () -> (i32)
  %3 = "test.op3"() : () -> (i8)
  %r:4 = scf.if %arg0 -> (i32, f32, i32, i8) {
    %a:2 = scf.if %arg1 -> (i32, f32) {
      %i = "test.inop"() : () -> (i32)
      %i1 = "test.inop1"() : () -> (f32)
      scf.yield %i, %i1 : i32, f32
    } else {
      scf.yield %2, %1 : i32, f32
    }
    scf.yield %0, %a#1, %a#0, %3 : i32, f32, i32, i8
  } else {
    scf.yield %0, %1, %2, %3 : i32, f32, i32, i8
  }
  return %r#0, %r#1, %r#2, %r#3 : i32, f32, i32, i8
}

// -----

// CHECK-LABEL: @merge_yielding_nested_if_nv1
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1)
func.func @merge_yielding_nested_if_nv1(%arg0: i1, %arg1: i1) {
// CHECK: %[[PRE0:.*]] = "test.op"() : () -> i32
// CHECK: %[[PRE1:.*]] = "test.op1"() : () -> f32
// CHECK: %[[COND:.*]] = arith.andi %[[ARG0]], %[[ARG1]]
// CHECK: scf.if %[[COND]]
// CHECK:   %[[IN0:.*]] = "test.inop"() : () -> i32
// CHECK:   %[[IN1:.*]] = "test.inop1"() : () -> f32
// CHECK: }
  %0 = "test.op"() : () -> (i32)
  %1 = "test.op1"() : () -> (f32)
  scf.if %arg0 {
    %a:2 = scf.if %arg1 -> (i32, f32) {
      %i = "test.inop"() : () -> (i32)
      %i1 = "test.inop1"() : () -> (f32)
      scf.yield %i, %i1 : i32, f32
    } else {
      scf.yield %0, %1 : i32, f32
    }
  }
  return
}

// -----

// CHECK-LABEL: @merge_yielding_nested_if_nv2
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1)
func.func @merge_yielding_nested_if_nv2(%arg0: i1, %arg1: i1) -> i32 {
// CHECK: %[[PRE0:.*]] = "test.op"() : () -> i32
// CHECK: %[[PRE1:.*]] = "test.op1"() : () -> i32
// CHECK: %[[COND:.*]] = arith.andi %[[ARG0]], %[[ARG1]]
// CHECK: %[[RES:.*]] = arith.select %[[ARG0]], %[[PRE0]], %[[PRE1]]
// CHECK: scf.if %[[COND]]
// CHECK:   "test.run"() : () -> ()
// CHECK: }
// CHECK: return %[[RES]]
  %0 = "test.op"() : () -> (i32)
  %1 = "test.op1"() : () -> (i32)
  %r = scf.if %arg0 -> i32 {
    scf.if %arg1 {
      "test.run"() : () -> ()
    }
    scf.yield %0 : i32
  } else {
    scf.yield %1 : i32
  }
  return %r : i32
}

// -----

// CHECK-LABEL: @merge_fail_yielding_nested_if
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1)
func.func @merge_fail_yielding_nested_if(%arg0: i1, %arg1: i1) -> (i32, f32, i32, i8) {
// CHECK-NOT: andi
  %0 = "test.op"() : () -> (i32)
  %1 = "test.op1"() : () -> (f32)
  %2 = "test.op2"() : () -> (i32)
  %3 = "test.op3"() : () -> (i8)
  %r:4 = scf.if %arg0 -> (i32, f32, i32, i8) {
    %a:2 = scf.if %arg1 -> (i32, f32) {
      %i = "test.inop"() : () -> (i32)
      %i1 = "test.inop1"() : () -> (f32)
      scf.yield %i, %i1 : i32, f32
    } else {
      scf.yield %0, %1 : i32, f32
    }
    scf.yield %0, %a#1, %a#0, %3 : i32, f32, i32, i8
  } else {
    scf.yield %0, %1, %2, %3 : i32, f32, i32, i8
  }
  return %r#0, %r#1, %r#2, %r#3 : i32, f32, i32, i8
}

// -----

// CHECK-LABEL:   func @if_condition_swap
// CHECK-NEXT:     %{{.*}} = scf.if %arg0 -> (index) {
// CHECK-NEXT:       %[[i1:.+]] = "test.origFalse"() : () -> index
// CHECK-NEXT:       scf.yield %[[i1]] : index
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[i2:.+]] = "test.origTrue"() : () -> index
// CHECK-NEXT:       scf.yield %[[i2]] : index
// CHECK-NEXT:     }
func.func @if_condition_swap(%cond: i1) -> index {
  %true = arith.constant true
  %not = arith.xori %cond, %true : i1
  %0 = scf.if %not -> (index) {
    %1 = "test.origTrue"() : () -> index
    scf.yield %1 : index
  } else {
    %1 = "test.origFalse"() : () -> index
    scf.yield %1 : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: @remove_zero_iteration_loop
func.func @remove_zero_iteration_loop() {
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
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
func.func @remove_zero_iteration_loop_vals(%arg0: index) {
  %c2 = arith.constant 2 : index
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
func.func @replace_single_iteration_loop_1() {
  // CHECK: %[[LB:.*]] = arith.constant 42
  %c42 = arith.constant 42 : index
  %c43 = arith.constant 43 : index
  %c1 = arith.constant 1 : index
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
func.func @replace_single_iteration_loop_2() {
  // CHECK: %[[LB:.*]] = arith.constant 5
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c11 = arith.constant 11 : index
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
func.func @replace_single_iteration_loop_non_unit_step() {
  // CHECK: %[[LB:.*]] = arith.constant 42
  %c42 = arith.constant 42 : index
  %c47 = arith.constant 47 : index
  %c5 = arith.constant 5 : index
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

// CHECK-LABEL: func @replace_single_iteration_const_diff(
//  CHECK-SAME: %[[A0:.*]]: index)
func.func @replace_single_iteration_const_diff(%arg0 : index) {
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 2
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %5 = arith.addi %arg0, %c1 : index
  // CHECK-NOT: scf.for
  scf.for %arg2 = %arg0 to %5 step %c1 {
    // CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[A0]], %[[CST]]
    %7 = arith.muli %c2, %arg2 : index
    // CHECK-NEXT: "test.consume"(%[[MUL]])
    "test.consume"(%7) : (index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @remove_empty_parallel_loop
func.func @remove_empty_parallel_loop(%lb: index, %ub: index, %s: index) {
  // CHECK: %[[INIT:.*]] = "test.init"
  %init = "test.init"() : () -> f32
  // CHECK-NOT: scf.parallel
  // CHECK-NOT: test.produce
  // CHECK-NOT: test.transform
  %0 = scf.parallel (%i, %j, %k) = (%lb, %ub, %lb) to (%ub, %ub, %ub) step (%s, %s, %s) init(%init) -> f32 {
    %1 = "test.produce"() : () -> f32
    scf.reduce(%1 : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = "test.transform"(%lhs, %rhs) : (f32, f32) -> f32
      scf.reduce.return %2 : f32
    }
  }
  // CHECK: "test.consume"(%[[INIT]])
  "test.consume"(%0) : (f32) -> ()
  return
}

// -----

// CHECK-LABEL: fold_away_iter_with_no_use_and_yielded_input
//  CHECK-SAME:   %[[A0:[0-9a-z]*]]: i32
func.func @fold_away_iter_with_no_use_and_yielded_input(%arg0 : i32,
                    %ub : index, %lb : index, %step : index) -> (i32, i32) {
  // CHECK-NEXT: %[[C32:.*]] = arith.constant 32 : i32
  %cst = arith.constant 32 : i32
  // CHECK-NEXT: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[A0]]) -> (i32) {
  %0:2 = scf.for %arg1 = %lb to %ub step %step iter_args(%arg2 = %arg0, %arg3 = %cst)
    -> (i32, i32) {
    %1 = arith.addi %arg2, %cst : i32
    scf.yield %1, %cst : i32, i32
  }

  // CHECK: return %[[FOR_RES]], %[[C32]] : i32, i32
  return %0#0, %0#1 : i32, i32
}

// -----

// CHECK-LABEL: fold_away_iter_and_result_with_no_use
//  CHECK-SAME:   %[[A0:[0-9a-z]*]]: i32
func.func @fold_away_iter_and_result_with_no_use(%arg0 : i32,
                    %ub : index, %lb : index, %step : index) -> (i32) {
  %cst = arith.constant 32 : i32
  // CHECK: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[A0]]) -> (i32) {
  %0:2 = scf.for %arg1 = %lb to %ub step %step iter_args(%arg2 = %arg0, %arg3 = %cst)
    -> (i32, i32) {
    %1 = arith.addi %arg2, %cst : i32
    scf.yield %1, %1 : i32, i32
  }

  // CHECK: return %[[FOR_RES]] : i32
  return %0#0 : i32
}

// -----

func.func private @do(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @matmul_on_tensors(%t0: tensor<32x1024xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  %0 = tensor.cast %t0 : tensor<32x1024xf32> to tensor<?x?xf32>
  %1 = scf.for %i = %c0 to %c1024 step %c32 iter_args(%iter_t0 = %0) -> (tensor<?x?xf32>) {
    %2 = func.call @do(%iter_t0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.yield %2 : tensor<?x?xf32>
  } {some_attr}
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: matmul_on_tensors
//  CHECK-SAME:   %[[T0:[0-9a-z]*]]: tensor<32x1024xf32>

//   CHECK-NOT: tensor.cast
//       CHECK: %[[FOR_RES:.*]] = scf.for {{.*}} iter_args(%[[ITER_T0:.*]] = %[[T0]]) -> (tensor<32x1024xf32>) {
//       CHECK:   %[[CAST:.*]] = tensor.cast %[[ITER_T0]] : tensor<32x1024xf32> to tensor<?x?xf32>
//       CHECK:   %[[DONE:.*]] = func.call @do(%[[CAST]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
//       CHECK:   %[[UNCAST:.*]] = tensor.cast %[[DONE]] : tensor<?x?xf32> to tensor<32x1024xf32>
//       CHECK:   scf.yield %[[UNCAST]] : tensor<32x1024xf32>
//       CHECK: } {some_attr}
//       CHECK: %[[RES:.*]] = tensor.cast
//       CHECK: return %[[RES]] : tensor<?x?xf32>

// -----

// CHECK-LABEL: @cond_prop
func.func @cond_prop(%arg0 : i1) -> index {
  %res = scf.if %arg0 -> index {
    %res1 = scf.if %arg0 -> index {
      %v1 = "test.get_some_value1"() : () -> index
      scf.yield %v1 : index
    } else {
      %v2 = "test.get_some_value2"() : () -> index
      scf.yield %v2 : index
    }
    scf.yield %res1 : index
  } else {
    %res2 = scf.if %arg0 -> index {
      %v3 = "test.get_some_value3"() : () -> index
      scf.yield %v3 : index
    } else {
      %v4 = "test.get_some_value4"() : () -> index
      scf.yield %v4 : index
    }
    scf.yield %res2 : index
  }
  return %res : index
}
// CHECK-NEXT:  %[[if:.+]] = scf.if %arg0 -> (index) {
// CHECK-NEXT:    %[[c1:.+]] = "test.get_some_value1"() : () -> index
// CHECK-NEXT:    scf.yield %[[c1]] : index
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[c4:.+]] = "test.get_some_value4"() : () -> index
// CHECK-NEXT:    scf.yield %[[c4]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[if]] : index
// CHECK-NEXT:}

// -----

// CHECK-LABEL: @replace_if_with_cond1
func.func @replace_if_with_cond1(%arg0 : i1) -> (i32, i1) {
  %true = arith.constant true
  %false = arith.constant false
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

// -----

// CHECK-LABEL: @replace_if_with_cond2
func.func @replace_if_with_cond2(%arg0 : i1) -> (i32, i1) {
  %true = arith.constant true
  %false = arith.constant false
  %res:2 = scf.if %arg0 -> (i32, i1) {
    %v = "test.get_some_value"() : () -> i32
    scf.yield %v, %false : i32, i1
  } else {
    %v2 = "test.get_some_value"() : () -> i32
    scf.yield %v2, %true : i32, i1
  }
  return %res#0, %res#1 : i32, i1
}
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %[[toret:.+]] = arith.xori %arg0, %true : i1
// CHECK-NEXT:     %[[if:.+]] = scf.if %arg0 -> (i32) {
// CHECK-NEXT:       %[[sv1:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:       scf.yield %[[sv1]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[sv2:.+]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:       scf.yield %[[sv2]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[if]], %[[toret]] : i32, i1

// -----

// CHECK-LABEL: @replace_if_with_cond3
func.func @replace_if_with_cond3(%arg0 : i1, %arg2: i64) -> (i32, i64) {
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

// -----

// CHECK-LABEL: @while_cond_true
func.func @while_cond_true() -> i1 {
  %0 = scf.while () : () -> i1 {
    %condition = "test.condition"() : () -> i1
    scf.condition(%condition) %condition : i1
  } do {
  ^bb0(%arg0: i1):
    "test.use"(%arg0) : (i1) -> ()
    scf.yield
  }
  return %0 : i1
}
// CHECK-NEXT:         %[[true:.+]] = arith.constant true
// CHECK-NEXT:         %{{.+}} = scf.while : () -> i1 {
// CHECK-NEXT:           %[[cmp:.+]] = "test.condition"() : () -> i1
// CHECK-NEXT:           scf.condition(%[[cmp]]) %[[cmp]] : i1
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%arg0: i1):
// CHECK-NEXT:           "test.use"(%[[true]]) : (i1) -> ()
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }

// -----

// CHECK-LABEL: @invariant_loop_args_in_same_order
// CHECK-SAME: (%[[FUNC_ARG0:.*]]: tensor<i32>)
func.func @invariant_loop_args_in_same_order(%f_arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<1> : tensor<i32>
  %cst_42 = arith.constant dense<42> : tensor<i32>

  %0:5 = scf.while (%arg0 = %cst_0, %arg1 = %f_arg0, %arg2 = %cst_1, %arg3 = %cst_1, %arg4 = %cst_0) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
    %1 = arith.cmpi slt, %arg0, %cst_42 : tensor<i32>
    %2 = tensor.extract %1[] : tensor<i1>
    scf.condition(%2) %arg0, %arg1, %arg2, %arg3, %arg4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  } do {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>): // no predecessors
    // %arg1 here will get replaced by %cst_1
    %1 = arith.addi %arg0, %arg1 : tensor<i32>
    %2 = arith.addi %arg2, %arg3 : tensor<i32>
    scf.yield %1, %arg1, %2, %2, %arg4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}
// CHECK:    %[[ZERO:.*]] = arith.constant dense<0>
// CHECK:    %[[ONE:.*]] = arith.constant dense<1>
// CHECK:    %[[CST42:.*]] = arith.constant dense<42>
// CHECK:    %[[WHILE:.*]]:3 = scf.while (%[[ARG0:.*]] = %[[ZERO]], %[[ARG2:.*]] = %[[ONE]], %[[ARG3:.*]] = %[[ONE]])
// CHECK:       arith.cmpi slt, %[[ARG0]], %{{.*}}
// CHECK:       tensor.extract %{{.*}}[]
// CHECK:       scf.condition(%{{.*}}) %[[ARG0]], %[[ARG2]], %[[ARG3]]
// CHECK:    } do {
// CHECK:     ^{{.*}}(%[[ARG0:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>):
// CHECK:       %[[VAL0:.*]] = arith.addi %[[ARG0]], %[[FUNC_ARG0]]
// CHECK:       %[[VAL1:.*]] = arith.addi %[[ARG2]], %[[ARG3]]
// CHECK:       scf.yield %[[VAL0]], %[[VAL1]], %[[VAL1]]
// CHECK:    }
// CHECK:    return %[[WHILE]]#0, %[[FUNC_ARG0]], %[[WHILE]]#1, %[[WHILE]]#2, %[[ZERO]]

// CHECK-LABEL: @while_loop_invariant_argument_different_order
func.func @while_loop_invariant_argument_different_order(%arg : tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<1> : tensor<i32>
  %cst_42 = arith.constant dense<42> : tensor<i32>

  %0:6 = scf.while (%arg0 = %cst_0, %arg1 = %cst_1, %arg2 = %cst_1, %arg3 = %cst_1, %arg4 = %cst_0) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
    %1 = arith.cmpi slt, %arg0, %arg : tensor<i32>
    %2 = tensor.extract %1[] : tensor<i1>
    scf.condition(%2) %arg1, %arg0, %arg2, %arg0, %arg3, %arg4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  } do {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>): // no predecessors
    %1 = arith.addi %arg0, %cst_1 : tensor<i32>
    %2 = arith.addi %arg2, %arg3 : tensor<i32>
    scf.yield %arg3, %arg1, %2, %2, %arg4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}
// CHECK-SAME: (%[[ARG:.+]]: tensor<i32>)
// CHECK:    %[[ZERO:.*]] = arith.constant dense<0>
// CHECK:    %[[ONE:.*]] = arith.constant dense<1>
// CHECK:    %[[WHILE:.*]]:2 = scf.while (%[[ARG1:.*]] = %[[ONE]], %[[ARG4:.*]] = %[[ZERO]])
// CHECK:       arith.cmpi sgt, %[[ARG]], %[[ZERO]]
// CHECK:       tensor.extract %{{.*}}[]
// CHECK:       scf.condition(%{{.*}}) %[[ARG1]], %[[ARG4]]
// CHECK:    } do {
// CHECK:     ^{{.*}}(%{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>):
// CHECK:       scf.yield %[[ZERO]], %[[ONE]]
// CHECK:    }
// CHECK:    return %[[WHILE]]#0, %[[ZERO]], %[[ONE]], %[[ZERO]], %[[ONE]], %[[WHILE]]#1

// -----

// CHECK-LABEL: @while_unused_result
func.func @while_unused_result() -> i32 {
  %0:2 = scf.while () : () -> (i32, i64) {
    %condition = "test.condition"() : () -> i1
    %v1 = "test.get_some_value"() : () -> i32
    %v2 = "test.get_some_value"() : () -> i64
    scf.condition(%condition) %v1, %v2 : i32, i64
  } do {
  ^bb0(%arg0: i32, %arg1: i64):
    "test.use"(%arg0) : (i32) -> ()
    scf.yield
  }
  return %0#0 : i32
}
// CHECK-NEXT:         %[[res:.*]] = scf.while : () -> i32 {
// CHECK-NEXT:           %[[cmp:.*]] = "test.condition"() : () -> i1
// CHECK-NEXT:           %[[val:.*]] = "test.get_some_value"() : () -> i32
// CHECK-NEXT:           %{{.*}} = "test.get_some_value"() : () -> i64
// CHECK-NEXT:           scf.condition(%[[cmp]]) %[[val]] : i32
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%[[arg:.*]]: i32):
// CHECK-NEXT:           "test.use"(%[[arg]]) : (i32) -> ()
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         return %[[res]] : i32

// -----

// CHECK-LABEL: @while_cmp_lhs
func.func @while_cmp_lhs(%arg0 : i32) {
  %0 = scf.while () : () -> i32 {
    %val = "test.val"() : () -> i32
    %condition = arith.cmpi ne, %val, %arg0 : i32
    scf.condition(%condition) %val : i32
  } do {
  ^bb0(%val2: i32):
    %condition2 = arith.cmpi ne, %val2, %arg0 : i32
    %negcondition2 = arith.cmpi eq, %val2, %arg0 : i32
    "test.use"(%condition2, %negcondition2, %val2) : (i1, i1, i32) -> ()
    scf.yield
  }
  return
}
// CHECK-DAG:         %[[true:.+]] = arith.constant true
// CHECK-DAG:         %[[false:.+]] = arith.constant false
// CHECK-DAG:         %{{.+}} = scf.while : () -> i32 {
// CHECK-NEXT:         %[[val:.+]] = "test.val"
// CHECK-NEXT:         %[[cmp:.+]] = arith.cmpi ne, %[[val]], %arg0 : i32
// CHECK-NEXT:           scf.condition(%[[cmp]]) %[[val]] : i32
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%arg1: i32):
// CHECK-NEXT:           "test.use"(%[[true]], %[[false]], %arg1) : (i1, i1, i32) -> ()
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }

// -----

// CHECK-LABEL: @while_cmp_rhs
func.func @while_cmp_rhs(%arg0 : i32) {
  %0 = scf.while () : () -> i32 {
    %val = "test.val"() : () -> i32
    %condition = arith.cmpi ne, %arg0, %val : i32
    scf.condition(%condition) %val : i32
  } do {
  ^bb0(%val2: i32):
    %condition2 = arith.cmpi ne, %arg0, %val2 : i32
    %negcondition2 = arith.cmpi eq, %arg0, %val2 : i32
    "test.use"(%condition2, %negcondition2, %val2) : (i1, i1, i32) -> ()
    scf.yield
  }
  return
}
// CHECK-DAG:         %[[true:.+]] = arith.constant true
// CHECK-DAG:         %[[false:.+]] = arith.constant false
// CHECK-DAG:         %{{.+}} = scf.while : () -> i32 {
// CHECK-NEXT:         %[[val:.+]] = "test.val"
// CHECK-NEXT:         %[[cmp:.+]] = arith.cmpi ne, %arg0, %[[val]] : i32
// CHECK-NEXT:           scf.condition(%[[cmp]]) %[[val]] : i32
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%arg1: i32):
// CHECK-NEXT:           "test.use"(%[[true]], %[[false]], %arg1) : (i1, i1, i32) -> ()
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }

// -----

// CHECK-LABEL: @while_duplicated_res
func.func @while_duplicated_res() -> (i32, i32) {
  %0:2 = scf.while () : () -> (i32, i32) {
    %val = "test.val"() : () -> i32
    %condition = "test.condition"() : () -> i1
    scf.condition(%condition) %val, %val : i32, i32
  } do {
  ^bb0(%val2: i32, %val3: i32):
    "test.use"(%val2, %val3) : (i32, i32) -> ()
    scf.yield
  }
  return %0#0, %0#1: i32, i32
}
// CHECK:         %[[RES:.*]] = scf.while : () -> i32 {
// CHECK:         %[[VAL:.*]] = "test.val"() : () -> i32
// CHECK:         %[[COND:.*]] = "test.condition"() : () -> i1
// CHECK:           scf.condition(%[[COND]]) %[[VAL]] : i32
// CHECK:         } do {
// CHECK:         ^bb0(%[[ARG:.*]]: i32):
// CHECK:           "test.use"(%[[ARG]], %[[ARG]]) : (i32, i32) -> ()
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         return %[[RES]], %[[RES]] : i32, i32


// -----

// CHECK-LABEL: @while_unused_arg1
func.func @while_unused_arg1(%x : i32, %y : f64) -> i32 {
  %0 = scf.while (%arg1 = %x, %arg2 = %y) : (i32, f64) -> (i32) {
    %condition = "test.condition"(%arg1) : (i32) -> i1
    scf.condition(%condition) %arg1 : i32
  } do {
  ^bb0(%arg1: i32):
    %next = "test.use"(%arg1) : (i32) -> (i32)
    scf.yield %next, %y : i32, f64
  }
  return %0 : i32
}
// CHECK-NEXT:         %[[res:.*]] = scf.while (%[[arg2:.*]] = %{{.*}}) : (i32) -> i32 {
// CHECK-NEXT:           %[[cmp:.*]] = "test.condition"(%[[arg2]]) : (i32) -> i1
// CHECK-NEXT:           scf.condition(%[[cmp]]) %[[arg2]] : i32
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%[[post:.*]]: i32):
// CHECK-NEXT:           %[[next:.*]] = "test.use"(%[[post]]) : (i32) -> i32
// CHECK-NEXT:           scf.yield %[[next]] : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         return %[[res]] : i32


// -----

// CHECK-LABEL: @while_unused_arg2
func.func @while_unused_arg2(%val0: i32) -> i32 {
  %0 = scf.while (%val1 = %val0) : (i32) -> i32 {
    %val = "test.val"() : () -> i32
    %condition = "test.condition"() : () -> i1
    scf.condition(%condition) %val: i32
  } do {
  ^bb0(%val2: i32):
    "test.use"(%val2) : (i32) -> ()
    %val1 = "test.val1"() : () -> i32
    scf.yield %val1 : i32
  }
  return %0 : i32
}
// CHECK:         %[[RES:.*]] = scf.while : () -> i32 {
// CHECK:         %[[VAL:.*]] = "test.val"() : () -> i32
// CHECK:         %[[COND:.*]] = "test.condition"() : () -> i1
// CHECK:           scf.condition(%[[COND]]) %[[VAL]] : i32
// CHECK:         } do {
// CHECK:         ^bb0(%[[ARG:.*]]: i32):
// CHECK:           "test.use"(%[[ARG]]) : (i32) -> ()
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         return %[[RES]] : i32


// -----

// CHECK-LABEL: func @test_align_args
//       CHECK:  %[[RES:.*]]:3 = scf.while (%[[ARG0:.*]] = %{{.*}}, %[[ARG1:.*]] = %{{.*}}, %[[ARG2:.*]] = %{{.*}}) : (f32, i32, i64) -> (f32, i32, i64) {
//       CHECK:  scf.condition(%{{.*}}) %[[ARG0]], %[[ARG1]], %[[ARG2]] : f32, i32, i64
//       CHECK:  ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i64):
//       CHECK:  %[[R1:.*]] = "test.test"(%[[ARG5]]) : (i64) -> f32
//       CHECK:  %[[R2:.*]] = "test.test"(%[[ARG3]]) : (f32) -> i32
//       CHECK:  %[[R3:.*]] = "test.test"(%[[ARG4]]) : (i32) -> i64
//       CHECK:  scf.yield %[[R1]], %[[R2]], %[[R3]] : f32, i32, i64
//       CHECK:  return %[[RES]]#2, %[[RES]]#0, %[[RES]]#1
func.func @test_align_args() -> (i64, f32, i32) {
  %0 = "test.test"() : () -> (f32)
  %1 = "test.test"() : () -> (i32)
  %2 = "test.test"() : () -> (i64)
  %3:3 = scf.while (%arg0 = %0, %arg1 = %1, %arg2 = %2) : (f32, i32, i64) -> (i64, f32, i32) {
    %cond = "test.test"() : () -> (i1)
    scf.condition(%cond) %arg2, %arg0, %arg1 : i64, f32, i32
  } do {
  ^bb0(%arg3: i64, %arg4: f32, %arg5: i32):
    %4 = "test.test"(%arg3) : (i64) -> (f32)
    %5 = "test.test"(%arg4) : (f32) -> (i32)
    %6 = "test.test"(%arg5) : (i32) -> (i64)
    scf.yield %4, %5, %6 : f32, i32, i64
  }
  return %3#0, %3#1, %3#2 : i64, f32, i32
}


// -----

// CHECK-LABEL: @combineIfs
func.func @combineIfs(%arg0 : i1, %arg2: i64) -> (i32, i32) {
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

// -----

// CHECK-LABEL: @combineIfs2
func.func @combineIfs2(%arg0 : i1, %arg2: i64) -> i32 {
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

// -----

// CHECK-LABEL: @combineIfs3
func.func @combineIfs3(%arg0 : i1, %arg2: i64) -> i32 {
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

// -----

// CHECK-LABEL: @combineIfs4
func.func @combineIfs4(%arg0 : i1, %arg2: i64) {
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

// -----

// CHECK-LABEL: @combineIfsUsed
// CHECK-SAME: %[[arg0:.+]]: i1
func.func @combineIfsUsed(%arg0 : i1, %arg2: i64) -> (i32, i32) {
  %res = scf.if %arg0 -> i32 {
    %v = "test.firstCodeTrue"() : () -> i32
    scf.yield %v : i32
  } else {
    %v2 = "test.firstCodeFalse"() : () -> i32
    scf.yield %v2 : i32
  }
  %res2 = scf.if %arg0 -> i32 {
    %v = "test.secondCodeTrue"(%res) : (i32) -> i32
    scf.yield %v : i32
  } else {
    %v2 = "test.secondCodeFalse"(%res) : (i32) -> i32
    scf.yield %v2 : i32
  }
  return %res, %res2 : i32, i32
}
// CHECK-NEXT:     %[[res:.+]]:2 = scf.if %[[arg0]] -> (i32, i32) {
// CHECK-NEXT:       %[[tval0:.+]] = "test.firstCodeTrue"() : () -> i32
// CHECK-NEXT:       %[[tval:.+]] = "test.secondCodeTrue"(%[[tval0]]) : (i32) -> i32
// CHECK-NEXT:       scf.yield %[[tval0]], %[[tval]] : i32, i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[fval0:.+]] = "test.firstCodeFalse"() : () -> i32
// CHECK-NEXT:       %[[fval:.+]] = "test.secondCodeFalse"(%[[fval0]]) : (i32) -> i32
// CHECK-NEXT:       scf.yield %[[fval0]], %[[fval]] : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[res]]#0, %[[res]]#1 : i32, i32

// -----

// CHECK-LABEL: @combineIfsNot
// CHECK-SAME: %[[arg0:.+]]: i1
func.func @combineIfsNot(%arg0 : i1, %arg2: i64) {
  %true = arith.constant true
  %not = arith.xori %arg0, %true : i1
  scf.if %arg0 {
    "test.firstCodeTrue"() : () -> ()
    scf.yield
  }
  scf.if %not {
    "test.secondCodeTrue"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-NEXT:     scf.if %[[arg0]] {
// CHECK-NEXT:       "test.firstCodeTrue"() : () -> ()
// CHECK-NEXT:     } else {
// CHECK-NEXT:       "test.secondCodeTrue"() : () -> ()
// CHECK-NEXT:     }

// -----

// CHECK-LABEL: @combineIfsNot2
// CHECK-SAME: %[[arg0:.+]]: i1
func.func @combineIfsNot2(%arg0 : i1, %arg2: i64) {
  %true = arith.constant true
  %not = arith.xori %arg0, %true : i1
  scf.if %not {
    "test.firstCodeTrue"() : () -> ()
    scf.yield
  }
  scf.if %arg0 {
    "test.secondCodeTrue"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-NEXT:     scf.if %[[arg0]] {
// CHECK-NEXT:       "test.secondCodeTrue"() : () -> ()
// CHECK-NEXT:     } else {
// CHECK-NEXT:       "test.firstCodeTrue"() : () -> ()
// CHECK-NEXT:     }

// -----

// CHECK-LABEL: func @propagate_into_execute_region
func.func @propagate_into_execute_region() {
  %cond = arith.constant 0 : i1
  affine.for %i = 0 to 100 {
    "test.foo"() : () -> ()
    %v = scf.execute_region -> i64 {
      cf.cond_br %cond, ^bb1, ^bb2

    ^bb1:
      %c1 = arith.constant 1 : i64
      cf.br ^bb3(%c1 : i64)

    ^bb2:
      %c2 = arith.constant 2 : i64
      cf.br ^bb3(%c2 : i64)

    ^bb3(%x : i64):
      scf.yield %x : i64
    }
    "test.bar"(%v) : (i64) -> ()
    // CHECK:      %[[C2:.*]] = arith.constant 2 : i64
    // CHECK: "test.foo"
    // CHECK-NEXT: "test.bar"(%[[C2]]) : (i64) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @execute_region_elim
func.func @execute_region_elim() {
  affine.for %i = 0 to 100 {
    "test.foo"() : () -> ()
    %v = scf.execute_region -> i64 {
      %x = "test.val"() : () -> i64
      scf.yield %x : i64
    }
    "test.bar"(%v) : (i64) -> ()
  }
  return
}

// CHECK-NEXT:     affine.for %arg0 = 0 to 100 {
// CHECK-NEXT:       "test.foo"() : () -> ()
// CHECK-NEXT:       %[[VAL:.*]] = "test.val"() : () -> i64
// CHECK-NEXT:       "test.bar"(%[[VAL]]) : (i64) -> ()
// CHECK-NEXT:     }

// -----

// CHECK-LABEL: func @func_execute_region_elim
func.func @func_execute_region_elim() {
    "test.foo"() : () -> ()
    %v = scf.execute_region -> i64 {
      %c = "test.cmp"() : () -> i1
      cf.cond_br %c, ^bb2, ^bb3
    ^bb2:
      %x = "test.val1"() : () -> i64
      cf.br ^bb4(%x : i64)
    ^bb3:
      %y = "test.val2"() : () -> i64
      cf.br ^bb4(%y : i64)
    ^bb4(%z : i64):
      scf.yield %z : i64
    }
    "test.bar"(%v) : (i64) -> ()
  return
}

// CHECK-NOT: execute_region
// CHECK:     "test.foo"
// CHECK:     %[[cmp:.+]] = "test.cmp"
// CHECK:     cf.cond_br %[[cmp]], ^[[bb1:.+]], ^[[bb2:.+]]
// CHECK:   ^[[bb1]]:
// CHECK:     %[[x:.+]] = "test.val1"
// CHECK:     cf.br ^[[bb3:.+]](%[[x]] : i64)
// CHECK:   ^[[bb2]]:
// CHECK:     %[[y:.+]] = "test.val2"
// CHECK:     cf.br ^[[bb3]](%[[y:.+]] : i64)
// CHECK:   ^[[bb3]](%[[z:.+]]: i64):
// CHECK:     "test.bar"(%[[z]])
// CHECK:     return

// -----

// CHECK-LABEL: func @func_execute_region_elim_multi_yield
func.func @func_execute_region_elim_multi_yield() {
    "test.foo"() : () -> ()
    %v = scf.execute_region -> i64 {
      %c = "test.cmp"() : () -> i1
      cf.cond_br %c, ^bb2, ^bb3
    ^bb2:
      %x = "test.val1"() : () -> i64
      scf.yield %x : i64
    ^bb3:
      %y = "test.val2"() : () -> i64
      scf.yield %y : i64
    }
    "test.bar"(%v) : (i64) -> ()
  return
}

// CHECK-NOT: execute_region
// CHECK:     "test.foo"
// CHECK:     %[[cmp:.+]] = "test.cmp"
// CHECK:     cf.cond_br %[[cmp]], ^[[bb1:.+]], ^[[bb2:.+]]
// CHECK:   ^[[bb1]]:
// CHECK:     %[[x:.+]] = "test.val1"
// CHECK:     cf.br ^[[bb3:.+]](%[[x]] : i64)
// CHECK:   ^[[bb2]]:
// CHECK:     %[[y:.+]] = "test.val2"
// CHECK:     cf.br ^[[bb3]](%[[y:.+]] : i64)
// CHECK:   ^[[bb3]](%[[z:.+]]: i64):
// CHECK:     "test.bar"(%[[z]])
// CHECK:     return

// -----

// CHECK-LABEL: func @canonicalize_parallel_insert_slice_indices(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<1x5xf32>, %[[arg1:.*]]: tensor<?x?xf32>
func.func @canonicalize_parallel_insert_slice_indices(
    %arg0 : tensor<1x5xf32>, %arg1: tensor<?x?xf32>, %num_threads : index) -> index
{
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index

  %2 = scf.forall (%tidx) in (%num_threads) shared_outs(%o = %arg1) -> (tensor<?x?xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %o[%tidx, 0] [1, 5] [1, 1] : tensor<1x5xf32> into tensor<?x?xf32>
    }
  }

  // CHECK: %[[dim:.*]] = tensor.dim %[[arg1]], %[[c1]]
  %dim = tensor.dim %2, %c1 : tensor<?x?xf32>
  // CHECK: return %[[dim]]
  return %dim : index
}

// -----

// CHECK-LABEL: func @forall_fold_control_operands
func.func @forall_fold_control_operands(
    %arg0 : tensor<?x10xf32>, %arg1: tensor<?x10xf32>) -> tensor<?x10xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x10xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x10xf32>

  %result = scf.forall (%i, %j) = (%c0, %c0) to (%dim0, %dim1)
      step (%c1, %c1) shared_outs(%o = %arg1) -> (tensor<?x10xf32>) {
    %slice = tensor.extract_slice %arg1[%i, %j] [1, 1] [1, 1]
      : tensor<?x10xf32> to tensor<1x1xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %o[%i, %j] [1, 1] [1, 1]
        : tensor<1x1xf32> into tensor<?x10xf32>
    }
  }

  return %result : tensor<?x10xf32>
}
// CHECK: forall (%{{.*}}, %{{.*}}) in (%{{.*}}, 10)

// -----

func.func @inline_forall_loop(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = scf.forall (%i, %j) = (%c0, %c0) to (%c1, %c1)
        step (%c8, %c8) shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %slice = tensor.extract_slice %out_[%i, %j] [2, 3] [1, 1]
      : tensor<8x8xf32> to tensor<2x3xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%slice : tensor<2x3xf32>)
          -> tensor<2x3xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %fill into %out_[%i, %j] [2, 3] [1, 1]
        : tensor<2x3xf32> into tensor<8x8xf32>
    }
  }
  return %1 : tensor<8x8xf32>
}
// CHECK-LABEL: @inline_forall_loop
// CHECK-NOT:     scf.forall
// CHECK:         %[[OUT:.*]] = tensor.empty

// CHECK-NEXT:    %[[SLICE:.*]] = tensor.extract_slice %[[OUT]]
// CHECK-SAME:      : tensor<8x8xf32> to tensor<2x3xf32>

// CHECK-NEXT:    %[[FILL:.*]] = linalg.fill
// CHECK-SAME:      outs(%[[SLICE]]

// CHECK-NEXT:    tensor.insert_slice %[[FILL]]
// CHECK-SAME:      : tensor<2x3xf32> into tensor<8x8xf32>

// -----

func.func @do_not_inline_distributed_forall_loop(
    %in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = scf.forall (%i, %j) = (0, 0) to (1, 1) step (8, 8)
      shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %slice = tensor.extract_slice %out_[%i, %j] [2, 3] [1, 1]
      : tensor<8x8xf32> to tensor<2x3xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%slice : tensor<2x3xf32>)
          -> tensor<2x3xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %fill into %out_[%i, %j] [2, 3] [1, 1]
        : tensor<2x3xf32> into tensor<8x8xf32>
    }
  }{ mapping = [#gpu.thread<y>, #gpu.thread<x>] }
  return %1 : tensor<8x8xf32>
}
// CHECK-LABEL: @do_not_inline_distributed_forall_loop
// CHECK: scf.forall

// -----

func.func @inline_empty_loop_with_empty_mapping(
    %in: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16xf32>
  %1 = scf.forall () in () shared_outs (%out_ = %0) -> (tensor<16xf32>) {
    %slice = tensor.extract_slice %out_[0] [16] [1]
      : tensor<16xf32> to tensor<16xf32>
    %generic = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%slice : tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %2 = arith.addf %b0, %b0 : f32
        linalg.yield %2 : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %generic into %out_[0] [16] [1]
        : tensor<16xf32> into tensor<16xf32>
    }
  }{ mapping = [] }
  return %1 : tensor<16xf32>
}
// CHECK-LABEL: func @inline_empty_loop_with_empty_mapping
//   CHECK-NOT:   scf.forall

// -----

func.func @collapse_one_dim_parallel(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = scf.forall (%i, %j) = (0, %c0) to (1, %c16)
        step (8, %c8) shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %fill = linalg.fill ins(%cst : f32) outs(%out_ : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %fill into %out_[%i, %j] [8, 8] [1, 1]
        : tensor<8x8xf32> into tensor<8x8xf32>
    }
  }
  return %1 : tensor<8x8xf32>
}
// CHECK-LABEL: @collapse_one_dim_parallel
// CHECK:         scf.forall (%[[ARG:.*]]) = (0) to (16) step (8)
// CHECK:           linalg.fill
// CHECK:           tensor.parallel_insert_slice

// -----

func.func @remove_empty_forall(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = scf.forall (%i, %j) = (%c0, %c16) to (%c1, %c16)
        step (%c8, %c8) shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %fill = linalg.fill ins(%cst : f32) outs(%out_ : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %fill into %out_[%i, %j] [8, 8] [1, 1]
        : tensor<8x8xf32> into tensor<8x8xf32>
    }
  }
  return %1 : tensor<8x8xf32>
}
// CHECK-LABEL: @remove_empty_forall
// CHECK-NOT:   scf.forall
// CHECK:       %[[EMPTY:.*]] = tensor.empty
// CHECK:       return %[[EMPTY]]

// -----

func.func @fold_tensor_cast_into_forall(
    %in: tensor<2xi32>, %out: tensor<2xi32>) -> tensor<2xi32> {
  %cst = arith.constant dense<[100500]> : tensor<1xi32>


  %out_cast = tensor.cast %out : tensor<2xi32> to tensor<?xi32>
  %result = scf.forall (%i) = (0) to (2) step (1)
      shared_outs (%out_ = %out_cast) -> tensor<?xi32> {

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %cst into %out_[%i] [1] [1]
        : tensor<1xi32> into tensor<?xi32>
    }
  }
  %result_cast = tensor.cast %result : tensor<?xi32> to tensor<2xi32>
  func.return %result_cast : tensor<2xi32>
}
// CHECK-LABEL: @fold_tensor_cast_into_forall
// CHECK-NOT:     tensor.cast
// CHECK:         parallel_insert_slice
// CHECK-SAME:      : tensor<1xi32> into tensor<2xi32>
// CHECK-NOT:     tensor.cast

// -----

func.func @do_not_fold_tensor_cast_from_dynamic_to_static_type_into_forall(
    %in: tensor<?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> {
  %cst = arith.constant dense<[100500]> : tensor<1xi32>


  %out_cast = tensor.cast %out : tensor<?xi32> to tensor<2xi32>
  %result = scf.forall (%i) = (0) to (2) step (1)
      shared_outs (%out_ = %out_cast) -> tensor<2xi32> {

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %cst into %out_[%i] [1] [1]
        : tensor<1xi32> into tensor<2xi32>
    }
  }
  %result_cast = tensor.cast %result : tensor<2xi32> to tensor<?xi32>
  func.return %result_cast : tensor<?xi32>
}
// CHECK-LABEL: @do_not_fold_tensor_cast_
// CHECK:         tensor.cast
// CHECK:         parallel_insert_slice
// CHECK-SAME:      : tensor<1xi32> into tensor<2xi32>
// CHECK:         tensor.cast

// -----

#map = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>
module {
  func.func @fold_iter_args_not_being_modified_within_scfforall(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 4.200000e+01 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map()[%dim, %arg0]
    %2:2 = scf.forall (%arg3) in (%1) shared_outs(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<?xf32>, tensor<?xf32>) {
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%dim, %arg0]
      %extracted_slice0 = tensor.extract_slice %arg4[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice1 = tensor.extract_slice %arg5[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %5 = linalg.elemwise_unary ins(%extracted_slice0 : tensor<?xf32>) outs(%extracted_slice1 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg5[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    return %2#0, %2#1 : tensor<?xf32>, tensor<?xf32>
  }
}
// CHECK-LABEL: @fold_iter_args_not_being_modified_within_scfforall
//  CHECK-SAME:   (%{{.*}}: index, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
//       CHECK:    %[[RESULT:.*]] = scf.forall 
//  CHECK-SAME:                       shared_outs(%[[ITER_ARG_5:.*]] = %[[ARG2]]) -> (tensor<?xf32>) {
//       CHECK:      %[[OPERAND0:.*]] = tensor.extract_slice %[[ARG1]]
//       CHECK:      %[[OPERAND1:.*]] = tensor.extract_slice %[[ITER_ARG_5]]
//       CHECK:      %[[ELEM:.*]] = linalg.elemwise_unary ins(%[[OPERAND0]] : tensor<?xf32>) outs(%[[OPERAND1]] : tensor<?xf32>) -> tensor<?xf32>
//       CHECK:      scf.forall.in_parallel {
//  CHECK-NEXT:         tensor.parallel_insert_slice %[[ELEM]] into %[[ITER_ARG_5]]
//  CHECK-NEXT:      }
//  CHECK-NEXT:    }
//  CHECK-NEXT:    return %[[ARG1]], %[[RESULT]]

// -----

#map = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>
module {
  func.func @fold_iter_args_with_no_use_of_result_scfforall(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map()[%dim, %arg0]
    %2:3 = scf.forall (%arg4) in (%1) shared_outs(%arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      %3 = affine.apply #map1(%arg4)[%arg0]
      %4 = affine.min #map2(%arg4)[%dim, %arg0]
      %extracted_slice = tensor.extract_slice %arg5[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %arg6[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_2 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %5 = linalg.elemwise_unary ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_1 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg6[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %extracted_slice into %arg5[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %extracted_slice_0 into %arg7[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %5 into %arg7[%4] [%3] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    return %2#1 : tensor<?xf32>
  }
}
// CHECK-LABEL: @fold_iter_args_with_no_use_of_result_scfforall
//  CHECK-SAME:   (%{{.*}}: index, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>, %[[ARG3:.*]]: tensor<?xf32>) -> tensor<?xf32> {
//       CHECK:    %[[RESULT:.*]] = scf.forall 
//  CHECK-SAME:                       shared_outs(%[[ITER_ARG_6:.*]] = %[[ARG2]]) -> (tensor<?xf32>) {
//       CHECK:      %[[OPERAND0:.*]] = tensor.extract_slice %[[ARG1]]
//       CHECK:      %[[OPERAND1:.*]] = tensor.extract_slice %[[ARG3]]
//       CHECK:      %[[ELEM:.*]] = linalg.elemwise_unary ins(%[[OPERAND0]] : tensor<?xf32>) outs(%[[OPERAND1]] : tensor<?xf32>) -> tensor<?xf32>
//       CHECK:      scf.forall.in_parallel {
//  CHECK-NEXT:         tensor.parallel_insert_slice %[[ELEM]] into %[[ITER_ARG_6]]
//  CHECK-NEXT:      }
//  CHECK-NEXT:    }
//  CHECK-NEXT:    return %[[RESULT]]

// -----

func.func @index_switch_fold() -> (f32, f32) {
  %switch_cst = arith.constant 1: index
  %0 = scf.index_switch %switch_cst -> f32
  case 1 {
    %y = arith.constant 1.0 : f32
    scf.yield %y : f32
  }
  default {
    %y = arith.constant 42.0 : f32
    scf.yield %y : f32
  }
  
  %switch_cst_2 = arith.constant 2: index
  %1 = scf.index_switch %switch_cst_2 -> f32
  case 0 {
    %y = arith.constant 0.0 : f32
    scf.yield %y : f32
  }
  default {
    %y = arith.constant 42.0 : f32
    scf.yield %y : f32
  }
  
  return %0, %1 : f32, f32
}

// CHECK-LABEL: func.func @index_switch_fold()
//  CHECK-NEXT:   %[[c1:.*]] = arith.constant 1.000000e+00 : f32
//  CHECK-NEXT:   %[[c42:.*]] = arith.constant 4.200000e+01 : f32
//  CHECK-NEXT:   return %[[c1]], %[[c42]] : f32, f32

// -----

func.func @index_switch_fold_no_res() {
  %c1 = arith.constant 1 : index
  scf.index_switch %c1
  case 0 {
    scf.yield
  }
  default {
    "test.op"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @index_switch_fold_no_res()
//  CHECK-NEXT: "test.op"() : () -> ()
