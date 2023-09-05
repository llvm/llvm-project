// RUN: mlir-opt --lift-cf-to-scf -split-input-file %s | FileCheck %s

func.func @simple_if() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  "test.test2"() : () -> ()
  cf.br ^bb3
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @simple_if
// CHECK:      %[[COND:.*]] = "test.test1"()
// CHECK-NEXT: scf.if %[[COND]]
// CHECK-NEXT:   "test.test2"()
// CHECK-NEXT: else
// CHECK-NEXT:   "test.test3"()
// CHECK-NEXT: }
// CHECK-NEXT: "test.test4"()
// CHECK-NEXT: return

// -----

func.func @if_with_block_args() -> index {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %1 = "test.test2"() : () -> (index)
  cf.br ^bb3(%1: index)
^bb2:
  %2 = "test.test3"() : () -> (index)
  cf.br ^bb3(%2: index)
^bb3(%3: index):
  "test.test4"() : () -> ()
  return %3 : index
}

// CHECK-LABEL: func @if_with_block_args
// CHECK:      %[[COND:.*]] = "test.test1"()
// CHECK-NEXT: %[[RES:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[VAL1:.*]]  = "test.test2"()
// CHECK-NEXT:   scf.yield %[[VAL1]]
// CHECK-NEXT: else
// CHECK-NEXT:   %[[VAL2:.*]]  = "test.test3"()
// CHECK-NEXT:   scf.yield %[[VAL2]]
// CHECK:      "test.test4"()
// CHECK-NEXT: return %[[RES]]

// -----

func.func @empty_else() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb3
^bb1:
  "test.test2"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @empty_else
// CHECK:      %[[COND:.*]] = "test.test1"()
// CHECK-NEXT: scf.if %[[COND]]
// CHECK-NEXT:   "test.test2"()
// CHECK:      else
// CHECK-NEXT:      }
// CHECK-NEXT: "test.test4"()
// CHECK-NEXT: return

// -----

func.func @while_loop() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb1
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @while_loop
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-NEXT: "test.test1"()
// CHECK-NEXT: scf.while
// CHECK-NEXT:   %[[COND:.*]] = "test.test2"()
// CHECK-NEXT:   %[[RES:.*]]:2 = scf.if %[[COND]]
// CHECK-NEXT:     "test.test3"()
// CHECK-NEXT:     scf.yield %[[C0]], %[[C1]]
// CHECK-NEXT:   else
// CHECK-NEXT:     scf.yield %[[C1]], %[[C0]]
// CHECK:        %[[COND:.*]] = arith.trunci %[[RES]]#1
// CHECK-NEXT:   scf.condition(%[[COND]])
// CHECK-NEXT: do
// CHECK-NEXT:   scf.yield
// CHECK:      "test.test4"()
// CHECK-NEXT: return

// -----

func.func @while_loop_with_block_args() -> i64{
  %1 = "test.test1"() : () -> index
  cf.br ^bb1(%1: index)
^bb1(%2: index):
  %cond:2 = "test.test2"() : () -> (i1, i64)
  cf.cond_br %cond#0, ^bb2(%cond#1: i64), ^bb3(%cond#1: i64)
^bb2(%3: i64):
  %4 = "test.test3"(%3) : (i64) -> index
  cf.br ^bb1(%4: index)
^bb3(%5: i64):
  "test.test4"() : () -> ()
  return %5 : i64
}

// CHECK-LABEL: func @while_loop_with_block_args
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[POISON_i64:.*]] = ub.poison : i64
// CHECK-DAG: %[[POISON_index:.*]] = ub.poison : index
// CHECK-NEXT: %[[VAL1:.*]] = "test.test1"()
// CHECK-NEXT: %[[RES:.*]]:2 = scf.while (%[[ARG0:.*]] = %[[VAL1]], %[[ARG1:.*]] = %[[POISON_i64]])
// CHECK-NEXT:   %[[COND:.*]]:2 = "test.test2"()
// CHECK-NEXT:   %[[IF_VALS:.*]]:4 = scf.if %[[COND]]#0
// CHECK-NEXT:     %[[VAL:.*]] = "test.test3"(%[[COND]]#1)
// CHECK-NEXT:     scf.yield %[[VAL]], %[[POISON_i64]], %[[C0]], %[[C1]]
// CHECK-NEXT:   else
// CHECK-NEXT:     scf.yield %[[POISON_index]], %[[COND]]#1, %[[C1]], %[[C0]]
// CHECK:        %[[TRUNC:.*]] = arith.trunci %[[IF_VALS]]#3
// CHECK-NEXT:   scf.condition(%[[TRUNC]]) %[[IF_VALS]]#0, %[[IF_VALS]]#1
// CHECK-NEXT: do
// CHECK-NEXT:   ^{{.+}}(
// CHECK-SAME: [[ARG0:[[:alnum:]]+]]:
// CHECK-SAME: [[ARG1:[[:alnum:]]+]]:
// CHECK-NEXT:   scf.yield %[[ARG0]], %[[ARG1]]
// CHECK:      "test.test4"() : () -> ()
// CHECK-NEXT: return %[[RES]]#1 : i64

// -----

func.func @multi_exit_loop() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb3, ^bb1
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @multi_exit_loop
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-NEXT: "test.test1"()
// CHECK-NEXT: scf.while
// CHECK-NEXT:   %[[COND:.*]] = "test.test2"()
// CHECK-NEXT:   %[[IF_PAIR:.*]]:2 = scf.if %[[COND]]
// CHECK-NEXT:     %[[COND2:.*]] = "test.test3"()
// CHECK-NEXT:     %[[IF_PAIR2:.*]]:2 = scf.if %[[COND2]]
// CHECK-NEXT:       scf.yield %[[C1]], %[[C0]]
// CHECK-NEXT:     else
// CHECK-NEXT:       scf.yield %[[C0]], %[[C1]]
// CHECK:          scf.yield %[[IF_PAIR2]]#0, %[[IF_PAIR2]]#1
// CHECK:        %[[TRUNC:.*]] = arith.trunci %[[IF_PAIR]]#1
// CHECK-NEXT:   scf.condition(%[[TRUNC]])
// CHECK-NEXT: do
// CHECK-NEXT:   scf.yield
// CHECK:      "test.test4"()
// CHECK-NEXT: return

// -----

func.func private @foo(%arg: f32) -> f32
func.func private @bar(%arg: f32)

func.func @switch_with_fallthrough(%flag: i32, %arg1 : f32, %arg2 : f32) {
  cf.switch %flag : i32, [
    default: ^bb1(%arg1 : f32),
    0: ^bb2(%arg2 : f32),
    1: ^bb3
  ]

^bb1(%arg3 : f32):
  %0 = call @foo(%arg3) : (f32) -> f32
  cf.br ^bb2(%0 : f32)

^bb2(%arg4 : f32):
  call @bar(%arg4) : (f32) -> ()
  cf.br ^bb3

^bb3:
  return
}

// CHECK-LABEL: func @switch_with_fallthrough
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[POISON:.*]] = ub.poison
// CHECK-NEXT: %[[INDEX_CAST:.*]] = arith.index_castui %[[ARG0]]
// CHECK-NEXT: %[[SWITCH_PAIR:.*]]:2 = scf.index_switch %[[INDEX_CAST]]
// CHECK-NEXT: case 0
// CHECK-NEXT:   scf.yield %[[ARG2]], %[[C0]]
// CHECK:      case 1
// CHECK-NEXT:   scf.yield %[[POISON]], %[[C1]]
// CHECK:      default
// CHECK-NEXT:   %[[RES:.*]] = func.call @foo(%[[ARG1]])
// CHECK-NEXT:   scf.yield %[[RES]], %[[C0]]
// CHECK:      %[[INDEX_CAST:.*]] = arith.index_castui %[[SWITCH_PAIR]]#1
// CHECK-NEXT: scf.index_switch %[[INDEX_CAST]]
// CHECK-NEXT: case 0
// CHECK-NEXT:   call @bar(%[[SWITCH_PAIR]]#0)
// CHECK-NEXT:   scf.yield
// CHECK:      default {
// CHECK-NEXT: }
// CHECK-NEXT: return

// -----

func.func private @bar(%arg: f32) -> (i1, f32)

func.func @already_structured_loop(%arg: f32) -> f32 {
  cf.br ^bb0

^bb0:
  %cond, %value = call @bar(%arg) : (f32) -> (i1, f32)
  cf.cond_br %cond, ^bb1, ^bb0

^bb1:
  return %value : f32
}

// CHECK-LABEL: @already_structured_loop
// CHECK-SAME: %[[ARG:.*]]: f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[POISON:.*]] = ub.poison
// CHECK-NEXT: %[[RES:.*]] = scf.while (%[[ARG1:.*]] = %[[POISON]])
// CHECK-NEXT:   %[[CALL_PAIR:.*]]:2 = func.call @bar(%[[ARG]])
// CHECK-NEXT:   %[[IF_PAIR:.*]]:2 = scf.if %[[CALL_PAIR]]#0
// CHECK-NEXT:     scf.yield %[[C1]], %[[C0]]
// CHECK-NEXT:   else
// CHECK-NEXT:     scf.yield %[[C0]], %[[C1]]
// CHECK:        %[[TRUNC:.*]] = arith.trunci %[[IF_PAIR]]#1
// CHECK-NEXT:   scf.condition(%[[TRUNC]]) %[[CALL_PAIR]]#1
// CHECK: return %[[RES]]

// -----

func.func private @bar(%arg: f32) -> (i1, f32)

// This test makes sure that the exit block using an iteration variable works
// correctly.

func.func @exit_loop_iter_use(%arg: f32) -> f32 {
  cf.br ^bb0(%arg : f32)

^bb0(%arg1: f32):
  %cond, %value = call @bar(%arg1) : (f32) -> (i1, f32)
  cf.cond_br %cond, ^bb1, ^bb0(%value : f32)

^bb1:
  return %arg1 : f32
}

// CHECK-LABEL: @exit_loop_iter_use
// CHECK-SAME: %[[ARG:.*]]:
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[POISON:.*]] = ub.poison
// CHECK-NEXT: %[[RES:.*]]:2 = scf.while
// CHECK-SAME:   %[[ARG1:.*]] = %[[ARG]]
// CHECK-SAME:   %[[ARG2:.*]] = %[[POISON]]
// CHECK-NEXT:   %[[CALL_PAIR:.*]]:2 = func.call @bar(%[[ARG1]])
// CHECK-NEXT:   %[[IF_RES:.*]]:3 = scf.if %[[CALL_PAIR]]#0
// CHECK-NEXT:     scf.yield %[[POISON]], %[[C1]], %[[C0]]
// CHECK-NEXT:   else
// CHECK-NEXT:     scf.yield %[[CALL_PAIR]]#1, %[[C0]], %[[C1]]
// CHECK:        %[[TRUNC:.*]] = arith.trunci %[[IF_RES]]#2
// CHECK-NEXT:   scf.condition(%[[TRUNC]]) %[[IF_RES]]#0, %[[ARG1]]
// CHECK: return %[[RES]]#1

// -----

func.func private @bar(%arg: f32) -> f32

func.func @infinite_loop(%arg: f32) -> f32 {
  cf.br ^bb1(%arg: f32)

^bb1(%arg1: f32):
  %0 = call @bar(%arg1) : (f32) -> f32
  cf.br ^bb1(%0 : f32)
}

// CHECK-LABEL: @infinite_loop
// CHECK-SAME: %[[ARG:.*]]:
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-NEXT: scf.while
// CHECK-SAME:   %[[ARG1:.*]] = %[[ARG]]
// CHECK-NEXT:   %[[CALL:.*]] = func.call @bar(%[[ARG1]])
// CHECK-NEXT:   %[[TRUNC:.*]] = arith.trunci %[[C1]]
// CHECK-NEXT:   scf.condition(%[[TRUNC]]) %[[CALL]]
// CHECK: %[[POISON:.*]] = ub.poison
// CHECK: return %[[POISON]]

// -----

func.func @multi_return() -> i32 {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb3
^bb1:
  %0 = "test.test2"() : () -> i32
  return %0 : i32
^bb3:
  %1 = "test.test4"() : () -> i32
  return %1 : i32
}

// CHECK-LABEL: func @multi_return
// CHECK:      %[[COND:.*]] = "test.test1"()
// CHECK-NEXT: %[[RES:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[VAL2:.*]] = "test.test2"()
// CHECK:        scf.yield %[[VAL2]]
// CHECK-NEXT: else
// CHECK-NEXT:   %[[VAL4:.*]] = "test.test4"()
// CHECK:        scf.yield %[[VAL4]]
// CHECK:      return %[[RES]]

// -----

func.func private @bar(%arg: f32) -> f32

func.func @conditional_infinite_loop(%arg: f32, %cond: i1) -> f32 {
  cf.cond_br %cond, ^bb1(%arg: f32), ^bb2

^bb1(%arg1: f32):
  %0 = call @bar(%arg1) : (f32) -> f32
  cf.br ^bb1(%0 : f32)

^bb2:
  return %arg : f32
}

// CHECK-LABEL: @conditional_infinite_loop
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]:
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]:
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-NEXT: %[[RES:.*]] = scf.if %[[ARG1]]
// CHECK-NEXT:   scf.while
// CHECK-SAME:     %[[ARG2:.*]] = %[[ARG0]]
// CHECK-NEXT:     %[[CALL:.*]] = func.call @bar(%[[ARG2]])
// CHECK-NEXT:     %[[TRUNC:.*]] = arith.trunci %[[C1]]
// CHECK-NEXT:     scf.condition(%[[TRUNC]]) %[[CALL]]
// CHECK:      else
// CHECK:        scf.yield %[[ARG0]]
// CHECK:      return %[[RES]]

// -----

// Different return-like terminators lead one control flow op remaining in the top level region.
// Each of the blocks the control flow op leads to are transformed into regions nevertheless.

func.func private @bar(%arg: i32)

func.func @mixing_return_like(%cond: i1, %cond2: i1, %cond3: i1) {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  cf.cond_br %cond, ^bb1, ^bb3

^bb1:
  cf.cond_br %cond2, ^bb2(%0 : i32), ^bb2(%1 : i32)
^bb2(%arg: i32):
  call @bar(%arg) : (i32) -> ()
  "test.returnLike"() : () -> ()

^bb3:
  cf.cond_br %cond3, ^bb4(%1 : i32), ^bb4(%0 : i32)
^bb4(%arg2: i32):
  call @bar(%arg2) : (i32) -> ()
  return
}

// CHECK-LABEL: @mixing_return_like
// CHECK-SAME: %[[COND1:[[:alnum:]]+]]:
// CHECK-SAME: %[[COND2:[[:alnum:]]+]]:
// CHECK-SAME: %[[COND3:[[:alnum:]]+]]:
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: cf.cond_br %[[COND1]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB1]]:
// CHECK: scf.if
// CHECK-NOT: cf.{{(switch|(cond_)?br)}}
// CHECK: call @bar
// CHECK: "test.returnLike"

// CHECK: ^[[BB2]]:
// CHECK: scf.if
// CHECK-NOT: cf.{{(switch|(cond_)?br)}}
// CHECK: call @bar
// CHECK: return

// -----

// cf.switch here only has some successors with different return-like ops.
// This test makes sure that if there are at least two successors branching to
// the same region, that this region gets properly turned to structured control
// flow.

func.func @some_successors_with_different_return(%flag: i32) -> i32 {
  %0 = arith.constant 5 : i32
  %1 = arith.constant 6 : i32
  cf.switch %flag : i32, [
    default: ^bb1,
    0: ^bb3(%0 : i32),
    1: ^bb3(%1 : i32)
  ]

^bb1:
  "test.returnLike"() : () -> ()

^bb3(%arg: i32):
  cf.br ^bb3(%arg : i32)
}

// CHECK-LABEL: @some_successors_with_different_return
// CHECK-SAME: %[[FLAG:[[:alnum:]]+]]:
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[POISON:.*]] = ub.poison
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : i32
// CHECK:      %[[INDEX_CAST:.*]] = arith.index_castui %[[FLAG]]
// CHECK-NEXT: %[[INDEX_SWITCH:.*]]:2 = scf.index_switch %[[INDEX_CAST]]
// CHECK:      case 0
// CHECK-NEXT:   scf.yield %[[C5]], %[[C1]]
// CHECK:      case 1
// CHECK-NEXT:   scf.yield %[[C6]], %[[C1]]
// CHECK:      default
// CHECK-NEXT:   scf.yield %[[POISON]], %[[C0]]
// CHECK:      cf.switch %[[INDEX_SWITCH]]#1
// CHECK-NEXT: default: ^[[BB2:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX_SWITCH]]#0
// CHECK-NEXT: 0: ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ]

// CHECK: ^[[BB2]]{{.*}}:
// CHECK: scf.while
// CHECK-NOT: cf.{{(switch|(cond_)?br)}}
// CHECK: return

// CHECK: ^[[BB1]]:
// CHECK-NEXT: "test.returnLike"

// -----

func.func @select_like(%cond: i1) -> i32 {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  cf.cond_br %cond, ^bb0(%0 : i32), ^bb0(%1 : i32)
^bb0(%arg: i32):
  return %arg : i32
}

// CHECK-LABEL: func @select_like
// CHECK-SAME: %[[COND:.*]]: i1
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-NEXT: %[[RES:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   scf.yield %[[C0]]
// CHECK-NEXT: else
// CHECK-NEXT:   scf.yield %[[C1]]
// CHECK:      return %[[RES]]

// -----

func.func @return_like_dominated_by_condition(%cond1: i1, %cond2: i1, %cond3: i1) {
  cf.cond_br %cond1, ^bb1, ^bb2

^bb1:
  return

^bb2:
  cf.cond_br %cond2, ^bb3, ^bb4

^bb3:
  "test.unreachable"() : () -> ()

^bb4:
  cf.cond_br %cond3, ^bb3, ^bb5

^bb5:
  return
}

// CHECK-LABEL: func @return_like_dominated_by_condition
// CHECK-SAME: %[[COND1:[[:alnum:]]+]]
// CHECK-SAME: %[[COND2:[[:alnum:]]+]]
// CHECK-SAME: %[[COND3:[[:alnum:]]+]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: %[[IF:.*]] = scf.if %[[COND1]]
// CHECK-NEXT:   scf.yield
// CHECK: else
// CHECK:   scf.if %[[COND2]]
// CHECK-NEXT:   scf.yield
// CHECK:   else
// CHECK:      scf.if %[[COND3]]
// CHECK-NEXT:    scf.yield
// CHECK-NEXT: else
// CHECK-NEXT:    scf.yield

// CHECK: cf.switch %[[IF]]
// CHECK-NEXT: default: ^[[BB1:.*]],
// CHECK-NEXT: 0: ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB2]]:
// CHECK-NEXT: return

// CHECK: ^[[BB1]]:
// CHECK-NEXT: "test.unreachable"

// -----

func.func @dominator_issue(%cond: i1, %cond2: i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  "test.unreachable"() : () -> ()

^bb2:
  %value = "test.def"() : () -> i32
  cf.cond_br %cond2, ^bb1, ^bb4

^bb4:
  return %value : i32
}

// CHECK-LABEL: func @dominator_issue
// CHECK-SAME: %[[COND1:[[:alnum:]]+]]
// CHECK-SAME: %[[COND2:[[:alnum:]]+]]
// CHECK: %[[IF:.*]]:2 = scf.if %[[COND1]]
// CHECK: else
// CHECK:   %[[VALUE:.*]] = "test.def"
// CHECK:   %[[IF_RES:.*]]:2 = scf.if %[[COND2]]
// CHECK:   else
// CHECK-NEXT: scf.yield %[[VALUE]]
// CHECK:   scf.yield %[[IF_RES]]#0
// CHECK: cf.switch %[[IF]]#1
// CHECK-NEXT: default: ^[[BB2:[[:alnum:]]+]](
// CHECK-SAME: %[[IF]]#0
// CHECK-NEXT: 0: ^[[BB1:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK-NEXT: "test.unreachable"
// CHECK: ^[[BB2]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]]

// -----

// Test that %value gets properly passed to ^bb4.

func.func private @bar(i32)

func.func @dominator_issue_loop(%cond: i1, %cond2: i1) -> i32 {
  %0 = arith.constant 5 : i32
  cf.br ^bb0

^bb0:
  cf.cond_br %cond, ^bb1, ^bb3

^bb1:
  %value = "test.def"() : () -> i32
  cf.cond_br %cond2, ^bb0, ^bb4

^bb3:
  return %0 : i32

^bb4:
  return %value : i32
}

// CHECK-LABEL: func @dominator_issue_loop
// CHECK-SAME: %[[COND1:[[:alnum:]]+]]
// CHECK-SAME: %[[COND2:[[:alnum:]]+]]
// CHECK: %[[WHILE:.*]]:2 = scf.while
// CHECK:   %[[IF:.*]]:3 = scf.if %[[COND1]]
// CHECK:     %[[DEF:.*]] = "test.def"
// CHECK:     %[[IF2:.*]]:3 = scf.if %[[COND2]]
// CHECK:       scf.yield
// CHECK-SAME:   %[[DEF]]
// CHECK:     else
// CHECK:       scf.yield %{{.*}}, %{{.*}}, %[[DEF]]
// CHECK:     scf.yield %[[IF2]]#0, %[[IF2]]#1, %[[IF2]]#2
// CHECK:   scf.condition(%{{.*}}) %[[IF]]#2, %[[IF]]#0

// CHECK: %[[SWITCH:.*]] = scf.index_switch
// CHECK:   scf.yield %[[WHILE]]#0
// CHECK: return %[[SWITCH]]

// -----

// Multi entry loops generally produce code in dire need of canonicalization.

func.func private @comp1(%arg: i32) -> i1
func.func private @comp2(%arg: i32) -> i1
func.func private @foo(%arg: i32)

func.func @multi_entry_loop(%cond: i1) {
  %0 = arith.constant 6 : i32
  %1 = arith.constant 5 : i32
  cf.cond_br %cond, ^bb0, ^bb1

^bb0:
  %exit = call @comp1(%0) : (i32) -> i1
  cf.cond_br %exit, ^bb2(%0 : i32), ^bb1

^bb1:
  %exit2 = call @comp2(%1) : (i32) -> i1
  cf.cond_br %exit2, ^bb2(%1 : i32), ^bb0

^bb2(%arg3 : i32):
  call @foo(%arg3) : (i32) -> ()
  return
}

// CHECK-LABEL: func @multi_entry_loop
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[UB:.*]] = ub.poison
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C6:.*]] = arith.constant 6
// CHECK-DAG: %[[C5:.*]] = arith.constant 5
// CHECK:      %[[IF:.*]]:{{.*}} = scf.if %[[ARG0]]
// CHECK-NEXT:   scf.yield %[[C1]], %[[UB]]
// CHECK-NEXT: else
// CHECK-NEXT:   scf.yield %[[C0]], %[[UB]]
// CHECK:      %[[WHILE:.*]]:{{.*}} = scf.while
// CHECK-SAME: %[[ARG1:.*]] = %[[IF]]#0
// CHECK-SAME: %[[ARG2:.*]] = %[[IF]]#1
// CHECK-NEXT:   %[[FLAG:.*]] = arith.index_castui %[[ARG1]]
// CHECK-NEXT:   %[[SWITCH:.*]]:{{.*}} = scf.index_switch %[[FLAG]]
// CHECK-NEXT:   case 0
// CHECK-NEXT:     %[[EXIT:.*]] = func.call @comp2(%[[C5]])
// CHECK-NEXT:     %[[IF_RES:.*]]:{{.*}} = scf.if %[[EXIT]]
// CHECK-NEXT:       scf.yield %[[UB]], %[[C5]], %[[C1]], %[[C0]]
// CHECK-NEXT:     else
// CHECK-NEXT:       scf.yield %[[C1]], %[[UB]], %[[C0]], %[[C1]]
// CHECK:          scf.yield %[[IF_RES]]#0, %[[IF_RES]]#1, %[[IF_RES]]#2, %[[IF_RES]]#3
// CHECK:        default
// CHECK-NEXT:     %[[EXIT:.*]] = func.call @comp1(%[[C6]])
// CHECK-NEXT:     %[[IF_RES:.*]]:{{.*}} = scf.if %[[EXIT]]
// CHECK-NEXT:       scf.yield %[[UB]], %[[C6]], %[[C1]], %[[C0]]
// CHECK-NEXT:     else
// CHECK-NEXT:       scf.yield %[[C0]], %[[UB]], %[[C0]], %[[C1]]
// CHECK:          scf.yield %[[IF_RES]]#0, %[[IF_RES]]#1, %[[IF_RES]]#2, %[[IF_RES]]#3
// CHECK:        %[[COND:.*]] = arith.trunci %[[SWITCH]]#3
// CHECK-NEXT:   scf.condition(%[[COND]]) %[[SWITCH]]#0, %[[SWITCH]]#1
// CHECK:      do
// CHECK:        scf.yield
// CHECK:      call @foo(%[[WHILE]]#1)
// CHECK-NEXT: return

// -----

func.func @nested_region() {
  scf.execute_region {
    %cond = "test.test1"() : () -> i1
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    "test.test2"() : () -> ()
    cf.br ^bb3
  ^bb2:
    "test.test3"() : () -> ()
    cf.br ^bb3
  ^bb3:
    "test.test4"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func @nested_region
// CHECK:      scf.execute_region {
// CHECK:      %[[COND:.*]] = "test.test1"()
// CHECK-NEXT: scf.if %[[COND]]
// CHECK-NEXT:   "test.test2"()
// CHECK-NEXT: else
// CHECK-NEXT:   "test.test3"()
// CHECK-NEXT: }
// CHECK-NEXT: "test.test4"()
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: return
