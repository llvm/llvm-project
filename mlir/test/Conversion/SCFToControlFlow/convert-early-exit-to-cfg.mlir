// RUN: mlir-opt -convert-scf-to-cf -split-input-file %s | FileCheck %s


func.func @loop_break(%cond : i1) {
  // CHECK: test.op1
   "test.op1"() : () -> ()
  // CHECK-NEXT: cf.br [[LOOP1_ENTRY:.*]]
  // CHECK-NEXT: [[LOOP1_ENTRY]]
   scf.loop token(%loop) {
    // CHECK-NEXT: test.op2
     "test.op2"() : () -> ()
    // CHECK-NEXT: cf.cond_br %arg0, [[IF_ENTRY:.*]], [[IF_CONTINUE:.*]]
    // CHECK-NEXT: [[IF_ENTRY]]
     scf.if %cond {
       "test.op3"() : () -> ()
       scf.break [%loop] loc("break1")
     }
     "test.op3"() : () -> ()
   } loc("loop1")
   "test.op4"() : () -> ()
   return
}

// -----

// Bug regression test: IfLowering was using thenTerminator on line 461 instead
// of elseTerminator, causing the else block's scf.yield to not be replaced with
// a cf.branch when the then-branch has a scf.break.
// CHECK-LABEL: func @if_break_then_yield_else
func.func @if_break_then_yield_else(%cond : i1) {
  // CHECK: cf.br ^[[LOOP:.*]]
  scf.loop token(%loop) {
    // CHECK: ^[[LOOP]]:
    // CHECK: cf.cond_br %arg0, ^[[THEN:.*]], ^[[ELSE:.*]]
    scf.if %cond {
      // CHECK: ^[[THEN]]:
      // CHECK-NEXT: cf.br ^[[EXIT:.*]]
      scf.break [%loop]
    } else {
      // CHECK: ^[[ELSE]]:
      // CHECK-NEXT: cf.br ^[[AFTER_IF:.*]]
    }
    // CHECK: ^[[AFTER_IF]]:
    // CHECK-NEXT: cf.br ^[[LOOP]]
    scf.continue [%loop]
  }
  // CHECK: ^[[EXIT]]:
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @nested_loops_and_ifs(
// CHECK-SAME: %[[COND1:.*]]: i1,
// CHECK-SAME: %[[COND2:.*]]: i1
func.func @nested_loops_and_ifs(%cond1 : i1, %cond2 : i1) {
  // CHECK: test.op1
   "test.op1"() : () -> ()
  // CHECK-NEXT: cf.br ^[[OUTER_LOOP_ENTRY:.*]]
   scf.loop token(%outer) {
  // CHECK-NEXT: ^[[OUTER_LOOP_ENTRY]]:
  // CHECK-NEXT: cf.cond_br %[[COND1]], ^[[IF1_THEN_BLOCK:.*]], ^[[IF1_EXIT:.*]]
     scf.if %cond1 {
  // CHECK-NEXT: ^[[IF1_THEN_BLOCK]]:
  // CHECK-NEXT: test.op2
       "test.op2"() : () -> ()
  // CHECK-NEXT: cf.br ^[[INNER_LOOP_ENTRY:.*]]
       scf.loop token(%inner) {
  // CHECK-NEXT: ^[[INNER_LOOP_ENTRY]]:
  // CHECK-NEXT: test.op3
         "test.op3"() : () -> ()
  // CHECK-NEXT: cf.cond_br %[[COND1]], ^[[IF2_THEN_BLOCK:.*]], ^[[IF2_EXIT:.*]]
         scf.if %cond1 {
  // CHECK-NEXT: ^[[IF2_THEN_BLOCK]]:
  // CHECK-NEXT: test.op4
           "test.op4"() : () -> ()
  // CHECK-NEXT: cf.br ^[[INNER_LOOP_ENTRY]]
           scf.continue [%inner] loc("continue1")
         }
  // CHECK-NEXT: ^[[IF2_EXIT]]:
  // CHECK-NEXT: test.op5
         "test.op5"() : () -> ()
  // CHECK-NEXT: cf.cond_br %[[COND2]], ^[[IF3_THEN_BLOCK:.*]], ^[[IF3_EXIT:.*]]
         scf.if %cond2 {
  // CHECK-NEXT: ^[[IF3_THEN_BLOCK]]:
  // CHECK-NEXT: test.op6
           "test.op6"() : () -> ()
  // CHECK-NEXT: cf.br ^[[FUNC_EXIT:.*]]
           scf.break [%outer] loc("break2")
         }
  // CHECK-NEXT: ^[[IF3_EXIT]]:
  // CHECK-NEXT: test.op7
         "test.op7"() : () -> ()
  // CHECK-NEXT: cf.br ^[[INNER_LOOP_ENTRY]]
         scf.continue [%inner] loc("continue2")
       } loc("loop3")
  // CHECK-NEXT: ^[[AFTER_INNER_LOOP:.*]]:
  // CHECK-NEXT: test.op8
       "test.op8"() : () -> ()
  // CHECK-NEXT: cf.br ^[[IF1_EXIT]]
     } loc("if1")
  // CHECK-NEXT: ^[[IF1_EXIT]]:
  // CHECK-NEXT: cf.br ^[[OUTER_LOOP_ENTRY]]
     scf.continue [%outer] loc("continue3")
   } loc("loop2")
  // CHECK-NEXT: ^[[FUNC_EXIT]]:
  // CHECK-NEXT: test.op9
   "test.op9"() : () -> ()
  // CHECK-NEXT: return
   return
}

// -----

// CHECK-LABEL: func @loop_with_iter_args
// CHECK-SAME: %[[INIT:.*]]: i32, %[[COND:.*]]: i1
func.func @loop_with_iter_args(%init: i32, %cond: i1) -> i32 {
  // CHECK: cf.br ^[[LOOP:.*]](%[[INIT]] : i32)
  %result = scf.loop token(%loop) iter_args(%arg = %init) : i32 -> i32 {
    // CHECK: ^[[LOOP]](%[[ARG:.*]]: i32):
    // CHECK: cf.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
    scf.if %cond {
      // CHECK: ^[[THEN]]:
      // CHECK-NEXT: cf.br ^[[EXIT:.*]](%[[ARG]] : i32)
      scf.break [%loop] %arg : i32
    }
    // CHECK: ^[[ELSE]]:
    // CHECK-NEXT: cf.br ^[[LOOP]](%[[ARG]] : i32)
    scf.continue [%loop] %arg : i32
  }
  // CHECK: ^[[EXIT]](%[[RES:.*]]: i32):
  // CHECK-NEXT: return %[[RES]]
  return %result : i32
}
