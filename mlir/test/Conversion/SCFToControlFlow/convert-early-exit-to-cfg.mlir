// RUN: mlir-opt -convert-scf-to-cf -split-input-file %s | FileCheck %s


func.func @loop_break(%cond : i1) {
  // CHECK: test.op1
   "test.op1"() : () -> ()
  // CHECK-NEXT: cf.br [[LOOP1_ENTRY:.*]]
  // CHECK-NEXT: [[LOOP1_ENTRY]]
   scf.loop {
    // CHECK-NEXT: test.op2
     "test.op2"() : () -> ()
    // CHECK-NEXT: cf.cond_br %arg0, [[IF_ENTRY:.*]], [[IF_CONTINUE:.*]]
    // CHECK-NEXT: [[IF_ENTRY]]
     scf.if %cond {
       "test.op3"() : () -> ()
       scf.break 2 loc("break1")
     }
     "test.op3"() : () -> ()
   } loc("loop1")
   "test.op4"() : () -> ()
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
   scf.loop {
  // CHECK-NEXT: ^[[OUTER_LOOP_ENTRY]]: 
  // CHECK-NEXT: cf.cond_br %[[COND1]], ^[[IF1_THEN_BLOCK:.*]], ^[[IF1_EXIT:.*]]
     scf.if %cond1 {
  // CHECK-NEXT: ^[[IF1_THEN_BLOCK]]:
  // CHECK-NEXT: test.op2
       "test.op2"() : () -> ()
  // CHECK-NEXT: cf.br ^[[INNER_LOOP_ENTRY:.*]]
       scf.loop {
  // CHECK-NEXT: ^[[INNER_LOOP_ENTRY]]:
  // CHECK-NEXT: test.op3
         "test.op3"() : () -> ()
  // CHECK-NEXT: cf.cond_br %[[COND1]], ^[[IF2_THEN_BLOCK:.*]], ^[[IF2_EXIT:.*]]
         scf.if %cond1 {
  // CHECK-NEXT: ^[[IF2_THEN_BLOCK]]:
  // CHECK-NEXT: test.op4
           "test.op4"() : () -> ()
  // CHECK-NEXT: cf.br ^[[INNER_LOOP_ENTRY]]
           scf.continue 2 loc("continue1")
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
           scf.break 4 loc("break2")
         }
  // CHECK-NEXT: ^[[IF3_EXIT]]:
  // CHECK-NEXT: test.op7
         "test.op7"() : () -> ()
  // CHECK-NEXT: cf.br ^[[INNER_LOOP_ENTRY]]
         scf.continue 1 loc("continue2")
       } loc("loop3")
  // CHECK-NEXT: ^[[AFTER_INNER_LOOP:.*]]:
  // CHECK-NEXT: test.op8
       "test.op8"() : () -> ()
  // CHECK-NEXT: cf.br ^[[IF1_EXIT]]
     } loc("if1")
  // CHECK-NEXT: ^[[IF1_EXIT]]:
  // CHECK-NEXT: cf.br ^[[OUTER_LOOP_ENTRY]]
     scf.continue 1 loc("continue3")
   } loc("loop2")
  // CHECK-NEXT: ^[[FUNC_EXIT]]:
  // CHECK-NEXT: test.op9
   "test.op9"() : () -> ()
  // CHECK-NEXT: return
   return
}
