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

func.func @loop_continue(%cond1 : i1, %cond2 : i1) {
   "test.op1"() : () -> ()
   scf.loop {
     "test.op2"() : () -> ()
     scf.loop {
       "test.op3"() : () -> ()
       scf.if %cond1 {
         "test.op4"() : () -> ()
         scf.continue 2 loc("continue1")
       }
       "test.op5"() : () -> ()
       scf.if %cond2 {
         "test.op6"() : () -> ()
         scf.break 3 loc("break2")
       }
       "test.op7"() : () -> ()
     } loc("loop3")
     "test.op8"() : () -> ()
   } loc("loop2")
   "test.op9"() : () -> ()
   return
}
