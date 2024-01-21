; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-pass-numbers -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NUMBER
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-before-pass-number=3 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BEFORE

define i32 @bar(i32 %arg) {
; BEFORE: *** IR Dump Before 3-IndVarSimplifyPass on bb1 ***
; BEFORE: define i32 @bar(i32 %arg) {

bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i32 [ 0, %bb ], [ %add, %bb1 ]
  %phi2 = phi i32 [ 0, %bb ], [ %add3, %bb1 ]
  %add = add i32 %phi, 1
  %add3 = add i32 %phi2, %add
  %icmp = icmp slt i32 %phi, %arg
  br i1 %icmp, label %bb1, label %bb4

bb4:                                              ; preds = %bb1
  ret i32 %add3
}

; NUMBER:  Running pass 1 LoopSimplifyPass on bar
; NUMBER-NEXT: Running pass 2 LCSSAPass on bar
; NUMBER-NEXT: Running pass 3 IndVarSimplifyPass on bb1
; NUMBER-NEXT: Running pass 4 LoopDeletionPass on bb1
; NUMBER-NOT: Running pass
