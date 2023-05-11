; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-pass-numbers -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NUMBER
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-at-pass-number=3 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=AT
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-at-pass-number=4 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=AT-INVALIDATE

define i32 @bar(i32 %arg) {
; AT: *** IR Dump At 3-IndVarSimplifyPass on bb1 ***
; AT: define i32 @bar(i32 %arg) {

; AT-INVALIDATE: *** IR Dump At 4-LoopDeletionPass on bb1 (invalidated) ***
; AT-INVALIDATE: define i32 @bar(i32 %arg) {

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

; NUMBER: Running pass 1 LoopSimplifyPass
; NUMBER-NEXT: Running pass 2 LCSSAPass
; NUMBER-NEXT: Running pass 3 IndVarSimplifyPass
; NUMBER-NEXT: Running pass 4 LoopDeletionPass
; NUMBER-NEXT: Running pass 5 VerifierPass
; NUMBER-NEXT: Running pass 6 PrintModulePass
