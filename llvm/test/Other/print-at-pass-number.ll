; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-pass-numbers -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NUMBER
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-pass-numbers -filter-print-funcs=baz -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NUMBER-FILTERED
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-before-pass-number=3 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BEFORE
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-after-pass-number=2 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=AFTER
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-before-pass-number=2,3 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BEFORE-MULTI
; RUN: opt -passes="loop(indvars,loop-deletion,loop-unroll-full)" -print-module-scope -print-after-pass-number=2,3 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=AFTER-MULTI

define i32 @bar(i32 %arg) {
; BEFORE: *** IR Dump Before 3-IndVarSimplifyPass on loop %bb1 in function bar ***
; BEFORE: define i32 @bar(i32 %arg) {
; AFTER:  *** IR Dump After 2-LCSSAPass on bar ***
; AFTER:  define i32 @bar(i32 %arg) {
; BEFORE-MULTI: *** IR Dump Before 2-LCSSAPass on bar ***
; BEFORE-MULTI: define i32 @bar(i32 %arg) {
; BEFORE-MULTI: *** IR Dump Before 3-IndVarSimplifyPass on loop %bb1 in function bar ***
; BEFORE-MULTI: define i32 @bar(i32 %arg) {
; AFTER-MULTI:  *** IR Dump After 2-LCSSAPass on bar ***
; AFTER-MULTI:  define i32 @bar(i32 %arg) {
; AFTER-MULTI:  *** IR Dump After 3-IndVarSimplifyPass on loop %bb1 in function bar ***
; AFTER-MULTI:  define i32 @bar(i32 %arg) {

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

define i32 @baz(i32 %arg) {
  ret i32 0;
}

; NUMBER:  Running pass 1 LoopSimplifyPass on bar
; NUMBER-NEXT: Running pass 2 LCSSAPass on bar
; NUMBER-NEXT: Running pass 3 IndVarSimplifyPass on loop %bb1 in function bar
; NUMBER-NEXT: Running pass 4 LoopDeletionPass on loop %bb1 in function bar
; NUMBER-NEXT: Running pass 5 LoopSimplifyPass on baz
; NUMBER-NEXT: Running pass 6 LCSSAPass on baz
; NUMBER-NOT: Running pass

; NUMBER-FILTERED: Running pass 1 LoopSimplifyPass on baz
; NUMBER-FILTERED-NEXT: Running pass 2 LCSSAPass on baz

