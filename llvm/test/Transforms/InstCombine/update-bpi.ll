; RUN: opt < %s -passes="print<branch-prob>,instcombine,print<branch-prob>" -S 2>&1 | FileCheck %s

; CHECK:      Printing analysis 'Branch Probability Analysis' for function 'invert_cond':
; CHECK-NEXT: ---- Branch Probabilities ----
; CHECK-NEXT:   edge %entry -> %bb1 probability is 0x06186186 / 0x80000000 = 4.76%
; CHECK-NEXT:   edge %entry -> %bb2 probability is 0x79e79e7a / 0x80000000 = 95.24% [HOT edge]
; CHECK-NEXT: Printing analysis 'Branch Probability Analysis' for function 'invert_cond':
; CHECK-NEXT: ---- Branch Probabilities ----
; CHECK-NEXT:   edge %entry -> %bb2 probability is 0x79e79e7a / 0x80000000 = 95.24% [HOT edge]
; CHECK-NEXT:   edge %entry -> %bb1 probability is 0x06186186 / 0x80000000 = 4.76%

define i32 @invert_cond(ptr %p) {
; CHECK-LABEL: define i32 @invert_cond(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[COND_NOT:%.*]] = icmp eq ptr [[P]], null
; CHECK-NEXT:    br i1 [[COND_NOT]], label [[BB2:%.*]], label [[BB1:%.*]], !prof [[PROF0:![0-9]+]]
; CHECK:       bb1:
; CHECK-NEXT:    ret i32 0
; CHECK:       bb2:
; CHECK-NEXT:    ret i32 1
;
entry:
  %cond = icmp ne ptr %p, null
  br i1 %cond, label %bb1, label %bb2, !prof !1

bb1:
  ret i32 0;

bb2:
  ret i32 1;
}

!1 = !{!"branch_weights", i32 1, i32 20}
; CHECK: [[PROF0]] = !{!"branch_weights", i32 20, i32 1}

