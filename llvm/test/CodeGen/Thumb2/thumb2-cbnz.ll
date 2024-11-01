; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -arm-atomic-cfg-tidy=0 | FileCheck %s
; rdar://7354379

declare double @foo(double) nounwind readnone

define void @t(i32 %c, double %b) {
entry:
; CHECK:      cmp	r0, #0
  %cmp1 = icmp ne i32 %c, 0
  br i1 %cmp1, label %bb3, label %bb1

bb1:                                              ; preds = %entry
  unreachable

bb3:                                              ; preds = %entry
  %cmp2 = icmp ne i32 %c, 0
  br i1 %cmp2, label %bb7, label %bb5

bb5:                                              ; preds = %bb3
  unreachable

bb7:                                              ; preds = %bb3
  %cmp3 = icmp ne i32 %c, 0
  br i1 %cmp3, label %bb11, label %bb9

bb9:                                              ; preds = %bb7
; CHECK:      cbnz
  %0 = tail call  double @foo(double %b) nounwind readnone ; <double> [#uses=0]
  br label %bb11

bb11:                                             ; preds = %bb9, %bb7
  store i32 0, ptr undef
  ret void
}
