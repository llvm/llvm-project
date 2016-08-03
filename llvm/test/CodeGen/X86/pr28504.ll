; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; The test case is rather involved, because we need to get to a state where
; We have a sext(setcc x, y, cc) -> (select (setcc x, y, cc), T, 0) combine,
; BUT this combine is only triggered post-legalization, so the setcc's return
; type is i8. So we can't have the combine opportunity be exposed too early.
; Basically, what we want to see is that the compare result zero-extended, and 
; then stored. Only one zext, and no sexts.

; CHECK-LABEL: main:
; CHECK: movzbl (%rdi), %[[EAX:.*]]
; CHECK-NEXT: xorl  %e[[C:.]]x, %e[[C]]x
; CHECK-NEXT: cmpl  $1, %[[EAX]]
; CHECK-NEXT: sete  %[[C]]l
; CHECK-NEXT: movl  %e[[C]]x, (%rsi)
define void @main(i8* %p, i32* %q) {
bb:
  %tmp4 = load i8, i8* %p, align 1
  %tmp5 = sext i8 %tmp4 to i32
  %tmp6 = load i8, i8* %p, align 1
  %tmp7 = zext i8 %tmp6 to i32
  %tmp8 = sub nsw i32 %tmp5, %tmp7
  %tmp11 = icmp eq i32 %tmp7, 1
  %tmp12 = zext i1 %tmp11 to i32
  %tmp13 = add nsw i32 %tmp8, %tmp12
  %tmp14 = trunc i32 %tmp13 to i8
  %tmp15 = sext i8 %tmp14 to i16
  %tmp16 = sext i16 %tmp15 to i32
  store i32 %tmp16, i32* %q, align 4
  br i1 %tmp11, label %bb21, label %bb22

bb21:                                             ; preds = %bb
  unreachable

bb22:                                             ; preds = %bb
  ret void
}
