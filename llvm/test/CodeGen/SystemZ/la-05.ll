; Test that a huge address offset is loaded into a register and then added
; separately.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@a = common dso_local global i32 0, align 4

define i64 @f1() {
; CHECK-LABEL: f1:
; CHECK: llihl   %r0, 829
; CHECK: oilf    %r0, 4294966308
; CHECK: larl    %r2, a
; CHECK: agr     %r2, %r0
; CHECK: br      %r14
  ret i64 add (i64 ptrtoint (i32* @a to i64), i64 3564822854692)
}

define signext i32 @f2() {
; CHECK-LABEL: f2:
; CHECK: llihl   %r0, 829
; CHECK: oilf    %r0, 4294966308
; CHECK: larl    %r1, a
; CHECK: agr     %r1, %r0
; CHECK: lgf     %r2, 0(%r1)
; CHECK: br      %r14
entry:
  %0 = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* @a to i64),
                                i64 3564822854692) to i32*)
  ret i32 %0
}

