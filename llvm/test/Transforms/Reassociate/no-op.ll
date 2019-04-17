; RUN: opt < %s -reassociate -S | FileCheck %s

; When there is nothing to do, or not much to do, check that reassociate leaves
; things alone.

declare void @use(i32)

define void @test1(i32 %a, i32 %b) {
; Shouldn't change or move any of the add instructions.  Should commute but
; otherwise not change or move any of the mul instructions.
; CHECK-LABEL: @test1(
  %a0 = add nsw i32 %a, 1
; CHECK-NEXT: %a0 = add nsw i32 %a, 1
  %m0 = mul nsw i32 3, %a
; CHECK-NEXT: %m0 = mul nsw i32 %a, 3
  %a1 = add nsw i32 %a0, %b
; CHECK-NEXT: %a1 = add nsw i32 %a0, %b
  %m1 = mul nsw i32 %b, %m0
; CHECK-NEXT: %m1 = mul nsw i32 %m0, %b
  call void @use(i32 %a1)
; CHECK-NEXT: call void @use
  call void @use(i32 %m1)
  ret void
}

define void @test2(i32 %a, i32 %b, i32 %c, i32 %d) {
; The initial add doesn't change so should not lose the nsw flag.
; CHECK-LABEL: @test2(
  %a0 = add nsw i32 %b, %a
; CHECK-NEXT: %a0 = add nsw i32 %b, %a
  %a1 = add nsw i32 %a0, %d
; CHECK-NEXT: %a1 = add i32 %a0, %c
  %a2 = add nsw i32 %a1, %c
; CHECK-NEXT: %a2 = add i32 %a1, %d
  call void @use(i32 %a2)
; CHECK-NEXT: call void @use
  ret void
}
