; RUN: llc -mtriple=arm64-windows -o - %s | FileCheck %s

declare void @f()
declare void @g()

; Function Attrs: nounwind
define dso_local void @SEHfilter() nounwind "frame-pointer"="all" {
; CHECK-LABEL: @SEHfilter
; CHECK:       %bb.0:
; CHECK-NEXT:  stp     x30, x29, [sp, #-32]!
; CHECK-NEXT:  str     x19, [sp, #16]
; CHECK-NEXT:  ldr     w19, [x8]
; CHECK-NEXT:  mov     x29, sp
; CHECK-NEXT:  bl      g
; CHECK-NEXT:  cbz     w19, .LBB0_2
; CHECK-NEXT:  ; %bb.1:
; CHECK-NEXT:  ldr     x19, [sp, #16]
; CHECK-NEXT:  ldp     x30, x29, [sp], #32
; CHECK-NEXT:  ret
; CHECK-NEXT:  .LBB0_2:                                ; %if.end.i
; CHECK-NEXT:  bl      f
; CHECK-NEXT:  brk     #0x1
  %1 = load i32, i32* undef, align 4
  tail call void @g()
  %tobool.i = icmp eq i32 %1, 0
  br i1 %tobool.i, label %if.end.i, label %exit

if.end.i:
  call void @f()
  unreachable

exit:
  ret void
}

