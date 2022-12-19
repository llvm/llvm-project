; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s

; Use sp, #imm to lower frame indices when the offset is multiple of 4
; and in the range of 0-1020. This saves code size by utilizing
; 16-bit instructions.
; rdar://9321541

define i32 @t() nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: sub sp, #12
; CHECK-NOT: sub
; CHECK: add r0, sp, #4
; CHECK: add r1, sp, #8
; CHECK: mov r2, sp
  %size = alloca i32, align 4
  %count = alloca i32, align 4
  %index = alloca i32, align 4
  %0 = call i32 @foo(ptr %count, ptr %size, ptr %index) nounwind
  ret i32 %0
}

declare i32 @foo(ptr, ptr, ptr)
