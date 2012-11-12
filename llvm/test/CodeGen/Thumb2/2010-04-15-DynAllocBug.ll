; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -O3 | FileCheck %s
; rdar://7493908

; Make sure the result of the first dynamic_alloc isn't copied back to sp more
; than once. We'll deal with poor codegen later.

define void @t() nounwind ssp {
entry:
; CHECK: t:
  %size = mul i32 8, 2
; CHECK:  subs  r0, #16
; CHECK:  mov sp, r0
  %vla_a = alloca i8, i32 %size, align 8
; CHECK:  subs  r0, #16
; CHECK:  mov sp, r0
  %vla_b = alloca i8, i32 %size, align 8
  unreachable
}
