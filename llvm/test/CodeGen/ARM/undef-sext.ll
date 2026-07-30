; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s

; No need to sign-extend undef.

define i32 @t(ptr %a) nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: ldr r0, [r0]
; CHECK: bx lr
  %0 = sext i16 undef to i32
  %1 = getelementptr inbounds i32, ptr %a, i32 %0
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}
