; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s

; No need to sign-extend undef.

define i32 @t(i32* %a) nounwind {
entry:
; CHECK: t:
; CHECK: ldr r0, [r0]
; CHECK: bx lr
  %0 = sext i16 undef to i32
  %1 = getelementptr inbounds i32* %a, i32 %0
  %2 = load i32* %1, align 4
  ret i32 %2
}
