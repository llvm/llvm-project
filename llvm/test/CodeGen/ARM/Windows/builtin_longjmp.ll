; RUN: llc -mtriple thumbv7--windows-itanium -filetype asm -o - %s | FileCheck %s

declare void @llvm.eh.sjlj.longjmp(i8*)

define arm_aapcs_vfpcc void @test___builtin_longjump(i8* %b) {
entry:
  tail call void @llvm.eh.sjlj.longjmp(i8* %b)
  unreachable
}

; CHECK: push.w  {r11, lr}
; CHECK: ldr     r[[SP:[0-9]+]], [r0, #8]
; CHECK: mov     sp, r[[SP]]
; CHECK: ldr     r[[PC:[0-9]+]], [r0, #4]
; CHECK: ldr     r11, [r0]
; CHECK: bx      r[[PC]]

