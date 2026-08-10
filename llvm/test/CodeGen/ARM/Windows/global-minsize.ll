; RUN: llc -mtriple=thumbv7-windows -filetype asm -o - %s | FileCheck %s

@i = internal global i32 0, align 4

; Function Attrs: minsize
define arm_aapcs_vfpcc ptr @function() #0 {
entry:
  ret ptr @i
}

attributes #0 = { minsize }

; CHECK: function:
; CHECK:   movw  r0, :lower16:i
; CHECK:   movt  r0, :upper16:i
; CHECK:   bx    lr
