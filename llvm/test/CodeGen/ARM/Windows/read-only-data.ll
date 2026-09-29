; RUN: llc -mtriple thumbv7-windows -filetype asm -o - %s | FileCheck %s

@.str = private unnamed_addr constant [7 x i8] c"string\00", align 1

declare arm_aapcs_vfpcc void @callee(ptr)

define arm_aapcs_vfpcc void @function() {
entry:
  call arm_aapcs_vfpcc void @callee(ptr @.str)
  ret void
}

; CHECK: .section .rdata,"dr"
; CHECK-NOT: .section ".rodata.str1.1"

