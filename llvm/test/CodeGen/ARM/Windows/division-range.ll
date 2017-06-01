; RUN: llc -O0 -mtriple thumbv7--windows-itanium -filetype obj -o - %s | llvm-objdump -disassemble - | FileCheck %s

declare i32 @llvm.arm.space(i32, i32)

define arm_aapcs_vfpcc i32 @f(i32 %n, i32 %d) local_unnamed_addr {
entry:
  %div = sdiv i32 %n, %d
  call i32 @llvm.arm.space(i32 128, i32 undef)
  ret i32 %div
}

; CHECK: cmp r1, #0
; CHECK: beq #
; CHECK: bl

