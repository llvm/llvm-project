; RUN: llc -mtriple thumbv7-windows-itanium -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-windows-msvc -filetype asm -o - %s | FileCheck %s

define arm_aapcs_vfpcc i32 @udiv32(i32 %divisor, i32 %divident) {
entry:
  %div = udiv i32 %divident, %divisor
  ret i32 %div
}

; CHECK-LABEL: udiv32:
; CHECK: cbz r0
; CHECK: bl __rt_udiv
; CHECK: udf.w #249

define arm_aapcs_vfpcc i64 @udiv64(i64 %divisor, i64 %divident) {
entry:
  %div = udiv i64 %divident, %divisor
  ret i64 %div
}

; CHECK-LABEL: udiv64:
; CHECK: orr.w r12, r0, r1
; CHECK-NEXT: cbz r12
; CHECK: bl __rt_udiv64
; CHECK: udf.w #249

