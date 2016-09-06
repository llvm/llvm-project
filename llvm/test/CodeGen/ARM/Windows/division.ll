; RUN: llc -mtriple thumbv7-windows-itanium -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-windows-msvc -filetype asm -o - %s | FileCheck %s

define arm_aapcs_vfpcc i32 @sdiv32(i32 %divisor, i32 %divident) {
entry:
  %div = sdiv i32 %divident, %divisor
  ret i32 %div
}

; CHECK-LABEL: sdiv32:
; CHECK: cbz r0
; CHECK: b
; CHECK: udf.w #249
; CHECK: bl __rt_sdiv

define arm_aapcs_vfpcc i32 @udiv32(i32 %divisor, i32 %divident) {
entry:
  %div = udiv i32 %divident, %divisor
  ret i32 %div
}

; CHECK-LABEL: udiv32:
; CHECK: cbz r0
; CHECK: b
; CHECK: udf.w #249
; CHECK: bl __rt_udiv

define arm_aapcs_vfpcc i64 @sdiv64(i64 %divisor, i64 %divident) {
entry:
  %div = sdiv i64 %divident, %divisor
  ret i64 %div
}

; CHECK-LABEL: sdiv64:
; CHECK: orr.w r4, r0, r1
; CHECK-NEXT: cbz r4
; CHECK: b
; CHECK: udf.w #249
; CHECK: bl __rt_sdiv64

define arm_aapcs_vfpcc i64 @udiv64(i64 %divisor, i64 %divident) {
entry:
  %div = udiv i64 %divident, %divisor
  ret i64 %div
}

; CHECK-LABEL: udiv64:
; CHECK: orr.w r4, r0, r1
; CHECK-NEXT: cbz r4
; CHECK: b
; CHECK: udf.w #249
; CHECK: bl __rt_udiv64

