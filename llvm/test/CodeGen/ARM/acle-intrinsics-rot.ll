; RUN: llc -mtriple=thumbv8m.main -mcpu=cortex-m33 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7em %s -o - | FileCheck %s
; RUN: llc -mtriple=armv6 %s -o - | FileCheck %s
; RUN: llc -mtriple=armv7 %s -o - | FileCheck %s
; RUN: llc -mtriple=armv8 %s -o - | FileCheck %s

; CHECK-LABEL: sxtb16_ror_8
; CHECK: sxtb16 r0, r0, ror #8
define i32 @sxtb16_ror_8(i32 %a) {
entry:
  %shr.i = lshr i32 %a, 8
  %shl.i = shl i32 %a, 24
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.sxtb16(i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: sxtb16_ror_16
; CHECK: sxtb16 r0, r0, ror #16
define i32 @sxtb16_ror_16(i32 %a) {
entry:
  %shr.i = lshr i32 %a, 16
  %shl.i = shl i32 %a, 16
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.sxtb16(i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: sxtb16_ror_24
; CHECK: sxtb16 r0, r0, ror #24
define i32 @sxtb16_ror_24(i32 %a) {
entry:
  %shr.i = lshr i32 %a, 24
  %shl.i = shl i32 %a, 8
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.sxtb16(i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: uxtb16_ror_8
; CHECK: uxtb16 r0, r0, ror #8
define i32 @uxtb16_ror_8(i32 %a) {
entry:
  %shr.i = lshr i32 %a, 8
  %shl.i = shl i32 %a, 24
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.uxtb16(i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: uxtb16_ror_16
; CHECK: uxtb16 r0, r0, ror #16
define i32 @uxtb16_ror_16(i32 %a) {
entry:
  %shr.i = lshr i32 %a, 16
  %shl.i = shl i32 %a, 16
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.uxtb16(i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: uxtb16_ror_24
; CHECK: uxtb16 r0, r0, ror #24
define i32 @uxtb16_ror_24(i32 %a) {
entry:
  %shr.i = lshr i32 %a, 24
  %shl.i = shl i32 %a, 8
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.uxtb16(i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: sxtab16_ror_8
; CHECK: sxtab16 r0, r0, r1, ror #8
define i32 @sxtab16_ror_8(i32 %a, i32 %b) {
entry:
  %shr.i = lshr i32 %b, 8
  %shl.i = shl i32 %b, 24
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.sxtab16(i32 %a, i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: sxtab16_ror_16
; CHECK: sxtab16 r0, r0, r1, ror #16
define i32 @sxtab16_ror_16(i32 %a, i32 %b) {
entry:
  %shr.i = lshr i32 %b, 16
  %shl.i = shl i32 %b, 16
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.sxtab16(i32 %a, i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: sxtab16_ror_24
; CHECK: sxtab16 r0, r0, r1, ror #24
define i32 @sxtab16_ror_24(i32 %a, i32 %b) {
entry:
  %shr.i = lshr i32 %b, 24
  %shl.i = shl i32 %b, 8
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.sxtab16(i32 %a, i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: uxtab16_ror_8
; CHECK: uxtab16 r0, r0, r1, ror #8
define i32 @uxtab16_ror_8(i32 %a, i32 %b) {
entry:
  %shr.i = lshr i32 %b, 8
  %shl.i = shl i32 %b, 24
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.uxtab16(i32 %a, i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: uxtab16_ror_16
; CHECK: uxtab16 r0, r0, r1, ror #16
define i32 @uxtab16_ror_16(i32 %a, i32 %b) {
entry:
  %shr.i = lshr i32 %b, 16
  %shl.i = shl i32 %b, 16
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.uxtab16(i32 %a, i32 %or.i)
  ret i32 %0
}

; CHECK-LABEL: uxtab16_ror_24
; CHECK: uxtab16 r0, r0, r1, ror #24
define i32 @uxtab16_ror_24(i32 %a, i32 %b) {
entry:
  %shr.i = lshr i32 %b, 24
  %shl.i = shl i32 %b, 8
  %or.i = or i32 %shl.i, %shr.i
  %0 = tail call i32 @llvm.arm.uxtab16(i32 %a, i32 %or.i)
  ret i32 %0
}

declare i32 @llvm.arm.sxtb16(i32)
declare i32 @llvm.arm.uxtb16(i32)
declare i32 @llvm.arm.sxtab16(i32, i32)
declare i32 @llvm.arm.uxtab16(i32, i32)

