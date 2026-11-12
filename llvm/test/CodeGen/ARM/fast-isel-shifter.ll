; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM

define i32 @shl() nounwind ssp {
entry:
; ARM: shl
; ARM: lsl r0, r0, #2
  %shl = shl i32 -1, 2
  ret i32 %shl
}

define i32 @shl_reg(i32 %src1, i32 %src2) nounwind ssp {
entry:
; ARM: shl_reg
; ARM: lsl r0, r0, r1
  %shl = shl i32 %src1, %src2
  ret i32 %shl
}

define i32 @lshr() nounwind ssp {
entry:
; ARM: lshr
; ARM: lsr r0, r0, #2
  %lshr = lshr i32 -1, 2
  ret i32 %lshr
}

define i32 @lshr_reg(i32 %src1, i32 %src2) nounwind ssp {
entry:
; ARM: lshr_reg
; ARM: lsr r0, r0, r1
  %lshr = lshr i32 %src1, %src2
  ret i32 %lshr
}

define i32 @ashr() nounwind ssp {
entry:
; ARM: ashr
; ARM: asr r0, r0, #2
  %ashr = ashr i32 -1, 2
  ret i32 %ashr
}

define i32 @ashr_reg(i32 %src1, i32 %src2) nounwind ssp {
entry:
; ARM: ashr_reg
; ARM: asr r0, r0, r1
  %ashr = ashr i32 %src1, %src2
  ret i32 %ashr
}

