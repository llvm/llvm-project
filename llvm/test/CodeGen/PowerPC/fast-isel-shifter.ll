; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64

define i32 @shl() nounwind {
entry:
; PPC64: shl
; PPC64: slw
  %shl = shl i32 -1, 2
  ret i32 %shl
}

define i32 @shl_reg(i32 %src1, i32 %src2) nounwind {
entry:
; PPC64: shl_reg
; PPC64: slw
  %shl = shl i32 %src1, %src2
  ret i32 %shl
}

define i32 @lshr() nounwind {
entry:
; PPC64: lshr
; PPC64: srw
  %lshr = lshr i32 -1, 2
  ret i32 %lshr
}

define i32 @lshr_reg(i32 %src1, i32 %src2) nounwind {
entry:
; PPC64: lshr_reg
; PPC64: srw
  %lshr = lshr i32 %src1, %src2
  ret i32 %lshr
}

define i32 @ashr() nounwind {
entry:
; PPC64: ashr
; PPC64: srawi
  %ashr = ashr i32 -1, 2
  ret i32 %ashr
}

define i32 @ashr_reg(i32 %src1, i32 %src2) nounwind {
entry:
; PPC64: ashr_reg
; PPC64: sraw
  %ashr = ashr i32 %src1, %src2
  ret i32 %ashr
}

