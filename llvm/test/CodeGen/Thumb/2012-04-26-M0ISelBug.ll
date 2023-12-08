; RUN: llc -mtriple=thumbv6-apple-ios -mcpu=cortex-m0 < %s | FileCheck %s
; Cortex-M0 doesn't have 32-bit Thumb2 instructions (except for dmb, mrs, etc.)
; rdar://11331541

define i32 @t(i32 %a) nounwind {
; CHECK-LABEL: t:
; CHECK: asrs [[REG1:(r[0-9]+)]], [[REG2:(r[0-9]+)]], #31
; CHECK: eors [[REG2]], [[REG1]]
  %tmp0 = ashr i32 %a, 31
  %tmp1 = xor i32 %tmp0, %a
  ret i32 %tmp1
}
