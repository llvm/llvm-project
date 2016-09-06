; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s
define <2 x i32> @fcmp_one(<2 x float> %x, <2 x float> %y) nounwind optsize readnone {
; CHECK-LABEL: fcmp_one:
; CHECK-NEXT: fcmgt.2s [[REG:v[0-9]+]], v0, v1
; CHECK-NEXT: fcmgt.2s [[REG2:v[0-9]+]], v1, v0
; CHECK-NEXT: orr.8b v0, [[REG2]], [[REG]]
; CHECK-NEXT: ret
  %tmp = fcmp one <2 x float> %x, %y
  %or = sext <2 x i1> %tmp to <2 x i32>
  ret <2 x i32> %or
}
