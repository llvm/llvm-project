; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=core-avx-i | FileCheck %s

define <16 x i16> @f_fu(<16 x i16> %bf) {
allocas:
  %avg.i.i = sdiv <16 x i16> %bf, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  ret <16 x i16> %avg.i.i
}

; CHECK: f_fu
; CHECK: psraw
; CHECK: psrlw
; CHECK: paddw
; CHECK: psraw
; CHECK: psraw
; CHECK: psrlw
; CHECK: paddw
; CHECK: psraw
; CHECK: ret
