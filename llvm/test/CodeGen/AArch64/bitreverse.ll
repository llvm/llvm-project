; RUN: llc -mtriple=aarch64-eabi %s -o - | FileCheck %s

; These tests just check that the plumbing is in place for @llvm.bitreverse. The
; actual output is massive at the moment as llvm.bitreverse is not yet legal.

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>) readnone

define <2 x i16> @f(<2 x i16> %a) {
; CHECK-LABEL: f:
; CHECK: rev32
; CHECK: ushr
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

declare i8 @llvm.bitreverse.i8(i8) readnone

define i8 @g(i8 %a) {
; CHECK-LABEL: g:
; CHECK-DAG: rev [[RV:w.*]], w0
; CHECK-DAG: and [[L4:w.*]], [[RV]], #0xf0f0f0f
; CHECK-DAG: and [[H4:w.*]], [[RV]], #0xf0f0f0f0
; CHECK-DAG: lsr [[S4:w.*]], [[H4]], #4
; CHECK-DAG: orr [[R4:w.*]], [[S4]], [[L4]], lsl #4

; CHECK-DAG: and [[L2:w.*]], [[R4]], #0x33333333
; CHECK-DAG: and [[H2:w.*]], [[R4]], #0xcccccccc
; CHECK-DAG: lsr [[S2:w.*]], [[H2]], #2
; CHECK-DAG: orr [[R2:w.*]], [[S2]], [[L2]], lsl #2

; CHECK-DAG: mov [[P1:w.*]], #1426063360
; CHECK-DAG: mov [[N1:w.*]], #-1442840576
; CHECK-DAG: and [[L1:w.*]], [[R2]], [[P1]]
; CHECK-DAG: and [[H1:w.*]], [[R2]], [[N1]]
; CHECK-DAG: lsr [[S1:w.*]], [[H1]], #1
; CHECK-DAG: orr [[R1:w.*]], [[S1]], [[L1]], lsl #1

; CHECK-DAG: lsr w0, [[R1]], #24
; CHECK-DAG: ret
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %b
}

declare <8 x i8> @llvm.bitreverse.v8i8(<8 x i8>) readnone

define <8 x i8> @g_vec(<8 x i8> %a) {
; CHECK-DAG: movi [[M1:v.*]], #15
; CHECK-DAG: movi [[M2:v.*]], #240
; CHECK:     and  [[A1:v.*]], v0.8b, [[M1]]
; CHECK:     and  [[A2:v.*]], v0.8b, [[M2]]
; CHECK-DAG: shl  [[L4:v.*]], [[A1]], #4
; CHECK-DAG: ushr [[R4:v.*]], [[A2]], #4
; CHECK-DAG: orr  [[V4:v.*]], [[R4]], [[L4]]

; CHECK-DAG: movi [[M3:v.*]], #51
; CHECK-DAG: movi [[M4:v.*]], #204
; CHECK:     and  [[A3:v.*]], [[V4]], [[M3]]
; CHECK:     and  [[A4:v.*]], [[V4]], [[M4]]
; CHECK-DAG: shl  [[L2:v.*]], [[A3]], #2
; CHECK-DAG: ushr [[R2:v.*]], [[A4]], #2
; CHECK-DAG: orr  [[V2:v.*]], [[R2]], [[L2]]

; CHECK-DAG: movi [[M5:v.*]], #85
; CHECK-DAG: movi [[M6:v.*]], #170
; CHECK:     and  [[A5:v.*]], [[V2]], [[M5]]
; CHECK:     and  [[A6:v.*]], [[V2]], [[M6]]
; CHECK-DAG: shl  [[L1:v.*]], [[A5]], #1
; CHECK-DAG: ushr [[R1:v.*]], [[A6]], #1
; CHECK:     orr  [[V1:v.*]], [[R1]], [[L1]]

; CHECK:     ret
  %b = call <8 x i8> @llvm.bitreverse.v8i8(<8 x i8> %a)
  ret <8 x i8> %b
}
