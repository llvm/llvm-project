; RUN: llc -mtriple=aarch64-eabi %s -o - | FileCheck %s

; These tests just check that the plumbing is in place for @llvm.bitreverse.

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>) readnone

define <2 x i16> @f(<2 x i16> %a) {
; CHECK-LABEL: f:
; CHECK: fmov [[REG1:w[0-9]+]], s0
; CHECK-DAG: rbit [[REG2:w[0-9]+]], [[REG1]]
; CHECK-DAG: fmov s0, [[REG2]]
; CHECK-DAG: mov [[REG3:w[0-9]+]], v0.s[1]
; CHECK-DAG: rbit [[REG4:w[0-9]+]], [[REG3]]
; CHECK-DAG: ins v0.s[1], [[REG4]]
; CHECK-DAG: ushr v0.2s, v0.2s, #16
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

declare i8 @llvm.bitreverse.i8(i8) readnone

define i8 @g(i8 %a) {
; CHECK-LABEL: g:
; CHECK: rbit [[REG:w[0-9]+]], w0
; CHECK-NEXT: lsr w0, [[REG]], #24
; CHECK-NEXT: ret
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
