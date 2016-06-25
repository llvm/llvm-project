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

; Unfortunately some of the shift-and-inserts become BFIs, and some do not :(
define i8 @g(i8 %a) {
; CHECK-LABEL: g:
; CHECK-DAG: lsr [[S5:w.*]], w0, #5
; CHECK-DAG: lsr [[S4:w.*]], w0, #4
; CHECK-DAG: lsr [[S3:w.*]], w0, #3
; CHECK-DAG: lsr [[S2:w.*]], w0, #2
; CHECK-DAG: lsl [[L1:w.*]], w0, #29
; CHECK-DAG: lsl [[L2:w.*]], w0, #19
; CHECK-DAG: lsl [[L3:w.*]], w0, #17

; CHECK-DAG: and [[T1:w.*]], [[L1]], #0x40000000
; CHECK-DAG: bfi [[T1]], w0, #31, #1
; CHECK-DAG: bfi [[T1]], [[S2]], #29, #1
; CHECK-DAG: bfi [[T1]], [[S3]], #28, #1
; CHECK-DAG: bfi [[T1]], [[S4]], #27, #1
; CHECK-DAG: bfi [[T1]], [[S5]], #26, #1
; CHECK-DAG: and [[T2:w.*]], [[L2]], #0x2000000
; CHECK-DAG: and [[T3:w.*]], [[L3]], #0x1000000
; CHECK-DAG: orr [[T4:w.*]], [[T1]], [[T2]]
; CHECK-DAG: orr [[T5:w.*]], [[T4]], [[T3]]
; CHECK:     lsr w0, [[T5]], #24

  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %b
}

declare <8 x i8> @llvm.bitreverse.v8i8(<8 x i8>) readnone

define <8 x i8> @g_vec(<8 x i8> %a) {
; Try and match as much of the sequence as precisely as possible.

; CHECK-LABEL: g_vec:
; CHECK-DAG: movi [[M1:v.*]], #128
; CHECK-DAG: movi [[M2:v.*]], #64
; CHECK-DAG: movi [[M3:v.*]], #32
; CHECK-DAG: movi [[M4:v.*]], #16
; CHECK-DAG: movi [[M5:v.*]], #8{{$}}
; CHECK-DAG: movi [[M6:v.*]], #4{{$}}
; CHECK-DAG: movi [[M7:v.*]], #2{{$}}
; CHECK-DAG: movi [[M8:v.*]], #1{{$}}
; CHECK-DAG: shl  [[S1:v.*]], v0.8b, #7
; CHECK-DAG: shl  [[S2:v.*]], v0.8b, #5
; CHECK-DAG: shl  [[S3:v.*]], v0.8b, #3
; CHECK-DAG: shl  [[S4:v.*]], v0.8b, #1
; CHECK-DAG: ushr [[S5:v.*]], v0.8b, #1
; CHECK-DAG: ushr [[S6:v.*]], v0.8b, #3
; CHECK-DAG: ushr [[S7:v.*]], v0.8b, #5
; CHECK-DAG: ushr [[S8:v.*]], v0.8b, #7
; CHECK-DAG: and  [[A1:v.*]], [[S1]], [[M1]]
; CHECK-DAG: and  [[A2:v.*]], [[S2]], [[M2]]
; CHECK-DAG: and  [[A3:v.*]], [[S3]], [[M3]]
; CHECK-DAG: and  [[A4:v.*]], [[S4]], [[M4]]
; CHECK-DAG: and  [[A5:v.*]], [[S5]], [[M5]]
; CHECK-DAG: and  [[A6:v.*]], [[S6]], [[M6]]
; CHECK-DAG: and  [[A7:v.*]], [[S7]], [[M7]]
; CHECK-DAG: and  [[A8:v.*]], [[S8]], [[M8]]

; The rest can be ORRed together in any order; it's not worth the test
; maintenance to match them precisely.
; CHECK-DAG: orr
; CHECK-DAG: orr
; CHECK-DAG: orr
; CHECK-DAG: orr
; CHECK-DAG: orr
; CHECK-DAG: orr
; CHECK-DAG: orr
; CHECK: ret
  %b = call <8 x i8> @llvm.bitreverse.v8i8(<8 x i8> %a)
  ret <8 x i8> %b
}
