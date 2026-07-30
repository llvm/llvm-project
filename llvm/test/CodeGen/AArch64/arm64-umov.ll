; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define zeroext i8 @f1(<16 x i8> %a) {
; CHECK-LABEL: f1:
; CHECK: mov.b w0, v0[3]
; CHECK-NEXT: ret
  %vecext = extractelement <16 x i8> %a, i32 3
  ret i8 %vecext
}

define zeroext i16 @f2(<4 x i16> %a) {
; CHECK-LABEL: f2:
; CHECK: mov.h w0, v0[2]
; CHECK-NEXT: ret
  %vecext = extractelement <4 x i16> %a, i32 2
  ret i16 %vecext
}

define i32 @f3(<2 x i32> %a) {
; CHECK-LABEL: f3:
; CHECK: mov.s w0, v0[1]
; CHECK-NEXT: ret
  %vecext = extractelement <2 x i32> %a, i32 1
  ret i32 %vecext
}

define i64 @f4(<2 x i64> %a) {
; CHECK-LABEL: f4:
; CHECK: mov.d x0, v0[1]
; CHECK-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 1
  ret i64 %vecext
}
