; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

; A vector TruncStore can not be selected.
; Test a trunc IR and a vector store IR can be selected correctly.
define void @truncStore.v2i64(<2 x i64> %a, <2 x i32>* %result) {
; CHECK-LABEL: truncStore.v2i64:
; CHECK: xtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
; CHECK: {{st1 { v[0-9]+.2s }|str d[0-9]+}}, [x{{[0-9]+|sp}}]
  %b = trunc <2 x i64> %a to <2 x i32>
  store <2 x i32> %b, <2 x i32>* %result
  ret void
}

define void @truncStore.v4i32(<4 x i32> %a, <4 x i16>* %result) {
; CHECK-LABEL: truncStore.v4i32:
; CHECK: xtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
; CHECK: {{st1 { v[0-9]+.4h }|str d[0-9]+}}, [x{{[0-9]+|sp}}]
  %b = trunc <4 x i32> %a to <4 x i16>
  store <4 x i16> %b, <4 x i16>* %result
  ret void
}

define void @truncStore.v4i8(<4 x i32> %a, <4 x i8>* %result) {
; CHECK-LABEL: truncStore.v4i8:
; CHECK:      xtn [[TMP:(v[0-9]+)]].4h, v{{[0-9]+}}.4s
; CHECK-NEXT: xtn [[TMP2:(v[0-9]+)]].8b, [[TMP]].8h
; CHECK-NEXT: str s{{[0-9]+}}, [x{{[0-9]+}}]
  %b = trunc <4 x i32> %a to <4 x i8>
  store <4 x i8> %b, <4 x i8>* %result
  ret void
}

define void @truncStore.v8i16(<8 x i16> %a, <8 x i8>* %result) {
; CHECK-LABEL: truncStore.v8i16:
; CHECK: xtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
; CHECK: {{st1 { v[0-9]+.8b }|str d[0-9]+}}, [x{{[0-9]+|sp}}]
  %b = trunc <8 x i16> %a to <8 x i8>
  store <8 x i8> %b, <8 x i8>* %result
  ret void
}
