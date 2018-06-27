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

; A vector LoadExt can not be selected.
; Test a vector load IR and a sext/zext IR can be selected correctly.
define <4 x i32> @loadSExt.v4i8(<4 x i8>* %ref) {
; CHECK-LABEL: loadSExt.v4i8:
; CHECK: ldrsb
  %a = load <4 x i8>, <4 x i8>* %ref
  %conv = sext <4 x i8> %a to <4 x i32>
  ret <4 x i32> %conv
}

define <4 x i32> @loadZExt.v4i8(<4 x i8>* %ref) {
; CHECK-LABEL: loadZExt.v4i8:
; CHECK: ldrb
  %a = load <4 x i8>, <4 x i8>* %ref
  %conv = zext <4 x i8> %a to <4 x i32>
  ret <4 x i32> %conv
}

define i32 @loadExt.i32(<4 x i8>* %ref) {
; CHECK-LABEL: loadExt.i32:
; CHECK: ldrb
  %a = load <4 x i8>, <4 x i8>* %ref
  %vecext = extractelement <4 x i8> %a, i32 0
  %conv = zext i8 %vecext to i32
  ret i32 %conv
}
