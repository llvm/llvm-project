; Test vector truncating stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8->v16i1 truncation.
define void @f1(<16 x i8> %val, <16 x i1> *%ptr) {
; No expected output, but must compile.
  %trunc = trunc <16 x i8> %val to <16 x i1>
  store <16 x i1> %trunc, <16 x i1> *%ptr
  ret void
}

; Test a v8i16->v8i1 truncation.
define void @f2(<8 x i16> %val, <8 x i1> *%ptr) {
; No expected output, but must compile.
  %trunc = trunc <8 x i16> %val to <8 x i1>
  store <8 x i1> %trunc, <8 x i1> *%ptr
  ret void
}

; Test a v8i16->v8i8 truncation.
define void @f3(<8 x i16> %val, <8 x i8> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vpkh [[REG1:%v[0-9]+]], %v24, %v24
; CHECK: vsteg [[REG1]], 0(%r2)
; CHECK: br %r14
  %trunc = trunc <8 x i16> %val to <8 x i8>
  store <8 x i8> %trunc, <8 x i8> *%ptr
  ret void
}

; Test a v4i32->v4i1 truncation.
define void @f4(<4 x i32> %val, <4 x i1> *%ptr) {
; No expected output, but must compile.
  %trunc = trunc <4 x i32> %val to <4 x i1>
  store <4 x i1> %trunc, <4 x i1> *%ptr
  ret void
}

; Test a v4i32->v4i8 truncation.  At the moment we use a VPERM rather than
; a chain of packs.
define void @f5(<4 x i32> %val, <4 x i8> *%ptr) {
; CHECK-LABEL: f5:
; CHECK: vperm [[REG:%v[0-9]+]],
; CHECK: vstef [[REG]], 0(%r2)
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i8>
  store <4 x i8> %trunc, <4 x i8> *%ptr
  ret void
}

; Test a v4i32->v4i16 truncation.
define void @f6(<4 x i32> %val, <4 x i16> *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vpkf [[REG1:%v[0-9]+]], %v24, %v24
; CHECK: vsteg [[REG1]], 0(%r2)
; CHECK: br %r14
  %trunc = trunc <4 x i32> %val to <4 x i16>
  store <4 x i16> %trunc, <4 x i16> *%ptr
  ret void
}

; Test a v2i64->v2i1 truncation.
define void @f7(<2 x i64> %val, <2 x i1> *%ptr) {
; CHECK-LABEL: f7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vlgvg %r0, %v24, 0
; CHECK-NEXT:    vlgvg %r1, %v24, 1
; CHECK-NEXT:    risbgn %r0, %r1, 32, 62, 1
; CHECK-NEXT:    nilf %r0, 3
; CHECK-NEXT:    stc %r0, 0(%r2)
; CHECK-NEXT:    br %r14
  %trunc = trunc <2 x i64> %val to <2 x i1>
  store <2 x i1> %trunc, <2 x i1> *%ptr
  ret void
}

; Test a v2i64->v2i8 truncation.  At the moment we use a VPERM rather than
; a chain of packs.
define void @f8(<2 x i64> %val, <2 x i8> *%ptr) {
; CHECK-LABEL: f8:
; CHECK: vperm [[REG:%v[0-9]+]],
; CHECK: vsteh [[REG]], 0(%r2)
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i8>
  store <2 x i8> %trunc, <2 x i8> *%ptr
  ret void
}

; Test a v2i64->v2i16 truncation.  At the moment we use a VPERM rather than
; a chain of packs.
define void @f9(<2 x i64> %val, <2 x i16> *%ptr) {
; CHECK-LABEL: f9:
; CHECK: vperm [[REG:%v[0-9]+]],
; CHECK: vstef [[REG]], 0(%r2)
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i16>
  store <2 x i16> %trunc, <2 x i16> *%ptr
  ret void
}

; Test a v2i64->v2i32 truncation.
define void @f10(<2 x i64> %val, <2 x i32> *%ptr) {
; CHECK-LABEL: f10:
; CHECK: vpkg [[REG1:%v[0-9]+]], %v24, %v24
; CHECK: vsteg [[REG1]], 0(%r2)
; CHECK: br %r14
  %trunc = trunc <2 x i64> %val to <2 x i32>
  store <2 x i32> %trunc, <2 x i32> *%ptr
  ret void
}
