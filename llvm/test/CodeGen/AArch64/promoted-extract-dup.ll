; RUN: llc -O2 -mtriple=aarch64-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O2 -mtriple=aarch64_be-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; A promoted EXTRACT_VECTOR_ELT has undefined high bits. When AArch64 lowers it
; into a scalar-fed DUP with wider lanes, choose sign-extension so all-ones
; masks remain all-ones after widening. Preserve explicit extension semantics.

define <4 x i1> @splat_i1_extract(<16 x i1> %v) {
; CHECK-LABEL: splat_i1_extract:
; CHECK:       smov w[[EXT:[0-9]+]], v{{[0-9]+}}.b[1]
; CHECK-NEXT:  dup v{{[0-9]+}}.4h, w[[EXT]]
entry:
  %elt = extractelement <16 x i1> %v, i64 1
  %ins = insertelement <4 x i1> poison, i1 %elt, i64 0
  %splat = shufflevector <4 x i1> %ins, <4 x i1> poison, <4 x i32> zeroinitializer
  ret <4 x i1> %splat
}

define <4 x i16> @splat_zext_i8_extract(<8 x i8> %v) {
; CHECK-LABEL: splat_zext_i8_extract:
; CHECK:       umov w[[EXT:[0-9]+]], v{{[0-9]+}}.b[1]
; CHECK-NEXT:  dup v{{[0-9]+}}.4h, w[[EXT]]
entry:
  %elt = extractelement <8 x i8> %v, i64 1
  %ext = zext i8 %elt to i16
  %ins = insertelement <4 x i16> poison, i16 %ext, i64 0
  %splat = shufflevector <4 x i16> %ins, <4 x i16> poison, <4 x i32> zeroinitializer
  ret <4 x i16> %splat
}

define <4 x i16> @splat_sext_i8_extract(<8 x i8> %v) {
; CHECK-LABEL: splat_sext_i8_extract:
; CHECK:       smov w[[EXT:[0-9]+]], v{{[0-9]+}}.b[1]
; CHECK-NEXT:  dup v{{[0-9]+}}.4h, w[[EXT]]
entry:
  %elt = extractelement <8 x i8> %v, i64 1
  %ext = sext i8 %elt to i16
  %ins = insertelement <4 x i16> poison, i16 %ext, i64 0
  %splat = shufflevector <4 x i16> %ins, <4 x i16> poison, <4 x i32> zeroinitializer
  ret <4 x i16> %splat
}
