; RUN: llc -mtriple=arm64-linux-gnuabi < %s | FileCheck %s

; The following tests is to check the correctness of reversing input operand 
; of vext by enumerating all cases of using two undefs in shuffle masks.

define <4 x i16> @vext_6701_0(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_0:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #4
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i16> %x
}

define <4 x i16> @vext_6701_12(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_12:
; CHECK: dup v0.2s, v0.s[0]
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  ret <4 x i16> %x
}

define <4 x i16> @vext_6701_13(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_13:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #4
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 7, i32 undef, i32 1>
  ret <4 x i16> %x
}

define <4 x i16> @vext_6701_14(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_14:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #4
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 7, i32 0, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_6701_23(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_23:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #4
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 6, i32 undef, i32 undef, i32 1>
  ret <4 x i16> %x
}

define <4 x i16> @vext_6701_24(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_24:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #4
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 6, i32 undef, i32 0, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_6701_34(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_6701_34:
; CHECK: dup  v0.2s, v1.s[1]
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 6, i32 7, i32 undef, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_0(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_0:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #2
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 5, i32 6, i32 7, i32 0>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_12(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_12:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #2
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 undef, i32 7, i32 0>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_13(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_13:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #2
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 6, i32 undef, i32 0>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_14(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_14:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #2
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 6, i32 7, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_23(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_23:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #2
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 5, i32 undef, i32 undef, i32 0>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_24(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_24:
; CHECK: rev32   v0.4h, v1.4h
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 5, i32 undef, i32 7, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_5670_34(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_5670_34:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #2
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 5, i32 6, i32 undef, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_0(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_0:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #6
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 7, i32 0, i32 1, i32 2>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_12(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_12:
; CHECK: ext	v0.8b, v0.8b, v0.8b, #6
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 undef, i32 1, i32 2>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_13(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_13:
; CHECK: rev32   v0.4h, v0.4h
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 0, i32 undef, i32 2>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_14(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_14:
; CHECK: ext	v0.8b, v0.8b, v0.8b, #6
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 undef, i32 0, i32 1, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_23(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_23:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #6
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 7, i32 undef, i32 undef, i32 2>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_24(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_24:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #6
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 7, i32 undef, i32 1, i32 undef>
  ret <4 x i16> %x
}

define <4 x i16> @vext_7012_34(<4 x i16> %a1, <4 x i16> %a2) {
entry:
; CHECK-LABEL: vext_7012_34:
; CHECK: ext	v0.8b, v1.8b, v0.8b, #6
  %x = shufflevector <4 x i16> %a1, <4 x i16> %a2, <4 x i32> <i32 7, i32 0, i32 undef, i32 undef>
  ret <4 x i16> %x
}
