; RUN: llc < %s -mtriple=armv8-linux-gnu -mattr=+neon | FileCheck %s

; CHECK-LABEL: t1
; CHECK: vmax.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <4 x i32> @t1(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp sgt <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t2
; CHECK: vmin.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <4 x i32> @t2(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp slt <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t3
; CHECK: vmax.u32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <4 x i32> @t3(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp ugt <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t4
; CHECK: vmin.u32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <4 x i32> @t4(<4 x i32> %a, <4 x i32> %b) {
  %t1 = icmp ult <4 x i32> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %t2
}

; CHECK-LABEL: t5
; CHECK: vmax.s32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <2 x i32> @t5(<2 x i32> %a, <2 x i32> %b) {
  %t1 = icmp sgt <2 x i32> %a, %b
  %t2 = select <2 x i1> %t1, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %t2
}

; CHECK-LABEL: t6
; CHECK: vmin.s32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <2 x i32> @t6(<2 x i32> %a, <2 x i32> %b) {
  %t1 = icmp slt <2 x i32> %a, %b
  %t2 = select <2 x i1> %t1, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %t2
}

; CHECK-LABEL: t7
; CHECK: vmax.u32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <2 x i32> @t7(<2 x i32> %a, <2 x i32> %b) {
  %t1 = icmp ugt <2 x i32> %a, %b
  %t2 = select <2 x i1> %t1, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %t2
}

; CHECK-LABEL: t8
; CHECK: vmin.u32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <2 x i32> @t8(<2 x i32> %a, <2 x i32> %b) {
  %t1 = icmp ult <2 x i32> %a, %b
  %t2 = select <2 x i1> %t1, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %t2
}

; CHECK-LABEL: t9
; CHECK: vmax.s16 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <8 x i16> @t9(<8 x i16> %a, <8 x i16> %b) {
  %t1 = icmp sgt <8 x i16> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %t2
}

; CHECK-LABEL: t10
; CHECK: vmin.s16 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <8 x i16> @t10(<8 x i16> %a, <8 x i16> %b) {
  %t1 = icmp slt <8 x i16> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %t2
}

; CHECK-LABEL: t11
; CHECK: vmax.u16 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <8 x i16> @t11(<8 x i16> %a, <8 x i16> %b) {
  %t1 = icmp ugt <8 x i16> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %t2
}

; CHECK-LABEL: t12
; CHECK: vmin.u16 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <8 x i16> @t12(<8 x i16> %a, <8 x i16> %b) {
  %t1 = icmp ult <8 x i16> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %t2
}

; CHECK-LABEL: t13
; CHECK: vmax.s16
define <4 x i16> @t13(<4 x i16> %a, <4 x i16> %b) {
  %t1 = icmp sgt <4 x i16> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i16> %a, <4 x i16> %b
  ret <4 x i16> %t2
}

; CHECK-LABEL: t14
; CHECK: vmin.s16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <4 x i16> @t14(<4 x i16> %a, <4 x i16> %b) {
  %t1 = icmp slt <4 x i16> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i16> %a, <4 x i16> %b
  ret <4 x i16> %t2
}

; CHECK-LABEL: t15
; CHECK: vmax.u16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <4 x i16> @t15(<4 x i16> %a, <4 x i16> %b) {
  %t1 = icmp ugt <4 x i16> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i16> %a, <4 x i16> %b
  ret <4 x i16> %t2
}

; CHECK-LABEL: t16
; CHECK: vmin.u16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <4 x i16> @t16(<4 x i16> %a, <4 x i16> %b) {
  %t1 = icmp ult <4 x i16> %a, %b
  %t2 = select <4 x i1> %t1, <4 x i16> %a, <4 x i16> %b
  ret <4 x i16> %t2
}

; CHECK-LABEL: t17
; CHECK: vmax.s8 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <16 x i8> @t17(<16 x i8> %a, <16 x i8> %b) {
  %t1 = icmp sgt <16 x i8> %a, %b
  %t2 = select <16 x i1> %t1, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %t2
}

; CHECK-LABEL: t18
; CHECK: vmin.s8 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <16 x i8> @t18(<16 x i8> %a, <16 x i8> %b) {
  %t1 = icmp slt <16 x i8> %a, %b
  %t2 = select <16 x i1> %t1, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %t2
}

; CHECK-LABEL: t19
; CHECK: vmax.u8 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <16 x i8> @t19(<16 x i8> %a, <16 x i8> %b) {
  %t1 = icmp ugt <16 x i8> %a, %b
  %t2 = select <16 x i1> %t1, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %t2
}

; CHECK-LABEL: t20
; CHECK: vmin.u8 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
define <16 x i8> @t20(<16 x i8> %a, <16 x i8> %b) {
  %t1 = icmp ult <16 x i8> %a, %b
  %t2 = select <16 x i1> %t1, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %t2
}

; CHECK-LABEL: t21
; CHECK: vmax.s8 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <8 x i8> @t21(<8 x i8> %a, <8 x i8> %b) {
  %t1 = icmp sgt <8 x i8> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i8> %a, <8 x i8> %b
  ret <8 x i8> %t2
}

; CHECK-LABEL: t22
; CHECK: vmin.s8 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <8 x i8> @t22(<8 x i8> %a, <8 x i8> %b) {
  %t1 = icmp slt <8 x i8> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i8> %a, <8 x i8> %b
  ret <8 x i8> %t2
}

; CHECK-LABEL: t23
; CHECK: vmax.u8 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <8 x i8> @t23(<8 x i8> %a, <8 x i8> %b) {
  %t1 = icmp ugt <8 x i8> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i8> %a, <8 x i8> %b
  ret <8 x i8> %t2
}

; CHECK-LABEL: t24
; CHECK: vmin.u8 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
define <8 x i8> @t24(<8 x i8> %a, <8 x i8> %b) {
  %t1 = icmp ult <8 x i8> %a, %b
  %t2 = select <8 x i1> %t1, <8 x i8> %a, <8 x i8> %b
  ret <8 x i8> %t2
}
