; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;; Splats of legal integer vector types

define <vscale x 16 x i8> @sve_splat_16xi8(i8 %val) {
; CHECK-LABEL: @sve_splat_16xi8
; CHECK: mov z0.b, w0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 16 x i8> undef, i8 %val, i32 0
  %splat = shufflevector <vscale x 16 x i8> %ins, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  ret <vscale x 16 x i8> %splat
}

define <vscale x 8 x i16> @sve_splat_8xi16(i16 %val) {
; CHECK-LABEL: @sve_splat_8xi16
; CHECK: mov z0.h, w0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 8 x i16> undef, i16 %val, i32 0
  %splat = shufflevector <vscale x 8 x i16> %ins, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  ret <vscale x 8 x i16> %splat
}

define <vscale x 4 x i32> @sve_splat_4xi32(i32 %val) {
; CHECK-LABEL: @sve_splat_4xi32
; CHECK: mov z0.s, w0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 4 x i32> undef, i32 %val, i32 0
  %splat = shufflevector <vscale x 4 x i32> %ins, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %splat
}

define <vscale x 2 x i64> @sve_splat_2xi64(i64 %val) {
; CHECK-LABEL: @sve_splat_2xi64
; CHECK: mov z0.d, x0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 2 x i64> undef, i64 %val, i32 0
  %splat = shufflevector <vscale x 2 x i64> %ins, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i64> %splat
}

;; Promote splats of smaller illegal integer vector types

define <vscale x 2 x i8> @sve_splat_2xi8(i8 %val) {
; CHECK-LABEL: @sve_splat_2xi8
; CHECK: mov z0.d, x0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 2 x i8> undef, i8 %val, i32 0
  %splat = shufflevector <vscale x 2 x i8> %ins, <vscale x 2 x i8> undef, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i8> %splat
}

define <vscale x 4 x i8> @sve_splat_4xi8(i8 %val) {
; CHECK-LABEL: @sve_splat_4xi8
; CHECK: mov z0.s, w0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 4 x i8> undef, i8 %val, i32 0
  %splat = shufflevector <vscale x 4 x i8> %ins, <vscale x 4 x i8> undef, <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i8> %splat
}

define <vscale x 8 x i8> @sve_splat_8xi8(i8 %val) {
; CHECK-LABEL: @sve_splat_8xi8
; CHECK: mov z0.h, w0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 8 x i8> undef, i8 %val, i32 0
  %splat = shufflevector <vscale x 8 x i8> %ins, <vscale x 8 x i8> undef, <vscale x 8 x i32> zeroinitializer
  ret <vscale x 8 x i8> %splat
}

define <vscale x 2 x i16> @sve_splat_2xi16(i16 %val) {
; CHECK-LABEL: @sve_splat_2xi16
; CHECK: mov z0.d, x0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 2 x i16> undef, i16 %val, i32 0
  %splat = shufflevector <vscale x 2 x i16> %ins, <vscale x 2 x i16> undef, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i16> %splat
}

define <vscale x 4 x i16> @sve_splat_4xi16(i16 %val) {
; CHECK-LABEL: @sve_splat_4xi16
; CHECK: mov z0.s, w0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 4 x i16> undef, i16 %val, i32 0
  %splat = shufflevector <vscale x 4 x i16> %ins, <vscale x 4 x i16> undef, <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i16> %splat
}

define <vscale x 2 x i32> @sve_splat_2xi32(i32 %val) {
; CHECK-LABEL: @sve_splat_2xi32
; CHECK: mov z0.d, x0
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 2 x i32> undef, i32 %val, i32 0
  %splat = shufflevector <vscale x 2 x i32> %ins, <vscale x 2 x i32> undef, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i32> %splat
}

define <vscale x 2 x i1> @sve_splat_2xi1(i1 %val) {
; CHECK-LABEL: @sve_splat_2xi1
; CHECK: sbfx x8, x0, #0, #1
; CHECK-NEXT: whilelo p0.d, xzr, x8
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 2 x i1> undef, i1 %val, i32 0
  %splat = shufflevector <vscale x 2 x i1> %ins, <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i1> %splat
}

define <vscale x 4 x i1> @sve_splat_4xi1(i1 %val) {
; CHECK-LABEL: @sve_splat_4xi1
; CHECK: sbfx x8, x0, #0, #1
; CHECK-NEXT: whilelo p0.s, xzr, x8
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 4 x i1> undef, i1 %val, i32 0
  %splat = shufflevector <vscale x 4 x i1> %ins, <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i1> %splat
}

define <vscale x 8 x i1> @sve_splat_8xi1(i1 %val) {
; CHECK-LABEL: @sve_splat_8xi1
; CHECK: sbfx x8, x0, #0, #1
; CHECK-NEXT: whilelo p0.h, xzr, x8
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 8 x i1> undef, i1 %val, i32 0
  %splat = shufflevector <vscale x 8 x i1> %ins, <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
  ret <vscale x 8 x i1> %splat
}

define <vscale x 16 x i1> @sve_splat_16xi1(i1 %val) {
; CHECK-LABEL: @sve_splat_16xi1
; CHECK: sbfx x8, x0, #0, #1
; CHECK-NEXT: whilelo p0.b, xzr, x8
; CHECK-NEXT: ret
  %ins = insertelement <vscale x 16 x i1> undef, i1 %val, i32 0
  %splat = shufflevector <vscale x 16 x i1> %ins, <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
  ret <vscale x 16 x i1> %splat
}
