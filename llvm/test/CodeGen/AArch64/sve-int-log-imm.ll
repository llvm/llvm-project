; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; SVE Logical Vector Immediate Unpredicated CodeGen
;

; ORR
define <vscale x 16 x i8> @orr_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: orr_i8:
; CHECK: orr z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 15, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res = or <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @orr_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: orr_i16:
; CHECK: orr z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 64519, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = or <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @orr_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: orr_i32:
; CHECK: orr z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 16776960, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = or <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @orr_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: orr_i64:
; CHECK: orr z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 18445618173802708992, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = or <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

; EOR
define <vscale x 16 x i8> @eor_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: eor_i8:
; CHECK: eor z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 15, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res = xor <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @eor_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: eor_i16:
; CHECK: eor z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 64519, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = xor <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @eor_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: eor_i32:
; CHECK: eor z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 16776960, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = xor <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @eor_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: eor_i64:
; CHECK: eor z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 18445618173802708992, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = xor <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

; AND
define <vscale x 16 x i8> @and_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: and_i8:
; CHECK: and z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 16 x i8> undef, i8 15, i32 0
  %splat = shufflevector <vscale x 16 x i8> %elt, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %res = and <vscale x 16 x i8> %a, %splat
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @and_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: and_i16:
; CHECK: and z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 8 x i16> undef, i16 64519, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = and <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @and_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: and_i32:
; CHECK: and z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 4 x i32> undef, i32 16776960, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = and <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @and_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: and_i64:
; CHECK: and z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %elt = insertelement <vscale x 2 x i64> undef, i64 18445618173802708992, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = and <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}
