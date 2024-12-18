; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; SVE Logical Vector Immediate Unpredicated CodeGen
;

; ORR
define <vscale x 16 x i8> @orr_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: orr_i8:
; CHECK: orr z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %res = or <vscale x 16 x i8> %a, splat(i8 15)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @orr_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: orr_i16:
; CHECK: orr z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %res = or <vscale x 8 x i16> %a, splat(i16 64519)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @orr_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: orr_i32:
; CHECK: orr z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %res = or <vscale x 4 x i32> %a, splat(i32 16776960)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @orr_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: orr_i64:
; CHECK: orr z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %res = or <vscale x 2 x i64> %a, splat(i64 18445618173802708992)
  ret <vscale x 2 x i64> %res
}

; EOR
define <vscale x 16 x i8> @eor_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: eor_i8:
; CHECK: eor z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %res = xor <vscale x 16 x i8> %a, splat(i8 15)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @eor_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: eor_i16:
; CHECK: eor z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %res = xor <vscale x 8 x i16> %a, splat(i16 64519)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @eor_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: eor_i32:
; CHECK: eor z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %res = xor <vscale x 4 x i32> %a, splat(i32 16776960)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @eor_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: eor_i64:
; CHECK: eor z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %res = xor <vscale x 2 x i64> %a, splat(i64 18445618173802708992)
  ret <vscale x 2 x i64> %res
}

; AND
define <vscale x 16 x i8> @and_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: and_i8:
; CHECK: and z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %res = and <vscale x 16 x i8> %a, splat(i8 15)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @and_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: and_i16:
; CHECK: and z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %res = and <vscale x 8 x i16> %a, splat(i16 64519)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @and_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: and_i32:
; CHECK: and z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %res = and <vscale x 4 x i32> %a, splat(i32 16776960)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @and_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: and_i64:
; CHECK: and z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %res = and <vscale x 2 x i64> %a, splat(i64 18445618173802708992)
  ret <vscale x 2 x i64> %res
}
