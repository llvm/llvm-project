; RUN: llc -mtriple=aarch64-apple-darwin                                                   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=2 -verify-machineinstrs < %s | FileCheck %s

; Vector Integer Add
define <8 x i8> @add_v8i8_rr(<8 x i8> %a, <8 x i8> %b) {
; CHECK: add_v8i8_rr
; CHECK: add.8b v0, v0, v1
  %1 = add <8 x i8> %a, %b
  ret <8 x i8> %1
}

define <16 x i8> @add_v16i8_rr(<16 x i8> %a, <16 x i8> %b) {
; CHECK: add_v16i8_rr
; CHECK: add.16b v0, v0, v1
  %1 = add <16 x i8> %a, %b
  ret <16 x i8> %1
}

define <4 x i16> @add_v4i16_rr(<4 x i16> %a, <4 x i16> %b) {
; CHECK: add_v4i16_rr
; CHECK: add.4h v0, v0, v1
  %1 = add <4 x i16> %a, %b
  ret <4 x i16> %1
}

define <8 x i16> @add_v8i16_rr(<8 x i16> %a, <8 x i16> %b) {
; CHECK: add_v8i16_rr
; CHECK: add.8h v0, v0, v1
  %1 = add <8 x i16> %a, %b
  ret <8 x i16> %1
}

define <2 x i32> @add_v2i32_rr(<2 x i32> %a, <2 x i32> %b) {
; CHECK: add_v2i32_rr
; CHECK: add.2s v0, v0, v1
  %1 = add <2 x i32> %a, %b
  ret <2 x i32> %1
}

define <4 x i32> @add_v4i32_rr(<4 x i32> %a, <4 x i32> %b) {
; CHECK: add_v4i32_rr
; CHECK: add.4s v0, v0, v1
  %1 = add <4 x i32> %a, %b
  ret <4 x i32> %1
}

define <2 x i64> @add_v2i64_rr(<2 x i64> %a, <2 x i64> %b) {
; CHECK: add_v2i64_rr
; CHECK: add.2d v0, v0, v1
  %1 = add <2 x i64> %a, %b
  ret <2 x i64> %1
}

; Vector Floating-point Add
define <2 x float> @add_v2f32_rr(<2 x float> %a, <2 x float> %b) {
; CHECK: add_v2f32_rr
; CHECK: fadd.2s v0, v0, v1
  %1 = fadd <2 x float> %a, %b
  ret <2 x float> %1
}

define <4 x float> @add_v4f32_rr(<4 x float> %a, <4 x float> %b) {
; CHECK: add_v4f32_rr
; CHECK: fadd.4s v0, v0, v1
  %1 = fadd <4 x float> %a, %b
  ret <4 x float> %1
}

define <2 x double> @add_v2f64_rr(<2 x double> %a, <2 x double> %b) {
; CHECK: add_v2f64_rr
; CHECK: fadd.2d v0, v0, v1
  %1 = fadd <2 x double> %a, %b
  ret <2 x double> %1
}
