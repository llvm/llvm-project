; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=fuse-address | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m3     | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m4     | FileCheck %s

target triple = "aarch64-unknown"

@var_8bit = global i8 0
@var_16bit = global i16 0
@var_32bit = global i32 0
@var_64bit = global i64 0
@var_128bit = global i128 0
@var_half = global half 0.0
@var_float = global float 0.0
@var_double = global double 0.0
@var_double2 = global <2 x double> <double 0.0, double 0.0>

define void @ldst_8bit() {
  %val8 = load volatile i8, i8* @var_8bit
  %ext = zext i8 %val8 to i64
  %add = add i64 %ext, 1
  %val16 = trunc i64 %add to i16
  store volatile i16 %val16, i16* @var_16bit
  ret void

; CHECK-LABEL: ldst_8bit:
; CHECK: adrp [[RB:x[0-9]+]], var_8bit
; CHECK-NEXT: ldrb {{w[0-9]+}}, {{\[}}[[RB]], {{#?}}:lo12:var_8bit{{\]}}
; CHECK: adrp [[RH:x[0-9]+]], var_16bit
; CHECK-NEXT: strh {{w[0-9]+}}, {{\[}}[[RH]], {{#?}}:lo12:var_16bit{{\]}}
}

define void @ldst_16bit() {
  %val16 = load volatile i16, i16* @var_16bit
  %ext = zext i16 %val16 to i64
  %add = add i64 %ext, 1
  %val32 = trunc i64 %add to i32
  store volatile i32 %val32, i32* @var_32bit
  ret void

; CHECK-LABEL: ldst_16bit:
; CHECK: adrp [[RH:x[0-9]+]], var_16bit
; CHECK-NEXT: ldrh {{w[0-9]+}}, {{\[}}[[RH]], {{#?}}:lo12:var_16bit{{\]}}
; CHECK: adrp [[RW:x[0-9]+]], var_32bit
; CHECK-NEXT: str {{w[0-9]+}}, {{\[}}[[RW]], {{#?}}:lo12:var_32bit{{\]}}
}

define void @ldst_32bit() {
  %val32 = load volatile i32, i32* @var_32bit
  %ext = zext i32 %val32 to i64
  %val64 = add i64 %ext, 1
  store volatile i64 %val64, i64* @var_64bit
  ret void

; CHECK-LABEL: ldst_32bit:
; CHECK: adrp [[RW:x[0-9]+]], var_32bit
; CHECK-NEXT: ldr {{w[0-9]+}}, {{\[}}[[RW]], {{#?}}:lo12:var_32bit{{\]}}
; CHECK: adrp [[RL:x[0-9]+]], var_64bit
; CHECK-NEXT: str {{x[0-9]+}}, {{\[}}[[RL]], {{#?}}:lo12:var_64bit{{\]}}
}

define void @ldst_64bit() {
  %val64 = load volatile i64, i64* @var_64bit
  %ext = zext i64 %val64 to i128
  %val128 = add i128 %ext, 1
  store volatile i128 %val128, i128* @var_128bit
  ret void

; CHECK-LABEL: ldst_64bit:
; CHECK: adrp [[RL:x[0-9]+]], var_64bit
; CHECK-NEXT: ldr {{x[0-9]+}}, {{\[}}[[RL]], {{#?}}:lo12:var_64bit{{\]}}
; CHECK: adrp [[RQ:x[0-9]+]], var_128bit
; CHECK-NEXT: add {{x[0-9]+}}, [[RQ]], {{#?}}:lo12:var_128bit
}

define void @ldst_half() {
  %valh = load volatile half, half* @var_half
  %valf = fpext half %valh to float
  store volatile float %valf, float* @var_float
  ret void

; CHECK-LABEL: ldst_half:
; CHECK: adrp [[RH:x[0-9]+]], var_half
; CHECK-NEXT: ldr {{h[0-9]+}}, {{\[}}[[RH]], {{#?}}:lo12:var_half{{\]}}
; CHECK: adrp [[RF:x[0-9]+]], var_float
; CHECK-NEXT: str {{s[0-9]+}}, {{\[}}[[RF]], {{#?}}:lo12:var_float{{\]}}
}

define void @ldst_float() {
  %valf = load volatile float, float* @var_float
  %vald = fpext float %valf to double
  store volatile double %vald, double* @var_double
  ret void

; CHECK-LABEL: ldst_float:
; CHECK: adrp [[RF:x[0-9]+]], var_float
; CHECK-NEXT: ldr {{s[0-9]+}}, {{\[}}[[RF]], {{#?}}:lo12:var_float{{\]}}
; CHECK: adrp [[RD:x[0-9]+]], var_double
; CHECK-NEXT: str {{d[0-9]+}}, {{\[}}[[RD]], {{#?}}:lo12:var_double{{\]}}
}

define void @ldst_double() {
  %valf = load volatile float, float* @var_float
  %vale = fpext float %valf to double
  %vald = load volatile double, double* @var_double
  %vald1 = insertelement <2 x double> undef, double %vald, i32 0
  %vald2 = insertelement <2 x double> %vald1, double %vale, i32 1
  store volatile <2 x double> %vald2, <2 x double>* @var_double2
  ret void

; CHECK-LABEL: ldst_double:
; CHECK: adrp [[RD:x[0-9]+]], var_double
; CHECK-NEXT: ldr {{d[0-9]+}}, {{\[}}[[RD]], {{#?}}:lo12:var_double{{\]}}
; CHECK: adrp [[RQ:x[0-9]+]], var_double2
; CHECK-NEXT: str {{q[0-9]+}}, {{\[}}[[RQ]], {{#?}}:lo12:var_double2{{\]}}
}
