; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-eabi | FileCheck %s --check-prefixes=DEFAULT,NOFP16
; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-eabi -mattr=+fullfp16 | FileCheck %s --check-prefixes=DEFAULT,FP16

define i1 @ct_i1(i1 %cond, i1 %a, i1 %b) {
; DEFAULT-LABEL: ct_i1:
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call i1 @llvm.ct.select.i1(i1 %cond, i1 %a, i1 %b)
  ret i1 %1
}

define i8 @ct_i8(i1 %cond, i8 %a, i8 %b) {
; DEFAULT-LABEL: ct_i8:
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call i8 @llvm.ct.select.i8(i1 %cond, i8 %a, i8 %b)
  ret i8 %1
}

define i16 @ct_i16(i1 %cond, i16 %a, i16 %b) {
; DEFAULT-LABEL: ct_i16:
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call i16 @llvm.ct.select.i16(i1 %cond, i16 %a, i16 %b)
  ret i16 %1
}

define i32 @ct_i32(i1 %cond, i32 %a, i32 %b) {
; DEFAULT-LABEL: ct_i32:
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call i32 @llvm.ct.select.i32(i1 %cond, i32 %a, i32 %b)
  ret i32 %1
}

define i64 @ct_i64(i1 %cond, i64 %a, i64 %b) {
; DEFAULT-LABEL: ct_i64:
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call i64 @llvm.ct.select.i64(i1 %cond, i64 %a, i64 %b)
  ret i64 %1
}

define i128 @ct_i128(i1 %cond, i128 %a, i128 %b) {
; DEFAULT-LABEL: ct_i128:
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call i128 @llvm.ct.select.i128(i1 %cond, i128 %a, i128 %b)
  ret i128 %1
}

define half @ct_f16(i1 %cond, half %a, half %b) {
; DEFAULT-LABEL: ct_f16:
; NOFP16: fcvt
; NOFP16: csel
; FP16: fcsel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
; NOFP16: fcvt
  %1 = call half @llvm.ct.select.f16(i1 %cond, half %a, half %b)
  ret half %1
}

define float @ct_f32(i1 %cond, float %a, float %b) {
; DEFAULT-LABEL: ct_f32:
; DEFAULT: fcsel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call float @llvm.ct.select.f32(i1 %cond, float %a, float %b)
  ret float %1
}

define double @ct_f64(i1 %cond, double %a, double %b) {
; DEFAULT-LABEL: ct_f64:
; DEFAULT: fcsel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{mov|ldr}}
  %1 = call double @llvm.ct.select.f64(i1 %cond, double %a, double %b)
  ret double %1
}

define <4 x i32> @ct_v4i32(i1 %cond, <4 x i32> %a, <4 x i32> %b) {
; DEFAULT-LABEL: ct_v4i32:
; DEFAULT: csel
; DEFAULT: csel
; DEFAULT: csel
; DEFAULT: csel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{ldr}}
  %1 = call <4 x i32> @llvm.ct.select.v4i32(i1 %cond, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %1
}

define <4 x float> @ct_v4f32(i1 %cond, <4 x float> %a, <4 x float> %b) {
; DEFAULT-LABEL: ct_v4f32:
; DEFAULT: fcsel
; DEFAULT: fcsel
; DEFAULT: fcsel
; DEFAULT: fcsel
; DEFAULT-NOT: b{{eq|ne}}
; DEFAULT-NOT: j
; DEFAULT-NOT: {{ldr}}
  %1 = call <4 x float> @llvm.ct.select.v4f32(i1 %cond, <4 x float> %a, <4 x float> %b)
  ret <4 x float> %1
}