; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -mattr=+ctselect -verify-machineinstrs | FileCheck --check-prefix=TEST-CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

define i8 @ct_int8(i1 %cond, i8 %a, i8 %b) {
; CT-LABEL: ct_int8:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; TEST-CT: and
; TEST-CT: sub
; TEST-CT: rsb
; TEST-CT-NEXT: and
; TEST-CT-NEXT: and
; TEST-CT-NEXT: orr
; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i8 @llvm.ct.select.i8(i1 %cond, i8 %a, i8 %b)
  ret i8 %sel
}

define i16 @ct_int16(i1 %cond, i16 %a, i16 %b) {
; CT-LABEL: ct_int16:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; TEST-CT: and
; TEST-CT: sub
; TEST-CT: rsb
; TEST-CT-NEXT: and
; TEST-CT-NEXT: and
; TEST-CT-NEXT: orr
; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i16 @llvm.ct.select.i16(i1 %cond, i16 %a, i16 %b)
  ret i16 %sel
}

define i32 @ct_int32(i1 %cond, i32 %a, i32 %b) {
; CT-LABEL: ct_int32:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; TEST-CT: and
; TEST-CT: sub
; TEST-CT: rsb
; TEST-CT-NEXT: and
; TEST-CT-NEXT: and
; TEST-CT-NEXT: orr
; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i32 @llvm.ct.select.i32(i1 %cond, i32 %a, i32 %b)
  ret i32 %sel
}

define i64 @ct_int64(i1 %cond, i64 %a, i64 %b) {
; CT-LABEL: ct_int64:
; CT: sub
; CT: rsb
; CT: and
; CT: and
; CT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i64 @llvm.ct.select.i64(i1 %cond, i64 %a, i64 %b)
  ret i64 %sel
}

define float @ct_float(i1 %cond, float %a, float %b) {
; CT-LABEL: ct_float:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr}}
entry:
  %sel = call float @llvm.ct.select.f32(i1 %cond, float %a, float %b)
  ret float %sel
}

define double @ct_f64(i1 %cond, double %a, double %b) {
; CT-LABEL: ct_f64:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr|vldr}}
entry:
  %sel = call double @llvm.ct.select.f64(i1 %cond, double %a, double %b)
  ret double %sel
}

define <8 x i8> @ct_v8i8(i1 %cond, <8 x i8> %a, <8 x i8> %b) {
; CT-LABEL: ct_v8i8:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr|vldr}}
entry:
  %sel = call <8 x i8> @llvm.ct.select.v8i8(i1 %cond, <8 x i8> %a, <8 x i8> %b)
  ret <8 x i8> %sel
}

define <4 x i16> @ct_v4i16(i1 %cond, <4 x i16> %a, <4 x i16> %b) {
; CT-LABEL: ct_v4i16:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr|vldr}}
entry:
  %sel = call <4 x i16> @llvm.ct.select.v4i16(i1 %cond, <4 x i16> %a, <4 x i16> %b)
  ret <4 x i16> %sel
}

; Technically this should be handled the exact same as double.
define <2 x i32> @ct_v2i32(i1 %cond, <2 x i32> %a, <2 x i32> %b) {
; CT-LABEL: ct_v2i32:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr|vldr}}
entry:
  %sel = call <2 x i32> @llvm.ct.select.v2i32(i1 %cond, <2 x i32> %a, <2 x i32> %b)
  ret <2 x i32> %sel
}

define <2 x float> @ct_v2f32(i1 %cond, <2 x float> %a, <2 x float> %b) {
; CT-LABEL: ct_v2f32:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr|vldr}}
entry:
  %sel = call <2 x float> @llvm.ct.select.v2f32(i1 %cond, <2 x float> %a, <2 x float> %b)
  ret <2 x float> %sel
}

define <4 x float> @ct_v4f32(i1 %cond, <4 x float> %a, <4 x float> %b) {
; CT-LABEL: ct_v4f32:
; CT: vand
; CT: vldr
; CT: vneg
; CT: vbsl
; CT-NOT: ldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; TEST-CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; TEST-CT-NOT: j
; TEST-CT-NOT: mov

; DEFAULT: {{mov|ldr|vldr}}
entry:
  %sel = call <4 x float> @llvm.ct.select.v4f32(i1 %cond, <4 x float> %a, <4 x float> %b)
  ret <4 x float> %sel
}