; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; ASR
;

define <vscale x 16 x i8> @asr_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: asr_i8:
; CHECK: asr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asr_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: asr_i16:
; CHECK: asr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asr_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: asr_i32:
; CHECK: asr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @asr_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_i64:
; CHECK: asr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @asr_wide_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_wide_i8:
; CHECK: asr z0.b, p0/m, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                    <vscale x 16 x i8> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asr_wide_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_wide_i16:
; CHECK: asr z0.h, p0/m, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x i16> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asr_wide_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_wide_i32:
; CHECK: asr z0.s, p0/m, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; ASRD
;

define <vscale x 16 x i8> @asrd_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: asrd_i8:
; CHECK: asrd z0.b, p0/m, z0.b, #1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asrd.nxv16i8(<vscale x 16 x i1> %pg,
                                                                <vscale x 16 x i8> %a,
                                                                i32 1)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asrd_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: asrd_i16:
; CHECK: asrd z0.h, p0/m, z0.h, #2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asrd.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                i32 2)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asrd_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: asrd_i32:
; CHECK: asrd z0.s, p0/m, z0.s, #31
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asrd.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                i32 31)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @asrd_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: asrd_i64:
; CHECK: asrd z0.d, p0/m, z0.d, #64
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.asrd.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                i32 64)
  ret <vscale x 2 x i64> %out
}

;
; INSR
;

define <vscale x 16 x i8> @insr_i8(<vscale x 16 x i8> %a, i8 %b) {
; CHECK-LABEL: insr_i8:
; CHECK: insr z0.b, w0
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.insr.nxv16i8(<vscale x 16 x i8> %a, i8 %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @insr_i16(<vscale x 8 x i16> %a, i16 %b) {
; CHECK-LABEL: insr_i16:
; CHECK: insr z0.h, w0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.insr.nxv8i16(<vscale x 8 x i16> %a, i16 %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @insr_i32(<vscale x 4 x i32> %a, i32 %b) {
; CHECK-LABEL: insr_i32:
; CHECK: insr z0.s, w0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.insr.nxv4i32(<vscale x 4 x i32> %a, i32 %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @insr_i64(<vscale x 2 x i64> %a, i64 %b) {
; CHECK-LABEL: insr_i64:
; CHECK: insr z0.d, x0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.insr.nxv2i64(<vscale x 2 x i64> %a, i64 %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @insr_f16(<vscale x 8 x half> %a, half %b) {
; CHECK-LABEL: insr_f16:
; CHECK: insr z0.h, h1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.insr.nxv8f16(<vscale x 8 x half> %a, half %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @insr_bf16(<vscale x 8 x bfloat> %a, bfloat %b) #0 {
; CHECK-LABEL: insr_bf16:
; CHECK: insr z0.h, h1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.insr.nxv8bf16(<vscale x 8 x bfloat> %a, bfloat %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @insr_f32(<vscale x 4 x float> %a, float %b) {
; CHECK-LABEL: insr_f32:
; CHECK: insr z0.s, s1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.insr.nxv4f32(<vscale x 4 x float> %a, float %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @insr_f64(<vscale x 2 x double> %a, double %b) {
; CHECK-LABEL: insr_f64:
; CHECK: insr z0.d, d1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.insr.nxv2f64(<vscale x 2 x double> %a, double %b)
  ret <vscale x 2 x double> %out
}

;
; LSL
;

define <vscale x 16 x i8> @lsl_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: lsl_i8:
; CHECK: lsl z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsl_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: lsl_i16:
; CHECK: lsl z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsl_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: lsl_i32:
; CHECK: lsl z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @lsl_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_i64:
; CHECK: lsl z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @lsl_wide_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_wide_i8:
; CHECK: lsl z0.b, p0/m, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                    <vscale x 16 x i8> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsl_wide_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_wide_i16:
; CHECK: lsl z0.h, p0/m, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x i16> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsl_wide_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_wide_i32:
; CHECK: lsl z0.s, p0/m, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; LSR
;

define <vscale x 16 x i8> @lsr_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: lsr_i8:
; CHECK: lsr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsr_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: lsr_i16:
; CHECK: lsr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsr_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: lsr_i32:
; CHECK: lsr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @lsr_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_i64:
; CHECK: lsr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @lsr_wide_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_wide_i8:
; CHECK: lsr z0.b, p0/m, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                    <vscale x 16 x i8> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsr_wide_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_wide_i16:
; CHECK: lsr z0.h, p0/m, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x i16> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsr_wide_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_wide_i32:
; CHECK: lsr z0.s, p0/m, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.asrd.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asrd.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asrd.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.asrd.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.insr.nxv16i8(<vscale x 16 x i8>, i8)
declare <vscale x 8 x i16> @llvm.aarch64.sve.insr.nxv8i16(<vscale x 8 x i16>, i16)
declare <vscale x 4 x i32> @llvm.aarch64.sve.insr.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.insr.nxv2i64(<vscale x 2 x i64>, i64)
declare <vscale x 8 x half> @llvm.aarch64.sve.insr.nxv8f16(<vscale x 8 x half>, half)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.insr.nxv8bf16(<vscale x 8 x bfloat>, bfloat)
declare <vscale x 4 x float> @llvm.aarch64.sve.insr.nxv4f32(<vscale x 4 x float>, float)
declare <vscale x 2 x double> @llvm.aarch64.sve.insr.nxv2f64(<vscale x 2 x double>, double)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsl.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsl.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsl.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsr.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsr.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
